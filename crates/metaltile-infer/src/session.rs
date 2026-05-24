//! `Session` — the main inference object.
//!
//! Lifecycle:
//! 1. `Session::new(model_dir, toml_src, config, dtype)` — upload weights +
//!    allocate GPU-resident KV cache.
//! 2. `session.step(token_id, temperature)` → `(next_token, gpu_seconds)` —
//!    single forward pass with GPU timing. Drive the autoregressive loop
//!    yourself for production control.
//! 3. `session.generate(prompt, max_tokens, temperature, on_token)` —
//!    convenience wrapper that returns `GenerateOutput` with timing stats.

use std::path::Path;

use metaltile_core::dtype::DType;
use metaltile_model::{
    CompileParams,
    FusionMode,
    KernelRegistry,
    ModelDef,
    PreparedDispatch,
    StateMap,
    WeightMap,
    compile,
    execute_prepared,
};
use metaltile_runtime::{Context, ResidentBuffer};
use rustc_hash::FxHashMap;
use tokenizers::Tokenizer;
use tracing::info;

use crate::{checkpoint::load_weights, config::ModelConfig, error::InferError};

/// Result of a `generate` call with timing breakdown.
#[derive(Debug, Clone)]
pub struct GenerateOutput {
    /// Generated text (decoded output tokens, excluding the prompt).
    pub text: String,
    /// Number of tokens generated.
    pub tokens_generated: usize,
    /// Number of prompt tokens processed during prefill.
    pub prompt_tokens: usize,
    /// Wall-clock time spent on prefill (prompt processing) in seconds.
    pub prefill_secs: f64,
    /// Wall-clock time spent on decode (token generation) in seconds.
    pub decode_secs: f64,
    /// Tokens per second during decode phase (decode only, excludes prefill).
    pub decode_tok_per_sec: f64,
}

/// Single-model inference session. Holds GPU-resident weights + KV cache.
pub struct Session {
    ctx: Context,
    plan: metaltile_model::ExecutionPlan,
    prepared: PreparedDispatch,
    state: StateMap,
    /// Packed scalar state array (6 × 4 bytes = 24 bytes) for the frequently
    /// updated scalars (token_id, position, n_kv, rms_eps, temperature, uniform).
    /// Eliminates per-token HashMap lookups and Vec<u8> heap allocations.
    scalars: ScalarState,
    tokenizer: Tokenizer,
    eos_token_id: u32,
    #[allow(dead_code)]
    _config: ModelConfig,
}

impl Session {
    /// Build a session from a model directory.
    ///
    /// `model_dir` must contain:
    /// - one or more `.safetensors` files
    /// - `tokenizer.json`
    ///
    /// `toml_src` is the TOML model definition (e.g. contents of
    /// `models/llama_decode.toml`). `config` is the parsed `config.json`.
    /// `fusion_mode` controls whether kernel fusion is applied.
    #[tracing::instrument(skip(toml_src, config), fields(model_dir = %model_dir.as_ref().display(), dtype = ?dtype, fusion = ?fusion_mode))]
    pub fn new(
        model_dir: impl AsRef<Path>,
        toml_src: &str,
        config: ModelConfig,
        dtype: DType,
        fusion_mode: FusionMode,
    ) -> Result<Self, InferError> {
        let model_dir = model_dir.as_ref();

        // ── Parse model definition ─────────────────────────────────────
        info!("parsing model definition");
        let def: ModelDef = toml::from_str(toml_src)?;

        // ── Load weights ───────────────────────────────────────────────
        // Weights are cast to the activation dtype at load time so that
        // kernels compiled for `dtype` read correctly-typed bytes.
        info!("loading weights");
        let mut weights: WeightMap = load_weights(model_dir, dtype)?;
        info!(n_tensors = weights.len(), "weights loaded");

        // Handle weight tying: if lm_head is missing but tok_embeddings
        // exists (tie_word_embeddings = true), alias them.
        if !weights.contains_key("lm_head")
            && let Some(emb) = weights.get("tok_embeddings").cloned()
        {
            weights.insert("lm_head".to_string(), emb);
        }

        // ── Build GPU context ──────────────────────────────────────────
        let ctx = Context::new()?;

        // ── Upload weights to GPU-resident buffers ─────────────────────
        info!(n_tensors = weights.len(), "uploading weights to GPU");
        let mut resident: FxHashMap<String, ResidentBuffer> = FxHashMap::default();
        for (name, bytes) in &weights {
            let rb = ctx.upload_resident(bytes)?;
            resident.insert(name.clone(), rb);
        }

        // ── Allocate GPU-resident KV cache ─────────────────────────────
        // Layout: a single large GPU allocation for all K and V caches,
        // replacing 2×n_layers separate Metal buffer allocations.
        // Each "kv_cache.{layer}.{k/v}" entry maps to a view of this buffer.
        let kv_bytes =
            config.n_kv_heads * config.max_seq_len * config.head_dim * dtype.size_bytes();
        let total_kv_bytes = kv_bytes * 2 * config.n_layers;
        let kv_base = ctx.alloc_resident(total_kv_bytes)?;
        for layer in 0..config.n_layers {
            for (ki, key) in ["k", "v"].iter().enumerate() {
                let name = format!("kv_cache.{layer}.{key}");
                let offset_bytes = (layer * 2 + ki) * kv_bytes;
                // Create a sub-buffer view into the single large allocation.
                // alloc_resident skips the zero-fill memcpy; the KV-update
                // kernel always writes before reading, so uninitialised data
                // at position ≥ n_kv is never consumed.
                let rb = kv_base.slice(offset_bytes, kv_bytes);
                resident.insert(name, rb);
            }
        }

        // ── Compile execution plan ─────────────────────────────────────
        let reg = KernelRegistry::build();
        info!(n_kernels = reg.len(), "compiling execution plan");
        let state_keys = vec![
            "token_id".to_string(),
            "position".to_string(),
            "n_kv".to_string(), // = position + 1, set before each forward pass
            "rms_eps".to_string(),
            "temperature".to_string(),
            "uniform".to_string(),
        ];
        // KV cache state keys (read position/constexpr from state)
        // These are GPU-resident, managed via resident map — just need
        // them registered as state for the compiler.
        let mut all_state_keys = state_keys.clone();
        for layer in 0..config.n_layers {
            all_state_keys.push(format!("kv_cache.{layer}.k"));
            all_state_keys.push(format!("kv_cache.{layer}.v"));
        }

        let params = build_compile_params(&config, dtype, all_state_keys);
        let plan = compile(&def, &params, &reg, fusion_mode)?;

        // ── Initial state ──────────────────────────────────────────────
        let mut state = StateMap::default();
        state.insert("position".to_string(), 0u32.to_le_bytes().to_vec());
        state.insert("n_kv".to_string(), 0u32.to_le_bytes().to_vec());
        state.insert("rms_eps".to_string(), 1e-5f32.to_le_bytes().to_vec());
        state.insert("temperature".to_string(), 1.0f32.to_le_bytes().to_vec());
        state.insert("uniform".to_string(), 0.5f32.to_le_bytes().to_vec());
        state.insert("token_id".to_string(), 0u32.to_le_bytes().to_vec());

        // ── Build PreparedDispatch ─────────────────────────────────────
        // Builds static binding maps once; only ~82 dynamic entries
        // (position, n_kv, token_id, temperature, uniform) are updated per token.
        let prepared = PreparedDispatch::build(&ctx, &plan, &resident, &state)?;

        // ── Tokenizer ─────────────────────────────────────────────────
        let tok_path = model_dir.join("tokenizer.json");
        info!(path = %tok_path.display(), "loading tokenizer");
        let tokenizer =
            Tokenizer::from_file(&tok_path).map_err(|e| InferError::Tokenizer(e.to_string()))?;

        // Common EOS token IDs for Llama family
        let eos_token_id = find_eos_token_id(&tokenizer);

        // Initialize packed scalar state from StateMap.
        let scalars = ScalarState::from_state(&state);

        info!(
            n_layers = config.n_layers,
            n_heads = config.n_heads,
            n_kv_heads = config.n_kv_heads,
            head_dim = config.head_dim,
            hidden_dim = config.hidden_dim,
            vocab_size = config.vocab_size,
            max_seq_len = config.max_seq_len,
            eos_token_id,
            n_plan_nodes = plan.nodes.len(),
            n_slots = plan.slots.len(),
            "session ready"
        );
        Ok(Session {
            ctx,
            plan,
            prepared,
            state,
            scalars,
            tokenizer,
            _config: config,
            eos_token_id,
        })
    }

    /// Run inference for `max_tokens` steps, calling `on_token` with each
    /// decoded piece. Returns the generated text with GPU timing breakdown.
    #[tracing::instrument(skip(self, on_token), fields(prompt_len = prompt.len(), max_tokens))]
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        mut on_token: impl FnMut(&str),
    ) -> Result<GenerateOutput, InferError> {
        // Tokenize prompt
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| InferError::Tokenizer(e.to_string()))?;
        let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
        let prompt_tokens = prompt_ids.len();
        info!(n_prompt_tokens = prompt_tokens, "starting prefill");

        // ── Prefill ───────────────────────────────────────────────
        // Feed each prompt token through the model to populate KV cache.
        // Non-final tokens skip the vocab-projection + sampling tail
        // (output_norm → lm_head → sampling) — those outputs are not
        // needed until the very last prefill step.
        let mut prefill_secs = 0.0f64;
        let mut last_token_id = 0u32;
        let n_prompt = prompt_ids.len();
        let prefill_limit = self.plan.prefill_node_count;
        for (i, &token_id) in prompt_ids.iter().enumerate() {
            let is_last = i + 1 == n_prompt;
            let max_nodes = if is_last { self.plan.nodes.len() } else { prefill_limit };
            let (tid, secs) = self.step_inner(token_id, temperature, max_nodes)?;
            if is_last {
                last_token_id = tid;
            }
            prefill_secs += secs;
        }

        // ── Decode ────────────────────────────────────────────────
        info!("starting decode");
        let mut output_ids = Vec::new();
        let mut token_id = last_token_id;
        let mut decode_secs = 0.0f64;
        for _ in 0..max_tokens {
            if token_id == self.eos_token_id {
                break;
            }
            output_ids.push(token_id);
            let piece = self
                .tokenizer
                .decode(&[token_id], true)
                .map_err(|e| InferError::Tokenizer(e.to_string()))?;
            on_token(&piece);
            let (tid, secs) = self.step(token_id, temperature)?;
            token_id = tid;
            decode_secs += secs;
        }

        let tokens_generated = output_ids.len();
        let decode_tok_per_sec =
            if decode_secs > 0.0 { tokens_generated as f64 / decode_secs } else { 0.0 };

        let generated = self
            .tokenizer
            .decode(&output_ids, true)
            .map_err(|e| InferError::Tokenizer(e.to_string()))?;

        info!(
            tokens_generated,
            prompt_tokens, prefill_secs, decode_secs, decode_tok_per_sec, "generation complete"
        );
        Ok(GenerateOutput {
            text: generated,
            tokens_generated,
            prompt_tokens,
            prefill_secs,
            decode_secs,
            decode_tok_per_sec,
        })
    }

    /// Single forward pass: set token_id + temperature + uniform in state,
    /// execute the plan, return the sampled next token id and wall-clock
    /// time in seconds.
    ///
    /// This is the low-level step API — drive the autoregressive loop
    /// yourself for fine-grained control over timing, sampling, etc.
    #[tracing::instrument(level = "debug", skip(self), fields(token_id, temperature))]
    pub fn step(&mut self, token_id: u32, temperature: f32) -> Result<(u32, f64), InferError> {
        self.step_inner(token_id, temperature, self.plan.nodes.len())
    }

    /// Internal step that runs at most `max_nodes` nodes of the plan.
    ///
    /// When `max_nodes < plan.nodes.len()` (prefill fast path), the plan
    /// output is empty — the returned token id is `0` and should be ignored.
    fn step_inner(
        &mut self,
        token_id: u32,
        temperature: f32,
        max_nodes: usize,
    ) -> Result<(u32, f64), InferError> {
        let pos = self.scalars.position();
        self.scalars.set_n_kv(pos + 1);
        self.scalars.set_token_id(token_id);
        self.scalars.set_temperature(temperature.max(1e-5));
        self.scalars.set_uniform(random_uniform());
        self.scalars.sync_to_state(&mut self.state);

        let t0 = std::time::Instant::now();
        let (out_bytes, _gpu_us) = execute_prepared(
            &mut self.prepared,
            &self.ctx,
            &self.plan,
            &mut self.state,
            max_nodes,
        )?;
        let wall_secs = t0.elapsed().as_secs_f64();

        self.scalars.set_u32(1, pos + 1);

        // Partial prefill: output is empty, return sentinel 0.
        if out_bytes.len() < 4 {
            return Ok((0, wall_secs));
        }
        let next_id = u32::from_le_bytes([out_bytes[0], out_bytes[1], out_bytes[2], out_bytes[3]]);
        Ok((next_id, wall_secs))
    }

    /// Reset KV cache and position counter (start a new conversation).
    pub fn reset(&mut self) {
        self.scalars.set_u32(1, 0);
        self.scalars.set_u32(2, 0);
        for key in &["position", "n_kv"] {
            if let Some(buf) = self.state.get_mut(*key) {
                buf[..4].copy_from_slice(&0u32.to_le_bytes());
            }
        }
    }
}

/// Packed scalar state (6 × 4 bytes) for frequently updated values
/// (token_id, position, n_kv, rms_eps, temperature, uniform).
#[derive(Clone)]
struct ScalarState([u8; 24]);

impl ScalarState {
    fn from_state(state: &StateMap) -> Self {
        const KEYS: [&str; 6] =
            ["token_id", "position", "n_kv", "rms_eps", "temperature", "uniform"];
        let mut s = [0u8; 24];
        for (i, key) in KEYS.iter().enumerate() {
            if let Some(buf) = state.get(*key) {
                let len = buf.len().min(4);
                s[i * 4..i * 4 + len].copy_from_slice(&buf[..len]);
            }
        }
        Self(s)
    }

    fn position(&self) -> u32 { self.get_u32(1) }
    fn set_token_id(&mut self, val: u32) { self.set_u32(0, val); }
    fn set_n_kv(&mut self, val: u32) { self.set_u32(2, val); }
    fn set_temperature(&mut self, val: f32) { self.set_f32(4, val); }
    fn set_uniform(&mut self, val: f32) { self.set_f32(5, val); }

    fn sync_to_state(&self, state: &mut StateMap) {
        const KEYS: [&str; 6] =
            ["token_id", "position", "n_kv", "rms_eps", "temperature", "uniform"];
        for (i, key) in KEYS.iter().enumerate() {
            let base = i * 4;
            if let Some(buf) = state.get_mut(*key) {
                buf[..4].copy_from_slice(&self.0[base..base + 4]);
            }
        }
    }

    fn get_u32(&self, slot: usize) -> u32 {
        let base = slot * 4;
        u32::from_le_bytes([self.0[base], self.0[base + 1], self.0[base + 2], self.0[base + 3]])
    }

    fn set_u32(&mut self, slot: usize, val: u32) {
        let base = slot * 4;
        self.0[base..base + 4].copy_from_slice(&val.to_le_bytes());
    }

    fn set_f32(&mut self, slot: usize, val: f32) {
        let base = slot * 4;
        self.0[base..base + 4].copy_from_slice(&val.to_le_bytes());
    }
}

fn build_compile_params(
    config: &ModelConfig,
    dtype: DType,
    state_keys: Vec<String>,
) -> CompileParams {
    let mut params = std::collections::HashMap::new();
    params.insert("n_layers".to_string(), config.n_layers as u32);
    params.insert("n_heads".to_string(), config.n_heads as u32);
    params.insert("n_kv_heads".to_string(), config.n_kv_heads as u32);
    params.insert("head_dim".to_string(), config.head_dim as u32);
    params.insert("hidden_dim".to_string(), config.hidden_dim as u32);
    params.insert("ffn_dim".to_string(), config.ffn_dim as u32);
    params.insert("vocab_size".to_string(), config.vocab_size as u32);
    params.insert("max_seq_len".to_string(), config.max_seq_len as u32);

    CompileParams {
        params,
        float_params: std::collections::HashMap::new(),
        activation_dtype: dtype,
        n_layers: config.n_layers,
        state_keys,
    }
}

use std::{
    cell::Cell,
    time::{SystemTime, UNIX_EPOCH},
};

thread_local! {
    static RNG: Cell<u64> = Cell::new({
        let t = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        let pid = std::process::id() as u64;
        let seed = t ^ pid.wrapping_mul(6364136223846793005);
        if seed == 0 { 1 } else { seed }
    });
}

/// OS-seeded xorshift64 uniform float in (0, 1).
///
/// Returns a value in [0, 1): mantissa bits are taken from the upper 53 bits
/// so the result never rounds to exactly 1.0. Values of 0.0 are possible but
/// benign — they cause the sampling CDF to hit the first token, which is correct.
#[inline]
fn random_uniform() -> f32 {
    RNG.with(|rng| {
        let mut x = rng.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        rng.set(x);
        (x >> 11) as f32 * (1.0f32 / (1u64 << 53) as f32)
    })
}

fn find_eos_token_id(tokenizer: &Tokenizer) -> u32 {
    // Try common EOS token strings for Llama models
    for candidate in &["</s>", "<|end_of_text|>", "<|eot_id|>", "<eos>"] {
        if let Some(id) = tokenizer.token_to_id(candidate) {
            return id;
        }
    }
    2 // Llama 1/2 default
}
