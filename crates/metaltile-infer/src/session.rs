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

use std::{collections::BTreeMap, path::Path};

use tracing::info;

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
use tokenizers::Tokenizer;

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
        let def: ModelDef =
            toml::from_str(toml_src).map_err(|e| InferError::Other(e.to_string()))?;

        // ── Load weights ───────────────────────────────────────────────
        // Weights are cast to the activation dtype at load time so that
        // kernels compiled for `dtype` read correctly-typed bytes.
        info!("loading weights");
        let mut weights: WeightMap = load_weights(model_dir, dtype)?;
        info!(n_tensors = weights.len(), "weights loaded");

        // Handle weight tying: if lm_head is missing but tok_embeddings
        // exists (tie_word_embeddings = true), alias them.
        if !weights.contains_key("lm_head") {
            if let Some(emb) = weights.get("tok_embeddings").cloned() {
                weights.insert("lm_head".to_string(), emb);
            }
        }

        // ── Build GPU context ──────────────────────────────────────────
        let ctx = Context::new().map_err(|e| InferError::Other(e.to_string()))?;

        // ── Upload weights to GPU-resident buffers ─────────────────────
        info!(n_tensors = weights.len(), "uploading weights to GPU");
        let mut resident: BTreeMap<String, ResidentBuffer> = BTreeMap::new();
        for (name, bytes) in &weights {
            let rb = ctx.upload_resident(bytes).map_err(|e| InferError::Other(e.to_string()))?;
            resident.insert(name.clone(), rb);
        }

        // ── Allocate GPU-resident KV cache ─────────────────────────────
        // Layout per layer, per K and V:
        //   n_kv_heads × max_seq_len × head_dim elements of `dtype`
        let kv_bytes =
            config.n_kv_heads * config.max_seq_len * config.head_dim * dtype.size_bytes();
        for layer in 0..config.n_layers {
            for key in ["k", "v"] {
                let name = format!("kv_cache.{layer}.{key}");
                // alloc_resident skips the zero-fill memcpy; the KV-update
                // kernel always writes before reading, so uninitialised data
                // at position ≥ n_kv is never consumed.
                let rb = ctx
                    .alloc_resident(kv_bytes)
                    .map_err(|e| InferError::Other(e.to_string()))?;
                resident.insert(name, rb);
            }
        }

        // ── Compile execution plan ─────────────────────────────────────
        let reg = KernelRegistry::build();
        info!(n_kernels = reg.len(), "compiling execution plan");
        let state_keys = vec![
            "token_id".to_string(),
            "position".to_string(),
            "n_kv".to_string(),   // = position + 1, set before each forward pass
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
        let mut state = StateMap::new();
        state.insert("position".to_string(), 0u32.to_le_bytes().to_vec());
        state.insert("n_kv".to_string(), 0u32.to_le_bytes().to_vec());
        state.insert("rms_eps".to_string(), 1e-5f32.to_le_bytes().to_vec());
        state.insert("temperature".to_string(), 1.0f32.to_le_bytes().to_vec());
        state.insert("uniform".to_string(), 0.5f32.to_le_bytes().to_vec());
        state.insert("token_id".to_string(), 0u32.to_le_bytes().to_vec());

        // ── Build PreparedDispatch ─────────────────────────────────────
        // Builds static binding maps once; only ~82 dynamic entries
        // (position, n_kv, token_id, temperature, uniform) are updated per token.
        let prepared = PreparedDispatch::build(&ctx, &plan, &resident, &state)
            .map_err(|e| InferError::Other(e.to_string()))?;

        // ── Tokenizer ─────────────────────────────────────────────────
        let tok_path = model_dir.join("tokenizer.json");
        info!(path = %tok_path.display(), "loading tokenizer");
        let tokenizer =
            Tokenizer::from_file(&tok_path).map_err(|e| InferError::Tokenizer(e.to_string()))?;

        // Common EOS token IDs for Llama family
        let eos_token_id = find_eos_token_id(&tokenizer);

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
        Ok(Session { ctx, plan, prepared, state, tokenizer, _config: config, eos_token_id })
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
        let decode_tok_per_sec = if decode_secs > 0.0 {
            tokens_generated as f64 / decode_secs
        } else {
            0.0
        };

        let generated = self
            .tokenizer
            .decode(&output_ids, true)
            .map_err(|e| InferError::Tokenizer(e.to_string()))?;

        info!(
            tokens_generated,
            prompt_tokens,
            prefill_secs,
            decode_secs,
            decode_tok_per_sec,
            "generation complete"
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
        let pos = position_from_state(&self.state);

        // n_kv = position + 1: the KV cache update will write token at slot `pos`,
        // so SDPA must attend to all pos+1 tokens (0..=pos inclusive).
        // Use in-place updates to avoid per-token String + Vec heap allocations.
        state_update_u32(&mut self.state, "n_kv", pos + 1);
        state_update_u32(&mut self.state, "token_id", token_id);
        state_update_f32(&mut self.state, "temperature", temperature);
        let uniform: f32 = pseudo_uniform(token_id, pos);
        state_update_f32(&mut self.state, "uniform", uniform);

        let t0 = std::time::Instant::now();
        let (out_bytes, _gpu_us) = execute_prepared(
            &mut self.prepared,
            &self.ctx,
            &self.plan,
            &mut self.state,
            max_nodes,
        )?;
        let wall_secs = t0.elapsed().as_secs_f64();

        // Advance position
        state_update_u32(&mut self.state, "position", pos + 1);

        // Partial prefill: output is empty, return sentinel 0.
        if out_bytes.len() < 4 {
            return Ok((0, wall_secs));
        }
        let next_id = u32::from_le_bytes([out_bytes[0], out_bytes[1], out_bytes[2], out_bytes[3]]);
        Ok((next_id, wall_secs))
    }

    /// Reset KV cache and position counter (start a new conversation).
    pub fn reset(&mut self) {
        state_update_u32(&mut self.state, "position", 0);
        state_update_u32(&mut self.state, "n_kv", 0);
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

/// Update a u32 state value in-place, avoiding a String key allocation and
/// a Vec<u8> heap allocation on every token step.
#[inline]
fn state_update_u32(state: &mut StateMap, key: &str, val: u32) {
    if let Some(buf) = state.get_mut(key) {
        buf[..4].copy_from_slice(&val.to_le_bytes());
    }
}

/// Update an f32 state value in-place.
#[inline]
fn state_update_f32(state: &mut StateMap, key: &str, val: f32) {
    if let Some(buf) = state.get_mut(key) {
        buf[..4].copy_from_slice(&val.to_le_bytes());
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

#[inline]
fn position_from_state(state: &StateMap) -> u32 {
    state
        .get("position")
        .and_then(|b| b.get(..4))
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .unwrap_or(0)
}

/// Deterministic pseudo-random float in [0,1) — good enough for greedy/temp sampling.
#[inline]
fn pseudo_uniform(token_id: u32, position: u32) -> f32 {
    let mut x = token_id.wrapping_mul(2654435761).wrapping_add(position.wrapping_mul(2246822519));
    x ^= x >> 13;
    x ^= x << 17;
    x ^= x >> 5;
    (x as f32) / (u32::MAX as f32)
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
