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

use metaltile_core::dtype::DType;
use metaltile_model::{
    CompileParams,
    FusionMode,
    KernelRegistry,
    ModelDef,
    StateMap,
    WeightMap,
    compile,
    execute_plan,
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
    resident: BTreeMap<String, ResidentBuffer>,
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
    pub fn new(
        model_dir: impl AsRef<Path>,
        toml_src: &str,
        config: ModelConfig,
        dtype: DType,
        fusion_mode: FusionMode,
    ) -> Result<Self, InferError> {
        let model_dir = model_dir.as_ref();

        // ── Parse model definition ─────────────────────────────────────
        let def: ModelDef =
            toml::from_str(toml_src).map_err(|e| InferError::Other(e.to_string()))?;

        // ── Load weights ───────────────────────────────────────────────
        // Weights are cast to the activation dtype at load time so that
        // kernels compiled for `dtype` read correctly-typed bytes.
        let mut weights: WeightMap = load_weights(model_dir, dtype)?;

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
                let rb = ctx
                    .upload_resident(&vec![0u8; kv_bytes])
                    .map_err(|e| InferError::Other(e.to_string()))?;
                resident.insert(name, rb);
            }
        }

        // ── Compile execution plan ─────────────────────────────────────
        let reg = KernelRegistry::build();
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

        // ── Tokenizer ─────────────────────────────────────────────────
        let tok_path = model_dir.join("tokenizer.json");
        let tokenizer =
            Tokenizer::from_file(&tok_path).map_err(|e| InferError::Tokenizer(e.to_string()))?;

        // Common EOS token IDs for Llama family
        let eos_token_id = find_eos_token_id(&tokenizer);

        Ok(Session { ctx, plan, resident, state, tokenizer, _config: config, eos_token_id })
    }

    /// Run inference for `max_tokens` steps, calling `on_token` with each
    /// decoded piece. Returns the generated text with GPU timing breakdown.
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

        // ── Prefill ───────────────────────────────────────────────
        // Feed each prompt token through the model to populate KV cache.
        let mut prefill_secs = 0.0f64;
        let mut last_token_id = 0u32;
        for &token_id in &prompt_ids {
            let (tid, secs) = self.step(token_id, temperature)?;
            last_token_id = tid;
            prefill_secs += secs;
        }

        // ── Decode ────────────────────────────────────────────────
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
    /// execute the plan, return the sampled next token id and GPU elapsed
    /// time in seconds.
    ///
    /// This is the low-level step API — drive the autoregressive loop
    /// yourself for fine-grained control over timing, sampling, etc.
    pub fn step(&mut self, token_id: u32, temperature: f32) -> Result<(u32, f64), InferError> {
        let pos = position_from_state(&self.state);

        // n_kv = position + 1: the KV cache update will write token at slot `pos`,
        // so SDPA must attend to all pos+1 tokens (0..=pos inclusive).
        self.state.insert("n_kv".to_string(), (pos + 1).to_le_bytes().to_vec());
        self.state.insert("token_id".to_string(), token_id.to_le_bytes().to_vec());
        self.state.insert("temperature".to_string(), temperature.to_le_bytes().to_vec());
        let uniform: f32 = pseudo_uniform(token_id, pos);
        self.state.insert("uniform".to_string(), uniform.to_le_bytes().to_vec());

        // Execute plan — wall-clock time captures both GPU compute and
        // CPU dispatch overhead (command-buffer creation, submission,
        // waitUntilCompleted). This reflects the real end-to-end cost per
        // token and shows the benefit of kernel fusion (fewer cmd buffers).
        let t0 = std::time::Instant::now();
        let (out_bytes, _gpu_us) = execute_plan(
            &self.ctx,
            &self.plan,
            &WeightMap::new(), // all weights are GPU-resident
            &mut self.state,
            &self.resident,
        )?;
        let wall_secs = t0.elapsed().as_secs_f64();

        // Advance position
        self.state.insert("position".to_string(), (pos + 1).to_le_bytes().to_vec());

        // Output is a u32 token id (4 bytes)
        if out_bytes.len() < 4 {
            return Err(InferError::Other("plan output too short".to_string()));
        }
        let next_id = u32::from_le_bytes([out_bytes[0], out_bytes[1], out_bytes[2], out_bytes[3]]);
        Ok((next_id, wall_secs))
    }

    /// Reset KV cache and position counter (start a new conversation).
    pub fn reset(&mut self) {
        self.state.insert("position".to_string(), 0u32.to_le_bytes().to_vec());
        self.state.insert("n_kv".to_string(), 0u32.to_le_bytes().to_vec());
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

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

fn position_from_state(state: &StateMap) -> u32 {
    state
        .get("position")
        .and_then(|b| b.get(..4))
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .unwrap_or(0)
}

/// Deterministic pseudo-random float in [0,1) — good enough for greedy/temp sampling.
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
