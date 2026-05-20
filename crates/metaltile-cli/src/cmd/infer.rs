//! `tile infer` — run a forward pass / generate text from a model checkpoint.

use std::path::PathBuf;

use metaltile_core::dtype::DType;
use metaltile_infer::{FusionMode, ModelConfig, Session};

/// Args for `tile infer`.
#[derive(clap::Args)]
pub struct InferArgs {
    /// Model directory (contains .safetensors + tokenizer.json + config.json)
    /// OR a HuggingFace repo ID (e.g. "meta-llama/Llama-3.2-1B-Instruct").
    pub model: String,

    /// Prompt text
    #[arg(long = "prompt", short = 'p', default_value = "Hello")]
    pub prompt: String,

    /// Maximum tokens to generate
    #[arg(long = "max-tokens", default_value = "200")]
    pub max_tokens: usize,

    /// Sampling temperature (0 = greedy)
    #[arg(long = "temperature", short = 't', default_value = "0.8")]
    pub temperature: f32,

    /// Path to TOML model definition (default: models/llama_decode.toml)
    #[arg(long = "model-toml")]
    pub model_toml: Option<String>,

    /// Activation dtype: f32, f16, bf16 (default: f16)
    #[arg(long = "dtype", default_value = "f16")]
    pub dtype: String,

    /// HuggingFace Hub revision (branch/tag/commit)
    #[arg(long = "revision", default_value = "main")]
    pub revision: String,

    /// Cap max_seq_len (KV cache size). Defaults to 2048 to avoid huge allocations.
    #[arg(long = "max-seq-len", default_value = "2048")]
    pub max_seq_len: usize,

    /// Disable graph-level kernel fusion. By default, the compiler
    /// automatically fuses adjacent dispatch nodes into a single
    /// MTLCommandBuffer (one waitUntilCompleted per token). Pass this
    /// flag to dispatch each kernel individually — useful for debugging
    /// but ~10× slower.
    #[arg(long = "no-fuse", default_value_t = false)]
    pub no_fuse: bool,
}

pub fn run(args: &InferArgs) {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    if let Err(e) = rt.block_on(run_async(args)) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
    // Force-exit to kill sleeping watchdog threads (one spawned per GPU dispatch).
    std::process::exit(0);
}

async fn run_async(args: &InferArgs) -> Result<(), metaltile_infer::InferError> {
    // ── Resolve model directory ────────────────────────────────────────
    let model_path = PathBuf::from(&args.model);
    let model_dir = if model_path.exists() {
        model_path
    } else {
        // Treat as HuggingFace repo ID
        eprintln!("Downloading {model} from HuggingFace Hub…", model = &args.model);
        metaltile_infer::hub::snapshot_download(&args.model, &args.revision).await?
    };

    // ── Parse config.json ──────────────────────────────────────────────
    let mut config = ModelConfig::from_file(model_dir.join("config.json"))?;
    config.max_seq_len = config.max_seq_len.min(args.max_seq_len);
    eprintln!(
        "Model: {} layers, {} heads ({} kv), dim={}, vocab={}",
        config.n_layers, config.n_heads, config.n_kv_heads, config.hidden_dim, config.vocab_size,
    );

    // ── Load TOML model definition ─────────────────────────────────────
    let toml_path = args.model_toml.as_deref().unwrap_or("models/llama_decode.toml");
    let toml_src = std::fs::read_to_string(toml_path).map_err(metaltile_infer::InferError::Io)?;

    // ── Dtype ──────────────────────────────────────────────────────────
    let dtype = match args.dtype.as_str() {
        "f32" => DType::F32,
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        other => {
            return Err(metaltile_infer::InferError::UnsupportedDtype(other.to_string()));
        },
    };

    // ── Fusion mode ──────────────────────────────────────────────────
    let fusion_mode = if args.no_fuse {
        FusionMode::None
    } else {
        FusionMode::GraphDriven
    };

    // ── Build session ──────────────────────────────────────────────────
    eprintln!("Loading weights and uploading to GPU…");
    let mut session = Session::new(&model_dir, &toml_src, config, dtype, fusion_mode)?;

    // ── Generate ───────────────────────────────────────────────────────
    eprintln!("\n--- generating ---");
    print!("{}", args.prompt);
    let output = session.generate(&args.prompt, args.max_tokens, args.temperature, |tok| {
        print!("{tok}");
        // Flush stdout so tokens appear incrementally
        use std::io::Write;
        let _ = std::io::stdout().flush();
    })?;
    println!();

    // ── Timing ─────────────────────────────────────────────────────────
    let total_secs = output.prefill_secs + output.decode_secs;
    eprintln!(
        "\n{pretok} prompt tokens processed in {prefill:.2}s",
        pretok = output.prompt_tokens,
        prefill = output.prefill_secs,
    );
    if output.prompt_tokens > 0 && output.prefill_secs > 0.0 {
        let prefill_tps = output.prompt_tokens as f64 / output.prefill_secs;
        eprintln!("  prefill:  {prefill_tps:.1} tok/s");
    }
    eprintln!(
        "{gentok} tokens generated in {decode:.2}s  ({tps:.1} tok/s)",
        gentok = output.tokens_generated,
        decode = output.decode_secs,
        tps = output.decode_tok_per_sec,
    );
    eprintln!("  total: {total:.2}s", total = total_secs);

    Ok(())
}
