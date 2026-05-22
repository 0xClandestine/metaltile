//! Training-data export: JSONL serialization of `TrainingRow`.

use std::{
    io::Write,
    path::{Path, PathBuf},
};

use metaltile::autotune::{Autotuner, TrainingRow};

use crate::AutotuneError;

/// Read the on-disk cache and return its rows. Caller decides where
/// to write them (stdout, default path, custom path).
pub fn collect_training_rows() -> Vec<TrainingRow> {
    let tuner = Autotuner::new(Autotuner::default_cache_dir(), /* enabled= */ true);
    tuner.export_training_data()
}

/// Default location for the JSONL export when the user passes
/// `--export-training-data` without a value.
pub fn default_training_data_path() -> PathBuf {
    Autotuner::default_cache_dir().join("training_data.jsonl")
}

/// Write rows to `path`, creating parent directories if needed.
pub fn write_training_jsonl_to_file(
    path: &Path,
    rows: &[TrainingRow],
) -> Result<(), AutotuneError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::File::create(path)?;
    write_training_jsonl(std::io::BufWriter::new(file), rows)
}

/// Write rows as JSONL to any `Write` sink — one row per line.
pub fn write_training_jsonl<W: Write>(mut w: W, rows: &[TrainingRow]) -> Result<(), AutotuneError> {
    for row in rows {
        let line = serde_json::to_string(row)
            .map_err(|e| AutotuneError::Other(format!("serialize training row: {e}")))?;
        writeln!(w, "{line}")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use metaltile::autotune::TuneConfig;

    use super::*;

    #[test]
    fn write_training_jsonl_emits_one_object_per_line() {
        let rows = vec![
            TrainingRow {
                kernel: "mt_a".into(),
                dtype: "f16".into(),
                family: "Elementwise".into(),
                bucket: [("N".to_string(), (0usize, 256usize))].into_iter().collect(),
                best_config: TuneConfig::default(),
                perf_us: 1.5,
                timestamp: 1,
            },
            TrainingRow {
                kernel: "mt_b".into(),
                dtype: "f32".into(),
                family: "Matmul".into(),
                bucket: [("N".to_string(), (256usize, 1024usize))].into_iter().collect(),
                best_config: TuneConfig::default(),
                perf_us: 2.5,
                timestamp: 2,
            },
        ];
        let mut buf: Vec<u8> = Vec::new();
        write_training_jsonl(&mut buf, &rows).unwrap();
        let s = String::from_utf8(buf).unwrap();
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines.len(), 2);
        for line in &lines {
            let _: TrainingRow = serde_json::from_str(line).expect("each line is valid JSON");
        }
        assert!(lines[0].contains("\"kernel\":\"mt_a\""));
        assert!(lines[1].contains("\"kernel\":\"mt_b\""));
    }

    #[test]
    fn write_training_jsonl_empty_rows_writes_nothing() {
        let mut buf: Vec<u8> = Vec::new();
        write_training_jsonl(&mut buf, &[]).unwrap();
        assert!(buf.is_empty());
    }
}
