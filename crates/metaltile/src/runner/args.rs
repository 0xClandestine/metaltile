//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Argument struct for the `__tile_runner` subprocess.
//!
//! The `tile` CLI serialises these as `--key value` flags when it spawns the
//! runner.  The runner binary parses them with [`RunnerArgs::from_env_args`].

/// The subcommand the runner should execute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunnerCommand {
    /// Run all (or filtered) benchmarks.
    Bench,
    /// Run all (or filtered) tests.
    Test,
    /// Build (compile) all (or filtered) kernels.
    Build,
    /// Inspect a kernel (dump MSL / IR / stats).
    Inspect,
}

/// Parsed arguments for the `__tile_runner` subprocess.
#[derive(Debug, Clone)]
pub struct RunnerArgs {
    /// Which subcommand to run.
    pub command: RunnerCommand,
    /// Optional name filter — only items whose name contains this substring
    /// are processed.
    pub filter: Option<String>,
    /// Dtype filter (e.g. `"f16"`). `None` means all supported dtypes.
    pub dtype: Option<String>,
    /// For `inspect`: which representation to emit (`msl`, `ir`, `stats`,
    /// `listing`).
    pub inspect_kind: Option<String>,
    /// Emit profiling data with each bench result.
    pub profile: bool,
    /// Number of warmup dispatches before timing (overrides `BENCH_WARMUP` default).
    pub warmup: Option<usize>,
    /// Number of timed iterations (overrides `BENCH_ITERS` default).
    pub iters: Option<usize>,
}

impl RunnerArgs {
    /// Parse [`std::env::args`] into a [`RunnerArgs`].
    ///
    /// Expected invocation format (produced by `ProjectRunner` in the CLI):
    /// ```text
    /// __tile_runner bench [--filter <pat>] [--dtype <dt>] [--profile]
    ///                     [--warmup-runs <n>] [--runs <n>]
    /// __tile_runner test  [--filter <pat>] [--dtype <dt>]
    /// __tile_runner build [--filter <pat>] [--dtype <dt>]
    /// __tile_runner inspect [--filter <pat>] [--kind <msl|ir|stats|listing>]
    /// ```
    pub fn from_env_args() -> Result<Self, String> {
        Self::parse(std::env::args().skip(1).collect())
    }

    pub fn parse(args: Vec<String>) -> Result<Self, String> {
        let mut it = args.into_iter();
        let cmd_str = it.next().ok_or("missing subcommand")?;
        let command = match cmd_str.as_str() {
            "bench" => RunnerCommand::Bench,
            "test" => RunnerCommand::Test,
            "build" => RunnerCommand::Build,
            "inspect" => RunnerCommand::Inspect,
            other => return Err(format!("unknown subcommand '{other}'")),
        };

        let mut filter = None;
        let mut dtype = None;
        let mut inspect_kind = None;
        let mut profile = false;
        let mut warmup: Option<usize> = None;
        let mut iters: Option<usize> = None;

        while let Some(flag) = it.next() {
            match flag.as_str() {
                "--filter" => filter = it.next(),
                "--dtype" => dtype = it.next(),
                "--kind" => inspect_kind = it.next(),
                "--profile" => profile = true,
                "--warmup-runs" => {
                    let v = it.next().ok_or("--warmup-runs requires a value")?;
                    warmup = Some(
                        v.parse::<usize>().map_err(|_| format!("invalid --warmup-runs '{v}'"))?,
                    );
                },
                "--runs" => {
                    let v = it.next().ok_or("--runs requires a value")?;
                    iters = Some(v.parse::<usize>().map_err(|_| format!("invalid --runs '{v}'"))?);
                },
                other => return Err(format!("unknown flag '{other}'")),
            }
        }

        Ok(RunnerArgs { command, filter, dtype, inspect_kind, profile, warmup, iters })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_bench_no_flags() {
        let a = RunnerArgs::parse(vec!["bench".into()]).unwrap();
        assert_eq!(a.command, RunnerCommand::Bench);
        assert!(a.filter.is_none());
        assert!(!a.profile);
    }

    #[test]
    fn parse_bench_with_filter_and_profile() {
        let a = RunnerArgs::parse(vec![
            "bench".into(),
            "--filter".into(),
            "exp".into(),
            "--profile".into(),
        ])
        .unwrap();
        assert_eq!(a.filter.as_deref(), Some("exp"));
        assert!(a.profile);
    }

    #[test]
    fn parse_inspect_with_kind() {
        let a = RunnerArgs::parse(vec!["inspect".into(), "--kind".into(), "msl".into()]).unwrap();
        assert_eq!(a.command, RunnerCommand::Inspect);
        assert_eq!(a.inspect_kind.as_deref(), Some("msl"));
    }

    #[test]
    fn parse_unknown_subcommand_errors() {
        assert!(RunnerArgs::parse(vec!["foo".into()]).is_err());
    }

    #[test]
    fn parse_unknown_flag_errors() {
        assert!(RunnerArgs::parse(vec!["bench".into(), "--unknown".into()]).is_err());
    }
}
