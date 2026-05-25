//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `tile completions` — Emit shell completion scripts.

use crate::CliError;

#[derive(clap::Args, Debug)]
pub struct CompletionsArgs {
    /// Shell to generate completions for.
    pub shell: clap_complete::Shell,
}

impl CompletionsArgs {
    pub fn run(&self) -> Result<(), CliError> {
        use clap::CommandFactory;
        let mut cmd = crate::Cli::command();
        clap_complete::generate(self.shell, &mut cmd, "tile", &mut std::io::stdout());
        Ok(())
    }
}
