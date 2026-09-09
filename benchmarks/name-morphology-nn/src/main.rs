#![allow(
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::unnecessary_wraps
)]

mod dataset;
mod output;
mod proxy;
mod text;

use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

use output::Result;
use proxy::ProxyInput;

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let mut arguments = env::args_os().skip(1);
    let command = arguments.next().ok_or_else(usage)?;
    match command.to_str() {
        Some("prepare") => run_prepare(arguments.collect()),
        Some("proxy") => run_proxy(arguments.collect()),
        Some("conditional-proxy") => run_conditional_proxy(arguments.collect()),
        _ => Err(usage().into()),
    }
}

fn run_prepare(arguments: Vec<OsString>) -> Result<()> {
    if arguments.len() != 4 {
        return Err(usage().into());
    }
    let artifact = PathBuf::from(&arguments[0]);
    let totals = PathBuf::from(&arguments[1]);
    let clean = PathBuf::from(&arguments[2]);
    let output = PathBuf::from(&arguments[3]);
    require_directory(&artifact, "artifact")?;
    require_file(&totals, "name totals")?;
    require_file(&clean, "clean-v1")?;
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures");
    dataset::prepare(&artifact, &totals, &clean, &fixtures, &output)
}

fn run_proxy(arguments: Vec<OsString>) -> Result<()> {
    let (artifact, inputs) = parse_proxy_arguments(arguments)?;
    proxy::stream_proxy(&artifact, inputs)
}

fn run_conditional_proxy(arguments: Vec<OsString>) -> Result<()> {
    let (artifact, inputs) = parse_proxy_arguments(arguments)?;
    proxy::stream_conditional_proxy(&artifact, inputs)
}

fn parse_proxy_arguments(arguments: Vec<OsString>) -> Result<(PathBuf, Vec<ProxyInput>)> {
    let mut arguments = arguments.into_iter();
    let artifact = PathBuf::from(arguments.next().ok_or_else(usage)?);
    require_directory(&artifact, "artifact")?;
    let mut sealed = Vec::new();
    let mut manifests = Vec::new();
    for argument in arguments {
        let value = argument.to_string_lossy();
        if let Some(path) = value.strip_prefix("--sealed=") {
            sealed.push(PathBuf::from(path));
        } else if let Some(path) = value.strip_prefix("--manifest=") {
            manifests.push(PathBuf::from(path));
        } else {
            return Err(format!("unknown proxy argument: {value}").into());
        }
    }
    if sealed.len() != manifests.len() {
        return Err("proxy mode requires aligned --sealed and --manifest arguments".into());
    }
    let inputs = sealed
        .into_iter()
        .zip(manifests)
        .map(|(sealed, manifest)| ProxyInput { sealed, manifest })
        .collect();
    Ok((artifact, inputs))
}

fn require_file(path: &Path, label: &str) -> Result<()> {
    if path.is_file() {
        Ok(())
    } else {
        Err(format!("{label} is not a file: {}", path.display()).into())
    }
}

fn require_directory(path: &Path, label: &str) -> Result<()> {
    if path.is_dir() {
        Ok(())
    } else {
        Err(format!("{label} is not a directory: {}", path.display()).into())
    }
}

fn usage() -> String {
    "usage:\n  name-morphology-nn-data prepare <artifact> <name-totals.csv> <clean-v1.csv> <new-output-directory>\n  name-morphology-nn-data proxy <artifact> [--sealed=FILE --manifest=FILE]x3\n  name-morphology-nn-data conditional-proxy <artifact> [--sealed=FILE --manifest=FILE]x3".to_string()
}
