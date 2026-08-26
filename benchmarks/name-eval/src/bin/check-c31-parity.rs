#![allow(
    dead_code,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::items_after_statements
)]

mod artifact {
    pub use bonjour::GenderHint;
}
#[path = "../dataset.rs"]
mod dataset;

use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::Command;

use bonjour::benchmark::{
    ALGORITHM_C2, ALGORITHM_C3, ALGORITHM_C31, RawInference, RoleInferenceDiagnostic,
    c31_inference_from_diagnostic, diagnose_role_inference, open_artifact,
};
use dataset::{Case, Split, generate_cases, load_regression};
use serde::Deserialize;
use sha2::{Digest, Sha256};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const BASELINE_BYTES: &[u8] =
    include_bytes!("../../fixtures/c31-parity-x86_64-linux-rust-1.93.0.json");

#[derive(Deserialize)]
struct Baseline {
    schema: u32,
    target: String,
    rust_version: String,
    production_manifest_sha256: String,
    case_counts: CaseCounts,
    behavior_sha256: String,
}

#[derive(Deserialize)]
struct CaseCounts {
    regression: usize,
    dev: usize,
    validation: usize,
    total: usize,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let artifact_path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: check-c31-parity ARTIFACT_DIRECTORY")?;
    let baseline: Baseline = serde_json::from_slice(BASELINE_BYTES)?;
    validate_environment(&baseline)?;

    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let fixtures = crate_root.join("fixtures");
    let regression = load_regression(&fixtures.join("regression.csv"))?;
    let mut cases = regression.clone();
    cases.extend(
        generate_cases(&fixtures, false)?
            .into_iter()
            .filter(|case| matches!(case.split, Split::Dev | Split::Validation)),
    );
    validate_case_counts(&baseline.case_counts, &regression, &cases)?;

    let corpus = open_artifact(&artifact_path)?;
    let mut behavior = Sha256::new();
    for case in &cases {
        let diagnostic = diagnose_role_inference(
            &corpus,
            ALGORITHM_C3,
            &case.input,
            case.country_hint.as_deref(),
            case.locale_hint.as_deref(),
        );
        let inference = c31_inference_from_diagnostic(&diagnostic, ALGORITHM_C2, ALGORITHM_C31);
        hash_behavior(&mut behavior, case, &diagnostic, &inference);
    }
    let actual = format!("{:x}", behavior.finalize());
    if actual != baseline.behavior_sha256 {
        return Err(format!(
            "C3.1 parity digest changed: expected {}, got {actual}",
            baseline.behavior_sha256
        )
        .into());
    }
    println!("cases={}", cases.len());
    println!("behavior_sha256={actual}");
    Ok(())
}

fn validate_environment(baseline: &Baseline) -> Result<()> {
    if baseline.schema != 1 || baseline.target != "x86_64-unknown-linux-gnu" {
        return Err("unsupported C3.1 parity baseline".into());
    }
    if !cfg!(all(target_arch = "x86_64", target_os = "linux")) {
        return Err(format!("parity baseline requires target {}", baseline.target).into());
    }
    let output = Command::new("rustc").arg("--version").output()?;
    let version = String::from_utf8(output.stdout)?;
    if !version.starts_with(&format!("rustc {} ", baseline.rust_version)) {
        return Err(format!("parity baseline requires Rust {}", baseline.rust_version).into());
    }
    let manifest = include_bytes!("../../../../data/name-v1/manifest.json");
    let digest = format!("{:x}", Sha256::digest(manifest));
    if digest != baseline.production_manifest_sha256 {
        return Err("production manifest digest changed".into());
    }
    Ok(())
}

fn validate_case_counts(baseline: &CaseCounts, regression: &[Case], cases: &[Case]) -> Result<()> {
    let dev = cases.iter().filter(|case| case.split == Split::Dev).count();
    let validation = cases
        .iter()
        .filter(|case| case.split == Split::Validation)
        .count();
    if baseline.regression != regression.len()
        || baseline.dev != dev
        || baseline.validation != validation
        || baseline.total != cases.len()
    {
        return Err("C3.1 parity case population changed".into());
    }
    Ok(())
}

fn hash_behavior(
    digest: &mut Sha256,
    case: &Case,
    diagnostic: &RoleInferenceDiagnostic,
    inference: &RawInference,
) {
    hash_text(digest, &case.id);
    hash_optional_text(digest, inference.greeting_candidate.as_deref());
    hash_optional_text(
        digest,
        diagnostic
            .candidates
            .first()
            .map(|candidate| candidate.origin),
    );
    let coordinates = diagnostic
        .candidates
        .first()
        .map_or([u64::MAX, u64::MAX], |candidate| {
            [candidate.start as u64, candidate.length as u64]
        });
    digest.update(coordinates[0].to_le_bytes());
    digest.update(coordinates[1].to_le_bytes());
    digest.update(inference.confidence.to_bits().to_le_bytes());
    digest.update(inference.gender_confidence.to_bits().to_le_bytes());
    hash_optional_text(digest, inference.greeting_at(ALGORITHM_C2.threshold));
    hash_optional_text(
        digest,
        inference
            .gender_at(ALGORITHM_C2.threshold)
            .map(|gender| gender.as_str()),
    );
    digest.update([u8::from(diagnostic.hard_organization_abstention)]);
}

fn hash_optional_text(digest: &mut Sha256, value: Option<&str>) {
    digest.update([u8::from(value.is_some())]);
    if let Some(value) = value {
        hash_text(digest, value);
    }
}

fn hash_text(digest: &mut Sha256, value: &str) {
    digest.update((value.len() as u64).to_le_bytes());
    digest.update(value.as_bytes());
}
