#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::imprecise_flops,
    clippy::items_after_statements,
    clippy::needless_pass_by_value,
    clippy::suboptimal_flops,
    clippy::too_many_lines
)]

mod artifact;
mod classifier;
mod corpus_audit;
mod dataset;
mod lexical;
mod metrics;

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::{Path, PathBuf};

use artifact::C32Artifact;
use classifier::{
    ALGORITHM_A, ALGORITHM_B, ALGORITHM_C, ALGORITHM_C1, AlgorithmConfig, RawInference,
    candidate_diagnostics, infer_prethreshold,
};
use corpus_audit::{LexicalAudit, audit_clean_v1};
use dataset::{
    C0_TEST_GENERATOR_SEED, C0_TEST_SHA256, Case, DEV_TARGET, FRESH_TEST_GENERATOR_SEED,
    FRESH_TEST_SHA256, GENERATOR_SEED, INSPECTED_TEST_SHA256, LARGE_GENERATOR_SEED, SeedStats,
    Split, TEST_TARGET, VALIDATION_TARGET, generate_cases, load_regression, seed_stats,
};
use metrics::{CaseOutcome, Metrics, greeting_matches, outcome};
use name_eval::holdout::{
    FrozenHoldout, SealedDecision, SealedEvaluation, evaluate_sealed, load_frozen,
    sealed_confidence_buckets_csv, sealed_summary_csv,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const DEFAULT_REFERENCE_THRESHOLD: f64 = 0.80;
const C0_FROZEN_THRESHOLD: f64 = 0.80;
const C1_SELECTED_THRESHOLD: f64 = 0.93;
const THRESHOLD_SWEEP: [f64; 12] = [
    0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99, 1.00,
];
const PRECISION_TARGETS: [f64; 3] = [0.990, 0.995, 0.999];
const ERROR_BUDGETS: [usize; 5] = [0, 1, 5, 10, 25];

struct AlgorithmRun {
    config: AlgorithmConfig,
    predictions: Vec<RawInference>,
}

#[derive(Clone, Copy)]
struct TargetResult {
    threshold: f64,
    metrics: Metrics,
}

#[derive(Clone)]
struct RoleSummary {
    algorithm: &'static str,
    split: Split,
    role: &'static str,
    values: Vec<f64>,
}

struct SealedRun {
    holdout: FrozenHoldout,
    evaluation: SealedEvaluation,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let arguments = parse_arguments()?;
    if arguments.output.exists() {
        return Err(format!("refusing to overwrite: {}", arguments.output.display()).into());
    }
    let output_parent = arguments.output.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(output_parent)?;
    let temporary = output_parent.join(format!(
        ".{}.tmp-{}",
        arguments
            .output
            .file_name()
            .ok_or("output path has no final component")?
            .to_string_lossy(),
        std::process::id()
    ));
    if temporary.exists() {
        return Err(format!("refusing to overwrite: {}", temporary.display()).into());
    }
    fs::create_dir(&temporary)?;

    let result = evaluate(&arguments, &temporary);
    match result {
        Ok(report) => {
            fs::write(temporary.join("report.md"), &report)?;
            fs::rename(&temporary, &arguments.output)?;
            println!("{report}");
            println!("Output: {}", arguments.output.display());
            Ok(())
        }
        Err(error) => {
            if let Err(cleanup_error) = fs::remove_dir_all(&temporary) {
                eprintln!(
                    "warning: failed to clean {}: {cleanup_error}",
                    temporary.display()
                );
            }
            Err(error)
        }
    }
}

struct Arguments {
    artifact: PathBuf,
    clean_csv: Option<PathBuf>,
    output: PathBuf,
    sealed: Option<PathBuf>,
    sealed_manifest: Option<PathBuf>,
    reference_threshold: f64,
    development_only: bool,
    sealed_only: bool,
}

fn parse_arguments() -> Result<Arguments> {
    let mut positional = Vec::new();
    let mut sealed = None;
    let mut sealed_manifest = None;
    let mut reference_threshold = DEFAULT_REFERENCE_THRESHOLD;
    let mut reference_threshold_set = false;
    let mut development_only = false;
    let mut sealed_only = false;
    for argument in std::env::args_os().skip(1) {
        let text = argument.to_string_lossy();
        if let Some(value) = text.strip_prefix("--sealed=") {
            sealed = Some(PathBuf::from(value));
        } else if let Some(value) = text.strip_prefix("--sealed-manifest=") {
            sealed_manifest = Some(PathBuf::from(value));
        } else if let Some(value) = text.strip_prefix("--reference-threshold=") {
            reference_threshold = value.parse::<f64>()?;
            reference_threshold_set = true;
            if !(0.0..=1.0).contains(&reference_threshold) {
                return Err("reference threshold must lie in 0..=1".into());
            }
        } else if text == "--development-only" {
            development_only = true;
        } else if text == "--sealed-only" {
            sealed_only = true;
        } else if text.starts_with('-') {
            return Err(format!("unknown option: {text}").into());
        } else {
            positional.push(PathBuf::from(argument));
        }
    }
    if sealed.is_some() != sealed_manifest.is_some() {
        return Err("--sealed and --sealed-manifest must be supplied together".into());
    }
    let (artifact, clean_csv, output) = if sealed_only {
        if positional.len() != 2 {
            return Err(usage().into());
        }
        if sealed.is_none() || development_only || reference_threshold_set {
            return Err("--sealed-only requires both sealed files and cannot be combined with --development-only or --reference-threshold".into());
        }
        (positional.remove(0), None, positional.remove(0))
    } else {
        if positional.len() != 3 {
            return Err(usage().into());
        }
        (
            positional.remove(0),
            Some(positional.remove(0)),
            positional.remove(0),
        )
    };
    Ok(Arguments {
        artifact,
        clean_csv,
        output,
        sealed,
        sealed_manifest,
        reference_threshold,
        development_only,
        sealed_only,
    })
}

fn usage() -> &'static str {
    "usage:\n  name-eval <c32-artifact-directory> <clean-v1.csv> <new-output-directory> [--sealed=FILE --sealed-manifest=FILE] [--reference-threshold=FLOAT] [--development-only]\n  name-eval <c32-artifact-directory> <new-output-directory> --sealed-only --sealed=FILE --sealed-manifest=FILE"
}

#[allow(clippy::too_many_lines)]
fn evaluate(arguments: &Arguments, output: &Path) -> Result<String> {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures");
    let frozen_holdout = arguments
        .sealed
        .as_deref()
        .zip(arguments.sealed_manifest.as_deref())
        .map(|(sealed, manifest)| load_frozen(sealed, manifest))
        .transpose()?;
    let corpus = C32Artifact::open(
        &arguments.artifact,
        &fixtures.join("artifact-manifest.csv"),
        &fixtures.join("surname-artifact-manifest.csv"),
    )?;
    if arguments.sealed_only {
        let holdout = frozen_holdout.ok_or("--sealed-only requires a frozen holdout")?;
        let sealed_run = evaluate_frozen_holdout(&corpus, holdout)?;
        fs::write(
            output.join("sealed_summary_metrics.csv"),
            sealed_summary_csv(&sealed_run.evaluation)?,
        )?;
        fs::write(
            output.join("sealed_confidence_buckets.csv"),
            sealed_confidence_buckets_csv(&sealed_run.evaluation)?,
        )?;
        return Ok(build_sealed_only_report(&sealed_run));
    }
    let clean_csv = arguments
        .clean_csv
        .as_deref()
        .ok_or("normal evaluation requires clean-v1.csv")?;
    let lexical_audit = audit_clean_v1(clean_csv)?;
    let seed_statistics = seed_stats(&fixtures)?;
    let regression = load_regression(&fixtures.join("regression.csv"))?;
    let generated = generate_cases(&fixtures, !arguments.development_only)?;
    let mut cases = Vec::with_capacity(regression.len() + generated.len());
    cases.extend(regression);
    cases.extend(generated);

    let algorithms = [ALGORITHM_A, ALGORITHM_B, ALGORITHM_C, ALGORITHM_C1]
        .into_iter()
        .map(|config| AlgorithmRun {
            config,
            predictions: cases
                .iter()
                .map(|case| {
                    infer_prethreshold(
                        &corpus,
                        config,
                        &case.input,
                        case.country_hint.as_deref(),
                        case.locale_hint.as_deref(),
                    )
                })
                .collect(),
        })
        .collect::<Vec<_>>();

    let sealed_run = frozen_holdout
        .map(|holdout| evaluate_frozen_holdout(&corpus, holdout))
        .transpose()?;
    if let Some(sealed_run) = &sealed_run {
        fs::write(
            output.join("sealed_summary_metrics.csv"),
            sealed_summary_csv(&sealed_run.evaluation)?,
        )?;
        fs::write(
            output.join("sealed_confidence_buckets.csv"),
            sealed_confidence_buckets_csv(&sealed_run.evaluation)?,
        )?;
    }

    write_generated_cases(output, &cases)?;
    write_lexical_audit(output, lexical_audit)?;
    write_dev_candidate_traces(output, &corpus, &cases, &algorithms)?;
    write_metrics(output, &cases, &algorithms, arguments.reference_threshold)?;
    write_threshold_curves(output, &cases, &algorithms)?;
    write_precision_targets(output, &cases, &algorithms)?;
    write_error_budget_operating_points(output, &cases, &algorithms)?;
    let role_summaries = write_role_llr_distribution(output, &corpus, &cases)?;
    write_results(
        &output.join("regression_results.csv"),
        &cases,
        &algorithms,
        arguments.reference_threshold,
        |case| case.split == Split::Regression,
    )?;
    write_results(
        &output.join("legacy_test_results.csv"),
        &cases,
        &algorithms,
        arguments.reference_threshold,
        |case| case.split == Split::LegacyTest,
    )?;
    write_results(
        &output.join("generated_failures.csv"),
        &cases,
        &algorithms,
        arguments.reference_threshold,
        |case| matches!(case.split, Split::Dev | Split::Validation),
    )?;
    write_comparison(
        &output.join("generated_comparison_ab.csv"),
        &cases,
        &algorithms[..2],
        arguments.reference_threshold,
        |case| matches!(case.split, Split::Dev | Split::Validation),
    )?;
    write_comparison(
        &output.join("generated_comparison_b_c0.csv"),
        &cases,
        &algorithms[1..3],
        arguments.reference_threshold,
        |case| matches!(case.split, Split::Dev | Split::Validation),
    )?;
    write_comparison(
        &output.join("generated_comparison_c0_c1.csv"),
        &cases,
        &algorithms[2..4],
        arguments.reference_threshold,
        |case| matches!(case.split, Split::Dev | Split::Validation),
    )?;
    write_comparison(
        &output.join("regression_comparison.csv"),
        &cases,
        &algorithms[..2],
        arguments.reference_threshold,
        |case| case.split == Split::Regression,
    )?;
    write_comparison(
        &output.join("regression_comparison_b_c0.csv"),
        &cases,
        &algorithms[1..3],
        arguments.reference_threshold,
        |case| case.split == Split::Regression,
    )?;
    write_comparison(
        &output.join("regression_comparison_c0_c1.csv"),
        &cases,
        &algorithms[2..4],
        arguments.reference_threshold,
        |case| case.split == Split::Regression,
    )?;
    write_comparison(
        &output.join("legacy_test_comparison.csv"),
        &cases,
        &algorithms[..2],
        arguments.reference_threshold,
        |case| case.split == Split::LegacyTest,
    )?;

    Ok(build_report(
        arguments,
        &corpus,
        lexical_audit,
        &seed_statistics,
        &cases,
        &algorithms,
        &role_summaries,
        sealed_run.as_ref(),
    ))
}

fn evaluate_frozen_holdout(corpus: &C32Artifact, holdout: FrozenHoldout) -> Result<SealedRun> {
    let decisions = holdout
        .cases
        .iter()
        .map(|case| {
            if !case.is_evaluable() {
                return None;
            }
            let inference = infer_prethreshold(
                corpus,
                ALGORITHM_C1,
                &case.display_name,
                nonempty(&case.country_hint),
                nonempty(&case.locale_hint),
            );
            Some(SealedDecision {
                greeting_candidate: inference.greeting_candidate,
                confidence: inference.confidence,
            })
        })
        .collect::<Vec<_>>();
    let evaluation = evaluate_sealed(&holdout, &decisions, C1_SELECTED_THRESHOLD)?;
    Ok(SealedRun {
        holdout,
        evaluation,
    })
}

fn nonempty(value: &str) -> Option<&str> {
    (!value.is_empty()).then_some(value)
}

fn indices_for(cases: &[Case], predicate: impl Fn(&Case) -> bool) -> Vec<usize> {
    cases
        .iter()
        .enumerate()
        .filter_map(|(index, case)| predicate(case).then_some(index))
        .collect()
}

fn metrics_for(
    cases: &[Case],
    predictions: &[RawInference],
    indices: &[usize],
    threshold: f64,
) -> Metrics {
    let case_refs = indices
        .iter()
        .map(|&index| &cases[index])
        .collect::<Vec<_>>();
    let prediction_refs = indices
        .iter()
        .map(|&index| &predictions[index])
        .collect::<Vec<_>>();
    Metrics::evaluate(&case_refs, &prediction_refs, threshold)
}

fn reported_threshold(algorithm: &AlgorithmRun, reference_threshold: f64) -> f64 {
    if algorithm.config.name == ALGORITHM_C1.name {
        C1_SELECTED_THRESHOLD
    } else {
        reference_threshold
    }
}

fn write_generated_cases(output: &Path, cases: &[Case]) -> Result<()> {
    let mut writer = csv::Writer::from_path(output.join("generated_cases.csv"))?;
    writer.write_record([
        "id",
        "split",
        "category",
        "input",
        "country_hint",
        "locale_hint",
        "expected_greeting",
        "expected_gender",
    ])?;
    for case in cases.iter().filter(|case| {
        matches!(
            case.split,
            Split::Dev | Split::Validation | Split::LegacyTest | Split::InspectedTest
        )
    }) {
        writer.write_record([
            case.id.as_str(),
            case.split.as_str(),
            case.category.as_str(),
            case.input.as_str(),
            case.country_hint.as_deref().unwrap_or(""),
            case.locale_hint.as_deref().unwrap_or(""),
            case.expected_greeting.as_deref().unwrap_or(""),
            case.expected_gender.map_or("", |gender| gender.as_str()),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn write_role_llr_distribution(
    output: &Path,
    corpus: &C32Artifact,
    cases: &[Case],
) -> Result<Vec<RoleSummary>> {
    let mut grouped = BTreeMap::<(&'static str, Split, &'static str), Vec<f64>>::new();
    for config in [ALGORITHM_C, ALGORITHM_C1] {
        for case in cases.iter().filter(|case| {
            case.is_person()
                && matches!(
                    case.split,
                    Split::Dev
                        | Split::Validation
                        | Split::InspectedTest
                        | Split::C0Test
                        | Split::Test
                )
        }) {
            let diagnostics = candidate_diagnostics(
                corpus,
                config,
                &case.input,
                case.country_hint.as_deref(),
                case.locale_hint.as_deref(),
            );
            let Some(expected) = case.expected_greeting.as_deref() else {
                continue;
            };
            let Some(expected_candidate) = diagnostics
                .iter()
                .find(|candidate| greeting_matches(Some(expected), Some(&candidate.display)))
            else {
                continue;
            };
            grouped
                .entry((config.name, case.split, "expected_given_candidate"))
                .or_default()
                .push(expected_candidate.role_llr);
            let expected_end = expected_candidate.start + expected_candidate.length;
            for candidate in diagnostics.iter().filter(|candidate| {
                let candidate_end = candidate.start + candidate.length;
                candidate_end <= expected_candidate.start || candidate.start >= expected_end
            }) {
                grouped
                    .entry((
                        config.name,
                        case.split,
                        "disjoint_competing_first_name_candidate",
                    ))
                    .or_default()
                    .push(candidate.role_llr);
            }
        }
    }

    let summaries = grouped
        .into_iter()
        .map(|((algorithm, split, role), mut values)| {
            values.sort_by(f64::total_cmp);
            RoleSummary {
                algorithm,
                split,
                role,
                values,
            }
        })
        .collect::<Vec<_>>();
    let mut writer = csv::Writer::from_path(output.join("role_llr_distribution.csv"))?;
    writer.write_record([
        "algorithm",
        "split",
        "role",
        "n",
        "mean",
        "p10",
        "p50",
        "p90",
    ])?;
    for summary in &summaries {
        writer.write_record([
            summary.algorithm.to_string(),
            summary.split.as_str().to_string(),
            summary.role.to_string(),
            summary.values.len().to_string(),
            format!("{:.6}", mean(&summary.values)),
            format!("{:.6}", quantile(&summary.values, 0.10)),
            format!("{:.6}", quantile(&summary.values, 0.50)),
            format!("{:.6}", quantile(&summary.values, 0.90)),
        ])?;
    }
    writer.flush()?;
    Ok(summaries)
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn quantile(values: &[f64], probability: f64) -> f64 {
    let index = ((values.len() - 1) as f64 * probability).round() as usize;
    values[index]
}

fn write_dev_candidate_traces(
    output: &Path,
    corpus: &C32Artifact,
    cases: &[Case],
    algorithms: &[AlgorithmRun],
) -> Result<()> {
    let mut selected = Vec::<usize>::new();
    for category in [
        "person_compound_given",
        "person_hyphenated",
        "organization_legal_suffix",
    ] {
        selected.extend(
            cases
                .iter()
                .enumerate()
                .filter(|(_, case)| case.split == Split::Dev && case.category == category)
                .take(3)
                .map(|(index, _)| index),
        );
    }
    selected.extend(
        cases
            .iter()
            .enumerate()
            .filter(|(index, case)| {
                case.split == Split::Dev
                    && outcome(
                        case,
                        &algorithms[3].predictions[*index],
                        DEFAULT_REFERENCE_THRESHOLD,
                    ) == CaseOutcome::Wrong
            })
            .take(3)
            .map(|(index, _)| index),
    );
    selected.sort_unstable();
    selected.dedup();
    let mut writer = csv::Writer::from_path(output.join("dev_candidate_traces.csv"))?;
    writer.write_record([
        "case_id",
        "category",
        "input",
        "expected",
        "a_result",
        "a_confidence",
        "b_result",
        "b_confidence",
        "c0_result",
        "c0_confidence",
        "c1_result",
        "c1_confidence",
        "candidate",
        "origin",
        "token_start",
        "token_length",
        "global_given_count",
        "country_given_count",
        "global_surname_count",
        "role_llr",
        "role_signal",
        "reliability",
        "country_support",
        "compound_evidence",
        "compositional_evidence",
        "remainder_evidence",
        "a_candidate_score",
        "b_candidate_score",
        "c1_candidate_score",
    ])?;
    for index in selected {
        let case = &cases[index];
        let diagnostics = candidate_diagnostics(
            corpus,
            ALGORITHM_C1,
            &case.input,
            case.country_hint.as_deref(),
            case.locale_hint.as_deref(),
        );
        if diagnostics.is_empty() {
            let mut row = vec![
                case.id.clone(),
                case.category.clone(),
                case.input.clone(),
                case.expected_greeting
                    .clone()
                    .unwrap_or_else(|| "NULL".to_string()),
                algorithms[0].predictions[index]
                    .greeting_candidate
                    .clone()
                    .unwrap_or_else(|| "NULL".to_string()),
                format!("{:.6}", algorithms[0].predictions[index].confidence),
                algorithms[1].predictions[index]
                    .greeting_candidate
                    .clone()
                    .unwrap_or_else(|| "NULL".to_string()),
                format!("{:.6}", algorithms[1].predictions[index].confidence),
                algorithms[2].predictions[index]
                    .greeting_candidate
                    .clone()
                    .unwrap_or_else(|| "NULL".to_string()),
                format!("{:.6}", algorithms[2].predictions[index].confidence),
                algorithms[3].predictions[index]
                    .greeting_candidate
                    .clone()
                    .unwrap_or_else(|| "NULL".to_string()),
                format!("{:.6}", algorithms[3].predictions[index].confidence),
                "NO_ELIGIBLE_LOOKUP".to_string(),
            ];
            row.resize(29, String::new());
            writer.write_record(row)?;
            continue;
        }
        for diagnostic in diagnostics {
            writer.write_record([
                case.id.clone(),
                case.category.clone(),
                case.input.clone(),
                case.expected_greeting
                    .clone()
                    .unwrap_or_else(|| "NULL".to_string()),
                algorithms[0].predictions[index]
                    .greeting_candidate
                    .clone()
                    .unwrap_or_else(|| "NULL".to_string()),
                format!("{:.6}", algorithms[0].predictions[index].confidence),
                algorithms[1].predictions[index]
                    .greeting_candidate
                    .clone()
                    .unwrap_or_else(|| "NULL".to_string()),
                format!("{:.6}", algorithms[1].predictions[index].confidence),
                algorithms[2].predictions[index]
                    .greeting_candidate
                    .clone()
                    .unwrap_or_else(|| "NULL".to_string()),
                format!("{:.6}", algorithms[2].predictions[index].confidence),
                algorithms[3].predictions[index]
                    .greeting_candidate
                    .clone()
                    .unwrap_or_else(|| "NULL".to_string()),
                format!("{:.6}", algorithms[3].predictions[index].confidence),
                diagnostic.display,
                diagnostic.origin.to_string(),
                diagnostic.start.to_string(),
                diagnostic.length.to_string(),
                diagnostic.global_given_count.to_string(),
                diagnostic.country_given_count.to_string(),
                diagnostic.global_surname_count.to_string(),
                format!("{:.6}", diagnostic.role_llr),
                format!("{:.6}", diagnostic.role_signal),
                format!("{:.6}", diagnostic.reliability),
                format!("{:.6}", diagnostic.country_support),
                format!("{:.6}", diagnostic.compound_evidence),
                format!("{:.6}", diagnostic.compositional_evidence),
                format!("{:.6}", diagnostic.remainder_evidence),
                format!("{:.6}", diagnostic.algorithm_a_score),
                format!("{:.6}", diagnostic.algorithm_b_score),
                format!("{:.6}", diagnostic.score),
            ])?;
        }
    }
    writer.flush()?;
    Ok(())
}

fn write_lexical_audit(output: &Path, audit: LexicalAudit) -> Result<()> {
    let mut writer = csv::Writer::from_path(output.join("lexical_eligibility_audit.csv"))?;
    writer.write_record([
        "total_keys",
        "ineligible_keys",
        "ineligible_key_percentage",
        "total_observations",
        "ineligible_observations",
        "ineligible_observation_percentage",
    ])?;
    writer.write_record([
        audit.total_keys.to_string(),
        audit.ineligible_keys.to_string(),
        format!(
            "{:.6}",
            audit.ineligible_keys as f64 / audit.total_keys as f64 * 100.0
        ),
        audit.total_observations.to_string(),
        audit.ineligible_observations.to_string(),
        format!(
            "{:.6}",
            audit.ineligible_observations as f64 / audit.total_observations as f64 * 100.0
        ),
    ])?;
    writer.flush()?;
    Ok(())
}

fn write_metrics(
    output: &Path,
    cases: &[Case],
    algorithms: &[AlgorithmRun],
    threshold: f64,
) -> Result<()> {
    let mut writer = csv::Writer::from_path(output.join("summary_metrics.csv"))?;
    writer.write_record(metric_header())?;
    let mut scopes = Vec::<(String, Vec<usize>)>::new();
    for split in [
        Split::Regression,
        Split::Dev,
        Split::Validation,
        Split::LegacyTest,
        Split::InspectedTest,
        Split::C0Test,
        Split::Test,
    ] {
        let indices = indices_for(cases, |case| case.split == split);
        if !indices.is_empty() {
            scopes.push((split.as_str().to_string(), indices));
        }
    }
    let mut categories = BTreeMap::<(Split, String), Vec<usize>>::new();
    for (index, case) in cases.iter().enumerate() {
        categories
            .entry((case.split, case.category.clone()))
            .or_default()
            .push(index);
    }
    for ((split, category), indices) in categories {
        scopes.push((format!("{}/{}", split.as_str(), category), indices));
    }
    for algorithm in algorithms {
        for (scope, indices) in &scopes {
            let metrics = metrics_for(cases, &algorithm.predictions, indices, threshold);
            write_metric_row(
                &mut writer,
                algorithm.config.name,
                scope,
                threshold,
                metrics,
            )?;
        }
    }
    writer.flush()?;
    Ok(())
}

fn write_threshold_curves(
    output: &Path,
    cases: &[Case],
    algorithms: &[AlgorithmRun],
) -> Result<()> {
    let mut writer = csv::Writer::from_path(output.join("threshold_curves.csv"))?;
    writer.write_record(metric_header())?;
    for algorithm in algorithms {
        for split in [Split::Dev, Split::Validation, Split::Test] {
            let indices = indices_for(cases, |case| case.split == split);
            if indices.is_empty() {
                continue;
            }
            for threshold in THRESHOLD_SWEEP {
                let metrics = metrics_for(cases, &algorithm.predictions, &indices, threshold);
                write_metric_row(
                    &mut writer,
                    algorithm.config.name,
                    split.as_str(),
                    threshold,
                    metrics,
                )?;
            }
        }
    }
    writer.flush()?;
    Ok(())
}

fn metric_header() -> [&'static str; 19] {
    [
        "algorithm",
        "scope",
        "threshold",
        "cases",
        "person_cases",
        "organization_cases",
        "emitted_greetings",
        "correct_greetings",
        "greeting_precision",
        "greeting_recall",
        "abstention_rate",
        "wrong_greetings",
        "organization_false_positive_rate",
        "organization_false_positives",
        "person_false_negative_rate",
        "person_false_negatives",
        "gender_precision",
        "gender_coverage",
        "gender_abstention_rate",
    ]
}

fn write_metric_row(
    writer: &mut csv::Writer<std::fs::File>,
    algorithm: &str,
    scope: &str,
    threshold: f64,
    metrics: Metrics,
) -> Result<()> {
    writer.write_record([
        algorithm.to_string(),
        scope.to_string(),
        format!("{threshold:.6}"),
        metrics.total.to_string(),
        metrics.person_cases.to_string(),
        metrics.organization_cases.to_string(),
        metrics.emitted.to_string(),
        metrics.correct_greetings.to_string(),
        format_optional(metrics.greeting_precision()),
        format_optional(metrics.greeting_recall()),
        format_optional(metrics.abstention_rate()),
        metrics.wrong_greetings.to_string(),
        format_optional(metrics.organization_false_positive_rate()),
        metrics.organization_false_positives.to_string(),
        format_optional(metrics.person_false_negative_rate()),
        metrics.person_false_negatives.to_string(),
        format_optional(metrics.gender_precision()),
        format_optional(metrics.gender_coverage()),
        format_optional(metrics.gender_abstention_rate()),
    ])?;
    Ok(())
}

fn write_precision_targets(
    output: &Path,
    cases: &[Case],
    algorithms: &[AlgorithmRun],
) -> Result<()> {
    let mut writer = csv::Writer::from_path(output.join("precision_targets.csv"))?;
    writer.write_record([
        "algorithm",
        "split",
        "precision_target",
        "threshold",
        "achieved_precision",
        "recall",
        "emitted",
        "correct_greetings",
        "wrong_greetings",
        "one_sided_95pct_wilson_lower_bound",
        "target_substantiated",
    ])?;
    for algorithm in algorithms {
        for split in [Split::Dev, Split::Validation, Split::Test] {
            let indices = indices_for(cases, |case| case.split == split);
            if indices.is_empty() {
                continue;
            }
            for target in PRECISION_TARGETS {
                let result = find_precision_target(cases, &algorithm.predictions, &indices, target);
                writer.write_record([
                    algorithm.config.name.to_string(),
                    split.as_str().to_string(),
                    format!("{target:.6}"),
                    result.map_or_else(String::new, |result| format!("{:.9}", result.threshold)),
                    result.map_or_else(String::new, |result| {
                        format_optional(result.metrics.greeting_precision())
                    }),
                    result.map_or_else(String::new, |result| {
                        format_optional(result.metrics.greeting_recall())
                    }),
                    result.map_or_else(String::new, |result| result.metrics.emitted.to_string()),
                    result.map_or_else(String::new, |result| {
                        result.metrics.correct_greetings.to_string()
                    }),
                    result.map_or_else(String::new, |result| {
                        result.metrics.wrong_greetings.to_string()
                    }),
                    result.map_or_else(String::new, |result| {
                        format!("{:.6}", precision_lower_bound(result.metrics))
                    }),
                    result.map_or_else(String::new, |result| {
                        (precision_lower_bound(result.metrics) >= target).to_string()
                    }),
                ])?;
            }
        }
    }
    writer.flush()?;
    Ok(())
}

fn write_error_budget_operating_points(
    output: &Path,
    cases: &[Case],
    algorithms: &[AlgorithmRun],
) -> Result<()> {
    let mut writer = csv::Writer::from_path(output.join("error_budget_operating_points.csv"))?;
    writer.write_record([
        "algorithm",
        "split",
        "wrong_greeting_budget",
        "threshold",
        "emitted",
        "correct_greetings",
        "wrong_greetings",
        "greeting_precision",
        "greeting_recall",
    ])?;
    for algorithm in algorithms {
        for split in [Split::Dev, Split::Validation, Split::Test] {
            let indices = indices_for(cases, |case| case.split == split);
            if indices.is_empty() {
                continue;
            }
            for budget in ERROR_BUDGETS {
                let result = find_error_budget(cases, &algorithm.predictions, &indices, budget);
                writer.write_record([
                    algorithm.config.name.to_string(),
                    split.as_str().to_string(),
                    budget.to_string(),
                    result.map_or_else(String::new, |result| format!("{:.9}", result.threshold)),
                    result.map_or_else(String::new, |result| result.metrics.emitted.to_string()),
                    result.map_or_else(String::new, |result| {
                        result.metrics.correct_greetings.to_string()
                    }),
                    result.map_or_else(String::new, |result| {
                        result.metrics.wrong_greetings.to_string()
                    }),
                    result.map_or_else(String::new, |result| {
                        format_optional(result.metrics.greeting_precision())
                    }),
                    result.map_or_else(String::new, |result| {
                        format_optional(result.metrics.greeting_recall())
                    }),
                ])?;
            }
        }
    }
    writer.flush()?;
    Ok(())
}

fn precision_lower_bound(metrics: Metrics) -> f64 {
    precision_lower_bound_counts(metrics.correct_greetings, metrics.emitted)
}

fn precision_lower_bound_counts(correct: usize, emitted: usize) -> f64 {
    if emitted == 0 {
        return 0.0;
    }
    const ONE_SIDED_95_PERCENT_Z: f64 = 1.644_853_626_951_472_2;
    let trials = emitted as f64;
    let proportion = correct as f64 / trials;
    let z_squared = ONE_SIDED_95_PERCENT_Z * ONE_SIDED_95_PERCENT_Z;
    let center = proportion + z_squared / (2.0 * trials);
    let radius = ONE_SIDED_95_PERCENT_Z
        * ((proportion * (1.0 - proportion) + z_squared / (4.0 * trials)) / trials).sqrt();
    ((center - radius) / (1.0 + z_squared / trials)).clamp(0.0, 1.0)
}

fn find_precision_target(
    cases: &[Case],
    predictions: &[RawInference],
    indices: &[usize],
    target: f64,
) -> Option<TargetResult> {
    let mut entries = indices
        .iter()
        .filter_map(|&index| {
            predictions[index]
                .greeting_candidate
                .as_ref()
                .map(|candidate| {
                    (
                        predictions[index].confidence,
                        greeting_matches(
                            cases[index].expected_greeting.as_deref(),
                            Some(candidate),
                        ),
                    )
                })
        })
        .collect::<Vec<_>>();
    entries.sort_by(|left, right| right.0.total_cmp(&left.0));
    let mut emitted = 0_usize;
    let mut correct = 0_usize;
    let mut best = None::<(f64, usize)>;
    let mut position = 0_usize;
    while position < entries.len() {
        let threshold = entries[position].0;
        let mut end = position;
        while end < entries.len() && entries[end].0.total_cmp(&threshold).is_eq() {
            emitted += 1;
            correct += usize::from(entries[end].1);
            end += 1;
        }
        if precision_lower_bound_counts(correct, emitted) >= target
            && best.is_none_or(|(_, best_correct)| correct > best_correct)
        {
            best = Some((threshold, correct));
        }
        position = end;
    }
    best.map(|(threshold, _)| TargetResult {
        threshold,
        metrics: metrics_for(cases, predictions, indices, threshold),
    })
}

fn find_error_budget(
    cases: &[Case],
    predictions: &[RawInference],
    indices: &[usize],
    wrong_budget: usize,
) -> Option<TargetResult> {
    let mut entries = indices
        .iter()
        .filter_map(|&index| {
            predictions[index]
                .greeting_candidate
                .as_ref()
                .map(|candidate| {
                    (
                        predictions[index].confidence,
                        greeting_matches(
                            cases[index].expected_greeting.as_deref(),
                            Some(candidate),
                        ),
                    )
                })
        })
        .collect::<Vec<_>>();
    entries.sort_by(|left, right| right.0.total_cmp(&left.0));
    let mut correct = 0_usize;
    let mut wrong = 0_usize;
    let mut best = None::<(f64, usize, usize)>;
    let mut position = 0_usize;
    while position < entries.len() {
        let threshold = entries[position].0;
        let mut end = position;
        while end < entries.len() && entries[end].0.total_cmp(&threshold).is_eq() {
            correct += usize::from(entries[end].1);
            wrong += usize::from(!entries[end].1);
            end += 1;
        }
        if wrong <= wrong_budget
            && best.is_none_or(|(_, best_correct, best_wrong)| {
                correct > best_correct || (correct == best_correct && wrong < best_wrong)
            })
        {
            best = Some((threshold, correct, wrong));
        }
        position = end;
    }
    best.map(|(threshold, _, _)| TargetResult {
        threshold,
        metrics: metrics_for(cases, predictions, indices, threshold),
    })
}

fn write_results(
    path: &Path,
    cases: &[Case],
    algorithms: &[AlgorithmRun],
    threshold: f64,
    include: impl Fn(&Case) -> bool,
) -> Result<()> {
    let failures_only = path
        .file_name()
        .is_some_and(|name| name == "generated_failures.csv");
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record([
        "algorithm",
        "id",
        "split",
        "category",
        "input",
        "expected",
        "result",
        "confidence",
        "outcome",
        "expected_gender",
        "gender_result",
        "gender_confidence",
        "notes",
    ])?;
    for algorithm in algorithms {
        for (index, case) in cases.iter().enumerate().filter(|(_, case)| include(case)) {
            let prediction = &algorithm.predictions[index];
            let case_outcome = outcome(case, prediction, threshold);
            if failures_only && case_outcome == CaseOutcome::Correct {
                continue;
            }
            writer.write_record([
                algorithm.config.name,
                case.id.as_str(),
                case.split.as_str(),
                case.category.as_str(),
                case.input.as_str(),
                case.expected_greeting.as_deref().unwrap_or("NULL"),
                prediction.greeting_at(threshold).unwrap_or("NULL"),
                &format!("{:.6}", prediction.confidence),
                outcome_name(case_outcome),
                case.expected_gender.map_or("", |gender| gender.as_str()),
                prediction
                    .gender_at(threshold)
                    .map_or("", |gender| gender.as_str()),
                &format!("{:.6}", prediction.gender_confidence),
                case.notes.as_str(),
            ])?;
        }
    }
    writer.flush()?;
    Ok(())
}

fn write_comparison(
    path: &Path,
    cases: &[Case],
    algorithms: &[AlgorithmRun],
    threshold: f64,
    include: impl Fn(&Case) -> bool,
) -> Result<()> {
    let [old, new] = algorithms else {
        return Err("comparison requires exactly two algorithms".into());
    };
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record([
        "change",
        "id",
        "split",
        "category",
        "input",
        "expected",
        "old_result",
        "old_confidence",
        "new_result",
        "new_confidence",
    ])?;
    for (index, case) in cases.iter().enumerate().filter(|(_, case)| include(case)) {
        let old_prediction = &old.predictions[index];
        let new_prediction = &new.predictions[index];
        let old_result = old_prediction.greeting_at(threshold);
        let new_result = new_prediction.greeting_at(threshold);
        let old_outcome = outcome(case, old_prediction, threshold);
        let new_outcome = outcome(case, new_prediction, threshold);
        if old_result == new_result && old_outcome == new_outcome {
            continue;
        }
        let change = if old_outcome != CaseOutcome::Correct && new_outcome == CaseOutcome::Correct {
            "improvement"
        } else if old_outcome == CaseOutcome::Correct && new_outcome != CaseOutcome::Correct {
            "regression"
        } else {
            "changed"
        };
        writer.write_record([
            change,
            case.id.as_str(),
            case.split.as_str(),
            case.category.as_str(),
            case.input.as_str(),
            case.expected_greeting.as_deref().unwrap_or("NULL"),
            old_result.unwrap_or("NULL"),
            &format!("{:.6}", old_prediction.confidence),
            new_result.unwrap_or("NULL"),
            &format!("{:.6}", new_prediction.confidence),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn build_report(
    arguments: &Arguments,
    corpus: &C32Artifact,
    lexical_audit: LexicalAudit,
    seed_statistics: &[SeedStats],
    cases: &[Case],
    algorithms: &[AlgorithmRun],
    role_summaries: &[RoleSummary],
    sealed_run: Option<&SealedRun>,
) -> String {
    let mut report = String::new();
    writeln!(report, "# Greeting-name classifier evaluation\n").unwrap();
    writeln!(
        report,
        "The fixed clean-v1 C32 + q8 baseline plus global-surname q8 sidecar validated before evaluation: {} MPHF keys and {} metadata rows. No surname-only key was added. The normalized likelihood calculation uses {} given-name observations and {} total non-empty surname observations as denominators. Corpus evidence is used only as classifier input; no label is derived from it.\n",
        corpus.key_count(),
        corpus.row_count(),
        corpus.given_total(),
        corpus.surname_total(),
    )
    .unwrap();
    writeln!(report, "## Candidate lexical eligibility\n").unwrap();
    writeln!(
        report,
        "Candidates are eligible only when they contain at least one Unicode alphabetic character and otherwise consist solely of Unicode alphabetic characters, Unicode mark categories, whitespace, apostrophe-like separators, or hyphen-like separators. The rule is applied during classifier lookup; clean-v1 and the C32 artifact are unchanged.\n"
    )
    .unwrap();
    writeln!(
        report,
        "A read-only scan of clean-v1 found **{} ineligible keys out of {} ({:.6}%)**, representing **{} observations out of {} ({:.6}%)**.\n",
        lexical_audit.ineligible_keys,
        lexical_audit.total_keys,
        lexical_audit.ineligible_keys as f64 / lexical_audit.total_keys as f64 * 100.0,
        lexical_audit.ineligible_observations,
        lexical_audit.total_observations,
        lexical_audit.ineligible_observations as f64 / lexical_audit.total_observations as f64
            * 100.0,
    )
    .unwrap();
    writeln!(report, "## Dataset sizes\n").unwrap();
    writeln!(
        report,
        "| Split | Cases | Person | Organization | Role |\n|---|---:|---:|---:|---|"
    )
    .unwrap();
    for split in [
        Split::Regression,
        Split::Dev,
        Split::Validation,
        Split::LegacyTest,
        Split::InspectedTest,
        Split::C0Test,
        Split::Test,
    ] {
        let indices = indices_for(cases, |case| case.split == split);
        if indices.is_empty() {
            continue;
        }
        let people = indices
            .iter()
            .filter(|&&index| cases[index].is_person())
            .count();
        let role = match split {
            Split::Regression => "inspectable behavior/bug corpus; excluded from quality claims",
            Split::Dev => "generated development",
            Split::Validation => "generated model-selection check",
            Split::LegacyTest => "frozen inspected TEST; regression/debug evidence only",
            Split::InspectedTest => "previous generated TEST; inspected regression evidence only",
            Split::C0Test => "C0 generated TEST; now inspected regression evidence only",
            Split::Test => "fresh C1 held-out name partition",
        };
        writeln!(
            report,
            "| {} | {} | {} | {} | {} |",
            split.as_str(),
            indices.len(),
            people,
            indices.len() - people,
            role
        )
        .unwrap();
    }
    writeln!(report, "\nFixture diversity before string generation:\n").unwrap();
    writeln!(report, "| Split | Given-name fixtures | Surname fixtures | Generic organization words | Legal markers |\n|---|---:|---:|---:|---:|").unwrap();
    for stats in seed_statistics {
        writeln!(
            report,
            "| {} | {} | {} | {} | {} |",
            stats.split.as_str(),
            stats.given_names,
            stats.surnames,
            stats.generic_organization_words,
            stats.legal_markers,
        )
        .unwrap();
    }
    writeln!(
        report,
        "\nThe frozen 116-case LEGACY_TEST uses SplitMix64 seed `0x{GENERATOR_SEED:016x}`. The large DEV/VALIDATION and INSPECTED_TEST use seed `0x{LARGE_GENERATOR_SEED:016x}`; INSPECTED_TEST is guarded by SHA-256 `{INSPECTED_TEST_SHA256}`. C0_TEST preserves C0's already-evaluated TEST using seed `0x{C0_TEST_GENERATOR_SEED:016x}` and SHA-256 `{C0_TEST_SHA256}`. The new TEST was snapshotted only after C1 and threshold `{C1_SELECTED_THRESHOLD:.2}` were frozen from DEV/VALIDATION; it uses seed `0x{FRESH_TEST_GENERATOR_SEED:016x}` and SHA-256 `{FRESH_TEST_SHA256}`. Targets are DEV={DEV_TARGET}, VALIDATION={VALIDATION_TARGET}, and each generated test={TEST_TARGET}. Given-name and surname atoms are assigned before generation; the loader rejects cross-split atom reuse. The seed labels are manually curated and were not extracted from clean-v1.\n"
    )
    .unwrap();

    writeln!(report, "## Reference-threshold summary\n").unwrap();
    writeln!(
        report,
        "`{:.2}` is a configurable comparison threshold, not a selected production threshold.\n",
        arguments.reference_threshold
    )
    .unwrap();
    writeln!(report, "| Algorithm | Split | Emitted | Correct | Wrong | Precision | Recall | Abstention | Org FPR | Person FNR | Gender precision | Gender coverage | Gender abstention |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|").unwrap();
    for algorithm in algorithms {
        for split in [
            Split::Regression,
            Split::Dev,
            Split::Validation,
            Split::LegacyTest,
            Split::InspectedTest,
            Split::C0Test,
            Split::Test,
        ] {
            let indices = indices_for(cases, |case| case.split == split);
            if indices.is_empty() {
                continue;
            }
            let metrics = metrics_for(
                cases,
                &algorithm.predictions,
                &indices,
                arguments.reference_threshold,
            );
            writeln!(
                report,
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                algorithm.config.name,
                split.as_str(),
                metrics.emitted,
                metrics.correct_greetings,
                metrics.wrong_greetings,
                percent(metrics.greeting_precision()),
                percent(metrics.greeting_recall()),
                percent(metrics.abstention_rate()),
                percent(metrics.organization_false_positive_rate()),
                percent(metrics.person_false_negative_rate()),
                percent(metrics.gender_precision()),
                percent(metrics.gender_coverage()),
                percent(metrics.gender_abstention_rate()),
            )
            .unwrap();
        }
    }

    writeln!(
        report,
        "\n## Frozen baseline and selected C1 operating points\n"
    )
    .unwrap();
    writeln!(report, "C0 remains frozen at `{C0_FROZEN_THRESHOLD:.2}`. C1's rounded `{C1_SELECTED_THRESHOLD:.2}` threshold was selected on VALIDATION before the new TEST snapshot was evaluated.\n").unwrap();
    writeln!(report, "| Algorithm | Split | Threshold | Emitted | Correct | Wrong | Precision | Recall | Org FPR |\n|---|---|---:|---:|---:|---:|---:|---:|---:|").unwrap();
    for (algorithm, threshold) in [
        (&algorithms[2], C0_FROZEN_THRESHOLD),
        (&algorithms[3], C1_SELECTED_THRESHOLD),
    ] {
        for split in [Split::Validation, Split::C0Test, Split::Test] {
            let indices = indices_for(cases, |case| case.split == split);
            if indices.is_empty() {
                continue;
            }
            let metrics = metrics_for(cases, &algorithm.predictions, &indices, threshold);
            writeln!(
                report,
                "| {} | {} | {:.2} | {} | {} | {} | {} | {} | {} |",
                algorithm.config.name,
                split.as_str(),
                threshold,
                metrics.emitted,
                metrics.correct_greetings,
                metrics.wrong_greetings,
                percent(metrics.greeting_precision()),
                percent(metrics.greeting_recall()),
                percent(metrics.organization_false_positive_rate()),
            )
            .unwrap();
        }
    }

    writeln!(report, "\n## Generated TEST threshold curves\n").unwrap();
    writeln!(report, "| Algorithm | Threshold | Emitted | Correct | Wrong | Precision | Recall | Abstention | Org FPR |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|").unwrap();
    let test_indices = indices_for(cases, |case| case.split == Split::Test);
    for algorithm in algorithms {
        for threshold in THRESHOLD_SWEEP {
            let metrics = metrics_for(cases, &algorithm.predictions, &test_indices, threshold);
            writeln!(
                report,
                "| {} | {:.2} | {} | {} | {} | {} | {} | {} | {} |",
                algorithm.config.name,
                threshold,
                metrics.emitted,
                metrics.correct_greetings,
                metrics.wrong_greetings,
                percent(metrics.greeting_precision()),
                percent(metrics.greeting_recall()),
                percent(metrics.abstention_rate()),
                percent(metrics.organization_false_positive_rate()),
            )
            .unwrap();
        }
    }

    writeln!(report, "\n## Precision-constrained operating points\n").unwrap();
    writeln!(report, "Each row maximizes correct emissions among observed thresholds whose one-sided 95% Wilson lower bound meets the requested target. An empirical percentage alone is not a real-world quality claim, and `n/a` means the generated split cannot substantiate that target.\n").unwrap();
    writeln!(report, "| Algorithm | Split | Target | Threshold | Achieved | Recall | Emitted | Correct | Wrong | 95% lower bound | Supported |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|").unwrap();
    for algorithm in algorithms {
        for split in [Split::Dev, Split::Validation, Split::Test] {
            let indices = indices_for(cases, |case| case.split == split);
            for target in PRECISION_TARGETS {
                if let Some(result) =
                    find_precision_target(cases, &algorithm.predictions, &indices, target)
                {
                    writeln!(
                        report,
                        "| {} | {} | {:.1}% | {:.6} | {} | {} | {} | {} | {} | {:.2}% | {} |",
                        algorithm.config.name,
                        split.as_str(),
                        target * 100.0,
                        result.threshold,
                        percent(result.metrics.greeting_precision()),
                        percent(result.metrics.greeting_recall()),
                        result.metrics.emitted,
                        result.metrics.correct_greetings,
                        result.metrics.wrong_greetings,
                        precision_lower_bound(result.metrics) * 100.0,
                        if precision_lower_bound(result.metrics) >= target {
                            "yes"
                        } else {
                            "no"
                        },
                    )
                    .unwrap();
                } else {
                    writeln!(
                        report,
                        "| {} | {} | {:.1}% | n/a | n/a | n/a | 0 | 0 | n/a | n/a | no |",
                        algorithm.config.name,
                        split.as_str(),
                        target * 100.0,
                    )
                    .unwrap();
                }
            }
        }
    }

    writeln!(report, "\n## Error-budget operating points\n").unwrap();
    writeln!(report, "Each row selects the threshold with maximum correct recall while allowing at most the stated number of wrong greetings. Model selection uses VALIDATION; TEST rows are confirmation only.\n").unwrap();
    writeln!(report, "| Algorithm | Split | Wrong budget | Threshold | Emitted | Correct | Wrong | Precision | Recall |\n|---|---|---:|---:|---:|---:|---:|---:|---:|").unwrap();
    for algorithm in algorithms {
        for split in [Split::Dev, Split::Validation, Split::Test] {
            let indices = indices_for(cases, |case| case.split == split);
            if indices.is_empty() {
                continue;
            }
            for budget in ERROR_BUDGETS {
                let Some(result) =
                    find_error_budget(cases, &algorithm.predictions, &indices, budget)
                else {
                    continue;
                };
                writeln!(
                    report,
                    "| {} | {} | {} | {:.6} | {} | {} | {} | {} | {} |",
                    algorithm.config.name,
                    split.as_str(),
                    budget,
                    result.threshold,
                    result.metrics.emitted,
                    result.metrics.correct_greetings,
                    result.metrics.wrong_greetings,
                    percent(result.metrics.greeting_precision()),
                    percent(result.metrics.greeting_recall()),
                )
                .unwrap();
            }
        }
    }

    write_role_distribution_report(&mut report, role_summaries);
    write_comparison_report(&mut report, arguments, cases, algorithms);
    write_category_report(&mut report, arguments, cases, algorithms);
    write_failure_report(&mut report, arguments, cases, algorithms);
    write_configuration_report(&mut report, algorithms);
    write_sealed_report(&mut report, sealed_run);
    writeln!(report, "\n## Interpretation boundaries\n").unwrap();
    writeln!(report, "- Regression metrics are behavior checks and are never pooled into DEV/VALIDATION/TEST quality metrics.").unwrap();
    writeln!(report, "- LEGACY_TEST, INSPECTED_TEST, and C0_TEST are frozen, inspected snapshots retained only as regression/debug evidence and excluded from primary TEST quality claims.").unwrap();
    writeln!(report, "- Fresh generated TEST was snapshotted before any classifier evaluation and was evaluated once after C1 and its threshold were frozen from DEV/VALIDATION. Its cases and failure rows are not written to the output. It is independent at the labeled-name atom level, but synthetic performance is not a substitute for a sealed real-world corpus.").unwrap();
    writeln!(report, "- Generated transformations share fixture atoms, so Wilson bounds are case-level diagnostics under an independence approximation; they do not measure uncertainty over the universe of names.").unwrap();
    writeln!(report, "- A sealed holdout is checksum-verified before inference, evaluated only with frozen C1 at `0.93`, and reported only through aggregate metrics and coarse confidence buckets. It never enters synthetic result, failure, comparison, threshold-sweep, or trace outputs.").unwrap();
    writeln!(report, "- Person false-negative rate counts person cases where the classifier abstains. Wrong emitted names are reported separately and also reduce greeting recall.").unwrap();
    writeln!(report, "- Precision-target rows are descriptive per split. `Supported=no` explicitly means that the emission count/error rate does not substantiate the target at a one-sided 95% Wilson bound. A production threshold must be selected on DEV/VALIDATION and confirmed once on genuinely sealed data.").unwrap();
    report
}

fn write_role_distribution_report(report: &mut String, summaries: &[RoleSummary]) {
    writeln!(report, "\n## Role-LLR distributions\n").unwrap();
    writeln!(report, "`role_llr = ln P(name|given) - ln P(name|surname)` with add-0.5 smoothing and the full given/surname observation denominators. Competitors below are first-name-index candidates occupying spans disjoint from the independently labeled greeting span.\n").unwrap();
    writeln!(
        report,
        "| Algorithm | Split | Role | n | Mean | p10 | p50 | p90 |\n|---|---|---|---:|---:|---:|---:|---:|"
    )
    .unwrap();
    for summary in summaries {
        writeln!(
            report,
            "| {} | {} | {} | {} | {:.3} | {:.3} | {:.3} | {:.3} |",
            summary.algorithm,
            summary.split.as_str(),
            summary.role,
            summary.values.len(),
            mean(&summary.values),
            quantile(&summary.values, 0.10),
            quantile(&summary.values, 0.50),
            quantile(&summary.values, 0.90),
        )
        .unwrap();
    }
}

fn build_sealed_only_report(sealed_run: &SealedRun) -> String {
    let mut report = String::new();
    writeln!(report, "# Sealed greeting-name evaluation\n").unwrap();
    writeln!(report, "This aggregate-only run did not generate or evaluate synthetic cases, Algorithms A/B/C0, threshold curves, comparisons, candidate traces, or row-level predictions.\n").unwrap();
    write_sealed_report(&mut report, Some(sealed_run));
    writeln!(report, "## Interpretation boundary\n").unwrap();
    writeln!(report, "This command evaluates only frozen C1 at `0.93`. Run the first real holdout evaluation once, and do not inspect individual rows or use these aggregates to retune its threshold. If a row is inspected to design a future algorithm, move it to DEV/regression and do not continue treating this holdout as sealed evidence.\n").unwrap();
    report
}

fn write_sealed_report(report: &mut String, sealed_run: Option<&SealedRun>) {
    writeln!(report, "## Sealed real-world holdout\n").unwrap();
    let Some(sealed_run) = sealed_run else {
        writeln!(report, "No sealed holdout was supplied. Use `--sealed=FILE --sealed-manifest=FILE` only after human labeling and checksum freezing are complete.\n").unwrap();
        return;
    };
    let manifest = &sealed_run.holdout.manifest;
    let evaluation = &sealed_run.evaluation;
    let metrics = evaluation.metrics;
    writeln!(
        report,
        "The holdout was checksum-verified as `{}` before inference. Provenance: {}. Only frozen `{}` at threshold `{:.2}` was evaluated; no row-level output was generated.\n",
        manifest.holdout_sha256,
        markdown(&manifest.provenance),
        ALGORITHM_C1.name,
        evaluation.threshold,
    )
    .unwrap();
    writeln!(report, "| Total labeled | Evaluable | Skipped | Expected greetings | Expected abstentions | Emitted | Correct | Wrong | Expected greetings missed | False emissions on expected abstentions | Precision | Recall | Abstention | Non-person cases | Non-person false positives | Non-person FPR |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|").unwrap();
    writeln!(
        report,
        "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
        metrics.total_labeled_cases,
        metrics.evaluable_cases,
        metrics.skipped_cases,
        metrics.expected_greetings,
        metrics.expected_abstentions,
        metrics.emitted_greetings,
        metrics.correct_greetings,
        metrics.wrong_greetings,
        metrics.expected_greetings_missed,
        metrics.false_emissions_on_expected_abstentions,
        percent(metrics.greeting_precision()),
        percent(metrics.greeting_recall()),
        percent(metrics.abstention_rate()),
        metrics.non_person_cases,
        metrics.non_person_false_positives,
        percent(metrics.non_person_false_positive_rate()),
    )
    .unwrap();
    writeln!(report, "\nCoarse emitted-confidence distribution (diagnostic only; the sealed data must not be used to retune the threshold):\n").unwrap();
    writeln!(
        report,
        "| Confidence | Emitted | Correct | Wrong |\n|---|---:|---:|---:|"
    )
    .unwrap();
    for bucket in evaluation.confidence_buckets {
        writeln!(
            report,
            "| {} | {} | {} | {} |",
            bucket.label, bucket.emitted, bucket.correct, bucket.wrong
        )
        .unwrap();
    }
    writeln!(report).unwrap();
}

fn write_comparison_report(
    report: &mut String,
    arguments: &Arguments,
    cases: &[Case],
    algorithms: &[AlgorithmRun],
) {
    let [a, b, c0, c1] = algorithms else { return };
    writeln!(
        report,
        "\n## A/B/C0/C1 comparison at {:.2}\n",
        arguments.reference_threshold
    )
    .unwrap();
    writeln!(
        report,
        "| Comparison | Split | Improvements | Regressions | Other changed decisions |\n|---|---|---:|---:|---:|"
    )
    .unwrap();
    for (label, old, new) in [("A→B", a, b), ("B→C0", b, c0), ("C0→C1", c0, c1)] {
        for split in [
            Split::Regression,
            Split::Dev,
            Split::Validation,
            Split::LegacyTest,
            Split::InspectedTest,
            Split::C0Test,
            Split::Test,
        ] {
            let mut improvements = 0;
            let mut regressions = 0;
            let mut changed = 0;
            for (index, case) in cases
                .iter()
                .enumerate()
                .filter(|(_, case)| case.split == split)
            {
                let old_outcome =
                    outcome(case, &old.predictions[index], arguments.reference_threshold);
                let new_outcome =
                    outcome(case, &new.predictions[index], arguments.reference_threshold);
                if old_outcome != CaseOutcome::Correct && new_outcome == CaseOutcome::Correct {
                    improvements += 1;
                } else if old_outcome == CaseOutcome::Correct && new_outcome != CaseOutcome::Correct
                {
                    regressions += 1;
                } else if old.predictions[index].greeting_at(arguments.reference_threshold)
                    != new.predictions[index].greeting_at(arguments.reference_threshold)
                {
                    changed += 1;
                }
            }
            writeln!(
                report,
                "| {label} | {} | {improvements} | {regressions} | {changed} |",
                split.as_str()
            )
            .unwrap();
        }
    }

    writeln!(
        report,
        "\nC0→C1 changed-case samples (DEV/VALIDATION only):\n"
    )
    .unwrap();
    writeln!(report, "| Change | Input | Expected | C0 result / confidence | C1 result / confidence |\n|---|---|---|---|---|").unwrap();
    let mut shown = 0;
    for (index, case) in cases
        .iter()
        .enumerate()
        .filter(|(_, case)| matches!(case.split, Split::Dev | Split::Validation))
    {
        let old_outcome = outcome(case, &c0.predictions[index], arguments.reference_threshold);
        let new_outcome = outcome(case, &c1.predictions[index], arguments.reference_threshold);
        let old_result = c0.predictions[index].greeting_at(arguments.reference_threshold);
        let new_result = c1.predictions[index].greeting_at(arguments.reference_threshold);
        if old_result == new_result && old_outcome == new_outcome {
            continue;
        }
        let change = if old_outcome != CaseOutcome::Correct && new_outcome == CaseOutcome::Correct {
            "improvement"
        } else if old_outcome == CaseOutcome::Correct && new_outcome != CaseOutcome::Correct {
            "regression"
        } else {
            "changed"
        };
        writeln!(
            report,
            "| {change} | {} | {} | {} / {:.3} | {} / {:.3} |",
            markdown(&case.input),
            markdown(case.expected_greeting.as_deref().unwrap_or("NULL")),
            markdown(old_result.unwrap_or("NULL")),
            c0.predictions[index].confidence,
            markdown(new_result.unwrap_or("NULL")),
            c1.predictions[index].confidence,
        )
        .unwrap();
        shown += 1;
        if shown == 12 {
            break;
        }
    }
    if shown == 0 {
        writeln!(report, "| none | — | — | — | — |").unwrap();
    }

    let dev = indices_for(cases, |case| case.split == Split::Dev);
    let test = indices_for(cases, |case| case.split == Split::Test);
    for algorithm in algorithms {
        if test.is_empty() {
            continue;
        }
        let dev_metrics = metrics_for(
            cases,
            &algorithm.predictions,
            &dev,
            arguments.reference_threshold,
        );
        let test_metrics = metrics_for(
            cases,
            &algorithm.predictions,
            &test,
            arguments.reference_threshold,
        );
        writeln!(report, "\nFor {}, the DEV→fresh-TEST greeting-precision gap is {} percentage points and recall gap is {} percentage points (TEST minus DEV).",
            algorithm.config.name,
            point_gap(test_metrics.greeting_precision(), dev_metrics.greeting_precision()),
            point_gap(test_metrics.greeting_recall(), dev_metrics.greeting_recall()),
        ).unwrap();
    }
    writeln!(report).unwrap();
}

fn write_category_report(
    report: &mut String,
    arguments: &Arguments,
    cases: &[Case],
    algorithms: &[AlgorithmRun],
) {
    writeln!(report, "## Metrics by new generated TEST category\n").unwrap();
    writeln!(report, "A/B/C0 use the reference threshold; C1 uses its VALIDATION-selected `{C1_SELECTED_THRESHOLD:.2}` operating threshold.\n").unwrap();
    let mut categories = BTreeMap::<String, Vec<usize>>::new();
    for (index, case) in cases
        .iter()
        .enumerate()
        .filter(|(_, case)| case.split == Split::Test)
    {
        categories
            .entry(case.category.clone())
            .or_default()
            .push(index);
    }
    for algorithm in algorithms {
        let threshold = reported_threshold(algorithm, arguments.reference_threshold);
        writeln!(
            report,
            "### {} at {:.2}\n",
            algorithm.config.name, threshold
        )
        .unwrap();
        writeln!(report, "| Category | Cases | Emitted | Correct | Wrong | Precision | Recall | Abstention | Org FPR |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|").unwrap();
        for (category, indices) in &categories {
            let metrics = metrics_for(cases, &algorithm.predictions, indices, threshold);
            writeln!(
                report,
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                markdown(category),
                metrics.total,
                metrics.emitted,
                metrics.correct_greetings,
                metrics.wrong_greetings,
                percent(metrics.greeting_precision()),
                percent(metrics.greeting_recall()),
                percent(metrics.abstention_rate()),
                percent(metrics.organization_false_positive_rate())
            )
            .unwrap();
        }
        writeln!(report).unwrap();
    }
}

fn write_failure_report(
    report: &mut String,
    arguments: &Arguments,
    cases: &[Case],
    algorithms: &[AlgorithmRun],
) {
    let Some(algorithm) = algorithms.last() else {
        return;
    };
    let threshold = reported_threshold(algorithm, arguments.reference_threshold);
    writeln!(
        report,
        "## Algorithm C1 DEV failure samples at {threshold:.2}\n"
    )
    .unwrap();
    writeln!(
        report,
        "| Input | Expected | Result | Confidence | Category |\n|---|---|---|---:|---|"
    )
    .unwrap();
    let mut shown = 0;
    for (index, case) in cases
        .iter()
        .enumerate()
        .filter(|(_, case)| case.split == Split::Dev)
    {
        let prediction = &algorithm.predictions[index];
        if outcome(case, prediction, threshold) == CaseOutcome::Correct {
            continue;
        }
        writeln!(
            report,
            "| {} | {} | {} | {:.3} | {} |",
            markdown(&case.input),
            markdown(case.expected_greeting.as_deref().unwrap_or("NULL")),
            markdown(prediction.greeting_at(threshold).unwrap_or("NULL")),
            prediction.confidence,
            markdown(&case.category)
        )
        .unwrap();
        shown += 1;
        if shown == 15 {
            break;
        }
    }
    if shown == 0 {
        writeln!(report, "| none | — | — | — | — |").unwrap();
    }
    writeln!(report).unwrap();
}

fn write_configuration_report(report: &mut String, algorithms: &[AlgorithmRun]) {
    writeln!(report, "## Algorithm configuration\n").unwrap();
    writeln!(report, "All algorithms return an uncalibrated score before thresholding. A and B are unchanged frequency-led baselines. C0 is the frozen global-role baseline. C1 changes only candidate construction: when an exact compound is absent, it may compose adjacent or hyphen-separated components that are independently given-like. Whitespace composition requires a remainder token because an unsupported two-token input is ambiguous. C1's `{C1_SELECTED_THRESHOLD:.2}` operating threshold was selected on VALIDATION.\n").unwrap();
    writeln!(
        report,
        "| Parameter | A | B | C0 | C1 |\n|---|---:|---:|---:|---:|"
    )
    .unwrap();
    let [a, b, c0, c1] = algorithms else { return };
    macro_rules! config_row {
        ($label:literal, $field:ident) => {
            writeln!(
                report,
                "| {} | {:.3} | {:.3} | {:.3} | {:.3} |",
                $label, a.config.$field, b.config.$field, c0.config.$field, c1.config.$field,
            )
            .unwrap();
        };
    }
    config_row!("frequency floor", frequency_floor);
    config_row!("frequency weight", frequency_weight);
    config_row!("country weight", country_weight);
    config_row!("first-position bonus", first_position_bonus);
    config_row!("last-position bonus", last_position_bonus);
    config_row!("multi-token bonus", multi_token_bonus);
    config_row!("single-display bonus", single_display_bonus);
    config_row!("competition penalty", competition_penalty);
    config_row!("strong-org multiplier", strong_organization_multiplier);
    config_row!("generic-org multiplier", generic_organization_multiplier);
    config_row!("gender emission threshold", gender_threshold);
    config_row!("role score floor", role_score_floor);
    config_row!("role weight", role_weight);
    config_row!("role center", role_center);
    config_row!("role scale", role_scale);
    config_row!("role smoothing", role_smoothing);
    config_row!("role reliability weight", role_reliability_weight);
    config_row!("compound evidence weight", compound_evidence_weight);
    config_row!("remainder role weight", remainder_role_weight);
    config_row!("compositional role floor", compositional_role_floor);
    config_row!(
        "compositional evidence weight",
        compositional_evidence_weight
    );
    config_row!("hyphen structure bonus", hyphen_structure_bonus);
    writeln!(report, "\nC0 and C1 hard-abstain on configured strong legal markers; A/B retain their old multipliers. C0/C1 use normalized global given-versus-surname role evidence, country given-name support, direct compound support, and disjoint competing-role evidence. C1 additionally supports compositional compounds without adding surname-only index keys or changing the binary artifact. Gender decoding is unchanged.\n").unwrap();
}

fn format_optional(value: Option<f64>) -> String {
    value.map_or_else(String::new, |value| format!("{value:.6}"))
}

fn percent(value: Option<f64>) -> String {
    value.map_or_else(
        || "n/a".to_string(),
        |value| format!("{:.2}%", value * 100.0),
    )
}

fn point_gap(left: Option<f64>, right: Option<f64>) -> String {
    match (left, right) {
        (Some(left), Some(right)) => format!("{:+.2}", (left - right) * 100.0),
        _ => "n/a".to_string(),
    }
}

fn markdown(value: &str) -> String {
    value.replace('|', "\\|").replace(['\n', '\r'], " ")
}

fn outcome_name(outcome: CaseOutcome) -> &'static str {
    match outcome {
        CaseOutcome::Correct => "correct",
        CaseOutcome::Wrong => "wrong",
        CaseOutcome::Abstained => "abstained",
    }
}

#[cfg(test)]
mod sealed_report_tests {
    use name_eval::holdout::{
        CaseKind, ConfidenceBucket, HoldoutCase, HoldoutManifest, LabelStatus, SealedMetrics,
    };

    use super::*;

    #[test]
    fn sealed_report_contains_aggregates_but_no_case_rows() {
        let private_name = "Private Display Name";
        let sealed = SealedRun {
            holdout: FrozenHoldout {
                cases: vec![HoldoutCase {
                    id: "case-00000000".to_string(),
                    display_name: private_name.to_string(),
                    country_hint: String::new(),
                    locale_hint: String::new(),
                    label_status: LabelStatus::Greeting,
                    expected_greeting: "Private".to_string(),
                    span_start: Some(0),
                    span_end: Some(7),
                    case_kind: CaseKind::Person,
                }],
                manifest: HoldoutManifest {
                    format_version: 1,
                    holdout_sha256: "0123456789abcdef".to_string(),
                    total_cases: 1,
                    evaluable_cases: 1,
                    skipped_cases: 0,
                    expected_greetings: 1,
                    expected_abstentions: 0,
                    person_cases: 1,
                    non_person_cases: 0,
                    unknown_kind_cases: 0,
                    provenance: "authorized test source".to_string(),
                },
            },
            evaluation: SealedEvaluation {
                threshold: 0.93,
                metrics: SealedMetrics {
                    total_labeled_cases: 1,
                    evaluable_cases: 1,
                    emitted_greetings: 1,
                    correct_greetings: 1,
                    expected_greetings: 1,
                    ..SealedMetrics::default()
                },
                confidence_buckets: [
                    ConfidenceBucket {
                        label: "0.93–0.95",
                        emitted: 1,
                        correct: 1,
                        wrong: 0,
                    },
                    ConfidenceBucket {
                        label: "0.95–0.97",
                        emitted: 0,
                        correct: 0,
                        wrong: 0,
                    },
                    ConfidenceBucket {
                        label: "0.97–0.99",
                        emitted: 0,
                        correct: 0,
                        wrong: 0,
                    },
                    ConfidenceBucket {
                        label: "0.99–1.00",
                        emitted: 0,
                        correct: 0,
                        wrong: 0,
                    },
                ],
            },
        };
        let mut report = String::new();
        write_sealed_report(&mut report, Some(&sealed));
        assert!(report.contains("| 1 | 1 | 0 | 1 |"));
        assert!(!report.contains(private_name));
        assert!(!report.contains("case-00000000"));
    }
}
