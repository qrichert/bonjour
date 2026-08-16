use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::Write as FmtWrite;
use std::path::Path;

use crate::artifact::EvidenceSource;
use crate::classifier::{
    ALGORITHM_C1, ALGORITHM_C2, ALGORITHM_C3, ALGORITHM_C31, RoleInferenceDiagnostic,
    WinnerFeatures, c2_inference_from_diagnostic, c31_inference_from_diagnostic,
    diagnose_role_inference, winner_features,
};
use crate::dataset::{Case, Split, generate_cases};
use crate::metrics::greeting_matches;
use name_eval::holdout::{FrozenHoldout, HoldoutCase};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const V1_SHA256: &str = "de95213f27fc1849032ee6788c8f16d7d515c1a991ae8b2e8414b7b155814c4e";
const V3_SHA256: &str = "d70e4d4b2ed7e49bed09dc1e8d2ba60ade8a752e3b86c772e964bd64883ee6fe";
const PENALTY_STEP: f64 = 0.0025;
const PENALTY_STEPS: usize = 80;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Population {
    SpentProxy,
    Validation,
}

impl Population {
    fn as_str(self) -> &'static str {
        match self {
            Self::SpentProxy => "SPENT_PROXY",
            Self::Validation => "VALIDATION",
        }
    }
}

struct DevelopmentCase {
    id: String,
    display_name: String,
    expected_greeting: Option<String>,
    category: String,
    c2: Decision,
    c3: Decision,
    c31: Decision,
    c3_features: Option<WinnerFeatures>,
}

#[derive(Clone)]
struct Decision {
    winner: Option<String>,
    score: f64,
    emitted: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct EmissionMetrics {
    total: usize,
    expected_greetings: usize,
    expected_nulls: usize,
    emitted: usize,
    correct: usize,
    wrong: usize,
    expected_null_emissions: usize,
}

impl EmissionMetrics {
    fn precision(self) -> Option<f64> {
        ratio(self.correct, self.emitted)
    }

    fn recall(self) -> Option<f64> {
        ratio(self.correct, self.expected_greetings)
    }
}

pub fn run_c31_development(
    output: &Path,
    corpus: &impl EvidenceSource,
    holdout: FrozenHoldout,
    fixtures: &Path,
) -> Result<String> {
    let proxy = build_proxy_cases(corpus, &holdout);
    let validation = build_validation_cases(corpus, fixtures)?;
    let checkpoints = baseline_metrics(&proxy, &validation);
    assert_frozen_checkpoints(&holdout.manifest.holdout_sha256, checkpoints)?;
    let selected = selected_metrics(&proxy, &validation);
    assert_selected_checkpoints(&holdout.manifest.holdout_sha256, selected)?;

    write_selected_metrics(output, checkpoints, selected)?;
    write_delta_cases(output, &proxy)?;
    write_delta_summary(output, &proxy)?;
    write_penalty_sweep(output, &proxy, &validation)?;
    build_report(&holdout, &proxy, checkpoints, selected)
}

fn build_proxy_cases(
    corpus: &impl EvidenceSource,
    holdout: &FrozenHoldout,
) -> Vec<DevelopmentCase> {
    holdout
        .cases
        .iter()
        .filter(|case| case.is_evaluable())
        .map(|case| development_case_from_holdout(corpus, case))
        .collect()
}

fn development_case_from_holdout(
    corpus: &impl EvidenceSource,
    case: &HoldoutCase,
) -> DevelopmentCase {
    development_case(
        corpus,
        &case.id,
        &case.display_name,
        case.expected_greeting(),
        nonempty(&case.country_hint),
        nonempty(&case.locale_hint),
        "real_proxy",
    )
}

fn build_validation_cases(
    corpus: &impl EvidenceSource,
    fixtures: &Path,
) -> Result<Vec<DevelopmentCase>> {
    Ok(generate_cases(fixtures, false)?
        .into_iter()
        .filter(|case| case.split == Split::Validation)
        .map(|case| development_case_from_generated(corpus, &case))
        .collect())
}

fn development_case_from_generated(corpus: &impl EvidenceSource, case: &Case) -> DevelopmentCase {
    development_case(
        corpus,
        &case.id,
        &case.input,
        case.expected_greeting.as_deref(),
        case.country_hint.as_deref(),
        case.locale_hint.as_deref(),
        &case.category,
    )
}

#[allow(clippy::too_many_arguments)]
fn development_case(
    corpus: &impl EvidenceSource,
    id: &str,
    display_name: &str,
    expected_greeting: Option<&str>,
    country_hint: Option<&str>,
    locale_hint: Option<&str>,
    category: &str,
) -> DevelopmentCase {
    let c2_diagnostic = diagnose_role_inference(
        corpus,
        ALGORITHM_C1,
        display_name,
        country_hint,
        locale_hint,
    );
    let c3_diagnostic = diagnose_role_inference(
        corpus,
        ALGORITHM_C3,
        display_name,
        country_hint,
        locale_hint,
    );
    DevelopmentCase {
        id: id.to_string(),
        display_name: display_name.to_string(),
        expected_greeting: expected_greeting.map(str::to_string),
        category: category.to_string(),
        c2: decision(&c2_diagnostic),
        c3: decision(&c3_diagnostic),
        c31: c31_decision(&c3_diagnostic),
        c3_features: winner_features(&c3_diagnostic),
    }
}

fn decision(diagnostic: &RoleInferenceDiagnostic) -> Decision {
    let inference = c2_inference_from_diagnostic(diagnostic, ALGORITHM_C2);
    Decision {
        winner: inference.greeting_candidate.clone(),
        score: inference.confidence,
        emitted: inference
            .greeting_at(ALGORITHM_C2.threshold)
            .map(str::to_string),
    }
}

fn c31_decision(diagnostic: &RoleInferenceDiagnostic) -> Decision {
    let inference = c31_inference_from_diagnostic(diagnostic, ALGORITHM_C2, ALGORITHM_C31);
    Decision {
        winner: inference.greeting_candidate.clone(),
        score: inference.confidence,
        emitted: inference
            .greeting_at(ALGORITHM_C2.threshold)
            .map(str::to_string),
    }
}

fn decision_with_handle_penalty(case: &DevelopmentCase, penalty: f64) -> Decision {
    let mut decision = case.c3.clone();
    if case
        .c3_features
        .as_ref()
        .is_some_and(|features| features.candidate_origin == "handle_segment")
    {
        decision.score = (decision.score - penalty).clamp(0.0, 1.0);
        decision.emitted = (decision.score >= ALGORITHM_C2.threshold)
            .then(|| decision.winner.clone())
            .flatten();
    }
    decision
}

fn baseline_metrics(
    proxy: &[DevelopmentCase],
    validation: &[DevelopmentCase],
) -> [EmissionMetrics; 4] {
    [
        evaluate(proxy, |case| case.c2.clone()),
        evaluate(proxy, |case| case.c3.clone()),
        evaluate(validation, |case| case.c2.clone()),
        evaluate(validation, |case| case.c3.clone()),
    ]
}

fn selected_metrics(
    proxy: &[DevelopmentCase],
    validation: &[DevelopmentCase],
) -> [EmissionMetrics; 2] {
    [
        evaluate(proxy, |case| case.c31.clone()),
        evaluate(validation, |case| case.c31.clone()),
    ]
}

fn evaluate(
    cases: &[DevelopmentCase],
    select: impl Fn(&DevelopmentCase) -> Decision,
) -> EmissionMetrics {
    let mut metrics = EmissionMetrics::default();
    for case in cases {
        let decision = select(case);
        metrics.total += 1;
        if case.expected_greeting.is_some() {
            metrics.expected_greetings += 1;
        } else {
            metrics.expected_nulls += 1;
        }
        if decision.emitted.is_some() {
            metrics.emitted += 1;
            if greeting_matches(
                case.expected_greeting.as_deref(),
                decision.emitted.as_deref(),
            ) {
                metrics.correct += 1;
            } else {
                metrics.wrong += 1;
                if case.expected_greeting.is_none() {
                    metrics.expected_null_emissions += 1;
                }
            }
        }
    }
    metrics
}

fn assert_frozen_checkpoints(digest: &str, metrics: [EmissionMetrics; 4]) -> Result<()> {
    let [c2_proxy, c3_proxy, c2_validation, c3_validation] = metrics;
    let (expected_c2, expected_c3) = match digest {
        V1_SHA256 => (
            EmissionMetrics {
                emitted: 207,
                correct: 207,
                ..c2_proxy
            },
            EmissionMetrics {
                emitted: 234,
                correct: 234,
                ..c3_proxy
            },
        ),
        V3_SHA256 => (
            EmissionMetrics {
                emitted: 205,
                correct: 200,
                wrong: 5,
                expected_null_emissions: 2,
                ..c2_proxy
            },
            EmissionMetrics {
                emitted: 223,
                correct: 217,
                wrong: 6,
                expected_null_emissions: 3,
                ..c3_proxy
            },
        ),
        _ => return Err(format!("unsupported spent proxy digest for C3.1: {digest}").into()),
    };
    if c2_proxy != expected_c2 || c3_proxy != expected_c3 {
        return Err(
            format!("frozen proxy checkpoints changed: C2={c2_proxy:?}, C3={c3_proxy:?}").into(),
        );
    }
    for (algorithm, actual) in [("C2", c2_validation), ("C3", c3_validation)] {
        if actual.emitted != 14_686
            || actual.correct != 14_686
            || actual.wrong != 0
            || actual.expected_null_emissions != 0
        {
            return Err(
                format!("frozen {algorithm} VALIDATION checkpoint changed: {actual:?}").into(),
            );
        }
    }
    Ok(())
}

fn assert_selected_checkpoints(digest: &str, metrics: [EmissionMetrics; 2]) -> Result<()> {
    let [proxy, validation] = metrics;
    let expected_proxy = match digest {
        V1_SHA256 => EmissionMetrics {
            emitted: 226,
            correct: 226,
            ..proxy
        },
        V3_SHA256 => EmissionMetrics {
            emitted: 219,
            correct: 214,
            wrong: 5,
            expected_null_emissions: 2,
            ..proxy
        },
        _ => return Err(format!("unsupported spent proxy digest for C3.1: {digest}").into()),
    };
    if proxy != expected_proxy {
        return Err(format!("selected C3.1 proxy checkpoint changed: {proxy:?}").into());
    }
    if validation.emitted != 14_686
        || validation.correct != 14_686
        || validation.wrong != 0
        || validation.expected_null_emissions != 0
    {
        return Err(format!("selected C3.1 VALIDATION checkpoint changed: {validation:?}").into());
    }
    Ok(())
}

fn write_selected_metrics(
    output: &Path,
    baseline: [EmissionMetrics; 4],
    selected: [EmissionMetrics; 2],
) -> Result<()> {
    let [c2_proxy, c3_proxy, c2_validation, c3_validation] = baseline;
    let [c31_proxy, c31_validation] = selected;
    let mut writer = csv::Writer::from_path(output.join("c31_selected_metrics.csv"))?;
    writer.write_record([
        "population",
        "algorithm",
        "handle_penalty",
        "emitted",
        "correct",
        "wrong",
        "expected_null_emissions",
        "precision",
        "recall",
    ])?;
    for (population, algorithm, penalty, metrics) in [
        (Population::SpentProxy, "C2", None, c2_proxy),
        (Population::SpentProxy, "C3", Some(0.0), c3_proxy),
        (
            Population::SpentProxy,
            "C3.1",
            Some(ALGORITHM_C31.handle_segment_penalty),
            c31_proxy,
        ),
        (Population::Validation, "C2", None, c2_validation),
        (Population::Validation, "C3", Some(0.0), c3_validation),
        (
            Population::Validation,
            "C3.1",
            Some(ALGORITHM_C31.handle_segment_penalty),
            c31_validation,
        ),
    ] {
        writer.write_record([
            population.as_str().to_string(),
            algorithm.to_string(),
            penalty.map_or_else(String::new, |value| format!("{value:.6}")),
            metrics.emitted.to_string(),
            metrics.correct.to_string(),
            metrics.wrong.to_string(),
            metrics.expected_null_emissions.to_string(),
            format_ratio(metrics.precision()),
            format_ratio(metrics.recall()),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn is_delta(case: &DevelopmentCase) -> bool {
    (case.c2.emitted.is_none() && case.c3.emitted.is_some()) || case.c2.winner != case.c3.winner
}

fn delta_outcome(case: &DevelopmentCase) -> &'static str {
    if case.c3.emitted.is_none() {
        "abstained"
    } else if greeting_matches(
        case.expected_greeting.as_deref(),
        case.c3.emitted.as_deref(),
    ) {
        "correct"
    } else if case.expected_greeting.is_none() {
        "expected_null_emission"
    } else {
        "wrong_greeting"
    }
}

fn write_delta_cases(output: &Path, proxy: &[DevelopmentCase]) -> Result<()> {
    let mut writer = csv::Writer::from_path(output.join("c3_delta_cases.csv"))?;
    writer.write_record([
        "id",
        "display_name",
        "expected_greeting",
        "category",
        "c2_winner",
        "c2_score",
        "c2_emitted",
        "c3_winner",
        "c3_candidate_score",
        "c3_emission_score",
        "c3_emitted",
        "c31_emission_score",
        "c31_emitted",
        "c2_abstained_c3_emitted",
        "winner_changed",
        "outcome",
        "candidate_origin",
        "segmentation_mechanism",
        "candidate_alphabetic_length",
        "role_llr",
        "winner_margin",
        "reliability",
        "global_given_count",
        "global_surname_count",
        "candidate_count",
    ])?;
    for case in proxy.iter().filter(|case| is_delta(case)) {
        let features = case.c3_features.as_ref();
        writer.write_record([
            case.id.clone(),
            case.display_name.clone(),
            case.expected_greeting.clone().unwrap_or_default(),
            case.category.clone(),
            case.c2.winner.clone().unwrap_or_default(),
            format!("{:.9}", case.c2.score),
            case.c2.emitted.clone().unwrap_or_default(),
            case.c3.winner.clone().unwrap_or_default(),
            format_optional(features.map(|features| features.winner_score)),
            format!("{:.9}", case.c3.score),
            case.c3.emitted.clone().unwrap_or_default(),
            format!("{:.9}", case.c31.score),
            case.c31.emitted.clone().unwrap_or_default(),
            (case.c2.emitted.is_none() && case.c3.emitted.is_some()).to_string(),
            (case.c2.winner != case.c3.winner).to_string(),
            delta_outcome(case).to_string(),
            features
                .map_or("", |features| features.candidate_origin)
                .to_string(),
            features
                .and_then(|features| features.segmentation_mechanism)
                .unwrap_or("")
                .to_string(),
            features
                .map_or(0, |features| features.alphabetic_length)
                .to_string(),
            format_optional(features.map(|features| features.role_llr)),
            format_optional(features.map(|features| features.winner_margin)),
            format_optional(features.map(|features| features.reliability)),
            features
                .map_or(0, |features| features.global_given_count)
                .to_string(),
            features
                .map_or(0, |features| features.global_surname_count)
                .to_string(),
            features
                .map_or(0, |features| features.candidate_count)
                .to_string(),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn write_delta_summary(output: &Path, proxy: &[DevelopmentCase]) -> Result<()> {
    let mut grouped = BTreeMap::<(&str, &str, &str), usize>::new();
    for case in proxy {
        let mechanism = case
            .c3_features
            .as_ref()
            .and_then(|features| features.segmentation_mechanism)
            .unwrap_or("native");
        let outcome = delta_outcome(case);
        if case.c2.emitted.is_none() && case.c3.emitted.is_some() {
            *grouped
                .entry(("c2_abstained_c3_emitted", mechanism, outcome))
                .or_default() += 1;
        }
        if case.c2.winner != case.c3.winner {
            *grouped
                .entry(("winner_changed", mechanism, outcome))
                .or_default() += 1;
        }
    }
    let mut writer = csv::Writer::from_path(output.join("c3_delta_summary.csv"))?;
    writer.write_record(["delta", "segmentation_mechanism", "outcome", "count"])?;
    for ((delta, mechanism, outcome), count) in grouped {
        writer.write_record([delta, mechanism, outcome, &count.to_string()])?;
    }
    writer.flush()?;
    Ok(())
}

fn write_penalty_sweep(
    output: &Path,
    proxy: &[DevelopmentCase],
    validation: &[DevelopmentCase],
) -> Result<()> {
    let mut writer = csv::Writer::from_path(output.join("c31_penalty_sweep.csv"))?;
    writer.write_record([
        "population",
        "handle_penalty",
        "effective_handle_threshold",
        "total",
        "expected_greetings",
        "expected_nulls",
        "emitted",
        "correct",
        "wrong",
        "expected_null_emissions",
        "precision",
        "recall",
    ])?;
    for step in 0..=PENALTY_STEPS {
        let penalty = step as f64 * PENALTY_STEP;
        for (population, cases) in [
            (Population::SpentProxy, proxy),
            (Population::Validation, validation),
        ] {
            let metrics = evaluate(cases, |case| decision_with_handle_penalty(case, penalty));
            writer.write_record([
                population.as_str().to_string(),
                format!("{penalty:.6}"),
                format!("{:.6}", ALGORITHM_C2.threshold + penalty),
                metrics.total.to_string(),
                metrics.expected_greetings.to_string(),
                metrics.expected_nulls.to_string(),
                metrics.emitted.to_string(),
                metrics.correct.to_string(),
                metrics.wrong.to_string(),
                metrics.expected_null_emissions.to_string(),
                format_ratio(metrics.precision()),
                format_ratio(metrics.recall()),
            ])?;
        }
    }
    writer.flush()?;
    Ok(())
}

fn build_report(
    holdout: &FrozenHoldout,
    proxy: &[DevelopmentCase],
    metrics: [EmissionMetrics; 4],
    selected: [EmissionMetrics; 2],
) -> Result<String> {
    let [c2_proxy, c3_proxy, c2_validation, c3_validation] = metrics;
    let [c31_proxy, c31_validation] = selected;
    let c3_only = proxy
        .iter()
        .filter(|case| case.c2.emitted.is_none() && case.c3.emitted.is_some())
        .count();
    let changed_winner = proxy
        .iter()
        .filter(|case| case.c2.winner != case.c3.winner)
        .count();
    let mut report = String::new();
    writeln!(report, "# C3.1 segmented-candidate delta diagnosis\n")?;
    writeln!(
        report,
        "This command deliberately inspects spent proxy `{}`. It evaluates no fresh proxy and no synthetic TEST split. C2 and C3 are reproduced unchanged before any provenance penalty is explored.\n",
        holdout.manifest.holdout_sha256,
    )?;
    writeln!(report, "## Frozen checkpoints\n")?;
    writeln!(
        report,
        "| Population | Algorithm | Emitted | Correct | Wrong | NULL emissions | Precision | Recall |"
    )?;
    writeln!(
        report,
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    )?;
    for (population, algorithm, values) in [
        ("SPENT_PROXY", "C2", c2_proxy),
        ("SPENT_PROXY", "C3", c3_proxy),
        ("SPENT_PROXY", "C3.1", c31_proxy),
        ("VALIDATION", "C2", c2_validation),
        ("VALIDATION", "C3", c3_validation),
        ("VALIDATION", "C3.1", c31_validation),
    ] {
        writeln!(
            report,
            "| {population} | {algorithm} | {} | {} | {} | {} | {} | {} |",
            values.emitted,
            values.correct,
            values.wrong,
            values.expected_null_emissions,
            percent(values.precision()),
            percent(values.recall()),
        )?;
    }
    writeln!(report, "\n## Delta scope\n")?;
    writeln!(
        report,
        "C2 abstained while C3 emitted in {c3_only} cases; the pre-threshold winner changed in {changed_winner} cases. `c3_delta_cases.csv` contains the spent-only row diagnostics requested for provenance analysis. `c3_delta_summary.csv` aggregates those cases by delta, segmentation mechanism, and outcome.\n"
    )?;
    writeln!(
        report,
        "`c31_penalty_sweep.csv` applies penalties only to handle-segment winners. Native winners retain exactly the frozen C2 score. C3.1 freezes one mechanism-independent penalty of `{:.3}`, making the effective handle-segment threshold `{:.15}` while keeping the public comparison threshold at `{:.15}`. This is spent development evidence, not a quality estimate; C3.1 requires a fresh V4 and must not be used to change C2 or C3.",
        ALGORITHM_C31.handle_segment_penalty,
        ALGORITHM_C2.threshold + ALGORITHM_C31.handle_segment_penalty,
        ALGORITHM_C2.threshold,
    )?;
    Ok(report)
}

fn nonempty(value: &str) -> Option<&str> {
    (!value.is_empty()).then_some(value)
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator != 0).then(|| numerator as f64 / denominator as f64)
}

fn format_optional(value: Option<f64>) -> String {
    value.map_or_else(String::new, |value| format!("{value:.9}"))
}

fn format_ratio(value: Option<f64>) -> String {
    value.map_or_else(String::new, |value| format!("{value:.6}"))
}

fn percent(value: Option<f64>) -> String {
    value.map_or_else(
        || "n/a".to_string(),
        |value| format!("{:.2}%", 100.0 * value),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case(origin: &'static str, score: f64) -> DevelopmentCase {
        DevelopmentCase {
            id: "test".to_string(),
            display_name: "Example42".to_string(),
            expected_greeting: Some("Example".to_string()),
            category: "test".to_string(),
            c2: Decision {
                winner: None,
                score: 0.0,
                emitted: None,
            },
            c3: Decision {
                winner: Some("Example".to_string()),
                score,
                emitted: (score >= ALGORITHM_C2.threshold).then(|| "Example".to_string()),
            },
            c31: Decision {
                winner: Some("Example".to_string()),
                score,
                emitted: (score >= ALGORITHM_C2.threshold).then(|| "Example".to_string()),
            },
            c3_features: Some(WinnerFeatures {
                greeting_candidate: "Example".to_string(),
                winner_score: 1.0,
                second_score: None,
                winner_margin: 1.0,
                no_competitor: true,
                role_llr: 2.0,
                role_signal: 0.9,
                reliability: 0.8,
                global_given_count: 1_000,
                global_surname_count: 10,
                candidate_origin: origin,
                segmentation_mechanism: (origin == "handle_segment").then_some("digit"),
                candidate_count: 1,
                alphabetic_length: 7,
                generic_organization_marker: false,
                ampersand_negative_evidence: false,
            }),
        }
    }

    #[test]
    fn penalty_changes_only_handle_segment_winners() {
        let score = ALGORITHM_C2.threshold + 0.01;
        let native = case("exact", score);
        let segmented = case("handle_segment", score);
        assert_eq!(decision_with_handle_penalty(&native, 0.02).score, score);
        assert!(
            decision_with_handle_penalty(&native, 0.02)
                .emitted
                .is_some()
        );
        assert!(
            decision_with_handle_penalty(&segmented, 0.02)
                .emitted
                .is_none()
        );
    }
}
