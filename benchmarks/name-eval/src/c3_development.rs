use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::Write as FmtWrite;
use std::path::Path;

use crate::artifact::EvidenceSource;
use crate::classifier::{
    ALGORITHM_C1, ALGORITHM_C2, ALGORITHM_C3, RoleInferenceDiagnostic,
    c2_inference_from_diagnostic, diagnose_role_inference,
};
use crate::dataset::{Case, Split, generate_cases};
use crate::metrics::greeting_matches;
use name_eval::holdout::{FrozenHoldout, HoldoutCase};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const C2_PROXY_CORRECT: usize = 207;
const C2_VALIDATION_CORRECT: usize = 14_686;
const C2_NAME: &str = "C2-proxy-calibrated-emission-v1";
const C3_NAME: &str = "C3-conservative-handle-candidates-v1";

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Population {
    Proxy,
    Validation,
}

impl Population {
    fn as_str(self) -> &'static str {
        match self {
            Self::Proxy => "REAL_PROXY_V1_DEV",
            Self::Validation => "VALIDATION",
        }
    }
}

struct DevelopmentCase {
    population: Population,
    id: String,
    display_name: String,
    expected_greeting: Option<String>,
    category: String,
    c2: Decision,
    c3: Decision,
    c1_expected_generated: bool,
    c3_expected_generated: bool,
    c3_expected_origin: Option<&'static str>,
}

#[derive(Clone)]
struct Decision {
    winner: Option<String>,
    score: f64,
    emitted: Option<String>,
}

#[derive(Clone, Copy, Debug, Default)]
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

pub fn run_c3_development(
    output: &Path,
    corpus: &impl EvidenceSource,
    holdout: FrozenHoldout,
    fixtures: &Path,
) -> Result<String> {
    let proxy = build_proxy_cases(corpus, &holdout);
    let validation = build_validation_cases(corpus, fixtures)?;
    let c2_proxy = evaluate(&proxy, |case| &case.c2);
    let c2_validation = evaluate(&validation, |case| &case.c2);
    assert_frozen_c2_checkpoint(c2_proxy, c2_validation)?;

    let c3_proxy = evaluate(&proxy, |case| &case.c3);
    let c3_validation = evaluate(&validation, |case| &case.c3);
    write_metrics(output, c2_proxy, c3_proxy, c2_validation, c3_validation)?;
    write_generation_recovery(output, &proxy)?;
    write_category_metrics(output, &proxy, &validation)?;
    write_changed_cases(output, &proxy, &validation)?;

    if c3_proxy.wrong != 0
        || c3_proxy.expected_null_emissions != 0
        || c3_validation.wrong != 0
        || c3_validation.expected_null_emissions != 0
    {
        let proxy_failures = proxy
            .iter()
            .filter(|case| {
                case.c3.emitted.is_some()
                    && !greeting_matches(
                        case.expected_greeting.as_deref(),
                        case.c3.emitted.as_deref(),
                    )
            })
            .map(|case| {
                format!(
                    "{} {:?} expected {:?} emitted {:?}",
                    case.id, case.display_name, case.expected_greeting, case.c3.emitted
                )
            })
            .collect::<Vec<_>>();
        return Err(format!(
            "C3 violates the zero-development-error constraint: proxy wrong={}, proxy NULL emissions={}, validation wrong={}, validation NULL emissions={}; proxy failures: {}",
            c3_proxy.wrong,
            c3_proxy.expected_null_emissions,
            c3_validation.wrong,
            c3_validation.expected_null_emissions,
            proxy_failures.join("; "),
        )
        .into());
    }

    build_report(
        &holdout,
        &proxy,
        c2_proxy,
        c3_proxy,
        c2_validation,
        c3_validation,
    )
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
        Population::Proxy,
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
        Population::Validation,
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
    population: Population,
    id: &str,
    display_name: &str,
    expected_greeting: Option<&str>,
    country_hint: Option<&str>,
    locale_hint: Option<&str>,
    category: &str,
) -> DevelopmentCase {
    let c1 = diagnose_role_inference(
        corpus,
        ALGORITHM_C1,
        display_name,
        country_hint,
        locale_hint,
    );
    let c3 = diagnose_role_inference(
        corpus,
        ALGORITHM_C3,
        display_name,
        country_hint,
        locale_hint,
    );
    let c1_expected_generated =
        expected_greeting.is_some_and(|expected| expected_candidate(&c1, expected).is_some());
    let c3_expected = expected_greeting.and_then(|expected| expected_candidate(&c3, expected));
    DevelopmentCase {
        population,
        id: id.to_string(),
        display_name: display_name.to_string(),
        expected_greeting: expected_greeting.map(str::to_string),
        category: category.to_string(),
        c2: decision(&c1),
        c3: decision(&c3),
        c1_expected_generated,
        c3_expected_generated: c3_expected.is_some(),
        c3_expected_origin: c3_expected.map(|candidate| candidate.origin),
    }
}

fn expected_candidate<'a>(
    diagnostic: &'a RoleInferenceDiagnostic,
    expected: &str,
) -> Option<&'a crate::classifier::CandidateDiagnostic> {
    diagnostic
        .candidates
        .iter()
        .find(|candidate| greeting_matches(Some(expected), Some(&candidate.display)))
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

fn evaluate(
    cases: &[DevelopmentCase],
    select: impl Fn(&DevelopmentCase) -> &Decision,
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

fn assert_frozen_c2_checkpoint(proxy: EmissionMetrics, validation: EmissionMetrics) -> Result<()> {
    if proxy.emitted != C2_PROXY_CORRECT
        || proxy.correct != C2_PROXY_CORRECT
        || proxy.wrong != 0
        || proxy.expected_null_emissions != 0
    {
        return Err(format!("frozen C2 proxy checkpoint changed: {proxy:?}").into());
    }
    if validation.emitted != C2_VALIDATION_CORRECT
        || validation.correct != C2_VALIDATION_CORRECT
        || validation.wrong != 0
        || validation.expected_null_emissions != 0
    {
        return Err(format!("frozen C2 validation checkpoint changed: {validation:?}").into());
    }
    Ok(())
}

fn write_metrics(
    output: &Path,
    c2_proxy: EmissionMetrics,
    c3_proxy: EmissionMetrics,
    c2_validation: EmissionMetrics,
    c3_validation: EmissionMetrics,
) -> Result<()> {
    let mut writer = csv::Writer::from_path(output.join("c2_c3_development_metrics.csv"))?;
    writer.write_record([
        "population",
        "algorithm",
        "threshold",
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
    for (population, algorithm, metrics) in [
        (Population::Proxy, C2_NAME, c2_proxy),
        (Population::Proxy, C3_NAME, c3_proxy),
        (Population::Validation, C2_NAME, c2_validation),
        (Population::Validation, C3_NAME, c3_validation),
    ] {
        writer.write_record(metric_row(population, algorithm, metrics))?;
    }
    writer.flush()?;
    Ok(())
}

fn metric_row(population: Population, algorithm: &str, metrics: EmissionMetrics) -> Vec<String> {
    vec![
        population.as_str().to_string(),
        algorithm.to_string(),
        format!("{:.17}", ALGORITHM_C2.threshold),
        metrics.total.to_string(),
        metrics.expected_greetings.to_string(),
        metrics.expected_nulls.to_string(),
        metrics.emitted.to_string(),
        metrics.correct.to_string(),
        metrics.wrong.to_string(),
        metrics.expected_null_emissions.to_string(),
        format_ratio(metrics.precision()),
        format_ratio(metrics.recall()),
    ]
}

fn write_generation_recovery(output: &Path, proxy: &[DevelopmentCase]) -> Result<()> {
    let expected = proxy
        .iter()
        .filter(|case| case.expected_greeting.is_some())
        .collect::<Vec<_>>();
    let c1_generated = expected
        .iter()
        .filter(|case| case.c1_expected_generated)
        .count();
    let c3_generated = expected
        .iter()
        .filter(|case| case.c3_expected_generated)
        .count();
    let newly_generated = expected
        .iter()
        .filter(|case| !case.c1_expected_generated && case.c3_expected_generated)
        .count();
    let c1_selected = expected
        .iter()
        .filter(|case| {
            greeting_matches(case.expected_greeting.as_deref(), case.c2.winner.as_deref())
        })
        .count();
    let c3_selected = expected
        .iter()
        .filter(|case| {
            greeting_matches(case.expected_greeting.as_deref(), case.c3.winner.as_deref())
        })
        .count();
    let newly_selected = expected
        .iter()
        .filter(|case| {
            !case.c1_expected_generated
                && case.c3_expected_generated
                && greeting_matches(case.expected_greeting.as_deref(), case.c3.winner.as_deref())
        })
        .count();
    let newly_emitted = expected
        .iter()
        .filter(|case| {
            !case.c1_expected_generated
                && case.c3_expected_generated
                && greeting_matches(
                    case.expected_greeting.as_deref(),
                    case.c3.emitted.as_deref(),
                )
        })
        .count();
    let handle_origin = expected
        .iter()
        .filter(|case| {
            !case.c1_expected_generated && case.c3_expected_origin == Some("handle_segment")
        })
        .count();
    let rows = [
        ("expected_greetings", expected.len()),
        ("c1_matching_candidate_generated", c1_generated),
        ("c3_matching_candidate_generated", c3_generated),
        ("c1_matching_candidate_selected", c1_selected),
        ("c3_matching_candidate_selected", c3_selected),
        ("new_matching_candidate_generated", newly_generated),
        ("new_handle_segment_origin", handle_origin),
        ("new_matching_candidate_selected", newly_selected),
        ("new_matching_candidate_emitted", newly_emitted),
    ];
    let mut writer = csv::Writer::from_path(output.join("c3_generation_recovery.csv"))?;
    writer.write_record(["stage", "count", "share_of_expected_greetings"])?;
    for (stage, count) in rows {
        writer.write_record([
            stage,
            &count.to_string(),
            &format!("{:.6}", count as f64 / expected.len() as f64),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn write_category_metrics(
    output: &Path,
    proxy: &[DevelopmentCase],
    validation: &[DevelopmentCase],
) -> Result<()> {
    let mut categories = BTreeMap::<(Population, String), Vec<&DevelopmentCase>>::new();
    for case in proxy.iter().chain(validation) {
        categories
            .entry((case.population, case.category.clone()))
            .or_default()
            .push(case);
    }
    let mut writer = csv::Writer::from_path(output.join("c2_c3_category_metrics.csv"))?;
    writer.write_record([
        "population",
        "category",
        "algorithm",
        "total",
        "expected_greetings",
        "expected_nulls",
        "emitted",
        "correct",
        "wrong",
        "expected_null_emissions",
    ])?;
    for ((population, category), cases) in categories {
        for (algorithm, select) in [
            (C2_NAME, c2_decision as fn(&DevelopmentCase) -> &Decision),
            (C3_NAME, c3_decision as fn(&DevelopmentCase) -> &Decision),
        ] {
            let metrics = evaluate_refs(&cases, select);
            writer.write_record([
                population.as_str().to_string(),
                category.clone(),
                algorithm.to_string(),
                metrics.total.to_string(),
                metrics.expected_greetings.to_string(),
                metrics.expected_nulls.to_string(),
                metrics.emitted.to_string(),
                metrics.correct.to_string(),
                metrics.wrong.to_string(),
                metrics.expected_null_emissions.to_string(),
            ])?;
        }
    }
    writer.flush()?;
    Ok(())
}

fn evaluate_refs(
    cases: &[&DevelopmentCase],
    select: fn(&DevelopmentCase) -> &Decision,
) -> EmissionMetrics {
    let mut metrics = EmissionMetrics::default();
    for &case in cases {
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

fn c2_decision(case: &DevelopmentCase) -> &Decision {
    &case.c2
}

fn c3_decision(case: &DevelopmentCase) -> &Decision {
    &case.c3
}

fn write_changed_cases(
    output: &Path,
    proxy: &[DevelopmentCase],
    validation: &[DevelopmentCase],
) -> Result<()> {
    let mut writer = csv::Writer::from_path(output.join("c2_c3_changed_cases.csv"))?;
    writer.write_record([
        "population",
        "id",
        "display_name",
        "expected_greeting",
        "category",
        "c2_winner",
        "c2_score",
        "c2_emitted",
        "c3_winner",
        "c3_score",
        "c3_emitted",
        "c1_expected_generated",
        "c3_expected_generated",
        "c3_expected_origin",
    ])?;
    for case in proxy.iter().chain(validation).filter(|case| {
        case.c2.winner != case.c3.winner
            || case.c2.emitted != case.c3.emitted
            || (case.c2.score - case.c3.score).abs() > f64::EPSILON
    }) {
        writer.write_record([
            case.population.as_str().to_string(),
            case.id.clone(),
            case.display_name.clone(),
            case.expected_greeting.clone().unwrap_or_default(),
            case.category.clone(),
            case.c2.winner.clone().unwrap_or_default(),
            format!("{:.6}", case.c2.score),
            case.c2.emitted.clone().unwrap_or_default(),
            case.c3.winner.clone().unwrap_or_default(),
            format!("{:.6}", case.c3.score),
            case.c3.emitted.clone().unwrap_or_default(),
            case.c1_expected_generated.to_string(),
            case.c3_expected_generated.to_string(),
            case.c3_expected_origin.unwrap_or("").to_string(),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn build_report(
    holdout: &FrozenHoldout,
    proxy: &[DevelopmentCase],
    c2_proxy: EmissionMetrics,
    c3_proxy: EmissionMetrics,
    c2_validation: EmissionMetrics,
    c3_validation: EmissionMetrics,
) -> Result<String> {
    let expected = proxy
        .iter()
        .filter(|case| case.expected_greeting.is_some())
        .count();
    let recovered = proxy
        .iter()
        .filter(|case| !case.c1_expected_generated && case.c3_expected_generated)
        .count();
    let c1_generated = proxy
        .iter()
        .filter(|case| case.expected_greeting.is_some() && case.c1_expected_generated)
        .count();
    let c3_generated = proxy
        .iter()
        .filter(|case| case.expected_greeting.is_some() && case.c3_expected_generated)
        .count();
    let c1_selected = proxy
        .iter()
        .filter(|case| {
            case.expected_greeting.is_some()
                && greeting_matches(case.expected_greeting.as_deref(), case.c2.winner.as_deref())
        })
        .count();
    let c3_selected = proxy
        .iter()
        .filter(|case| {
            case.expected_greeting.is_some()
                && greeting_matches(case.expected_greeting.as_deref(), case.c3.winner.as_deref())
        })
        .count();
    let selected = proxy
        .iter()
        .filter(|case| {
            !case.c1_expected_generated
                && case.c3_expected_generated
                && greeting_matches(case.expected_greeting.as_deref(), case.c3.winner.as_deref())
        })
        .count();
    let emitted = proxy
        .iter()
        .filter(|case| {
            !case.c1_expected_generated
                && case.c3_expected_generated
                && greeting_matches(
                    case.expected_greeting.as_deref(),
                    case.c3.emitted.as_deref(),
                )
        })
        .count();
    let mut report = String::new();
    writeln!(report, "# C3 conservative handle-candidate development\n")?;
    writeln!(
        report,
        "This command deliberately uses spent REAL_PROXY_V1_DEV `{}` plus synthetic VALIDATION. It does not load or evaluate V2 or any synthetic TEST split. C2's emission weights and threshold `{:.17}` remain frozen.\n",
        holdout.manifest.holdout_sha256, ALGORITHM_C2.threshold,
    )?;
    writeln!(report, "## Metrics\n")?;
    writeln!(
        report,
        "| Population | Algorithm | Emitted | Correct | Wrong | NULL emissions | Precision | Recall |"
    )?;
    writeln!(
        report,
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    )?;
    for (population, algorithm, metrics) in [
        (Population::Proxy, C2_NAME, c2_proxy),
        (Population::Proxy, C3_NAME, c3_proxy),
        (Population::Validation, C2_NAME, c2_validation),
        (Population::Validation, C3_NAME, c3_validation),
    ] {
        writeln!(
            report,
            "| {} | {algorithm} | {} | {} | {} | {} | {} | {} |",
            population.as_str(),
            metrics.emitted,
            metrics.correct,
            metrics.wrong,
            metrics.expected_null_emissions,
            percent(metrics.precision()),
            percent(metrics.recall()),
        )?;
    }
    writeln!(report, "\n## V1 candidate-generation recovery\n")?;
    writeln!(report, "| Measure | Count | Expected-greeting share |")?;
    writeln!(report, "| --- | ---: | ---: |")?;
    for (label, value) in [
        ("C1 matching candidates generated", c1_generated),
        ("C3 matching candidates generated", c3_generated),
        ("C1 matching candidates selected", c1_selected),
        ("C3 matching candidates selected", c3_selected),
        ("New matching candidates generated", recovered),
        ("New matching candidates selected", selected),
        ("New matching candidates emitted", emitted),
    ] {
        writeln!(
            report,
            "| {label} | {value} | {:.2}% |",
            100.0 * value as f64 / expected as f64
        )?;
    }
    writeln!(report, "\n## Frozen semantics\n")?;
    writeln!(
        report,
        "C3 adds only maximal corpus-backed substrings exposed by ASCII digit runs, `_`/`.` separators, or Unicode lowercase-to-uppercase transitions. Tokens containing any other non-name punctuation or symbol produce no handle-derived candidates. C3 does not scan arbitrary prefixes, split all-lower/all-uppercase concatenations, repair repeated letters, parse URLs/emails, or change scoring, organization evidence, gender, C2 weights, or C2's threshold.\n"
    )?;
    writeln!(
        report,
        "The zero observed development errors are a selection constraint over machine-labeled V1 and synthetic VALIDATION, not held-out quality evidence. C3 must remain frozen and be evaluated once on a fresh, disjoint REAL_PROXY_V3 before any generalization claim."
    )?;
    Ok(report)
}

fn nonempty(value: &str) -> Option<&str> {
    (!value.is_empty()).then_some(value)
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator != 0).then(|| numerator as f64 / denominator as f64)
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

    #[test]
    fn metrics_count_wrong_and_null_emissions_separately() {
        let cases = vec![
            DevelopmentCase {
                population: Population::Proxy,
                id: "positive".to_string(),
                display_name: "Expected Wrong".to_string(),
                expected_greeting: Some("Expected".to_string()),
                category: "test".to_string(),
                c2: Decision {
                    winner: Some("Wrong".to_string()),
                    score: 1.0,
                    emitted: Some("Wrong".to_string()),
                },
                c3: Decision {
                    winner: Some("Expected".to_string()),
                    score: 1.0,
                    emitted: Some("Expected".to_string()),
                },
                c1_expected_generated: false,
                c3_expected_generated: true,
                c3_expected_origin: Some("handle_segment"),
            },
            DevelopmentCase {
                population: Population::Proxy,
                id: "null".to_string(),
                display_name: "Null".to_string(),
                expected_greeting: None,
                category: "test".to_string(),
                c2: Decision {
                    winner: None,
                    score: 0.0,
                    emitted: None,
                },
                c3: Decision {
                    winner: Some("Null".to_string()),
                    score: 1.0,
                    emitted: Some("Null".to_string()),
                },
                c1_expected_generated: false,
                c3_expected_generated: false,
                c3_expected_origin: None,
            },
        ];
        let c2 = evaluate(&cases, |case| &case.c2);
        let c3 = evaluate(&cases, |case| &case.c3);
        assert_eq!(c2.wrong, 1);
        assert_eq!(c2.expected_null_emissions, 0);
        assert_eq!(c3.correct, 1);
        assert_eq!(c3.wrong, 1);
        assert_eq!(c3.expected_null_emissions, 1);
    }
}
