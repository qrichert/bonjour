use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::Path;

use unicode_general_category::{GeneralCategory, get_general_category};

use crate::artifact::EvidenceSource;
use crate::classifier::{
    ALGORITHM_C1, ALGORITHM_C2, C2EmissionConfig, WinnerFeatures, c2_config_is_valid,
    c2_decision_score, c2_inference_from_diagnostic, diagnose_role_inference,
    expected_composition_diagnostic, expected_lookup_diagnostic, winner_features,
};
use crate::dataset::{Case, Split, generate_cases};
use crate::metrics::greeting_matches;
use name_eval::holdout::{FrozenHoldout, HoldoutCase, LabelStatus};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const C1_THRESHOLD: f64 = 0.93;
const C1_PROXY_CORRECT: usize = 34;
const MARGIN_SCALES: [f64; 5] = [0.10, 0.20, 0.30, 0.50, 1.00];
const MINIMUM_LETTERS: [usize; 3] = [1, 2, 3];

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

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum LabelGroup {
    CorrectWinner,
    WrongWinnerPositive,
    CandidateOnExpectedNull,
    NoCandidatePositive,
    NoCandidateNull,
}

impl LabelGroup {
    const ALL: [Self; 5] = [
        Self::CorrectWinner,
        Self::WrongWinnerPositive,
        Self::CandidateOnExpectedNull,
        Self::NoCandidatePositive,
        Self::NoCandidateNull,
    ];

    fn as_str(self) -> &'static str {
        match self {
            Self::CorrectWinner => "correct_winner",
            Self::WrongWinnerPositive => "wrong_winner_positive",
            Self::CandidateOnExpectedNull => "candidate_on_expected_null",
            Self::NoCandidatePositive => "no_candidate_positive",
            Self::NoCandidateNull => "no_candidate_null",
        }
    }
}

#[derive(Clone)]
struct DevelopmentCase {
    population: Population,
    id: String,
    display_name: String,
    expected_greeting: Option<String>,
    category: String,
    group: LabelGroup,
    features: Option<WinnerFeatures>,
    c1_confidence: f64,
}

impl DevelopmentCase {
    fn winner_is_correct(&self) -> bool {
        self.group == LabelGroup::CorrectWinner
    }

    fn c2_score(&self, config: C2EmissionConfig) -> Option<f64> {
        self.features
            .as_ref()
            .map(|features| c2_decision_score(features, config))
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct EmissionMetrics {
    total: usize,
    expected_greetings: usize,
    expected_nulls: usize,
    candidates: usize,
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

    fn wilson_lower_bound(self) -> Option<f64> {
        wilson_lower_bound(self.correct, self.emitted)
    }
}

#[derive(Clone, Copy, Debug)]
struct OperatingPoint {
    config: C2EmissionConfig,
    proxy: EmissionMetrics,
    validation: EmissionMetrics,
}

pub fn run_c2_calibration(
    output: &Path,
    corpus: &impl EvidenceSource,
    holdout: FrozenHoldout,
    fixtures: &Path,
) -> Result<String> {
    let proxy = build_proxy_cases(corpus, &holdout);
    let validation = build_validation_cases(corpus, fixtures)?;
    write_development_cases(output, &proxy, &validation)?;
    write_feature_distributions(output, &proxy, &validation)?;
    let generation_categories = categorize_generation_misses(corpus, &holdout);
    write_generation_categories(output, &generation_categories)?;

    let mut feasible = search_operating_points(&proxy, &validation);
    feasible.sort_by(compare_operating_points);
    write_operating_points(output, &feasible)?;
    let selected = feasible.first().copied();
    if selected.map(|point| point.config) != Some(ALGORITHM_C2) {
        return Err(format!(
            "frozen C2 configuration {:?} differs from deterministic selection {:?}",
            ALGORITHM_C2,
            selected.map(|point| point.config)
        )
        .into());
    }
    write_selected_config(output, selected)?;
    write_metrics(output, &proxy, &validation, selected)?;
    build_report(
        &holdout,
        &proxy,
        &validation,
        &generation_categories,
        selected,
        feasible.len(),
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
        .map(|case| {
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
        })
        .collect()
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
    let diagnostic = diagnose_role_inference(
        corpus,
        ALGORITHM_C1,
        display_name,
        country_hint,
        locale_hint,
    );
    let features = winner_features(&diagnostic);
    let c2 = c2_inference_from_diagnostic(&diagnostic, ALGORITHM_C2);
    debug_assert_eq!(
        c2.greeting_candidate,
        diagnostic.inference.greeting_candidate
    );
    debug_assert_eq!(c2.gender_hint, diagnostic.inference.gender_hint);
    debug_assert_eq!(c2.gender_confidence, diagnostic.inference.gender_confidence);
    let winner = features
        .as_ref()
        .map(|features| features.greeting_candidate.as_str());
    let group = match (expected_greeting, winner) {
        (Some(expected), Some(actual)) if greeting_matches(Some(expected), Some(actual)) => {
            LabelGroup::CorrectWinner
        }
        (Some(_), Some(_)) => LabelGroup::WrongWinnerPositive,
        (None, Some(_)) => LabelGroup::CandidateOnExpectedNull,
        (Some(_), None) => LabelGroup::NoCandidatePositive,
        (None, None) => LabelGroup::NoCandidateNull,
    };
    DevelopmentCase {
        population,
        id: id.to_string(),
        display_name: display_name.to_string(),
        expected_greeting: expected_greeting.map(str::to_string),
        category: category.to_string(),
        group,
        features,
        c1_confidence: diagnostic.inference.confidence,
    }
}

fn search_operating_points(
    proxy: &[DevelopmentCase],
    validation: &[DevelopmentCase],
) -> Vec<OperatingPoint> {
    search_operating_points_above(proxy, validation, C1_PROXY_CORRECT)
}

fn search_operating_points_above(
    proxy: &[DevelopmentCase],
    validation: &[DevelopmentCase],
    minimum_proxy_correct: usize,
) -> Vec<OperatingPoint> {
    let all = proxy.iter().chain(validation).collect::<Vec<_>>();
    let mut feasible = Vec::new();
    for quality_tenths in 0..=10 {
        for margin_tenths in 0..=10 - quality_tenths {
            for role_tenths in 0..=10 - quality_tenths - margin_tenths {
                let reliability_tenths = 10 - quality_tenths - margin_tenths - role_tenths;
                for margin_scale in MARGIN_SCALES {
                    for minimum_candidate_letters in MINIMUM_LETTERS {
                        let mut config = C2EmissionConfig {
                            quality_weight: f64::from(quality_tenths) / 10.0,
                            margin_weight: f64::from(margin_tenths) / 10.0,
                            role_weight: f64::from(role_tenths) / 10.0,
                            reliability_weight: f64::from(reliability_tenths) / 10.0,
                            margin_scale,
                            minimum_candidate_letters,
                            threshold: 1.0,
                        };
                        let maximum_negative = all
                            .iter()
                            .filter(|case| !case.winner_is_correct())
                            .filter_map(|case| case.c2_score(config))
                            .max_by(f64::total_cmp)
                            .unwrap_or(0.0);
                        let threshold = all
                            .iter()
                            .filter(|case| case.winner_is_correct())
                            .filter_map(|case| case.c2_score(config))
                            .filter(|score| *score > maximum_negative)
                            .min_by(f64::total_cmp);
                        let Some(threshold) = threshold else {
                            continue;
                        };
                        config.threshold = threshold;
                        debug_assert!(c2_config_is_valid(config));
                        let proxy_metrics = evaluate_c2(proxy, config);
                        let validation_metrics = evaluate_c2(validation, config);
                        if proxy_metrics.wrong == 0
                            && validation_metrics.wrong == 0
                            && validation_metrics.expected_null_emissions == 0
                            && proxy_metrics.correct > minimum_proxy_correct
                        {
                            feasible.push(OperatingPoint {
                                config,
                                proxy: proxy_metrics,
                                validation: validation_metrics,
                            });
                        }
                    }
                }
            }
        }
    }
    feasible
}

fn compare_operating_points(left: &OperatingPoint, right: &OperatingPoint) -> std::cmp::Ordering {
    right
        .proxy
        .correct
        .cmp(&left.proxy.correct)
        .then_with(|| right.validation.correct.cmp(&left.validation.correct))
        .then_with(|| {
            right
                .config
                .margin_weight
                .total_cmp(&left.config.margin_weight)
        })
        .then_with(|| {
            right
                .config
                .quality_weight
                .total_cmp(&left.config.quality_weight)
        })
        .then_with(|| right.config.role_weight.total_cmp(&left.config.role_weight))
        .then_with(|| {
            right
                .config
                .reliability_weight
                .total_cmp(&left.config.reliability_weight)
        })
        .then_with(|| {
            left.config
                .margin_scale
                .total_cmp(&right.config.margin_scale)
        })
        .then_with(|| {
            right
                .config
                .minimum_candidate_letters
                .cmp(&left.config.minimum_candidate_letters)
        })
        .then_with(|| right.config.threshold.total_cmp(&left.config.threshold))
}

fn evaluate_c2(cases: &[DevelopmentCase], config: C2EmissionConfig) -> EmissionMetrics {
    evaluate(cases, |case| {
        case.c2_score(config)
            .is_some_and(|score| score >= config.threshold)
    })
}

fn evaluate_c1(cases: &[DevelopmentCase]) -> EmissionMetrics {
    evaluate(cases, |case| {
        case.features.is_some() && case.c1_confidence >= C1_THRESHOLD
    })
}

fn evaluate(
    cases: &[DevelopmentCase],
    emitted: impl Fn(&DevelopmentCase) -> bool,
) -> EmissionMetrics {
    let mut metrics = EmissionMetrics::default();
    for case in cases {
        metrics.total += 1;
        if case.expected_greeting.is_some() {
            metrics.expected_greetings += 1;
        } else {
            metrics.expected_nulls += 1;
        }
        if case.features.is_some() {
            metrics.candidates += 1;
        }
        if emitted(case) {
            metrics.emitted += 1;
            if case.winner_is_correct() {
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

fn write_development_cases(
    output: &Path,
    proxy: &[DevelopmentCase],
    validation: &[DevelopmentCase],
) -> Result<()> {
    let mut writer = csv::Writer::from_path(output.join("c2_development_cases.csv"))?;
    writer.write_record([
        "population",
        "id",
        "display_name",
        "expected_greeting",
        "category",
        "label_group",
        "c1_confidence",
        "winner",
        "winner_score",
        "second_score",
        "winner_margin",
        "no_competitor",
        "role_llr",
        "role_signal",
        "reliability",
        "global_given_count",
        "global_surname_count",
        "candidate_origin",
        "candidate_count",
        "alphabetic_length",
        "generic_organization_marker",
        "ampersand_negative_evidence",
    ])?;
    for case in proxy.iter().chain(validation) {
        let features = case.features.as_ref();
        writer.write_record([
            case.population.as_str().to_string(),
            case.id.clone(),
            case.display_name.clone(),
            case.expected_greeting.clone().unwrap_or_default(),
            case.category.clone(),
            case.group.as_str().to_string(),
            format!("{:.6}", case.c1_confidence),
            features.map_or_else(String::new, |features| features.greeting_candidate.clone()),
            optional_feature(features, |features| features.winner_score),
            features
                .and_then(|features| features.second_score)
                .map_or_else(String::new, |value| format!("{value:.6}")),
            optional_feature(features, |features| features.winner_margin),
            features.map_or_else(String::new, |features| features.no_competitor.to_string()),
            optional_feature(features, |features| features.role_llr),
            optional_feature(features, |features| features.role_signal),
            optional_feature(features, |features| features.reliability),
            features.map_or_else(String::new, |features| {
                features.global_given_count.to_string()
            }),
            features.map_or_else(String::new, |features| {
                features.global_surname_count.to_string()
            }),
            features.map_or_else(String::new, |features| {
                features.candidate_origin.to_string()
            }),
            features.map_or_else(String::new, |features| features.candidate_count.to_string()),
            features.map_or_else(String::new, |features| {
                features.alphabetic_length.to_string()
            }),
            features.map_or_else(String::new, |features| {
                features.generic_organization_marker.to_string()
            }),
            features.map_or_else(String::new, |features| {
                features.ampersand_negative_evidence.to_string()
            }),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn write_feature_distributions(
    output: &Path,
    proxy: &[DevelopmentCase],
    validation: &[DevelopmentCase],
) -> Result<()> {
    let mut writer = csv::Writer::from_path(output.join("c2_feature_distributions.csv"))?;
    writer.write_record([
        "population",
        "label_group",
        "group_cases",
        "feature",
        "n",
        "minimum",
        "p10",
        "median",
        "p90",
        "maximum",
    ])?;
    for (population, cases) in [
        (Population::Proxy, proxy),
        (Population::Validation, validation),
    ] {
        for group in LabelGroup::ALL {
            let grouped = cases
                .iter()
                .filter(|case| case.group == group)
                .collect::<Vec<_>>();
            for (name, values) in feature_values(&grouped) {
                let summary = summarize(values);
                writer.write_record([
                    population.as_str().to_string(),
                    group.as_str().to_string(),
                    grouped.len().to_string(),
                    name.to_string(),
                    summary.count.to_string(),
                    optional_f64(summary.minimum),
                    optional_f64(summary.p10),
                    optional_f64(summary.median),
                    optional_f64(summary.p90),
                    optional_f64(summary.maximum),
                ])?;
            }
        }
    }
    writer.flush()?;
    Ok(())
}

fn feature_values(cases: &[&DevelopmentCase]) -> Vec<(&'static str, Vec<f64>)> {
    let values = |feature: fn(&DevelopmentCase, &WinnerFeatures) -> f64| {
        cases
            .iter()
            .filter_map(|case| {
                case.features
                    .as_ref()
                    .map(|features| feature(case, features))
            })
            .collect::<Vec<_>>()
    };
    vec![
        ("winner_score", values(|_, features| features.winner_score)),
        ("c1_confidence", values(|case, _| case.c1_confidence)),
        (
            "winner_margin",
            values(|_, features| features.winner_margin),
        ),
        ("role_llr", values(|_, features| features.role_llr)),
        ("role_signal", values(|_, features| features.role_signal)),
        ("reliability", values(|_, features| features.reliability)),
        (
            "global_given_count",
            values(|_, features| features.global_given_count as f64),
        ),
        (
            "global_surname_count",
            values(|_, features| features.global_surname_count as f64),
        ),
        (
            "candidate_count",
            values(|_, features| features.candidate_count as f64),
        ),
        (
            "alphabetic_length",
            values(|_, features| features.alphabetic_length as f64),
        ),
    ]
}

fn write_operating_points(output: &Path, feasible: &[OperatingPoint]) -> Result<()> {
    let mut writer = csv::Writer::from_path(output.join("c2_operating_points.csv"))?;
    writer.write_record(operating_point_header())?;
    for (rank, point) in feasible.iter().take(200).enumerate() {
        writer.write_record(operating_point_row(rank + 1, point))?;
    }
    writer.flush()?;
    Ok(())
}

fn operating_point_header() -> [&'static str; 21] {
    [
        "rank",
        "quality_weight",
        "margin_weight",
        "role_weight",
        "reliability_weight",
        "margin_scale",
        "minimum_candidate_letters",
        "threshold",
        "proxy_emitted",
        "proxy_correct",
        "proxy_wrong",
        "proxy_precision",
        "proxy_recall",
        "proxy_wilson_lower_95_one_sided",
        "validation_emitted",
        "validation_correct",
        "validation_wrong",
        "validation_precision",
        "validation_recall",
        "validation_wilson_lower_95_one_sided",
        "validation_expected_null_emissions",
    ]
}

fn operating_point_row(rank: usize, point: &OperatingPoint) -> Vec<String> {
    vec![
        rank.to_string(),
        format!("{:.2}", point.config.quality_weight),
        format!("{:.2}", point.config.margin_weight),
        format!("{:.2}", point.config.role_weight),
        format!("{:.2}", point.config.reliability_weight),
        format!("{:.2}", point.config.margin_scale),
        point.config.minimum_candidate_letters.to_string(),
        format!("{:.17}", point.config.threshold),
        point.proxy.emitted.to_string(),
        point.proxy.correct.to_string(),
        point.proxy.wrong.to_string(),
        format_ratio(point.proxy.precision()),
        format_ratio(point.proxy.recall()),
        format_ratio(point.proxy.wilson_lower_bound()),
        point.validation.emitted.to_string(),
        point.validation.correct.to_string(),
        point.validation.wrong.to_string(),
        format_ratio(point.validation.precision()),
        format_ratio(point.validation.recall()),
        format_ratio(point.validation.wilson_lower_bound()),
        point.validation.expected_null_emissions.to_string(),
    ]
}

fn write_selected_config(output: &Path, selected: Option<OperatingPoint>) -> Result<()> {
    let mut writer = csv::Writer::from_path(output.join("c2_selected_config.csv"))?;
    writer.write_record(operating_point_header())?;
    if let Some(point) = selected {
        writer.write_record(operating_point_row(1, &point))?;
    }
    writer.flush()?;
    Ok(())
}

fn write_metrics(
    output: &Path,
    proxy: &[DevelopmentCase],
    validation: &[DevelopmentCase],
    selected: Option<OperatingPoint>,
) -> Result<()> {
    let mut writer = csv::Writer::from_path(output.join("c1_c2_development_metrics.csv"))?;
    writer.write_record([
        "algorithm",
        "population",
        "threshold",
        "total",
        "expected_greetings",
        "expected_nulls",
        "candidates",
        "emitted",
        "correct",
        "wrong",
        "expected_null_emissions",
        "precision",
        "recall",
        "wilson_lower_95_one_sided",
    ])?;
    for (population, cases) in [
        (Population::Proxy, proxy),
        (Population::Validation, validation),
    ] {
        write_metric_row(
            &mut writer,
            "C1-compositional-role-v1",
            population,
            C1_THRESHOLD,
            evaluate_c1(cases),
        )?;
        if let Some(point) = selected {
            let metrics = if population == Population::Proxy {
                point.proxy
            } else {
                point.validation
            };
            write_metric_row(
                &mut writer,
                "C2-proxy-calibrated-emission-v1",
                population,
                point.config.threshold,
                metrics,
            )?;
        }
    }
    writer.flush()?;
    Ok(())
}

fn write_metric_row(
    writer: &mut csv::Writer<fs::File>,
    algorithm: &str,
    population: Population,
    threshold: f64,
    metrics: EmissionMetrics,
) -> Result<()> {
    writer.write_record([
        algorithm.to_string(),
        population.as_str().to_string(),
        format!("{threshold:.17}"),
        metrics.total.to_string(),
        metrics.expected_greetings.to_string(),
        metrics.expected_nulls.to_string(),
        metrics.candidates.to_string(),
        metrics.emitted.to_string(),
        metrics.correct.to_string(),
        metrics.wrong.to_string(),
        metrics.expected_null_emissions.to_string(),
        format_ratio(metrics.precision()),
        format_ratio(metrics.recall()),
        format_ratio(metrics.wilson_lower_bound()),
    ])?;
    Ok(())
}

fn categorize_generation_misses(
    corpus: &impl EvidenceSource,
    holdout: &FrozenHoldout,
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for case in holdout
        .cases
        .iter()
        .filter(|case| case.label_status == LabelStatus::Greeting)
    {
        let expected = case
            .expected_greeting()
            .expect("greeting-labeled case has an expected greeting");
        let lookup = expected_lookup_diagnostic(
            corpus,
            ALGORITHM_C1,
            expected,
            nonempty(&case.country_hint),
            nonempty(&case.locale_hint),
        );
        let composition = expected_composition_diagnostic(
            corpus,
            ALGORITHM_C1,
            expected,
            nonempty(&case.country_hint),
            nonempty(&case.locale_hint),
        );
        let diagnostic = diagnose_role_inference(
            corpus,
            ALGORITHM_C1,
            &case.display_name,
            nonempty(&case.country_hint),
            nonempty(&case.locale_hint),
        );
        let generated = diagnostic
            .candidates
            .iter()
            .any(|candidate| greeting_matches(Some(expected), Some(&candidate.display)));
        if generated || (lookup.evidence.is_none() && !composition.supported) {
            continue;
        }
        increment(&mut counts, "all_generation_misses");
        categorize_generation_miss(case, &mut counts);
    }
    counts
}

fn categorize_generation_miss(case: &HoldoutCase, counts: &mut BTreeMap<&'static str, usize>) {
    let expected = case
        .expected_greeting()
        .expect("greeting-labeled case has an expected greeting");
    let start = case.span_start.unwrap_or(0);
    let end = case.span_end.unwrap_or(start + expected.len());
    let (token_start, token_end) = containing_token_bounds(&case.display_name, start, end);
    let containing_token = &case.display_name[token_start..token_end];
    let embedded = start > token_start || end < token_end;
    if embedded {
        increment(counts, "embedded_in_larger_whitespace_free_token");
    }
    if containing_token.bytes().any(|byte| byte.is_ascii_digit()) {
        increment(counts, "containing_token_has_ascii_digit");
    }
    if containing_token
        .chars()
        .any(is_ineligible_punctuation_or_symbol)
    {
        increment(
            counts,
            "containing_token_has_ineligible_punctuation_or_symbol",
        );
    }
    if embedded && looks_camel_case(containing_token) {
        increment(counts, "concatenated_or_camel_case_like");
    }
    let whitespace_tokens = expected.split_whitespace().count();
    if expected.contains(['\'', '’', 'ʼ', 'ʻ']) {
        increment(counts, "expected_contains_apostrophe");
    }
    if expected.contains(['-', '‐', '‑', '‒', '–', '—']) {
        increment(counts, "expected_contains_hyphen");
    }
    if whitespace_tokens == 2 {
        increment(counts, "expected_two_token_whitespace");
    } else if whitespace_tokens >= 3 {
        increment(counts, "expected_three_or_more_tokens");
    } else if !embedded {
        increment(counts, "ordinary_standalone_single_token");
    }
    if !embedded
        && whitespace_tokens == 1
        && !expected.contains(['\'', '’', 'ʼ', 'ʻ', '-', '‐', '‑', '‒', '–', '—'])
    {
        increment(counts, "other_or_unclassified");
    }
}

fn containing_token_bounds(display: &str, start: usize, end: usize) -> (usize, usize) {
    let token_start = display[..start]
        .char_indices()
        .rev()
        .find(|(_, character)| character.is_whitespace())
        .map_or(0, |(index, character)| index + character.len_utf8());
    let token_end = display[end..]
        .char_indices()
        .find(|(_, character)| character.is_whitespace())
        .map_or(display.len(), |(index, _)| end + index);
    (token_start, token_end)
}

fn is_ineligible_punctuation_or_symbol(character: char) -> bool {
    if character.is_alphabetic()
        || character.is_ascii_digit()
        || matches!(
            get_general_category(character),
            GeneralCategory::NonspacingMark
                | GeneralCategory::SpacingMark
                | GeneralCategory::EnclosingMark
        )
        || matches!(
            character,
            '\'' | '‘' | '’' | '‛' | 'ʻ' | 'ʼ' | '＇' | '-' | '‐' | '‑' | '‒' | '–' | '—'
        )
    {
        return false;
    }
    true
}

fn looks_camel_case(value: &str) -> bool {
    value
        .chars()
        .zip(value.chars().skip(1))
        .any(|(left, right)| left.is_lowercase() && right.is_uppercase())
}

fn increment(counts: &mut BTreeMap<&'static str, usize>, category: &'static str) {
    *counts.entry(category).or_default() += 1;
}

fn write_generation_categories(
    output: &Path,
    counts: &BTreeMap<&'static str, usize>,
) -> Result<()> {
    let total = counts.get("all_generation_misses").copied().unwrap_or(0);
    let mut writer = csv::Writer::from_path(output.join("candidate_generation_categories.csv"))?;
    writer.write_record(["category", "count", "share_of_generation_misses"])?;
    for (&category, &value) in counts {
        writer.write_record([
            category,
            &value.to_string(),
            &format_ratio(ratio(value, total)),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn build_report(
    holdout: &FrozenHoldout,
    proxy: &[DevelopmentCase],
    validation: &[DevelopmentCase],
    generation_categories: &BTreeMap<&'static str, usize>,
    selected: Option<OperatingPoint>,
    feasible_count: usize,
) -> Result<String> {
    let mut report = String::new();
    writeln!(report, "# C2 proxy-calibrated emission development\n")?;
    writeln!(
        report,
        "REAL_PROXY_V1 checksum `{}` is inspected DEV evidence. C2 reuses C1 candidate generation and ranking unchanged; it may only emit C1's winner or abstain. Synthetic TEST splits were not inferred.\n",
        holdout.manifest.holdout_sha256
    )?;
    writeln!(report, "## Development group sizes\n")?;
    writeln!(report, "| Population | Group | Cases |")?;
    writeln!(report, "| --- | --- | ---: |")?;
    for (population, cases) in [
        (Population::Proxy, proxy),
        (Population::Validation, validation),
    ] {
        for group in LabelGroup::ALL {
            writeln!(
                report,
                "| {} | `{}` | {} |",
                population.as_str(),
                group.as_str(),
                cases.iter().filter(|case| case.group == group).count()
            )?;
        }
    }
    writeln!(report, "\n## Selected operating point\n")?;
    if let Some(point) = selected {
        writeln!(
            report,
            "The deterministic search found {feasible_count} feasible configurations. The selected score is `quality={:.2}, margin={:.2}, role={:.2}, reliability={:.2}`, with margin scale `{:.2}`, minimum candidate letters `{}`, and threshold `{:.17}`.\n",
            point.config.quality_weight,
            point.config.margin_weight,
            point.config.role_weight,
            point.config.reliability_weight,
            point.config.margin_scale,
            point.config.minimum_candidate_letters,
            point.config.threshold,
        )?;
        writeln!(
            report,
            "| Population | Algorithm | Emitted | Correct | Wrong | Precision | Recall | One-sided 95% Wilson lower bound |"
        )?;
        writeln!(
            report,
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"
        )?;
        for (population, c1, c2) in [
            (Population::Proxy, evaluate_c1(proxy), point.proxy),
            (
                Population::Validation,
                evaluate_c1(validation),
                point.validation,
            ),
        ] {
            for (algorithm, metrics) in [("C1", c1), ("C2", c2)] {
                writeln!(
                    report,
                    "| {} | {algorithm} | {} | {} | {} | {} | {} | {} |",
                    population.as_str(),
                    metrics.emitted,
                    metrics.correct,
                    metrics.wrong,
                    percent(metrics.precision()),
                    percent(metrics.recall()),
                    percent(metrics.wilson_lower_bound()),
                )?;
            }
        }
    } else {
        writeln!(
            report,
            "No configuration satisfied the locked zero-error constraints while improving on C1's 34 proxy emissions. C2 must not be frozen.\n"
        )?;
    }
    writeln!(report, "\n## Candidate-generation diagnostics\n")?;
    writeln!(report, "| Cross-cutting category | Count |")?;
    writeln!(report, "| --- | ---: |")?;
    for (category, count) in generation_categories {
        writeln!(report, "| `{category}` | {count} |")?;
    }
    writeln!(
        report,
        "\nFeature quantiles, the selected row, bounded operating points, aggregate generation categories, and the private development rows are written as CSV files. V1 labels remain unchanged. Scores are decision scores, not probabilities; proxy and synthetic Wilson intervals are reported separately and are not pooled."
    )?;
    Ok(report)
}

#[derive(Default)]
struct DistributionSummary {
    count: usize,
    minimum: Option<f64>,
    p10: Option<f64>,
    median: Option<f64>,
    p90: Option<f64>,
    maximum: Option<f64>,
}

fn summarize(mut values: Vec<f64>) -> DistributionSummary {
    if values.is_empty() {
        return DistributionSummary::default();
    }
    values.sort_by(f64::total_cmp);
    DistributionSummary {
        count: values.len(),
        minimum: values.first().copied(),
        p10: quantile(&values, 0.10),
        median: quantile(&values, 0.50),
        p90: quantile(&values, 0.90),
        maximum: values.last().copied(),
    }
}

fn quantile(values: &[f64], probability: f64) -> Option<f64> {
    (!values.is_empty()).then(|| {
        let index = ((values.len() - 1) as f64 * probability).round() as usize;
        values[index]
    })
}

fn wilson_lower_bound(correct: usize, emitted: usize) -> Option<f64> {
    if emitted == 0 {
        return None;
    }
    const Z: f64 = 1.644_853_626_951_472_2;
    let n = emitted as f64;
    let p = correct as f64 / n;
    let z2 = Z * Z;
    Some((p + z2 / (2.0 * n) - Z * ((p * (1.0 - p) + z2 / (4.0 * n)) / n).sqrt()) / (1.0 + z2 / n))
}

fn optional_feature(
    features: Option<&WinnerFeatures>,
    feature: impl Fn(&WinnerFeatures) -> f64,
) -> String {
    features.map_or_else(String::new, |features| format!("{:.6}", feature(features)))
}

fn optional_f64(value: Option<f64>) -> String {
    value.map_or_else(String::new, |value| format!("{value:.6}"))
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

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator != 0).then(|| numerator as f64 / denominator as f64)
}

fn nonempty(value: &str) -> Option<&str> {
    (!value.is_empty()).then_some(value)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::artifact::Evidence;
    use name_eval::holdout::{CaseKind, HoldoutManifest};

    use super::*;

    struct FakeCorpus(HashMap<String, Evidence>);

    impl EvidenceSource for FakeCorpus {
        fn lookup(&self, name: &str, _country_hint: Option<[u8; 2]>) -> Option<Evidence> {
            self.0.get(name).copied()
        }
    }

    fn evidence() -> Evidence {
        Evidence {
            global_count: 100_000,
            country_count: 0,
            effective_count: 100_000,
            female_count: 0,
            male_count: 100_000,
            surname_count: 100,
            given_total: 444_154_759,
            surname_total: 489_631_377,
        }
    }

    fn feature_case(
        population: Population,
        id: &str,
        group: LabelGroup,
        role_signal: f64,
        reliability: f64,
    ) -> DevelopmentCase {
        DevelopmentCase {
            population,
            id: id.to_string(),
            display_name: id.to_string(),
            expected_greeting: (group != LabelGroup::CandidateOnExpectedNull)
                .then(|| "Expected".to_string()),
            category: "test".to_string(),
            group,
            features: Some(WinnerFeatures {
                greeting_candidate: "Winner".to_string(),
                winner_score: 0.7,
                second_score: Some(0.5),
                winner_margin: 0.2,
                no_competitor: false,
                role_llr: 1.0,
                role_signal,
                reliability,
                global_given_count: 1_000,
                global_surname_count: 10,
                candidate_origin: "exact",
                candidate_count: 2,
                alphabetic_length: 6,
                generic_organization_marker: false,
                ampersand_negative_evidence: false,
            }),
            c1_confidence: 0.7,
        }
    }

    #[test]
    fn weight_grid_contains_expected_number_of_convex_combinations() {
        let count = (0..=10)
            .flat_map(|quality| (0..=10 - quality).map(move |margin| (quality, margin)))
            .flat_map(|(quality, margin)| 0..=10 - quality - margin)
            .count();
        assert_eq!(count, 286);
    }

    #[test]
    fn quantiles_are_deterministic() {
        let summary = summarize(vec![5.0, 1.0, 4.0, 2.0, 3.0]);
        assert_eq!(summary.minimum, Some(1.0));
        assert_eq!(summary.p10, Some(1.0));
        assert_eq!(summary.median, Some(3.0));
        assert_eq!(summary.p90, Some(5.0));
        assert_eq!(summary.maximum, Some(5.0));
    }

    #[test]
    fn wilson_bounds_are_not_pooled_and_require_emissions() {
        assert_eq!(wilson_lower_bound(0, 0), None);
        assert!(wilson_lower_bound(100, 100).is_some_and(|value| value < 1.0));
    }

    #[test]
    fn containing_token_bounds_preserve_unicode_byte_offsets() {
        let display = "Pré_Élodie42 Dupont";
        let start = display.find("Élodie").unwrap();
        let end = start + "Élodie".len();
        let (token_start, token_end) = containing_token_bounds(display, start, end);
        assert_eq!(&display[token_start..token_end], "Pré_Élodie42");
    }

    #[test]
    fn development_groups_cover_positive_null_and_missing_candidates() {
        let corpus = FakeCorpus(HashMap::from([("Quentin".to_string(), evidence())]));
        assert_eq!(
            development_case(
                &corpus,
                Population::Proxy,
                "correct",
                "Quentin",
                Some("Quentin"),
                None,
                None,
                "test"
            )
            .group,
            LabelGroup::CorrectWinner
        );
        assert_eq!(
            development_case(
                &corpus,
                Population::Proxy,
                "wrong",
                "Quentin",
                Some("Alex"),
                None,
                None,
                "test"
            )
            .group,
            LabelGroup::WrongWinnerPositive
        );
        assert_eq!(
            development_case(
                &corpus,
                Population::Proxy,
                "null",
                "Quentin",
                None,
                None,
                None,
                "test"
            )
            .group,
            LabelGroup::CandidateOnExpectedNull
        );
        assert_eq!(
            development_case(
                &corpus,
                Population::Proxy,
                "missing-positive",
                "Unknown",
                Some("Unknown"),
                None,
                None,
                "test"
            )
            .group,
            LabelGroup::NoCandidatePositive
        );
        assert_eq!(
            development_case(
                &corpus,
                Population::Proxy,
                "missing-null",
                "Unknown",
                None,
                None,
                None,
                "test"
            )
            .group,
            LabelGroup::NoCandidateNull
        );
    }

    #[test]
    fn generation_categories_detect_embedded_digit_handles() {
        let corpus = FakeCorpus(HashMap::from([("Quentin".to_string(), evidence())]));
        let holdout = FrozenHoldout {
            cases: vec![HoldoutCase {
                id: "embedded".to_string(),
                display_name: "Quentin123".to_string(),
                country_hint: String::new(),
                locale_hint: String::new(),
                label_status: LabelStatus::Greeting,
                expected_greeting: "Quentin".to_string(),
                span_start: Some(0),
                span_end: Some("Quentin".len()),
                case_kind: CaseKind::Person,
            }],
            manifest: HoldoutManifest {
                format_version: 1,
                holdout_sha256: "test".to_string(),
                total_cases: 1,
                evaluable_cases: 1,
                skipped_cases: 0,
                expected_greetings: 1,
                expected_abstentions: 0,
                person_cases: 1,
                non_person_cases: 0,
                unknown_kind_cases: 0,
                provenance: "test".to_string(),
            },
        };
        let counts = categorize_generation_misses(&corpus, &holdout);
        assert_eq!(counts.get("all_generation_misses"), Some(&1));
        assert_eq!(
            counts.get("embedded_in_larger_whitespace_free_token"),
            Some(&1)
        );
        assert_eq!(counts.get("containing_token_has_ascii_digit"), Some(&1));
    }

    #[test]
    fn deterministic_search_enforces_zero_error_constraints() {
        let proxy = vec![
            feature_case(
                Population::Proxy,
                "proxy-correct",
                LabelGroup::CorrectWinner,
                0.9,
                0.9,
            ),
            feature_case(
                Population::Proxy,
                "proxy-wrong",
                LabelGroup::WrongWinnerPositive,
                0.1,
                0.1,
            ),
        ];
        let validation = vec![
            feature_case(
                Population::Validation,
                "validation-correct",
                LabelGroup::CorrectWinner,
                0.8,
                0.8,
            ),
            feature_case(
                Population::Validation,
                "validation-null",
                LabelGroup::CandidateOnExpectedNull,
                0.2,
                0.2,
            ),
        ];
        let mut first = search_operating_points_above(&proxy, &validation, 0);
        let mut second = search_operating_points_above(&proxy, &validation, 0);
        first.sort_by(compare_operating_points);
        second.sort_by(compare_operating_points);
        let first = first.first().unwrap();
        let second = second.first().unwrap();
        assert_eq!(first.config, second.config);
        assert_eq!(first.proxy.correct, 1);
        assert_eq!(first.proxy.wrong, 0);
        assert_eq!(first.validation.correct, 1);
        assert_eq!(first.validation.wrong, 0);
        assert_eq!(first.validation.expected_null_emissions, 0);
    }
}
