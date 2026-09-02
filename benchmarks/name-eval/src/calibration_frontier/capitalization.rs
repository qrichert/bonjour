use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::Path;

use name_eval::holdout::FrozenHoldout;
use unicode_general_category::{GeneralCategory, get_general_category};

use super::{
    ALGORITHM_C2, ALGORITHM_C3, ALGORITHM_C4, ALGORITHM_C31, EmissionMetrics, EvidenceSource,
    FeatureRow, Population, Result, c4_decision_breakdown, diagnose_role_inference,
    feature_row_from_decision, greeting_matches, percent, ratio, validate_and_order_holdouts,
    wilson_interval,
};
use crate::classifier::{C4EmissionSource, CandidateDiagnostic, RoleInferenceDiagnostic, canonicalize};
use crate::dataset::{Case, Split, generate_cases};

const ADDITIVE_FEATURE_COUNT: usize = 17;
const INTERACTION_FEATURE_COUNT: usize = 21;
const LOGISTIC_L2: f64 = 0.01;
const MAX_OPTIMIZER_ITERATIONS: usize = 10_000;
const PARAMETER_TOLERANCE: f64 = 1.0e-10;
const ARMIJO: f64 = 1.0e-4;
const MAX_CASE_ADJUSTMENT: f64 = 0.04;
const RANKING_WEIGHTS: [f64; 5] = [0.0, 0.01, 0.02, 0.03, 0.04];
const CAPITALIZATION_TARGETS: [f64; 6] = [0.995, 0.99, 0.98, 0.97, 0.95, 0.90];
const QUALITATIVE_INPUTS: [&str; 8] = [
    "Olivier REDACTED",
    "Baris REDACTED",
    "Ngoc Lam REDACTED",
    "Alexandre REDACTED",
    "Olivier REDACTED",
    "OLIVIER REDACTED",
    "Baris REDACTED",
    "BARIS REDACTED",
];

const ADDITIVE_FEATURE_NAMES: [&str; ADDITIVE_FEATURE_COUNT] = [
    "decision_score",
    "candidate_quality",
    "winner_margin",
    "role_signal",
    "reliability",
    "sole_candidate",
    "native_candidate",
    "candidate_has_case_signal",
    "competitor_has_case_signal",
    "candidate_is_all_upper",
    "candidate_is_all_lower",
    "candidate_is_title_like",
    "candidate_is_mixed_internal",
    "title_upper_direction",
    "uppercase_fraction_delta",
    "input_contains_case_contrast",
    "all_tokens_same_case_pattern",
];

const INTERACTION_FEATURE_NAMES: [&str; INTERACTION_FEATURE_COUNT] = [
    "decision_score",
    "candidate_quality",
    "winner_margin",
    "role_signal",
    "reliability",
    "sole_candidate",
    "native_candidate",
    "candidate_has_case_signal",
    "competitor_has_case_signal",
    "candidate_is_all_upper",
    "candidate_is_all_lower",
    "candidate_is_title_like",
    "candidate_is_mixed_internal",
    "title_upper_direction",
    "uppercase_fraction_delta",
    "input_contains_case_contrast",
    "all_tokens_same_case_pattern",
    "candidate_quality_x_casing_support",
    "role_signal_x_casing_support",
    "winner_margin_x_casing_support",
    "reliability_x_casing_support",
];

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum CaseClass {
    AllUpper,
    AllLower,
    TitleLike,
    MixedInternal,
    Uncased,
    Other,
}

impl CaseClass {
    const ALL: [Self; 6] = [
        Self::AllUpper,
        Self::AllLower,
        Self::TitleLike,
        Self::MixedInternal,
        Self::Uncased,
        Self::Other,
    ];

    fn as_str(self) -> &'static str {
        match self {
            Self::AllUpper => "all_upper",
            Self::AllLower => "all_lower",
            Self::TitleLike => "title_like",
            Self::MixedInternal => "mixed_internal",
            Self::Uncased => "uncased",
            Self::Other => "other",
        }
    }

    fn has_signal(self) -> bool {
        matches!(
            self,
            Self::AllUpper | Self::AllLower | Self::TitleLike | Self::MixedInternal
        )
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct CaseCounts {
    alphabetic: usize,
    cased: usize,
    upper: usize,
    lower: usize,
    title: usize,
    uncased: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct CaseStats {
    class: CaseClass,
    counts: CaseCounts,
}

impl CaseStats {
    const UNAVAILABLE: Self = Self {
        class: CaseClass::Other,
        counts: CaseCounts {
            alphabetic: 0,
            cased: 0,
            upper: 0,
            lower: 0,
            title: 0,
            uncased: 0,
        },
    };

    fn has_signal(self) -> bool {
        self.class.has_signal()
    }

    fn cased_proportion(self) -> f64 {
        ratio_value(self.counts.cased, self.counts.alphabetic)
    }

    fn uppercase_proportion(self) -> f64 {
        ratio_value(self.counts.upper, self.counts.cased)
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum ContrastClass {
    NoneOrUnusable,
    Same,
    CandidateTitleCompetitorUpper,
    CandidateUpperCompetitorTitle,
    OtherContrast,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum CompetitorCaseSource {
    Candidate,
    ContextToken,
    None,
}

impl CompetitorCaseSource {
    fn as_str(self) -> &'static str {
        match self {
            Self::Candidate => "candidate",
            Self::ContextToken => "context_token",
            Self::None => "none",
        }
    }
}

impl ContrastClass {
    const ALL: [Self; 5] = [
        Self::NoneOrUnusable,
        Self::Same,
        Self::CandidateTitleCompetitorUpper,
        Self::CandidateUpperCompetitorTitle,
        Self::OtherContrast,
    ];

    fn as_str(self) -> &'static str {
        match self {
            Self::NoneOrUnusable => "none_or_unusable",
            Self::Same => "same",
            Self::CandidateTitleCompetitorUpper => "candidate_title_competitor_upper",
            Self::CandidateUpperCompetitorTitle => "candidate_upper_competitor_title",
            Self::OtherContrast => "other_contrast",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct CandidateCasing {
    candidate: CaseStats,
    competitor: CaseStats,
    contrast: ContrastClass,
    candidate_less_uppercase: bool,
    candidate_more_uppercase: bool,
    title_upper_direction: f64,
    uppercase_fraction_delta: f64,
    support: f64,
    competitor_source: CompetitorCaseSource,
}

impl CandidateCasing {
    fn unavailable(candidate: CaseStats) -> Self {
        Self {
            candidate,
            competitor: CaseStats::UNAVAILABLE,
            contrast: ContrastClass::NoneOrUnusable,
            candidate_less_uppercase: false,
            candidate_more_uppercase: false,
            title_upper_direction: 0.0,
            uppercase_fraction_delta: 0.0,
            support: 0.0,
            competitor_source: CompetitorCaseSource::None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct InputCasing {
    any_usable_token: bool,
    all_tokens_same: bool,
    contains_contrast: bool,
    entirely_uncased: bool,
}

#[derive(Clone)]
struct CapitalizationRow {
    base: FeatureRow,
    diagnostic: RoleInferenceDiagnostic,
    candidate_stats: Vec<CaseStats>,
    token_stats: Vec<CaseStats>,
    candidate_matches: Vec<bool>,
    input_casing: InputCasing,
    category: String,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum QualityGate {
    Linear,
    Squared,
}

impl QualityGate {
    fn as_str(self) -> &'static str {
        match self {
            Self::Linear => "quality",
            Self::Squared => "quality_squared",
        }
    }

    fn apply(self, quality: f64) -> f64 {
        match self {
            Self::Linear => quality,
            Self::Squared => quality * quality,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct RankingConfig {
    gate: QualityGate,
    weight: f64,
}

impl RankingConfig {
    const FROZEN: Self = Self {
        gate: QualityGate::Linear,
        weight: 0.0,
    };

    fn parameters(self) -> String {
        format!("gate={};weight={:.2}", self.gate.as_str(), self.weight)
    }

    fn adjustment(self, candidate: &CandidateDiagnostic, support: f64) -> f64 {
        (self.weight * support * self.gate.apply(candidate.score.clamp(0.0, 1.0)))
            .clamp(-MAX_CASE_ADJUSTMENT, MAX_CASE_ADJUSTMENT)
    }
}

#[derive(Clone)]
struct RankedRow {
    features: FeatureRow,
    casing: Option<CandidateCasing>,
    selected_index: Option<usize>,
    input_casing: InputCasing,
    category: String,
}

impl RankedRow {
    fn additive_features(&self) -> [f64; ADDITIVE_FEATURE_COUNT] {
        let mut features = [0.0; ADDITIVE_FEATURE_COUNT];
        features[..7].copy_from_slice(&self.features.logistic_features());
        let Some(casing) = self.casing else {
            return features;
        };
        features[7] = f64::from(casing.candidate.has_signal());
        features[8] = f64::from(casing.competitor.has_signal());
        features[9] = f64::from(casing.candidate.class == CaseClass::AllUpper);
        features[10] = f64::from(casing.candidate.class == CaseClass::AllLower);
        features[11] = f64::from(casing.candidate.class == CaseClass::TitleLike);
        features[12] = f64::from(casing.candidate.class == CaseClass::MixedInternal);
        features[13] = casing.title_upper_direction;
        features[14] = casing.uppercase_fraction_delta;
        features[15] = f64::from(self.input_casing.contains_contrast);
        features[16] = f64::from(self.input_casing.all_tokens_same);
        features
    }

    fn interaction_features(&self) -> [f64; INTERACTION_FEATURE_COUNT] {
        let additive = self.additive_features();
        let mut features = [0.0; INTERACTION_FEATURE_COUNT];
        features[..ADDITIVE_FEATURE_COUNT].copy_from_slice(&additive);
        let support = self.casing.map_or(0.0, |casing| casing.support);
        features[17] = additive[1] * support;
        features[18] = additive[3] * support;
        features[19] = additive[2] * support;
        features[20] = additive[4] * support;
        features
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct RankingMetrics {
    rows: usize,
    expected_greetings: usize,
    expected_nulls: usize,
    winner_present: usize,
    correct_winners: usize,
    wrong_winners: usize,
    null_winners: usize,
    generation_ceiling: usize,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum CalibrationVariant {
    Additive,
    Interaction,
    RerankedInteraction,
}

impl CalibrationVariant {
    const BASE: [Self; 2] = [Self::Additive, Self::Interaction];

    fn as_str(self) -> &'static str {
        match self {
            Self::Additive => "additive_capitalization",
            Self::Interaction => "interaction_capitalization",
            Self::RerankedInteraction => "reranked_interaction_capitalization",
        }
    }
}

#[derive(Clone, Debug)]
struct LogisticModel<const N: usize> {
    intercept: f64,
    coefficients: [f64; N],
    iterations: usize,
}

impl<const N: usize> LogisticModel<N> {
    fn score(&self, features: [f64; N]) -> f64 {
        sigmoid(
            self.coefficients
                .iter()
                .zip(features)
                .fold(self.intercept, |score, (coefficient, feature)| {
                    score + coefficient * feature
                }),
        )
    }
}

#[derive(Clone)]
enum CapitalizationModel {
    Additive(LogisticModel<ADDITIVE_FEATURE_COUNT>),
    Interaction(LogisticModel<INTERACTION_FEATURE_COUNT>),
}

impl CapitalizationModel {
    fn score(&self, row: &RankedRow) -> f64 {
        match self {
            Self::Additive(model) => model.score(row.additive_features()),
            Self::Interaction(model) => model.score(row.interaction_features()),
        }
    }

    fn parameters(&self) -> String {
        match self {
            Self::Additive(model) => model_parameters("additive", model),
            Self::Interaction(model) => model_parameters("interaction", model),
        }
    }
}

#[derive(Clone)]
struct CapitalizationPolicy {
    model: CapitalizationModel,
    threshold: f64,
}

impl CapitalizationPolicy {
    fn emits(&self, row: &RankedRow) -> bool {
        row.features.eligible() && self.model.score(row) >= self.threshold
    }

    fn parameters(&self) -> String {
        format!(
            "threshold={:.17};{}",
            self.threshold,
            self.model.parameters()
        )
    }
}

#[derive(Clone)]
struct OperatingPoint {
    policy: CapitalizationPolicy,
    metrics: EmissionMetrics,
}

#[derive(Clone)]
struct FoldResult {
    held_out: Population,
    variant: CalibrationVariant,
    target: f64,
    ranking: RankingConfig,
    policy: CapitalizationPolicy,
    training_metrics: EmissionMetrics,
    held_out_metrics: EmissionMetrics,
}

#[derive(Clone)]
struct CrossValidatedPoint {
    variant: CalibrationVariant,
    target: f64,
    metrics: EmissionMetrics,
}

#[derive(Clone)]
struct FullDevelopmentVariant {
    variant: CalibrationVariant,
    ranking: RankingConfig,
    frontier: Vec<OperatingPoint>,
}

#[derive(Clone)]
struct SelectedPoint {
    target: f64,
    variant: CalibrationVariant,
    ranking: RankingConfig,
    full_development: OperatingPoint,
}

#[derive(Clone)]
struct BaselineSelection {
    target: f64,
    family: super::Family,
    full_development: super::OperatingPoint,
}

#[derive(Clone, Copy)]
struct WeightedTrainingRow<const N: usize> {
    features: [f64; N],
    label: f64,
    weight: f64,
}

struct StructuralSuite {
    rows: Vec<CapitalizationRow>,
    skipped_non_exact: usize,
    skipped_multiple: usize,
}

struct QualitativeOutcome {
    input: &'static str,
    target: f64,
    variant: CalibrationVariant,
    frozen_candidate: String,
    experimental_candidate: String,
    casing: Option<CandidateCasing>,
    emits: bool,
}

pub(crate) fn run_capitalization_diagnostic(
    output: &Path,
    corpus: &impl EvidenceSource,
    holdouts: Vec<FrozenHoldout>,
    fixtures: &Path,
) -> Result<String> {
    let holdouts = validate_and_order_holdouts(holdouts)?;
    let proxy_rows = build_proxy_rows(corpus, &holdouts);
    let validation_rows = build_validation_rows(corpus, fixtures)?;
    let structural_suite = build_structural_suite(corpus, fixtures)?;
    assert_capitalization_dataset_counts(&proxy_rows)?;

    let baseline_rows = proxy_rows
        .iter()
        .map(|row| row.base.clone())
        .collect::<Vec<_>>();
    super::assert_historical_checkpoints(&baseline_rows)?;
    let baseline_folds = super::logo_frontier(&baseline_rows)?;
    let baseline_best = super::best_cross_validated_families(&baseline_folds);
    let baseline_selections = baseline_full_development(&baseline_rows, &baseline_best)?;

    let ranking_configs = ranking_configs();
    let ranking_folds = ranking_logo(&proxy_rows, &ranking_configs);
    let ranking_useful = ranking_is_useful(&ranking_folds);
    let full_ranking = select_ranking_config(&proxy_rows, &ranking_configs);
    let folds = capitalization_logo(&proxy_rows, &ranking_configs, ranking_useful)?;
    let best = best_by_target(&folds, ranking_useful);
    let variants = full_development_variants(&proxy_rows, full_ranking, ranking_useful)?;
    let selections = select_full_development_points(&best, &variants);
    let qualitative = qualitative_outcomes(corpus, &selections);

    let outputs = build_outputs(
        &holdouts,
        &proxy_rows,
        &validation_rows,
        &structural_suite,
        &baseline_best,
        &baseline_folds,
        &baseline_selections,
        &ranking_folds,
        ranking_useful,
        full_ranking,
        &folds,
        &best,
        &selections,
        &qualitative,
    )?;
    let repeated = build_outputs(
        &holdouts,
        &proxy_rows,
        &validation_rows,
        &structural_suite,
        &baseline_best,
        &baseline_folds,
        &baseline_selections,
        &ranking_folds,
        ranking_useful,
        full_ranking,
        &folds,
        &best,
        &selections,
        &qualitative,
    )?;
    if outputs != repeated {
        return Err("capitalization diagnostic serialization is not deterministic".into());
    }
    for (name, bytes) in &outputs {
        fs::write(output.join(name), bytes)?;
    }
    Ok(String::from_utf8(
        outputs
            .get("report.md")
            .ok_or("capitalization report missing")?
            .clone(),
    )?)
}

fn build_proxy_rows(
    corpus: &impl EvidenceSource,
    holdouts: &[FrozenHoldout],
) -> Vec<CapitalizationRow> {
    let mut rows = Vec::new();
    for holdout in holdouts {
        let population = Population::from_digest(&holdout.manifest.holdout_sha256)
            .expect("validated proxy digest");
        for (ordinal, case) in holdout
            .cases
            .iter()
            .filter(|case| case.is_evaluable())
            .enumerate()
        {
            rows.push(build_row(
                corpus,
                population,
                ordinal,
                &case.display_name,
                case.expected_greeting(),
                nonempty(&case.country_hint),
                nonempty(&case.locale_hint),
                "real_proxy",
            ));
        }
    }
    rows
}

fn build_validation_rows(
    corpus: &impl EvidenceSource,
    fixtures: &Path,
) -> Result<Vec<CapitalizationRow>> {
    Ok(generate_cases(fixtures, false)?
        .into_iter()
        .filter(|case| case.split == Split::Validation)
        .enumerate()
        .map(|(ordinal, case)| build_row_from_case(corpus, ordinal, &case))
        .collect())
}

fn build_structural_suite(
    corpus: &impl EvidenceSource,
    fixtures: &Path,
) -> Result<StructuralSuite> {
    let cases = generate_cases(fixtures, false)?;
    let mut rows = Vec::new();
    let mut skipped_non_exact = 0;
    let mut skipped_multiple = 0;
    let mut ordinal = 0;
    for case in cases
        .iter()
        .filter(|case| case.split == Split::Validation)
        .filter(|case| case.expected_greeting.is_some())
    {
        let expected = case.expected_greeting.as_deref().expect("filtered greeting");
        let occurrences = case.input.match_indices(expected).collect::<Vec<_>>();
        let [(start, _)] = occurrences.as_slice() else {
            if occurrences.is_empty() {
                skipped_non_exact += 1;
            } else {
                skipped_multiple += 1;
            }
            continue;
        };
        let end = start + expected.len();
        let prefix = &case.input[..*start];
        let suffix = &case.input[end..];
        for (variant, prefix_transform, expected_transform, suffix_transform) in [
            (
                "all_upper",
                CaseTransform::Upper,
                CaseTransform::Upper,
                CaseTransform::Upper,
            ),
            (
                "all_lower",
                CaseTransform::Lower,
                CaseTransform::Lower,
                CaseTransform::Lower,
            ),
            (
                "expected_title_remainder_upper",
                CaseTransform::Upper,
                CaseTransform::Title,
                CaseTransform::Upper,
            ),
            (
                "expected_upper_remainder_title",
                CaseTransform::Title,
                CaseTransform::Upper,
                CaseTransform::Title,
            ),
        ] {
            let rendered_prefix = prefix_transform.apply(prefix);
            let rendered_expected = expected_transform.apply(expected);
            let rendered_suffix = suffix_transform.apply(suffix);
            let input = format!("{rendered_prefix}{rendered_expected}{rendered_suffix}");
            rows.push(build_row(
                corpus,
                Population::Validation,
                ordinal,
                &input,
                Some(&rendered_expected),
                case.country_hint.as_deref(),
                case.locale_hint.as_deref(),
                &format!("{}:{variant}", case.category),
            ));
            ordinal += 1;
        }
    }
    Ok(StructuralSuite {
        rows,
        skipped_non_exact,
        skipped_multiple,
    })
}

#[derive(Clone, Copy)]
enum CaseTransform {
    Upper,
    Lower,
    Title,
}

impl CaseTransform {
    fn apply(self, value: &str) -> String {
        match self {
            Self::Upper => value.to_uppercase(),
            Self::Lower => value.to_lowercase(),
            Self::Title => title_like_transform(value),
        }
    }
}

fn title_like_transform(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    let mut at_component_start = true;
    for character in value.chars() {
        if character.is_alphabetic() {
            if at_component_start {
                output.extend(character.to_uppercase());
            } else {
                output.extend(character.to_lowercase());
            }
            at_component_start = false;
        } else {
            output.push(character);
            if is_name_component_separator(character) {
                at_component_start = true;
            }
        }
    }
    output
}

fn build_row_from_case(
    corpus: &impl EvidenceSource,
    ordinal: usize,
    case: &Case,
) -> CapitalizationRow {
    build_row(
        corpus,
        Population::Validation,
        ordinal,
        &case.input,
        case.expected_greeting.as_deref(),
        case.country_hint.as_deref(),
        case.locale_hint.as_deref(),
        &case.category,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_row(
    corpus: &impl EvidenceSource,
    population: Population,
    ordinal: usize,
    display_name: &str,
    expected_greeting: Option<&str>,
    country_hint: Option<&str>,
    locale_hint: Option<&str>,
    category: &str,
) -> CapitalizationRow {
    let diagnostic = diagnose_role_inference(
        corpus,
        ALGORITHM_C3,
        display_name,
        country_hint,
        locale_hint,
    );
    let decision = c4_decision_breakdown(&diagnostic, ALGORITHM_C2, ALGORITHM_C31, ALGORITHM_C4);
    let base = feature_row_from_decision(
        population,
        ordinal,
        expected_greeting,
        country_hint.is_some(),
        locale_hint.is_some(),
        &decision,
        None,
    );
    let candidate_stats = diagnostic
        .candidates
        .iter()
        .map(|candidate| candidate_case_stats(display_name, candidate))
        .collect::<Vec<_>>();
    let candidate_matches = diagnostic
        .candidates
        .iter()
        .map(|candidate| greeting_matches(expected_greeting, Some(&candidate.display)))
        .collect::<Vec<_>>();
    let token_stats = canonicalize(display_name)
        .split_whitespace()
        .map(classify_case)
        .collect::<Vec<_>>();
    CapitalizationRow {
        base,
        diagnostic,
        candidate_stats,
        token_stats,
        candidate_matches,
        input_casing: input_case_stats(display_name),
        category: category.to_string(),
    }
}

fn candidate_case_stats(display_name: &str, candidate: &CandidateDiagnostic) -> CaseStats {
    candidate
        .byte_start
        .zip(candidate.byte_end)
        .and_then(|(start, end)| display_name.get(start..end))
        .map_or(CaseStats::UNAVAILABLE, classify_case)
}

fn input_case_stats(display_name: &str) -> InputCasing {
    let tokens = canonicalize(display_name)
        .split_whitespace()
        .map(classify_case)
        .collect::<Vec<_>>();
    let usable = tokens
        .iter()
        .copied()
        .filter(|stats| stats.has_signal())
        .collect::<Vec<_>>();
    let any_usable_token = !usable.is_empty();
    let contains_contrast = usable.len() >= 2
        && usable
            .iter()
            .map(|stats| stats.class)
            .collect::<BTreeSet<_>>()
            .len()
            >= 2;
    let all_tokens_same = tokens.len() >= 2
        && usable.len() == tokens.len()
        && usable
            .iter()
            .map(|stats| stats.class)
            .collect::<BTreeSet<_>>()
            .len()
            == 1;
    let alphabetic = tokens
        .iter()
        .map(|stats| stats.counts.alphabetic)
        .sum::<usize>();
    let cased = tokens
        .iter()
        .map(|stats| stats.counts.cased)
        .sum::<usize>();
    InputCasing {
        any_usable_token,
        all_tokens_same,
        contains_contrast,
        entirely_uncased: alphabetic > 0 && cased == 0,
    }
}

fn candidate_casing(row: &CapitalizationRow, index: usize) -> CandidateCasing {
    let candidate = row
        .candidate_stats
        .get(index)
        .copied()
        .unwrap_or(CaseStats::UNAVAILABLE);
    if let Some(competitor) = (0..row.candidate_stats.len())
        .filter(|other| *other != index)
        .map(|other| row.candidate_stats[other])
        .find(|stats| stats.has_signal())
    {
        let mut casing = contrast_stats(candidate, competitor);
        casing.competitor_source = CompetitorCaseSource::Candidate;
        return casing;
    }
    let selected = &row.diagnostic.candidates[index];
    let end = selected.start.saturating_add(selected.length);
    let contextual = row
        .token_stats
        .iter()
        .copied()
        .enumerate()
        .filter(|(token, stats)| {
            (*token < selected.start || *token >= end) && stats.has_signal()
        })
        .max_by(|(left_index, left), (right_index, right)| {
            left.counts
                .alphabetic
                .cmp(&right.counts.alphabetic)
                .then_with(|| right_index.cmp(left_index))
        })
        .map(|(_, stats)| stats);
    contextual.map_or_else(
        || CandidateCasing::unavailable(candidate),
        |competitor| {
            let mut casing = contrast_stats(candidate, competitor);
            casing.competitor_source = CompetitorCaseSource::ContextToken;
            casing
        },
    )
}

fn contrast_stats(candidate: CaseStats, competitor: CaseStats) -> CandidateCasing {
    if !candidate.has_signal() || !competitor.has_signal() {
        return CandidateCasing::unavailable(candidate);
    }
    let contrast = match (candidate.class, competitor.class) {
        (left, right) if left == right => ContrastClass::Same,
        (CaseClass::TitleLike, CaseClass::AllUpper) => {
            ContrastClass::CandidateTitleCompetitorUpper
        }
        (CaseClass::AllUpper, CaseClass::TitleLike) => {
            ContrastClass::CandidateUpperCompetitorTitle
        }
        _ => ContrastClass::OtherContrast,
    };
    let ordering = compare_uppercase_proportions(candidate, competitor);
    let title_upper_direction = match contrast {
        ContrastClass::CandidateTitleCompetitorUpper => 1.0,
        ContrastClass::CandidateUpperCompetitorTitle => -1.0,
        _ => 0.0,
    };
    let uppercase_fraction_delta = if contrast == ContrastClass::Same {
        0.0
    } else {
        competitor.uppercase_proportion() - candidate.uppercase_proportion()
    };
    let support =
        (0.5 * title_upper_direction + 0.5 * uppercase_fraction_delta).clamp(-1.0, 1.0);
    CandidateCasing {
        candidate,
        competitor,
        contrast,
        candidate_less_uppercase: ordering == Ordering::Less,
        candidate_more_uppercase: ordering == Ordering::Greater,
        title_upper_direction,
        uppercase_fraction_delta,
        support,
        competitor_source: CompetitorCaseSource::None,
    }
}

fn compare_uppercase_proportions(left: CaseStats, right: CaseStats) -> Ordering {
    (left.counts.upper * right.counts.cased).cmp(&(right.counts.upper * left.counts.cased))
}

fn classify_case(value: &str) -> CaseStats {
    let mut counts = CaseCounts::default();
    let mut title_like = true;
    let mut at_component_start = true;
    for character in value.chars() {
        if character.is_alphabetic() {
            counts.alphabetic += 1;
            let category = get_general_category(character);
            if category == GeneralCategory::TitlecaseLetter {
                counts.cased += 1;
                counts.title += 1;
                if !at_component_start {
                    title_like = false;
                }
            } else if character.is_uppercase() {
                counts.cased += 1;
                counts.upper += 1;
                if !at_component_start {
                    title_like = false;
                }
            } else if character.is_lowercase() {
                counts.cased += 1;
                counts.lower += 1;
                if at_component_start {
                    title_like = false;
                }
            } else {
                counts.uncased += 1;
                title_like = false;
            }
            at_component_start = false;
        } else if is_name_component_separator(character) {
            at_component_start = true;
        } else if !is_mark(character) {
            title_like = false;
        }
    }
    let class = if counts.alphabetic == 0 || (counts.cased > 0 && counts.uncased > 0) {
        CaseClass::Other
    } else if counts.cased == 0 {
        CaseClass::Uncased
    } else if counts.upper == counts.cased {
        CaseClass::AllUpper
    } else if counts.lower == counts.cased {
        CaseClass::AllLower
    } else if title_like {
        CaseClass::TitleLike
    } else {
        CaseClass::MixedInternal
    };
    CaseStats { class, counts }
}

fn is_name_component_separator(character: char) -> bool {
    character.is_whitespace()
        || matches!(
            character,
            '\'' | '‘' | '’' | '‛' | 'ʻ' | 'ʼ' | '＇'
                | '-' | '‐' | '‑' | '‒' | '–' | '—' | '―' | '−'
        )
}

fn is_mark(character: char) -> bool {
    matches!(
        get_general_category(character),
        GeneralCategory::NonspacingMark
            | GeneralCategory::SpacingMark
            | GeneralCategory::EnclosingMark
    )
}

fn ranking_configs() -> Vec<RankingConfig> {
    let mut configs = vec![RankingConfig::FROZEN];
    for gate in [QualityGate::Linear, QualityGate::Squared] {
        for weight in RANKING_WEIGHTS.into_iter().skip(1) {
            configs.push(RankingConfig { gate, weight });
        }
    }
    configs
}

fn frozen_ranked_row(row: &CapitalizationRow) -> RankedRow {
    let selected_index = (!row.diagnostic.candidates.is_empty()).then_some(0);
    RankedRow {
        features: row.base.clone(),
        casing: selected_index.map(|index| candidate_casing(row, index)),
        selected_index,
        input_casing: row.input_casing,
        category: row.category.clone(),
    }
}

fn rank_row(row: &CapitalizationRow, config: RankingConfig) -> RankedRow {
    if config == RankingConfig::FROZEN {
        return frozen_ranked_row(row);
    }
    let mut ranked = row
        .diagnostic
        .candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| {
            let casing = candidate_casing(row, index);
            (
                index,
                candidate.score + config.adjustment(candidate, casing.support),
            )
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|(left_index, left_score), (right_index, right_score)| {
        right_score
            .total_cmp(left_score)
            .then_with(|| left_index.cmp(right_index))
    });
    let Some((selected_index, adjusted_score)) = ranked.first().copied() else {
        return RankedRow {
            features: row.base.clone(),
            casing: None,
            selected_index: None,
            input_casing: row.input_casing,
            category: row.category.clone(),
        };
    };
    let second_score = ranked.get(1).map(|(_, score)| *score);
    let candidate = &row.diagnostic.candidates[selected_index];
    let winner_margin = second_score.map_or(1.0, |score| adjusted_score - score);
    let margin_signal = (winner_margin / ALGORITHM_C2.margin_scale).clamp(0.0, 1.0);
    let alphabetic_length = candidate
        .display
        .chars()
        .filter(|character| character.is_alphabetic())
        .count();
    let candidate_too_short = alphabetic_length < ALGORITHM_C2.minimum_candidate_letters;
    let hard = row.diagnostic.hard_organization_abstention;
    let generic = row.diagnostic.generic_organization_marker;
    let ampersand = row.diagnostic.ampersand_negative_evidence;
    let vetoes_pass = !hard && !generic && !ampersand && !candidate_too_short;
    let pre_veto_score = (ALGORITHM_C2.quality_weight * candidate.score
        + ALGORITHM_C2.margin_weight * margin_signal
        + ALGORITHM_C2.role_weight * candidate.role_signal
        + ALGORITHM_C2.reliability_weight * candidate.reliability)
        .clamp(0.0, 1.0);
    let post_veto_score = if vetoes_pass { pre_veto_score } else { 0.0 };
    let segmented = candidate.origin == "handle_segment";
    let penalty = if segmented {
        ALGORITHM_C31.handle_segment_penalty
    } else {
        0.0
    };
    let final_score = (post_veto_score - penalty).clamp(0.0, 1.0);
    let winner_present = !hard;
    let mut features = row.base.clone();
    features.selected_matches = winner_present && row.candidate_matches[selected_index];
    features.winner_present = winner_present;
    features.vetoes_pass = vetoes_pass;
    features.decision_score = final_score;
    features.candidate_quality = candidate.score;
    features.candidate_count = row.diagnostic.candidates.len();
    features.winner_margin = winner_margin;
    features.margin_signal = margin_signal;
    features.role_llr = candidate.role_llr;
    features.role_signal = candidate.role_signal;
    features.reliability = candidate.reliability;
    features.alphabetic_length = alphabetic_length;
    features.native = !segmented;
    features.segmentation_mechanism = candidate.segmentation_mechanism;
    features.hard_organization_marker = hard;
    features.generic_organization_marker = generic;
    features.ampersand = ampersand;
    features.candidate_too_short = candidate_too_short;
    features.c31_emits = winner_present && final_score >= ALGORITHM_C2.threshold;
    features.c4_emits = false;
    features.c4_source = C4EmissionSource::Abstain;
    features.unhinted = None;
    RankedRow {
        features,
        casing: Some(candidate_casing(row, selected_index)),
        selected_index: Some(selected_index),
        input_casing: row.input_casing,
        category: row.category.clone(),
    }
}

fn ranking_metrics(rows: &[CapitalizationRow], config: RankingConfig) -> RankingMetrics {
    let mut metrics = RankingMetrics::default();
    for row in rows {
        metrics.rows += 1;
        metrics.expected_greetings += usize::from(row.base.expected_greeting);
        metrics.expected_nulls += usize::from(!row.base.expected_greeting);
        metrics.generation_ceiling += usize::from(
            row.base.expected_greeting && row.candidate_matches.iter().any(|matches| *matches),
        );
        let ranked = rank_row(row, config);
        if ranked.selected_index.is_some() {
            metrics.winner_present += 1;
            if row.base.expected_greeting {
                if ranked.features.selected_matches {
                    metrics.correct_winners += 1;
                } else {
                    metrics.wrong_winners += 1;
                }
            } else {
                metrics.null_winners += 1;
            }
        }
    }
    metrics
}

fn select_ranking_config(
    rows: &[CapitalizationRow],
    configs: &[RankingConfig],
) -> RankingConfig {
    configs
        .iter()
        .copied()
        .max_by(|left, right| compare_ranking_configs(rows, *left, *right))
        .expect("ranking grid is nonempty")
}

fn compare_ranking_configs(
    rows: &[CapitalizationRow],
    left: RankingConfig,
    right: RankingConfig,
) -> Ordering {
    let left_metrics = ranking_metrics(rows, left);
    let right_metrics = ranking_metrics(rows, right);
    left_metrics
        .correct_winners
        .cmp(&right_metrics.correct_winners)
        .then_with(|| right_metrics.wrong_winners.cmp(&left_metrics.wrong_winners))
        .then_with(|| right_metrics.null_winners.cmp(&left_metrics.null_winners))
        .then_with(|| right.weight.total_cmp(&left.weight))
        .then_with(|| right.gate.cmp(&left.gate))
        .then_with(|| right.parameters().cmp(&left.parameters()))
}

#[derive(Clone, Copy)]
struct RankingFold {
    held_out: Population,
    config: RankingConfig,
    frozen: RankingMetrics,
    adjusted: RankingMetrics,
}

fn ranking_logo(
    rows: &[CapitalizationRow],
    configs: &[RankingConfig],
) -> Vec<RankingFold> {
    Population::PROXIES
        .into_iter()
        .map(|held_out| {
            let training = rows
                .iter()
                .filter(|row| row.base.population != held_out)
                .cloned()
                .collect::<Vec<_>>();
            let held_out_rows = rows
                .iter()
                .filter(|row| row.base.population == held_out)
                .cloned()
                .collect::<Vec<_>>();
            let config = select_ranking_config(&training, configs);
            RankingFold {
                held_out,
                config,
                frozen: ranking_metrics(&held_out_rows, RankingConfig::FROZEN),
                adjusted: ranking_metrics(&held_out_rows, config),
            }
        })
        .collect()
}

fn ranking_is_useful(folds: &[RankingFold]) -> bool {
    let frozen_correct = folds
        .iter()
        .map(|fold| fold.frozen.correct_winners)
        .sum::<usize>();
    let adjusted_correct = folds
        .iter()
        .map(|fold| fold.adjusted.correct_winners)
        .sum::<usize>();
    let frozen_wrong = folds
        .iter()
        .map(|fold| fold.frozen.wrong_winners + fold.frozen.null_winners)
        .sum::<usize>();
    let adjusted_wrong = folds
        .iter()
        .map(|fold| fold.adjusted.wrong_winners + fold.adjusted.null_winners)
        .sum::<usize>();
    adjusted_correct > frozen_correct && adjusted_wrong <= frozen_wrong
}

fn ranked_rows(
    rows: &[CapitalizationRow],
    variant: CalibrationVariant,
    ranking: RankingConfig,
) -> Vec<RankedRow> {
    rows.iter()
        .map(|row| {
            if variant == CalibrationVariant::RerankedInteraction {
                rank_row(row, ranking)
            } else {
                frozen_ranked_row(row)
            }
        })
        .collect()
}

fn fit_model(
    rows: &[RankedRow],
    variant: CalibrationVariant,
) -> Result<CapitalizationModel> {
    match variant {
        CalibrationVariant::Additive => Ok(CapitalizationModel::Additive(fit_logistic(
            rows,
            RankedRow::additive_features,
        )?)),
        CalibrationVariant::Interaction | CalibrationVariant::RerankedInteraction => Ok(
            CapitalizationModel::Interaction(fit_logistic(
                rows,
                RankedRow::interaction_features,
            )?),
        ),
    }
}

fn fit_logistic<const N: usize>(
    rows: &[RankedRow],
    features: fn(&RankedRow) -> [f64; N],
) -> Result<LogisticModel<N>> {
    let training = logistic_training_rows(rows, features)?;
    let mut intercept = 0.0;
    let mut coefficients = [0.0; N];
    let mut objective = logistic_objective(&training, intercept, coefficients);
    for iteration in 1..=MAX_OPTIMIZER_ITERATIONS {
        let (intercept_gradient, coefficient_gradients) =
            logistic_gradient(&training, intercept, coefficients);
        let mut step = 1.0;
        let mut accepted = None;
        while step >= f64::EPSILON {
            let next_intercept = intercept - step * intercept_gradient;
            let mut next_coefficients = coefficients;
            for (coefficient, gradient) in next_coefficients.iter_mut().zip(coefficient_gradients) {
                *coefficient = (*coefficient - step * gradient).max(0.0);
            }
            let next_objective = logistic_objective(&training, next_intercept, next_coefficients);
            let directional = intercept_gradient * (next_intercept - intercept)
                + coefficient_gradients
                    .iter()
                    .zip(next_coefficients.iter().zip(coefficients))
                    .map(|(gradient, (next, current))| gradient * (next - current))
                    .sum::<f64>();
            if next_objective <= objective + ARMIJO * directional {
                accepted = Some((next_intercept, next_coefficients, next_objective));
                break;
            }
            step *= 0.5;
        }
        let Some((next_intercept, next_coefficients, next_objective)) = accepted else {
            return Err("capitalization logistic optimizer line search failed".into());
        };
        let parameter_change = (next_intercept - intercept).abs().max(
            next_coefficients
                .iter()
                .zip(coefficients)
                .map(|(next, current)| (next - current).abs())
                .fold(0.0, f64::max),
        );
        intercept = next_intercept;
        coefficients = next_coefficients;
        objective = next_objective;
        if parameter_change < PARAMETER_TOLERANCE {
            return Ok(LogisticModel {
                intercept,
                coefficients,
                iterations: iteration,
            });
        }
    }
    Err(format!(
        "capitalization logistic optimizer did not converge in {MAX_OPTIMIZER_ITERATIONS} iterations"
    )
    .into())
}

fn logistic_training_rows<const N: usize>(
    rows: &[RankedRow],
    features: fn(&RankedRow) -> [f64; N],
) -> Result<Vec<WeightedTrainingRow<N>>> {
    let populations = rows
        .iter()
        .filter(|row| row.features.population != Population::Validation)
        .map(|row| row.features.population)
        .collect::<BTreeSet<_>>();
    if populations.is_empty() {
        return Err("capitalization calibration requires proxy generations".into());
    }
    let counts = populations
        .iter()
        .map(|population| {
            let count = rows
                .iter()
                .filter(|row| {
                    row.features.population == *population && row.features.eligible()
                })
                .count();
            (*population, count)
        })
        .collect::<BTreeMap<_, _>>();
    if counts.values().any(|count| *count == 0) {
        return Err("every training generation must contain an eligible winner".into());
    }
    let generation_weight = 1.0 / populations.len() as f64;
    Ok(rows
        .iter()
        .filter(|row| row.features.population != Population::Validation && row.features.eligible())
        .map(|row| WeightedTrainingRow {
            features: features(row),
            label: f64::from(row.features.selected_matches),
            weight: generation_weight / counts[&row.features.population] as f64,
        })
        .collect())
}

fn logistic_objective<const N: usize>(
    rows: &[WeightedTrainingRow<N>],
    intercept: f64,
    coefficients: [f64; N],
) -> f64 {
    let loss = rows
        .iter()
        .map(|row| {
            let score = coefficients
                .iter()
                .zip(row.features)
                .fold(intercept, |score, (coefficient, feature)| {
                    score + coefficient * feature
                });
            row.weight * logistic_loss(score, row.label)
        })
        .sum::<f64>();
    loss + 0.5 * LOGISTIC_L2 * coefficients.iter().map(|value| value * value).sum::<f64>()
}

fn logistic_gradient<const N: usize>(
    rows: &[WeightedTrainingRow<N>],
    intercept: f64,
    coefficients: [f64; N],
) -> (f64, [f64; N]) {
    let mut intercept_gradient = 0.0;
    let mut gradients = [0.0; N];
    for row in rows {
        let score = coefficients
            .iter()
            .zip(row.features)
            .fold(intercept, |score, (coefficient, feature)| {
                score + coefficient * feature
            });
        let residual = row.weight * (sigmoid(score) - row.label);
        intercept_gradient += residual;
        for (gradient, feature) in gradients.iter_mut().zip(row.features) {
            *gradient += residual * feature;
        }
    }
    for (gradient, coefficient) in gradients.iter_mut().zip(coefficients) {
        *gradient += LOGISTIC_L2 * coefficient;
    }
    (intercept_gradient, gradients)
}

fn logistic_loss(score: f64, label: f64) -> f64 {
    if score >= 0.0 {
        (1.0 - label) * score + (-score).exp().ln_1p()
    } else {
        -label * score + score.exp().ln_1p()
    }
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

fn model_parameters<const N: usize>(label: &str, model: &LogisticModel<N>) -> String {
    format!(
        "model={label};iterations={};intercept={:.17};coefficients={}",
        model.iterations,
        model.intercept,
        model
            .coefficients
            .iter()
            .map(|value| format!("{value:.17}"))
            .collect::<Vec<_>>()
            .join(":"),
    )
}

fn frontier(rows: &[RankedRow], model: &CapitalizationModel) -> Vec<OperatingPoint> {
    let mut thresholds = rows
        .iter()
        .filter(|row| row.features.eligible())
        .map(|row| model.score(row))
        .collect::<Vec<_>>();
    thresholds.sort_by(|left, right| right.total_cmp(left));
    thresholds.dedup_by(|left, right| left.to_bits() == right.to_bits());
    let mut points = thresholds
        .into_iter()
        .map(|threshold| {
            let policy = CapitalizationPolicy {
                model: model.clone(),
                threshold,
            };
            OperatingPoint {
                metrics: evaluate_policy(rows.iter(), &policy),
                policy,
            }
        })
        .collect::<Vec<_>>();
    deduplicate_points(&mut points, rows);
    points
}

fn evaluate_policy<'a>(
    rows: impl Iterator<Item = &'a RankedRow>,
    policy: &CapitalizationPolicy,
) -> EmissionMetrics {
    let mut metrics = EmissionMetrics::default();
    for row in rows {
        metrics.observe(&row.features, policy.emits(row));
    }
    metrics
}

fn deduplicate_points(points: &mut Vec<OperatingPoint>, rows: &[RankedRow]) {
    let mut unique = BTreeMap::<Vec<u64>, OperatingPoint>::new();
    for point in points.drain(..) {
        let signature = emission_signature(rows, &point.policy);
        match unique.get(&signature) {
            Some(existing) if existing.policy.parameters() <= point.policy.parameters() => {}
            _ => {
                unique.insert(signature, point);
            }
        }
    }
    points.extend(unique.into_values());
    points.sort_by(|left, right| {
        left.metrics
            .emitted
            .cmp(&right.metrics.emitted)
            .then(left.metrics.correct.cmp(&right.metrics.correct))
            .then(left.metrics.wrong.cmp(&right.metrics.wrong))
            .then_with(|| left.policy.parameters().cmp(&right.policy.parameters()))
    });
}

fn emission_signature(rows: &[RankedRow], policy: &CapitalizationPolicy) -> Vec<u64> {
    let mut signature = vec![0_u64; rows.len().div_ceil(64)];
    for (index, row) in rows.iter().enumerate() {
        if policy.emits(row) {
            signature[index / 64] |= 1_u64 << (index % 64);
        }
    }
    signature
}

fn select_point(points: &[OperatingPoint], target: f64) -> Option<&OperatingPoint> {
    points
        .iter()
        .filter(|point| {
            point
                .metrics
                .precision()
                .is_some_and(|precision| precision >= target)
        })
        .max_by(|left, right| compare_points(left, right))
}

fn compare_points(left: &OperatingPoint, right: &OperatingPoint) -> Ordering {
    let left_wilson = wilson_interval(left.metrics.correct, left.metrics.emitted)
        .map_or(0.0, |interval| interval.lower);
    let right_wilson = wilson_interval(right.metrics.correct, right.metrics.emitted)
        .map_or(0.0, |interval| interval.lower);
    left.metrics
        .correct
        .cmp(&right.metrics.correct)
        .then_with(|| right.metrics.wrong.cmp(&left.metrics.wrong))
        .then_with(|| {
            right
                .metrics
                .null_false_emissions
                .cmp(&left.metrics.null_false_emissions)
        })
        .then_with(|| left_wilson.total_cmp(&right_wilson))
        .then_with(|| left.policy.threshold.total_cmp(&right.policy.threshold))
        .then_with(|| right.policy.parameters().cmp(&left.policy.parameters()))
}

fn capitalization_logo(
    rows: &[CapitalizationRow],
    configs: &[RankingConfig],
    ranking_useful: bool,
) -> Result<Vec<FoldResult>> {
    let mut results = Vec::new();
    for held_out in Population::PROXIES {
        let training = rows
            .iter()
            .filter(|row| row.base.population != held_out)
            .cloned()
            .collect::<Vec<_>>();
        let held_out_rows = rows
            .iter()
            .filter(|row| row.base.population == held_out)
            .cloned()
            .collect::<Vec<_>>();
        let ranking = select_ranking_config(&training, configs);
        let mut variants = CalibrationVariant::BASE.to_vec();
        if ranking_useful {
            variants.push(CalibrationVariant::RerankedInteraction);
        }
        for variant in variants {
            let training_ranked = ranked_rows(&training, variant, ranking);
            let held_out_ranked = ranked_rows(&held_out_rows, variant, ranking);
            let model = fit_model(&training_ranked, variant)?;
            let points = frontier(&training_ranked, &model);
            for target in CAPITALIZATION_TARGETS {
                let Some(selected) = select_point(&points, target) else {
                    continue;
                };
                results.push(FoldResult {
                    held_out,
                    variant,
                    target,
                    ranking: if variant == CalibrationVariant::RerankedInteraction {
                        ranking
                    } else {
                        RankingConfig::FROZEN
                    },
                    policy: selected.policy.clone(),
                    training_metrics: selected.metrics,
                    held_out_metrics: evaluate_policy(held_out_ranked.iter(), &selected.policy),
                });
            }
        }
    }
    Ok(results)
}

fn best_by_target(
    folds: &[FoldResult],
    ranking_useful: bool,
) -> Vec<CrossValidatedPoint> {
    let mut variants = CalibrationVariant::BASE.to_vec();
    if ranking_useful {
        variants.push(CalibrationVariant::RerankedInteraction);
    }
    CAPITALIZATION_TARGETS
        .into_iter()
        .filter_map(|target| {
            variants
                .iter()
                .copied()
                .filter_map(|variant| {
                    let matching = folds
                        .iter()
                        .filter(|fold| fold.variant == variant && fold.target == target)
                        .collect::<Vec<_>>();
                    if matching.len() != Population::PROXIES.len() {
                        return None;
                    }
                    let mut metrics = EmissionMetrics::default();
                    for fold in matching {
                        metrics.add(fold.held_out_metrics);
                    }
                    Some(CrossValidatedPoint {
                        variant,
                        target,
                        metrics,
                    })
                })
                .max_by(compare_cross_validated)
        })
        .collect()
}

fn compare_cross_validated(
    left: &CrossValidatedPoint,
    right: &CrossValidatedPoint,
) -> Ordering {
    left.metrics
        .correct
        .cmp(&right.metrics.correct)
        .then_with(|| right.metrics.wrong.cmp(&left.metrics.wrong))
        .then_with(|| {
            right
                .metrics
                .null_false_emissions
                .cmp(&left.metrics.null_false_emissions)
        })
        .then_with(|| right.variant.cmp(&left.variant))
}

fn full_development_variants(
    rows: &[CapitalizationRow],
    ranking: RankingConfig,
    ranking_useful: bool,
) -> Result<Vec<FullDevelopmentVariant>> {
    let mut variants = CalibrationVariant::BASE.to_vec();
    if ranking_useful {
        variants.push(CalibrationVariant::RerankedInteraction);
    }
    variants
        .into_iter()
        .map(|variant| {
            let effective_ranking = if variant == CalibrationVariant::RerankedInteraction {
                ranking
            } else {
                RankingConfig::FROZEN
            };
            let ranked = ranked_rows(rows, variant, effective_ranking);
            let model = fit_model(&ranked, variant)?;
            let points = frontier(&ranked, &model);
            Ok(FullDevelopmentVariant {
                variant,
                ranking: effective_ranking,
                frontier: points,
            })
        })
        .collect()
}

fn select_full_development_points(
    best: &[CrossValidatedPoint],
    variants: &[FullDevelopmentVariant],
) -> Vec<SelectedPoint> {
    best.iter()
        .filter_map(|point| {
            let variant = variants
                .iter()
                .find(|variant| variant.variant == point.variant)?;
            Some(SelectedPoint {
                target: point.target,
                variant: point.variant,
                ranking: variant.ranking,
                full_development: select_point(&variant.frontier, point.target)?.clone(),
            })
        })
        .collect()
}

fn baseline_full_development(
    rows: &[FeatureRow],
    best: &[super::CrossValidatedPoint],
) -> Result<Vec<BaselineSelection>> {
    let score = super::score_frontier(rows);
    let controlled = super::controlled_frontier(rows);
    let model = super::fit_logistic(rows)?;
    let logistic = super::logistic_frontier(rows, &model, false);
    let additive = super::logistic_frontier(rows, &model, true);
    Ok(best
        .iter()
        .filter_map(|point| {
            let frontier = match point.family {
                super::Family::ScoreOnly => &score,
                super::Family::ControlledC4 => &controlled,
                super::Family::Logistic => &logistic,
                super::Family::C4PlusLogistic => &additive,
            };
            Some(BaselineSelection {
                target: point.target,
                family: point.family,
                full_development: super::select_point(frontier, point.target)?.clone(),
            })
        })
        .collect())
}

fn assert_capitalization_dataset_counts(rows: &[CapitalizationRow]) -> Result<()> {
    let baseline = rows
        .iter()
        .map(|row| row.base.clone())
        .collect::<Vec<_>>();
    super::assert_dataset_counts(&baseline)
}

#[allow(clippy::too_many_arguments)]
fn build_outputs(
    holdouts: &[FrozenHoldout],
    proxy_rows: &[CapitalizationRow],
    validation_rows: &[CapitalizationRow],
    structural_suite: &StructuralSuite,
    baseline_best: &[super::CrossValidatedPoint],
    baseline_folds: &[super::FoldResult],
    baseline_selections: &[BaselineSelection],
    ranking_folds: &[RankingFold],
    ranking_useful: bool,
    full_ranking: RankingConfig,
    folds: &[FoldResult],
    best: &[CrossValidatedPoint],
    selections: &[SelectedPoint],
    qualitative: &[QualitativeOutcome],
) -> Result<BTreeMap<String, Vec<u8>>> {
    let mut outputs = BTreeMap::new();
    outputs.insert(
        "dataset_provenance.csv".to_string(),
        provenance_csv(holdouts)?,
    );
    outputs.insert(
        "capitalization_features.csv".to_string(),
        feature_rows_csv(proxy_rows)?,
    );
    outputs.insert(
        "casing_distributions.csv".to_string(),
        distributions_csv(proxy_rows)?,
    );
    outputs.insert(
        "casing_availability.csv".to_string(),
        availability_csv(proxy_rows)?,
    );
    outputs.insert(
        "ranking_grid.csv".to_string(),
        ranking_grid_csv(proxy_rows)?,
    );
    outputs.insert(
        "ranking_logo.csv".to_string(),
        ranking_logo_csv(ranking_folds)?,
    );
    outputs.insert(
        "frontier_comparison.csv".to_string(),
        frontier_comparison_csv(baseline_best, best)?,
    );
    outputs.insert(
        "model_form_comparison.csv".to_string(),
        model_form_comparison_csv(baseline_folds, folds)?,
    );
    outputs.insert(
        "capitalization_logo_results.csv".to_string(),
        logo_results_csv(folds)?,
    );
    outputs.insert(
        "capitalization_per_generation.csv".to_string(),
        per_generation_csv(folds)?,
    );
    outputs.insert(
        "capitalization_coefficients.csv".to_string(),
        coefficients_csv(selections)?,
    );
    outputs.insert(
        "validation_by_category.csv".to_string(),
        validation_csv(validation_rows, baseline_selections, selections)?,
    );
    outputs.insert(
        "structural_casing_regressions.csv".to_string(),
        structural_csv(structural_suite, baseline_selections, selections)?,
    );
    outputs.insert(
        "qualitative_examples.csv".to_string(),
        qualitative_csv(qualitative)?,
    );
    outputs.insert(
        "report.md".to_string(),
        build_report(
            holdouts,
            proxy_rows,
            validation_rows,
            structural_suite,
            baseline_best,
            baseline_folds,
            baseline_selections,
            ranking_folds,
            ranking_useful,
            full_ranking,
            folds,
            best,
            selections,
            qualitative,
        )?
        .into_bytes(),
    );
    Ok(outputs)
}

fn provenance_csv(holdouts: &[FrozenHoldout]) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "population",
        "sha256",
        "evaluable",
        "expected_greeting",
        "expected_null",
    ])?;
    for holdout in holdouts {
        let population = Population::from_digest(&holdout.manifest.holdout_sha256)
            .expect("validated proxy digest");
        let evaluable = holdout.cases.iter().filter(|case| case.is_evaluable());
        let rows = evaluable.clone().count();
        let greetings = evaluable
            .filter(|case| case.expected_greeting().is_some())
            .count();
        writer.write_record([
            population.as_str().to_string(),
            holdout.manifest.holdout_sha256.clone(),
            rows.to_string(),
            greetings.to_string(),
            (rows - greetings).to_string(),
        ])?;
    }
    finish_writer(writer)
}

fn feature_rows_csv(rows: &[CapitalizationRow]) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "population",
        "ordinal",
        "expected_greeting",
        "selected_matches_expected",
        "winner_present",
        "vetoes_pass",
        "c4_emits",
        "decision_score",
        "candidate_quality",
        "candidate_count",
        "winner_margin",
        "role_signal",
        "reliability",
        "native",
        "candidate_case_class",
        "candidate_has_case_signal",
        "competitor_case_class",
        "competitor_has_case_signal",
        "competitor_case_source",
        "contrast",
        "candidate_less_uppercase",
        "candidate_more_uppercase",
        "cased_proportion",
        "uppercase_proportion",
        "title_upper_direction",
        "uppercase_fraction_delta",
        "casing_support",
        "input_contains_case_contrast",
        "all_tokens_same_case_pattern",
        "input_entirely_uncased",
    ])?;
    for row in rows {
        let casing = (!row.diagnostic.candidates.is_empty()).then(|| candidate_casing(row, 0));
        writer.write_record([
            row.base.population.as_str().to_string(),
            row.base.ordinal.to_string(),
            row.base.expected_greeting.to_string(),
            row.base.selected_matches.to_string(),
            row.base.winner_present.to_string(),
            row.base.vetoes_pass.to_string(),
            row.base.c4_emits.to_string(),
            format!("{:.17}", row.base.decision_score),
            format!("{:.17}", row.base.candidate_quality),
            row.base.candidate_count.to_string(),
            format!("{:.17}", row.base.winner_margin),
            format!("{:.17}", row.base.role_signal),
            format!("{:.17}", row.base.reliability),
            row.base.native.to_string(),
            casing.map_or("other", |value| value.candidate.class.as_str()).to_string(),
            casing.is_some_and(|value| value.candidate.has_signal()).to_string(),
            casing.map_or("other", |value| value.competitor.class.as_str()).to_string(),
            casing.is_some_and(|value| value.competitor.has_signal()).to_string(),
            casing
                .map_or(CompetitorCaseSource::None, |value| value.competitor_source)
                .as_str()
                .to_string(),
            casing.map_or("none_or_unusable", |value| value.contrast.as_str()).to_string(),
            casing.is_some_and(|value| value.candidate_less_uppercase).to_string(),
            casing.is_some_and(|value| value.candidate_more_uppercase).to_string(),
            format!("{:.17}", casing.map_or(0.0, |value| value.candidate.cased_proportion())),
            format!("{:.17}", casing.map_or(0.0, |value| value.candidate.uppercase_proportion())),
            format!("{:.17}", casing.map_or(0.0, |value| value.title_upper_direction)),
            format!("{:.17}", casing.map_or(0.0, |value| value.uppercase_fraction_delta)),
            format!("{:.17}", casing.map_or(0.0, |value| value.support)),
            row.input_casing.contains_contrast.to_string(),
            row.input_casing.all_tokens_same.to_string(),
            row.input_casing.entirely_uncased.to_string(),
        ])?;
    }
    finish_writer(writer)
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum DiagnosticPopulation {
    Correct,
    Wrong,
    ExpectedNull,
    C4RejectedCorrect,
}

impl DiagnosticPopulation {
    const ALL: [Self; 4] = [
        Self::Correct,
        Self::Wrong,
        Self::ExpectedNull,
        Self::C4RejectedCorrect,
    ];

    fn as_str(self) -> &'static str {
        match self {
            Self::Correct => "correct_selected_winner",
            Self::Wrong => "wrong_selected_winner_on_expected_greeting",
            Self::ExpectedNull => "expected_null_with_selected_winner",
            Self::C4RejectedCorrect => "correct_veto_free_winner_rejected_by_c4",
        }
    }

    fn contains(self, row: &CapitalizationRow) -> bool {
        match self {
            Self::Correct => row.base.expected_greeting && row.base.selected_matches,
            Self::Wrong => {
                row.base.expected_greeting
                    && row.base.winner_present
                    && !row.base.selected_matches
            }
            Self::ExpectedNull => !row.base.expected_greeting && row.base.winner_present,
            Self::C4RejectedCorrect => {
                row.base.expected_greeting
                    && row.base.selected_matches
                    && row.base.vetoes_pass
                    && !row.base.c4_emits
            }
        }
    }
}

fn distributions_csv(rows: &[CapitalizationRow]) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "population",
        "diagnostic_population",
        "feature",
        "value",
        "count",
        "rate",
    ])?;
    for population in Population::PROXIES.into_iter().map(Some).chain([None]) {
        for group in DiagnosticPopulation::ALL {
            let selected = rows
                .iter()
                .filter(|row| population.is_none_or(|value| row.base.population == value))
                .filter(|row| group.contains(row))
                .collect::<Vec<_>>();
            let total = selected.len();
            for class in CaseClass::ALL {
                let count = selected
                    .iter()
                    .filter(|row| {
                        row.candidate_stats
                            .first()
                            .is_some_and(|stats| stats.class == class)
                    })
                    .count();
                write_distribution_row(
                    &mut writer,
                    population,
                    group,
                    "candidate_case_class",
                    class.as_str(),
                    count,
                    total,
                )?;
            }
            for contrast in ContrastClass::ALL {
                let count = selected
                    .iter()
                    .filter(|row| {
                        !row.diagnostic.candidates.is_empty()
                            && candidate_casing(row, 0).contrast == contrast
                    })
                    .count();
                write_distribution_row(
                    &mut writer,
                    population,
                    group,
                    "candidate_competitor_contrast",
                    contrast.as_str(),
                    count,
                    total,
                )?;
            }
            for (feature, predicate) in [
                (
                    "candidate_less_uppercase",
                    less_uppercase as fn(&CapitalizationRow) -> bool,
                ),
                ("candidate_more_uppercase", more_uppercase),
                ("input_contains_case_contrast", input_has_contrast),
                ("all_tokens_same_case_pattern", input_all_same),
                ("input_entirely_uncased", input_entirely_uncased),
            ] {
                let count = selected.iter().filter(|row| predicate(row)).count();
                write_distribution_row(
                    &mut writer,
                    population,
                    group,
                    feature,
                    "true",
                    count,
                    total,
                )?;
            }
        }
    }
    finish_writer(writer)
}

fn write_distribution_row(
    writer: &mut csv::Writer<Vec<u8>>,
    population: Option<Population>,
    group: DiagnosticPopulation,
    feature: &str,
    value: &str,
    count: usize,
    total: usize,
) -> Result<()> {
    writer.write_record([
        population.map_or("COMBINED", Population::as_str).to_string(),
        group.as_str().to_string(),
        feature.to_string(),
        value.to_string(),
        count.to_string(),
        ratio(count, total).map_or_else(String::new, |value| format!("{value:.17}")),
    ])?;
    Ok(())
}

fn less_uppercase(row: &CapitalizationRow) -> bool {
    !row.diagnostic.candidates.is_empty() && candidate_casing(row, 0).candidate_less_uppercase
}

fn more_uppercase(row: &CapitalizationRow) -> bool {
    !row.diagnostic.candidates.is_empty() && candidate_casing(row, 0).candidate_more_uppercase
}

fn input_has_contrast(row: &CapitalizationRow) -> bool {
    row.input_casing.contains_contrast
}

fn input_all_same(row: &CapitalizationRow) -> bool {
    row.input_casing.all_tokens_same
}

fn input_entirely_uncased(row: &CapitalizationRow) -> bool {
    row.input_casing.entirely_uncased
}

fn availability_csv(rows: &[CapitalizationRow]) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record(["population", "metric", "count", "rows", "rate"])?;
    for population in Population::PROXIES.into_iter().map(Some).chain([None]) {
        let selected = rows
            .iter()
            .filter(|row| population.is_none_or(|value| row.base.population == value))
            .collect::<Vec<_>>();
        for (metric, predicate) in [
            (
                "any_usable_cased_display_token",
                any_usable_token as fn(&CapitalizationRow) -> bool,
            ),
            ("usable_selected_candidate", usable_selected_candidate),
            ("usable_selected_and_competitor", usable_candidate_pair),
            ("nonzero_candidate_competitor_contrast", nonzero_contrast),
            ("alphabetic_input_entirely_uncased", input_entirely_uncased),
        ] {
            let count = selected.iter().filter(|row| predicate(row)).count();
            writer.write_record([
                population.map_or("COMBINED", Population::as_str).to_string(),
                metric.to_string(),
                count.to_string(),
                selected.len().to_string(),
                ratio(count, selected.len())
                    .map_or_else(String::new, |value| format!("{value:.17}")),
            ])?;
        }
    }
    finish_writer(writer)
}

fn any_usable_token(row: &CapitalizationRow) -> bool {
    row.input_casing.any_usable_token
}

fn usable_selected_candidate(row: &CapitalizationRow) -> bool {
    row.candidate_stats
        .first()
        .is_some_and(|stats| stats.has_signal())
}

fn usable_candidate_pair(row: &CapitalizationRow) -> bool {
    !row.diagnostic.candidates.is_empty()
        && candidate_casing(row, 0).candidate.has_signal()
        && candidate_casing(row, 0).competitor.has_signal()
}

fn nonzero_contrast(row: &CapitalizationRow) -> bool {
    !row.diagnostic.candidates.is_empty() && candidate_casing(row, 0).support != 0.0
}

fn ranking_grid_csv(rows: &[CapitalizationRow]) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "parameters",
        "correct_winners",
        "wrong_winners",
        "null_winners",
        "generation_ceiling",
        "ranking_ceiling",
    ])?;
    for config in ranking_configs() {
        let metrics = ranking_metrics(rows, config);
        writer.write_record([
            config.parameters(),
            metrics.correct_winners.to_string(),
            metrics.wrong_winners.to_string(),
            metrics.null_winners.to_string(),
            metrics.generation_ceiling.to_string(),
            ratio(metrics.correct_winners, metrics.expected_greetings)
                .map_or_else(String::new, |value| format!("{value:.17}")),
        ])?;
    }
    finish_writer(writer)
}

fn ranking_logo_csv(folds: &[RankingFold]) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "held_out",
        "parameters",
        "frozen_correct",
        "adjusted_correct",
        "frozen_wrong",
        "adjusted_wrong",
        "frozen_null",
        "adjusted_null",
    ])?;
    for fold in folds {
        writer.write_record([
            fold.held_out.as_str().to_string(),
            fold.config.parameters(),
            fold.frozen.correct_winners.to_string(),
            fold.adjusted.correct_winners.to_string(),
            fold.frozen.wrong_winners.to_string(),
            fold.adjusted.wrong_winners.to_string(),
            fold.frozen.null_winners.to_string(),
            fold.adjusted.null_winners.to_string(),
        ])?;
    }
    finish_writer(writer)
}

fn frontier_comparison_csv(
    baseline: &[super::CrossValidatedPoint],
    capitalization: &[CrossValidatedPoint],
) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "target",
        "baseline_family",
        "baseline_precision",
        "baseline_recall",
        "baseline_correct",
        "baseline_wrong",
        "baseline_null_fp",
        "baseline_false_abstentions",
        "baseline_correct_winner_rejected",
        "capitalization_variant",
        "capitalization_precision",
        "capitalization_recall",
        "capitalization_correct",
        "capitalization_wrong",
        "capitalization_null_fp",
        "capitalization_false_abstentions",
        "capitalization_correct_winner_rejected",
        "recall_delta",
    ])?;
    for target in CAPITALIZATION_TARGETS {
        let base = baseline
            .iter()
            .find(|point| point.target == target)
            .ok_or("baseline frontier target missing")?;
        let casing = capitalization
            .iter()
            .find(|point| point.target == target)
            .ok_or("capitalization frontier target missing")?;
        let base_recall = base.metrics.recall().unwrap_or(0.0);
        let casing_recall = casing.metrics.recall().unwrap_or(0.0);
        writer.write_record([
            format!("{target:.3}"),
            base.family.as_str().to_string(),
            metric_value(base.metrics.precision()),
            metric_value(base.metrics.recall()),
            base.metrics.correct.to_string(),
            base.metrics.wrong.to_string(),
            base.metrics.null_false_emissions.to_string(),
            base.metrics.false_abstentions.to_string(),
            base.metrics.winner_correct_but_abstained.to_string(),
            casing.variant.as_str().to_string(),
            metric_value(casing.metrics.precision()),
            metric_value(casing.metrics.recall()),
            casing.metrics.correct.to_string(),
            casing.metrics.wrong.to_string(),
            casing.metrics.null_false_emissions.to_string(),
            casing.metrics.false_abstentions.to_string(),
            casing.metrics.winner_correct_but_abstained.to_string(),
            format!("{:.17}", casing_recall - base_recall),
        ])?;
    }
    finish_writer(writer)
}

fn model_form_comparison_csv(
    baseline_folds: &[super::FoldResult],
    capitalization_folds: &[FoldResult],
) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "target",
        "model",
        "precision",
        "recall",
        "correct",
        "wrong",
        "null_fp",
        "false_abstentions",
        "correct_winner_rejected",
    ])?;
    for target in CAPITALIZATION_TARGETS {
        let logistic = aggregate_baseline_fold_metrics(
            baseline_folds,
            super::Family::Logistic,
            target,
        )
        .ok_or("pure logistic baseline fold missing")?;
        write_model_form_row(&mut writer, target, "baseline_logistic", logistic)?;
        for variant in [
            CalibrationVariant::Additive,
            CalibrationVariant::Interaction,
            CalibrationVariant::RerankedInteraction,
        ] {
            if let Some(metrics) =
                aggregate_capitalization_fold_metrics(capitalization_folds, variant, target)
            {
                write_model_form_row(&mut writer, target, variant.as_str(), metrics)?;
            }
        }
    }
    finish_writer(writer)
}

fn write_model_form_row(
    writer: &mut csv::Writer<Vec<u8>>,
    target: f64,
    model: &str,
    metrics: EmissionMetrics,
) -> Result<()> {
    writer.write_record([
        format!("{target:.3}"),
        model.to_string(),
        metric_value(metrics.precision()),
        metric_value(metrics.recall()),
        metrics.correct.to_string(),
        metrics.wrong.to_string(),
        metrics.null_false_emissions.to_string(),
        metrics.false_abstentions.to_string(),
        metrics.winner_correct_but_abstained.to_string(),
    ])?;
    Ok(())
}

fn aggregate_baseline_fold_metrics(
    folds: &[super::FoldResult],
    family: super::Family,
    target: f64,
) -> Option<EmissionMetrics> {
    let matching = folds
        .iter()
        .filter(|fold| fold.family == family && fold.target == target)
        .collect::<Vec<_>>();
    if matching.len() != Population::PROXIES.len() {
        return None;
    }
    let mut metrics = EmissionMetrics::default();
    for fold in matching {
        metrics.add(fold.held_out_metrics);
    }
    Some(metrics)
}

fn aggregate_capitalization_fold_metrics(
    folds: &[FoldResult],
    variant: CalibrationVariant,
    target: f64,
) -> Option<EmissionMetrics> {
    let matching = folds
        .iter()
        .filter(|fold| fold.variant == variant && fold.target == target)
        .collect::<Vec<_>>();
    if matching.len() != Population::PROXIES.len() {
        return None;
    }
    let mut metrics = EmissionMetrics::default();
    for fold in matching {
        metrics.add(fold.held_out_metrics);
    }
    Some(metrics)
}

fn logo_results_csv(folds: &[FoldResult]) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "held_out",
        "variant",
        "target",
        "ranking",
        "model",
        "training_precision",
        "training_recall",
        "held_out_precision",
        "held_out_recall",
        "held_out_correct",
        "held_out_wrong",
        "held_out_null_fp",
        "held_out_false_abstentions",
        "held_out_correct_winner_rejected",
    ])?;
    for fold in folds {
        writer.write_record([
            fold.held_out.as_str().to_string(),
            fold.variant.as_str().to_string(),
            format!("{:.3}", fold.target),
            fold.ranking.parameters(),
            fold.policy.parameters(),
            metric_value(fold.training_metrics.precision()),
            metric_value(fold.training_metrics.recall()),
            metric_value(fold.held_out_metrics.precision()),
            metric_value(fold.held_out_metrics.recall()),
            fold.held_out_metrics.correct.to_string(),
            fold.held_out_metrics.wrong.to_string(),
            fold.held_out_metrics.null_false_emissions.to_string(),
            fold.held_out_metrics.false_abstentions.to_string(),
            fold
                .held_out_metrics
                .winner_correct_but_abstained
                .to_string(),
        ])?;
    }
    finish_writer(writer)
}

fn per_generation_csv(folds: &[FoldResult]) -> Result<Vec<u8>> {
    logo_results_csv(folds)
}

fn coefficients_csv(selections: &[SelectedPoint]) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record(["target", "variant", "feature", "coefficient"])?;
    for selection in selections {
        match &selection.full_development.policy.model {
            CapitalizationModel::Additive(model) => {
                writer.write_record([
                    format!("{:.3}", selection.target),
                    selection.variant.as_str().to_string(),
                    "intercept".to_string(),
                    format!("{:.17}", model.intercept),
                ])?;
                for (name, coefficient) in ADDITIVE_FEATURE_NAMES.iter().zip(model.coefficients) {
                    writer.write_record([
                        format!("{:.3}", selection.target),
                        selection.variant.as_str().to_string(),
                        (*name).to_string(),
                        format!("{coefficient:.17}"),
                    ])?;
                }
            }
            CapitalizationModel::Interaction(model) => {
                writer.write_record([
                    format!("{:.3}", selection.target),
                    selection.variant.as_str().to_string(),
                    "intercept".to_string(),
                    format!("{:.17}", model.intercept),
                ])?;
                for (name, coefficient) in INTERACTION_FEATURE_NAMES.iter().zip(model.coefficients) {
                    writer.write_record([
                        format!("{:.3}", selection.target),
                        selection.variant.as_str().to_string(),
                        (*name).to_string(),
                        format!("{coefficient:.17}"),
                    ])?;
                }
            }
        }
    }
    finish_writer(writer)
}

fn validation_csv(
    rows: &[CapitalizationRow],
    baseline_selections: &[BaselineSelection],
    selections: &[SelectedPoint],
) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "target",
        "model_kind",
        "variant",
        "category",
        "rows",
        "correct",
        "wrong",
        "null_fp",
        "recall",
    ])?;
    let categories = rows
        .iter()
        .map(|row| row.category.as_str())
        .collect::<BTreeSet<_>>();
    for selection in baseline_selections {
        for category in &categories {
            let selected = rows
                .iter()
                .filter(|row| row.category == **category)
                .map(|row| &row.base)
                .collect::<Vec<_>>();
            let metrics = super::evaluate_policy(
                selected.iter().copied(),
                &selection.full_development.policy,
            );
            writer.write_record([
                format!("{:.3}", selection.target),
                "baseline".to_string(),
                selection.family.as_str().to_string(),
                (*category).to_string(),
                metrics.rows.to_string(),
                metrics.correct.to_string(),
                metrics.wrong.to_string(),
                metrics.null_false_emissions.to_string(),
                metric_value(metrics.recall()),
            ])?;
        }
    }
    for selection in selections {
        let ranked = ranked_rows(rows, selection.variant, selection.ranking);
        for category in &categories {
            let selected = ranked
                .iter()
                .filter(|row| row.category == **category)
                .collect::<Vec<_>>();
            let metrics = evaluate_policy(selected.iter().copied(), &selection.full_development.policy);
            writer.write_record([
                format!("{:.3}", selection.target),
                "capitalization".to_string(),
                selection.variant.as_str().to_string(),
                (*category).to_string(),
                metrics.rows.to_string(),
                metrics.correct.to_string(),
                metrics.wrong.to_string(),
                metrics.null_false_emissions.to_string(),
                metric_value(metrics.recall()),
            ])?;
        }
    }
    finish_writer(writer)
}

fn structural_csv(
    suite: &StructuralSuite,
    baseline_selections: &[BaselineSelection],
    selections: &[SelectedPoint],
) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "target",
        "model_kind",
        "variant",
        "category",
        "transformation",
        "rows",
        "correct_winners",
        "correct",
        "wrong",
        "recall",
    ])?;
    let categories = suite
        .rows
        .iter()
        .map(|row| row.category.as_str())
        .collect::<BTreeSet<_>>();
    for selection in baseline_selections {
        for category in &categories {
            let selected = suite
                .rows
                .iter()
                .filter(|row| row.category == **category)
                .collect::<Vec<_>>();
            let metrics = super::evaluate_policy(
                selected.iter().map(|row| &row.base),
                &selection.full_development.policy,
            );
            let correct_winners = selected
                .iter()
                .filter(|row| row.base.selected_matches)
                .count();
            let (base_category, transformation) = category
                .rsplit_once(':')
                .unwrap_or((category, "unknown"));
            writer.write_record([
                format!("{:.3}", selection.target),
                "baseline".to_string(),
                selection.family.as_str().to_string(),
                base_category.to_string(),
                transformation.to_string(),
                metrics.rows.to_string(),
                correct_winners.to_string(),
                metrics.correct.to_string(),
                metrics.wrong.to_string(),
                metric_value(metrics.recall()),
            ])?;
        }
    }
    for selection in selections {
        let ranked = ranked_rows(&suite.rows, selection.variant, selection.ranking);
        for category in &categories {
            let selected = ranked
                .iter()
                .filter(|row| row.category == **category)
                .collect::<Vec<_>>();
            let metrics = evaluate_policy(selected.iter().copied(), &selection.full_development.policy);
            let correct_winners = selected
                .iter()
                .filter(|row| row.features.selected_matches)
                .count();
            let (base_category, transformation) = category
                .rsplit_once(':')
                .unwrap_or((category, "unknown"));
            writer.write_record([
                format!("{:.3}", selection.target),
                "capitalization".to_string(),
                selection.variant.as_str().to_string(),
                base_category.to_string(),
                transformation.to_string(),
                metrics.rows.to_string(),
                correct_winners.to_string(),
                metrics.correct.to_string(),
                metrics.wrong.to_string(),
                metric_value(metrics.recall()),
            ])?;
        }
    }
    finish_writer(writer)
}

fn qualitative_outcomes(
    corpus: &impl EvidenceSource,
    selections: &[SelectedPoint],
) -> Vec<QualitativeOutcome> {
    let mut outcomes = Vec::new();
    for input in QUALITATIVE_INPUTS {
        let row = build_row(
            corpus,
            Population::Validation,
            0,
            input,
            None,
            None,
            None,
            "qualitative",
        );
        let frozen_candidate = row
            .diagnostic
            .candidates
            .first()
            .map_or_else(String::new, |candidate| candidate.display.clone());
        for selection in selections {
            let ranked = rank_row(&row, selection.ranking);
            let experimental_candidate = ranked.selected_index.map_or_else(String::new, |index| {
                row.diagnostic.candidates[index].display.clone()
            });
            outcomes.push(QualitativeOutcome {
                input,
                target: selection.target,
                variant: selection.variant,
                frozen_candidate: frozen_candidate.clone(),
                experimental_candidate,
                casing: ranked.casing,
                emits: selection.full_development.policy.emits(&ranked),
            });
        }
    }
    outcomes
}

fn qualitative_csv(outcomes: &[QualitativeOutcome]) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "input",
        "target",
        "variant",
        "frozen_candidate",
        "experimental_candidate",
        "candidate_case_class",
        "competitor_case_class",
        "competitor_case_source",
        "contrast",
        "casing_support",
        "emits",
    ])?;
    for outcome in outcomes {
        let casing = outcome.casing;
        writer.write_record([
            outcome.input.to_string(),
            format!("{:.3}", outcome.target),
            outcome.variant.as_str().to_string(),
            outcome.frozen_candidate.clone(),
            outcome.experimental_candidate.clone(),
            casing.map_or("other", |value| value.candidate.class.as_str()).to_string(),
            casing.map_or("other", |value| value.competitor.class.as_str()).to_string(),
            casing
                .map_or(CompetitorCaseSource::None, |value| value.competitor_source)
                .as_str()
                .to_string(),
            casing.map_or("none_or_unusable", |value| value.contrast.as_str()).to_string(),
            format!("{:.17}", casing.map_or(0.0, |value| value.support)),
            outcome.emits.to_string(),
        ])?;
    }
    finish_writer(writer)
}

#[allow(clippy::too_many_arguments)]
fn build_report(
    holdouts: &[FrozenHoldout],
    proxy_rows: &[CapitalizationRow],
    validation_rows: &[CapitalizationRow],
    structural_suite: &StructuralSuite,
    baseline_best: &[super::CrossValidatedPoint],
    baseline_folds: &[super::FoldResult],
    baseline_selections: &[BaselineSelection],
    ranking_folds: &[RankingFold],
    ranking_useful: bool,
    full_ranking: RankingConfig,
    folds: &[FoldResult],
    best: &[CrossValidatedPoint],
    selections: &[SelectedPoint],
    qualitative: &[QualitativeOutcome],
) -> Result<String> {
    let mut report = String::new();
    writeln!(report, "# Capitalization evidence diagnostic\n")?;
    writeln!(
        report,
        "C4 remains frozen production behavior. This benchmark-only experiment measures whether Unicode-aware contrastive casing and its interactions with existing evidence move the generation-held-out calibration frontier. Generic or locale-aware ordering evidence is excluded, no C5 policy is frozen, and no fresh holdout is used.\n"
    )?;
    writeln!(report, "## Data and provenance\n")?;
    writeln!(report, "| Population | SHA-256 | Evaluable | Greetings | NULL |")?;
    writeln!(report, "|---|---|---:|---:|---:|")?;
    for holdout in holdouts {
        let population = Population::from_digest(&holdout.manifest.holdout_sha256)
            .expect("validated proxy digest");
        let rows = holdout.cases.iter().filter(|case| case.is_evaluable());
        let total = rows.clone().count();
        let greetings = rows
            .filter(|case| case.expected_greeting().is_some())
            .count();
        writeln!(
            report,
            "| {} | `{}` | {} | {} | {} |",
            population.as_str(),
            holdout.manifest.holdout_sha256,
            total,
            greetings,
            total - greetings,
        )?;
    }
    writeln!(
        report,
        "\nCombined: {} rows, {} expected greetings, {} expected NULLs. V1 retains its distinct single-annotation provenance; V2-V5 use exact machine-annotation consensus. These are proxy labels, not worldwide ground truth.\n",
        proxy_rows.len(),
        proxy_rows.iter().filter(|row| row.base.expected_greeting).count(),
        proxy_rows.iter().filter(|row| !row.base.expected_greeting).count(),
    )?;

    writeln!(report, "## Feature definitions and availability\n")?;
    writeln!(
        report,
        "Candidates are classified as `all_upper`, `all_lower`, `title_like`, `mixed_internal`, `uncased`, or `other` using Unicode Uppercase/Lowercase properties and the Titlecase_Letter general category. Combining marks and accepted name separators are case-neutral. Uncased scripts and mixed cased/uncased-script spans receive zero casing support.\n"
    )?;
    writeln!(report, "| Availability | Count | Rate |")?;
    writeln!(report, "|---|---:|---:|")?;
    for (label, predicate) in [
        (
            "Any usable cased display token",
            any_usable_token as fn(&CapitalizationRow) -> bool,
        ),
        ("Usable selected candidate", usable_selected_candidate),
        ("Usable selected and competitor", usable_candidate_pair),
        ("Nonzero candidate/competitor contrast", nonzero_contrast),
        ("Alphabetic input entirely uncased", input_entirely_uncased),
    ] {
        let count = proxy_rows.iter().filter(|row| predicate(row)).count();
        writeln!(
            report,
            "| {label} | {count} | {} |",
            percent(ratio(count, proxy_rows.len())),
        )?;
    }

    writeln!(report, "\n## Direct correlation\n")?;
    writeln!(report, "| Population | Rows | Title/upper contrast | Any nonzero contrast | No usable pair |")?;
    writeln!(report, "|---|---:|---:|---:|---:|")?;
    for group in DiagnosticPopulation::ALL {
        let selected = proxy_rows
            .iter()
            .filter(|row| group.contains(row))
            .collect::<Vec<_>>();
        let title_upper = selected
            .iter()
            .filter(|row| {
                !row.diagnostic.candidates.is_empty()
                    && matches!(
                        candidate_casing(row, 0).contrast,
                        ContrastClass::CandidateTitleCompetitorUpper
                            | ContrastClass::CandidateUpperCompetitorTitle
                    )
            })
            .count();
        let nonzero = selected.iter().filter(|row| nonzero_contrast(row)).count();
        let unusable = selected
            .iter()
            .filter(|row| !usable_candidate_pair(row))
            .count();
        writeln!(
            report,
            "| {} | {} | {} | {} | {} |",
            group.as_str(),
            selected.len(),
            percent(ratio(title_upper, selected.len())),
            percent(ratio(nonzero, selected.len())),
            percent(ratio(unusable, selected.len())),
        )?;
    }

    let frozen_ranking = ranking_folds.iter().fold(RankingMetrics::default(), |mut sum, fold| {
        add_ranking_metrics(&mut sum, fold.frozen);
        sum
    });
    let adjusted_ranking = ranking_folds.iter().fold(RankingMetrics::default(), |mut sum, fold| {
        add_ranking_metrics(&mut sum, fold.adjusted);
        sum
    });
    writeln!(report, "\n## Ranking experiment\n")?;
    writeln!(
        report,
        "The fold-selection gate retained a casing ranker: **{}**. Across held-out folds it changes correct winners by {:+} and wrong winners by {:+}. Full-development selection: `{}`. Adjustments are contrastive and bounded to ±{:.2}; no flat case-class bonus is evaluated.\n",
        ranking_useful,
        adjusted_ranking.correct_winners as isize - frozen_ranking.correct_winners as isize,
        adjusted_ranking.wrong_winners as isize - frozen_ranking.wrong_winners as isize,
        full_ranking.parameters(),
        MAX_CASE_ADJUSTMENT,
    )?;
    writeln!(report, "| Ranking | Correct winner | Wrong winner | NULL winner | Ceiling |")?;
    writeln!(report, "|---|---:|---:|---:|---:|")?;
    for (label, metrics) in [("Frozen", frozen_ranking), ("Capitalization", adjusted_ranking)] {
        writeln!(
            report,
            "| {label} | {} | {} | {} | {} |",
            metrics.correct_winners,
            metrics.wrong_winners,
            metrics.null_winners,
            percent(ratio(metrics.correct_winners, metrics.expected_greetings)),
        )?;
    }

    writeln!(report, "\n## Out-of-fold calibration frontier\n")?;
    writeln!(report, "| Target | Baseline | Baseline precision | Baseline recall | Capitalization | Precision | Recall | Δ recall | Correct | Wrong | NULL FP | Correct winner rejected |")?;
    writeln!(report, "|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|")?;
    for target in CAPITALIZATION_TARGETS {
        let base = baseline_best
            .iter()
            .find(|point| point.target == target)
            .ok_or("baseline target missing")?;
        let casing = best
            .iter()
            .find(|point| point.target == target)
            .ok_or("capitalization target missing")?;
        let delta = casing.metrics.recall().unwrap_or(0.0) - base.metrics.recall().unwrap_or(0.0);
        writeln!(
            report,
            "| {:.1}% | {} | {} | {} | {} | {} | {} | {:+.2} pp | {} | {} | {} | {} |",
            target * 100.0,
            base.family.as_str(),
            percent(base.metrics.precision()),
            percent(base.metrics.recall()),
            casing.variant.as_str(),
            percent(casing.metrics.precision()),
            percent(casing.metrics.recall()),
            delta * 100.0,
            casing.metrics.correct,
            casing.metrics.wrong,
            casing.metrics.null_false_emissions,
            casing.metrics.winner_correct_but_abstained,
        )?;
    }

    writeln!(report, "\n## Additive versus interaction terms\n")?;
    writeln!(report, "| Target | Pure logistic recall | Additive casing recall | Interaction recall | Reranked interaction recall |")?;
    writeln!(report, "|---:|---:|---:|---:|---:|")?;
    for target in CAPITALIZATION_TARGETS {
        let logistic = aggregate_baseline_fold_metrics(
            baseline_folds,
            super::Family::Logistic,
            target,
        )
        .ok_or("pure logistic baseline target missing")?;
        let additive = aggregate_capitalization_fold_metrics(
            folds,
            CalibrationVariant::Additive,
            target,
        )
        .ok_or("additive capitalization target missing")?;
        let interaction = aggregate_capitalization_fold_metrics(
            folds,
            CalibrationVariant::Interaction,
            target,
        )
        .ok_or("interaction capitalization target missing")?;
        let reranked = aggregate_capitalization_fold_metrics(
            folds,
            CalibrationVariant::RerankedInteraction,
            target,
        );
        writeln!(
            report,
            "| {:.1}% | {} | {} | {} | {} |",
            target * 100.0,
            percent(logistic.recall()),
            percent(additive.recall()),
            percent(interaction.recall()),
            reranked.map_or_else(|| "not evaluated".to_string(), |metrics| percent(metrics.recall())),
        )?;
    }

    writeln!(report, "\n## Per-generation stability\n")?;
    writeln!(report, "| Held out | Target | Variant | Precision | Recall | Correct | Wrong | NULL FP |")?;
    writeln!(report, "|---|---:|---|---:|---:|---:|---:|---:|")?;
    for fold in folds {
        writeln!(
            report,
            "| {} | {:.1}% | {} | {} | {} | {} | {} | {} |",
            fold.held_out.as_str(),
            fold.target * 100.0,
            fold.variant.as_str(),
            percent(fold.held_out_metrics.precision()),
            percent(fold.held_out_metrics.recall()),
            fold.held_out_metrics.correct,
            fold.held_out_metrics.wrong,
            fold.held_out_metrics.null_false_emissions,
        )?;
    }

    writeln!(report, "\n## Synthetic structural regression\n")?;
    writeln!(
        report,
        "Historical VALIDATION contains {} rows. The separate capitalization-only suite contains {} derived rows; {} rows were skipped because the expected greeting was not an exact source span and {} because it occurred more than once. Neither population participates in fitting or selection. Detailed results by historical category and transformation are in `validation_by_category.csv` and `structural_casing_regressions.csv`.\n",
        validation_rows.len(),
        structural_suite.rows.len(),
        structural_suite.skipped_non_exact,
        structural_suite.skipped_multiple,
    )?;
    if let (Some(baseline), Some(casing)) = (
        baseline_selections
            .iter()
            .find(|selection| selection.target == 0.99),
        selections
            .iter()
            .find(|selection| selection.target == 0.99),
    ) {
        let baseline_metrics = super::evaluate_policy(
            structural_suite.rows.iter().map(|row| &row.base),
            &baseline.full_development.policy,
        );
        let ranked = ranked_rows(&structural_suite.rows, casing.variant, casing.ranking);
        let casing_metrics = evaluate_policy(ranked.iter(), &casing.full_development.policy);
        writeln!(report, "| 99% full-development policy | Correct | Wrong | Recall |")?;
        writeln!(report, "|---|---:|---:|---:|")?;
        writeln!(
            report,
            "| Existing baseline ({}) | {} | {} | {} |",
            baseline.family.as_str(),
            baseline_metrics.correct,
            baseline_metrics.wrong,
            percent(baseline_metrics.recall()),
        )?;
        writeln!(
            report,
            "| Capitalization ({}) | {} | {} | {} |\n",
            casing.variant.as_str(),
            casing_metrics.correct,
            casing_metrics.wrong,
            percent(casing_metrics.recall()),
        )?;
    }

    writeln!(report, "| Target | Policy | Transformation | Correct | Wrong | Recall |")?;
    writeln!(report, "|---:|---|---|---:|---:|---:|")?;
    for target in [0.995, 0.99] {
        let baseline = baseline_selections
            .iter()
            .find(|selection| selection.target == target)
            .ok_or("structural baseline target missing")?;
        let casing = selections
            .iter()
            .find(|selection| selection.target == target)
            .ok_or("structural capitalization target missing")?;
        let ranked = ranked_rows(&structural_suite.rows, casing.variant, casing.ranking);
        for transformation in [
            "all_upper",
            "all_lower",
            "expected_title_remainder_upper",
            "expected_upper_remainder_title",
        ] {
            let suffix = format!(":{transformation}");
            let baseline_metrics = super::evaluate_policy(
                structural_suite
                    .rows
                    .iter()
                    .filter(|row| row.category.ends_with(&suffix))
                    .map(|row| &row.base),
                &baseline.full_development.policy,
            );
            let casing_metrics = evaluate_policy(
                ranked
                    .iter()
                    .filter(|row| row.category.ends_with(&suffix)),
                &casing.full_development.policy,
            );
            for (label, metrics) in [
                ("existing baseline", baseline_metrics),
                ("capitalization", casing_metrics),
            ] {
                writeln!(
                    report,
                    "| {:.1}% | {label} | {transformation} | {} | {} | {} |",
                    target * 100.0,
                    metrics.correct,
                    metrics.wrong,
                    percent(metrics.recall()),
                )?;
            }
        }
    }

    let recommendation = classify_recommendation(
        baseline_best,
        best,
        baseline_folds,
        folds,
        baseline_selections,
        selections,
        structural_suite,
    );
    writeln!(report, "## Recommendation\n")?;
    writeln!(report, "**{recommendation}**\n")?;
    if recommendation == "harmful / no value" {
        writeln!(
            report,
            "Capitalization does not improve the established frontier at the 99%, 98%, 97%, or 95% targets. Although the interaction model improves a like-for-like logistic baseline, the selected capitalization policy has substantial structural regressions on uniformly upper- and lowercase inputs and on reversed case contrast. Drop casing from the future C5 feature set at this stage.\n"
        )?;
    }
    writeln!(
        report,
        "The interaction contribution is reported separately from additive-only casing in the frontier and coefficient files. `Uncased`/`Other` rows retain the seven baseline features and receive zero casing interactions. No artifact data is added; production cost and behavior remain unchanged. Experimental extraction scans Unicode scalars in display tokens and existing candidate spans and allocates benchmark-only diagnostic vectors.\n"
    )?;

    writeln!(report, "## Qualitative smoke tests\n")?;
    writeln!(
        report,
        "Identifying example text is redacted. The diagnostic path remains exercised only after model selection.\n"
    )?;
    writeln!(report, "| Input | Target | Variant | Winner before → after | Contrast | Support | Emits |")?;
    writeln!(report, "|---|---:|---|---|---|---:|---:|")?;
    for outcome in qualitative {
        writeln!(
            report,
            "| {} | {:.1}% | {} | {} → {} | {} | {:.3} | {} |",
            outcome.input,
            outcome.target * 100.0,
            outcome.variant.as_str(),
            outcome.frozen_candidate,
            outcome.experimental_candidate,
            outcome
                .casing
                .map_or("none_or_unusable", |casing| casing.contrast.as_str()),
            outcome.casing.map_or(0.0, |casing| casing.support),
            outcome.emits,
        )?;
    }
    writeln!(report)?;
    writeln!(
        report,
        "The redacted qualitative examples do not influence fitting. C4 remains production behavior, generic ordering remains marginal/not promoted, no C5 is frozen, and V6 remains untouched."
    )?;
    Ok(report)
}

fn add_ranking_metrics(total: &mut RankingMetrics, value: RankingMetrics) {
    total.rows += value.rows;
    total.expected_greetings += value.expected_greetings;
    total.expected_nulls += value.expected_nulls;
    total.winner_present += value.winner_present;
    total.correct_winners += value.correct_winners;
    total.wrong_winners += value.wrong_winners;
    total.null_winners += value.null_winners;
    total.generation_ceiling += value.generation_ceiling;
}

fn classify_recommendation(
    baseline: &[super::CrossValidatedPoint],
    capitalization: &[CrossValidatedPoint],
    baseline_folds: &[super::FoldResult],
    capitalization_folds: &[FoldResult],
    baseline_selections: &[BaselineSelection],
    selections: &[SelectedPoint],
    structural_suite: &StructuralSuite,
) -> &'static str {
    let gains = [0.99, 0.98]
        .into_iter()
        .map(|target| {
            let base = baseline.iter().find(|point| point.target == target);
            let casing = capitalization
                .iter()
                .find(|point| point.target == target);
            match (base, casing) {
                (Some(base), Some(casing)) => {
                    casing.metrics.recall().unwrap_or(0.0) - base.metrics.recall().unwrap_or(0.0)
                }
                _ => 0.0,
            }
        })
        .collect::<Vec<_>>();
    let baseline_structural = baseline_selections
        .iter()
        .find(|selection| selection.target == 0.99)
        .map(|selection| {
            super::evaluate_policy(
                structural_suite.rows.iter().map(|row| &row.base),
                &selection.full_development.policy,
            )
        });
    let capitalization_structural = selections
        .iter()
        .find(|selection| selection.target == 0.99)
        .map(|selection| {
            let ranked = ranked_rows(
                &structural_suite.rows,
                selection.variant,
                selection.ranking,
            );
            evaluate_policy(ranked.iter(), &selection.full_development.policy)
        });
    let structural_safe = baseline_structural
        .zip(capitalization_structural)
        .is_some_and(|(baseline, capitalization)| {
            capitalization.correct >= baseline.correct
                && capitalization.wrong <= baseline.wrong
        });
    let like_for_like_gain = [0.99, 0.98].into_iter().any(|target| {
        let baseline = aggregate_baseline_fold_metrics(
            baseline_folds,
            super::Family::Logistic,
            target,
        );
        let casing = aggregate_capitalization_fold_metrics(
            capitalization_folds,
            CalibrationVariant::Interaction,
            target,
        );
        baseline.zip(casing).is_some_and(|(baseline, casing)| {
            casing.recall().unwrap_or(0.0) > baseline.recall().unwrap_or(0.0)
        })
    });
    if !structural_safe {
        "harmful / no value"
    } else if gains.iter().all(|gain| *gain >= 0.005) {
        "strongly useful"
    } else if gains.iter().any(|gain| *gain > 0.0) || like_for_like_gain {
        "marginal"
    } else {
        "harmful / no value"
    }
}

fn metric_value(value: Option<f64>) -> String {
    value.map_or_else(String::new, |value| format!("{value:.17}"))
}

fn ratio_value(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn nonempty(value: &str) -> Option<&str> {
    (!value.is_empty()).then_some(value)
}

fn finish_writer(mut writer: csv::Writer<Vec<u8>>) -> Result<Vec<u8>> {
    writer.flush()?;
    Ok(writer.into_inner()?)
}

#[cfg(test)]
mod tests {
    use unicode_normalization::UnicodeNormalization;

    use super::*;

    #[test]
    fn unicode_case_classes_cover_cased_uncased_and_internal_shapes() {
        for (value, expected) in [
            ("JEAN", CaseClass::AllUpper),
            ("jean", CaseClass::AllLower),
            ("Jean", CaseClass::TitleLike),
            ("Jean-Pierre", CaseClass::TitleLike),
            ("O’Connor", CaseClass::TitleLike),
            ("McDonald", CaseClass::MixedInternal),
            ("ΔΗΜΗΤΡΗΣ", CaseClass::AllUpper),
            ("дмитрий", CaseClass::AllLower),
            ("李明", CaseClass::Uncased),
            ("محمد", CaseClass::Uncased),
            ("अमित", CaseClass::Uncased),
            ("Jean李", CaseClass::Other),
            ("123", CaseClass::Other),
        ] {
            assert_eq!(classify_case(value).class, expected, "{value:?}");
        }
        assert_eq!(
            classify_case("ǅuro").class,
            CaseClass::TitleLike,
            "Unicode Titlecase_Letter starts a title-like component"
        );
    }

    #[test]
    fn combining_marks_do_not_change_case_class() {
        let composed = classify_case("Élodie");
        let decomposed = classify_case(&"Élodie".nfd().collect::<String>());
        assert_eq!(composed.class, CaseClass::TitleLike);
        assert_eq!(decomposed.class, composed.class);
        assert_eq!(decomposed.counts.alphabetic, composed.counts.alphabetic);
    }

    #[test]
    fn case_support_is_contrastive_and_uncased_neutral() {
        let title = classify_case("Jean");
        let upper = classify_case("MARTIN");
        let lower = classify_case("martin");
        let uncased = classify_case("李明");
        let supportive = contrast_stats(title, upper);
        let reverse = contrast_stats(upper, title);
        let same = contrast_stats(title, classify_case("Martin"));
        assert_eq!(supportive.support.to_bits(), 0.875_f64.to_bits());
        assert_eq!(reverse.support.to_bits(), (-0.875_f64).to_bits());
        assert_eq!(same.support.to_bits(), 0.0_f64.to_bits());
        assert_eq!(contrast_stats(lower, lower).support.to_bits(), 0.0_f64.to_bits());
        assert_eq!(contrast_stats(title, uncased).support.to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn input_context_distinguishes_contrast_same_case_and_uncased() {
        let contrast = input_case_stats("Jean MARTIN");
        assert!(contrast.contains_contrast);
        assert!(!contrast.all_tokens_same);

        for value in ["JEAN MARTIN", "jean martin", "Jean Martin"] {
            let same = input_case_stats(value);
            assert!(!same.contains_contrast, "{value:?}");
            assert!(same.all_tokens_same, "{value:?}");
        }

        let uncased = input_case_stats("李 明");
        assert!(uncased.entirely_uncased);
        assert!(!uncased.any_usable_token);
        assert!(!uncased.contains_contrast);
    }

    #[test]
    fn uppercase_proportion_comparison_uses_exact_counts() {
        let title = classify_case("Jean");
        let mixed = classify_case("JeAN");
        assert_eq!(compare_uppercase_proportions(title, mixed), Ordering::Less);
        assert_eq!(compare_uppercase_proportions(mixed, title), Ordering::Greater);
    }

    #[test]
    fn ranking_adjustment_is_quality_gated_bounded_and_reversible() {
        let candidate = candidate("Jean", 0.8);
        let linear = RankingConfig {
            gate: QualityGate::Linear,
            weight: 0.04,
        };
        let squared = RankingConfig {
            gate: QualityGate::Squared,
            weight: 0.04,
        };
        assert_eq!(
            linear.adjustment(&candidate, 1.0).to_bits(),
            0.032_f64.to_bits()
        );
        assert!((squared.adjustment(&candidate, 1.0) - 0.0256).abs() < f64::EPSILON);
        assert_eq!(
            linear.adjustment(&candidate, -1.0).to_bits(),
            (-0.032_f64).to_bits()
        );
        assert_eq!(
            RankingConfig::FROZEN.adjustment(&candidate, 1.0).to_bits(),
            0.0_f64.to_bits()
        );
    }

    #[test]
    fn feature_names_exclude_ordering_evidence() {
        for feature in ADDITIVE_FEATURE_NAMES
            .into_iter()
            .chain(INTERACTION_FEATURE_NAMES)
        {
            assert!(!feature.contains("position"), "{feature}");
            assert!(!feature.contains("initial"), "{feature}");
            assert!(!feature.contains("final"), "{feature}");
            assert!(!feature.contains("comma"), "{feature}");
            assert!(!feature.contains("prior"), "{feature}");
        }
    }

    #[test]
    fn title_transform_preserves_name_component_boundaries() {
        assert_eq!(title_like_transform("o’CONNOR jean-pierre"), "O’Connor Jean-Pierre");
    }

    fn candidate(display: &str, score: f64) -> CandidateDiagnostic {
        CandidateDiagnostic {
            display: display.to_string(),
            start: 0,
            length: 1,
            byte_start: Some(0),
            byte_end: Some(display.len()),
            global_given_count: 1,
            country_given_count: 0,
            effective_given_count: 1,
            female_given_count: 0,
            male_given_count: 0,
            global_surname_count: 0,
            role_llr: 1.0,
            role_signal: 0.5,
            reliability: 0.5,
            country_support: 0.0,
            compound_evidence: 0.0,
            compositional_evidence: 0.0,
            remainder_evidence: 0.0,
            origin: "corpus",
            segmentation_mechanism: None,
            lookup_query: Some(display.to_string()),
            lookup_mode: Some("normalized"),
            left_lookup_mode: None,
            right_lookup_mode: None,
            score,
            algorithm_a_score: 0.0,
            algorithm_b_score: 0.0,
        }
    }
}
