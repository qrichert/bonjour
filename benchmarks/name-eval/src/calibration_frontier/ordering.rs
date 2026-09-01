use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as FmtWrite;
use std::fs;
use std::mem;
use std::path::Path;

use name_eval::holdout::FrozenHoldout;

use super::{
    ALGORITHM_C2, ALGORITHM_C3, ALGORITHM_C4, ALGORITHM_C31, EmissionMetrics, EvidenceSource,
    FeatureRow, Population, Result, TARGETS, c4_decision_breakdown, diagnose_role_inference,
    feature_row_from_decision, greeting_matches, percent, ratio, validate_and_order_holdouts,
    wilson_interval,
};
use crate::classifier::{
    C4EmissionSource, CandidateDiagnostic, RoleInferenceDiagnostic, canonicalize,
};
use crate::dataset::{Case, Split, generate_cases};

const MAX_ORDER_ADJUSTMENT: f64 = 0.06;
const ORDER_ADDITIVE_FEATURE_COUNT: usize = 14;
const ORDER_INTERACTION_FEATURE_COUNT: usize = 22;
const LOGISTIC_L2: f64 = 0.01;
const MAX_OPTIMIZER_ITERATIONS: usize = 10_000;
const PARAMETER_TOLERANCE: f64 = 1.0e-10;
const ARMIJO: f64 = 1.0e-4;

const GENERIC_WEIGHTS: [f64; 7] = [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03];
const PRIOR_WEIGHTS: [f64; 4] = [0.0, 0.01, 0.02, 0.03];
const COMMA_WEIGHTS: [f64; 4] = [0.0, 0.02, 0.04, 0.06];

/// Derived from Unicode CLDR release 48 supplementalData.xml.
const SURNAME_FIRST_LANGUAGES: [&str; 11] = [
    "hu", "ja", "km", "ko", "mn", "si", "ta", "te", "vi", "yue", "zh",
];

/// Regions with an und_REGION entry in CLDR 48 likelySubtags.xml.
const KNOWN_REGIONS: [[u8; 2]; 192] = [
    *b"AD", *b"AE", *b"AF", *b"AL", *b"AM", *b"AO", *b"AR", *b"AS", *b"AT", *b"AW", *b"AX", *b"AZ",
    *b"BA", *b"BD", *b"BE", *b"BF", *b"BG", *b"BH", *b"BI", *b"BJ", *b"BL", *b"BN", *b"BO", *b"BQ",
    *b"BR", *b"BT", *b"BV", *b"BY", *b"CC", *b"CD", *b"CF", *b"CG", *b"CH", *b"CI", *b"CL", *b"CM",
    *b"CN", *b"CO", *b"CR", *b"CU", *b"CV", *b"CW", *b"CY", *b"CZ", *b"DE", *b"DJ", *b"DK", *b"DO",
    *b"DZ", *b"EA", *b"EC", *b"EE", *b"EG", *b"EH", *b"ER", *b"ES", *b"ET", *b"FI", *b"FO", *b"FR",
    *b"GA", *b"GE", *b"GF", *b"GH", *b"GL", *b"GN", *b"GP", *b"GQ", *b"GR", *b"GT", *b"GW", *b"HK",
    *b"HN", *b"HR", *b"HT", *b"HU", *b"IC", *b"ID", *b"IL", *b"IN", *b"IQ", *b"IR", *b"IS", *b"IT",
    *b"JO", *b"JP", *b"KE", *b"KG", *b"KH", *b"KM", *b"KP", *b"KR", *b"KW", *b"KZ", *b"LA", *b"LB",
    *b"LI", *b"LK", *b"LS", *b"LT", *b"LU", *b"LV", *b"LY", *b"MA", *b"MC", *b"MD", *b"ME", *b"MF",
    *b"MG", *b"MK", *b"ML", *b"MM", *b"MN", *b"MO", *b"MQ", *b"MR", *b"MT", *b"MU", *b"MV", *b"MX",
    *b"MY", *b"MZ", *b"NA", *b"NC", *b"NE", *b"NI", *b"NL", *b"NO", *b"NP", *b"OM", *b"PA", *b"PE",
    *b"PF", *b"PG", *b"PH", *b"PK", *b"PL", *b"PM", *b"PR", *b"PS", *b"PT", *b"PW", *b"PY", *b"QA",
    *b"RE", *b"RO", *b"RS", *b"RU", *b"RW", *b"SA", *b"SC", *b"SD", *b"SE", *b"SI", *b"SJ", *b"SK",
    *b"SM", *b"SN", *b"SO", *b"SR", *b"SS", *b"ST", *b"SV", *b"SY", *b"TD", *b"TF", *b"TG", *b"TH",
    *b"TJ", *b"TK", *b"TL", *b"TM", *b"TN", *b"TO", *b"TR", *b"TV", *b"TW", *b"TZ", *b"UA", *b"UG",
    *b"UY", *b"UZ", *b"VA", *b"VE", *b"VN", *b"VU", *b"WF", *b"WS", *b"XK", *b"YE", *b"YT", *b"ZW",
];

/// CLDR 48 regions whose likely language has surname-first name order.
const SURNAME_FIRST_REGIONS: [[u8; 2]; 12] = [
    *b"CN", *b"HK", *b"HU", *b"JP", *b"KH", *b"KP", *b"KR", *b"LK", *b"MN", *b"MO", *b"TW", *b"VN",
];

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum NameOrderPrior {
    GivenFirst,
    SurnameFirst,
    Neutral,
}

impl NameOrderPrior {
    fn as_str(self) -> &'static str {
        match self {
            Self::GivenFirst => "given_first",
            Self::SurnameFirst => "surname_first",
            Self::Neutral => "neutral",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum CandidatePosition {
    Initial,
    Interior,
    Final,
    Whole,
}

impl CandidatePosition {
    fn as_str(self) -> &'static str {
        match self {
            Self::Initial => "initial",
            Self::Interior => "interior",
            Self::Final => "final",
            Self::Whole => "whole",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum CandidateRelation {
    Before,
    After,
    Overlap,
    None,
}

impl CandidateRelation {
    fn as_str(self) -> &'static str {
        match self {
            Self::Before => "before",
            Self::After => "after",
            Self::Overlap => "overlap",
            Self::None => "none",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct CandidateOrdering {
    start: usize,
    end: usize,
    token_count: usize,
    display_token_count: usize,
    position: CandidatePosition,
    comma_present: bool,
    comma_inversion: bool,
    prior: NameOrderPrior,
    agrees_with_prior: bool,
    conflicts_with_prior: bool,
}

impl CandidateOrdering {
    fn is_initial(self) -> bool {
        matches!(
            self.position,
            CandidatePosition::Initial | CandidatePosition::Whole
        )
    }

    fn is_final(self) -> bool {
        matches!(
            self.position,
            CandidatePosition::Final | CandidatePosition::Whole
        )
    }

    fn generic_signal(self) -> f64 {
        match self.position {
            CandidatePosition::Initial => 1.0,
            CandidatePosition::Final => -1.0,
            CandidatePosition::Interior | CandidatePosition::Whole => 0.0,
        }
    }

    fn prior_signal(self) -> f64 {
        if self.agrees_with_prior {
            1.0
        } else if self.conflicts_with_prior {
            -1.0
        } else {
            0.0
        }
    }
}

#[derive(Clone)]
struct OrderingRow {
    base: FeatureRow,
    diagnostic: RoleInferenceDiagnostic,
    candidate_ordering: Vec<CandidateOrdering>,
    candidate_matches: Vec<bool>,
    category: String,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum RankingFamily {
    Frozen,
    Flat,
    Confirmatory,
}

impl RankingFamily {
    fn as_str(self) -> &'static str {
        match self {
            Self::Frozen => "frozen",
            Self::Flat => "flat",
            Self::Confirmatory => "confirmatory",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct RankingConfig {
    family: RankingFamily,
    generic_weight: f64,
    prior_weight: f64,
    comma_weight: f64,
}

impl RankingConfig {
    const FROZEN: Self = Self {
        family: RankingFamily::Frozen,
        generic_weight: 0.0,
        prior_weight: 0.0,
        comma_weight: 0.0,
    };

    fn parameters(self) -> String {
        format!(
            "family={};generic={:.2};prior={:.2};comma={:.2}",
            self.family.as_str(),
            self.generic_weight,
            self.prior_weight,
            self.comma_weight,
        )
    }

    fn complexity(self) -> usize {
        [self.generic_weight, self.prior_weight, self.comma_weight]
            .into_iter()
            .filter(|weight| *weight != 0.0)
            .count()
    }

    fn maximum_weight(self) -> f64 {
        self.generic_weight
            .abs()
            .max(self.prior_weight.abs())
            .max(self.comma_weight.abs())
    }

    fn adjustment(self, candidate: &CandidateDiagnostic, order: CandidateOrdering) -> f64 {
        if self.family == RankingFamily::Frozen {
            return 0.0;
        }
        let flat = self.generic_weight * order.generic_signal()
            + self.prior_weight * order.prior_signal()
            + self.comma_weight * f64::from(order.comma_inversion);
        let adjusted = if self.family == RankingFamily::Confirmatory {
            flat * candidate.score.clamp(0.0, 1.0)
        } else {
            flat
        };
        adjusted.clamp(-MAX_ORDER_ADJUSTMENT, MAX_ORDER_ADJUSTMENT)
    }
}

#[derive(Clone)]
struct RankedRow {
    features: FeatureRow,
    ordering: Option<CandidateOrdering>,
    competitor_relation: CandidateRelation,
    selected_index: Option<usize>,
}

impl RankedRow {
    fn additive_features(&self) -> [f64; ORDER_ADDITIVE_FEATURE_COUNT] {
        let mut features = [0.0; ORDER_ADDITIVE_FEATURE_COUNT];
        features[..7].copy_from_slice(&self.features.logistic_features());
        let Some(ordering) = self.ordering else {
            return features;
        };
        features[7] = f64::from(ordering.is_initial());
        features[8] = f64::from(ordering.is_final());
        features[9] = f64::from(self.competitor_relation == CandidateRelation::Before);
        features[10] = f64::from(self.competitor_relation == CandidateRelation::After);
        features[11] = f64::from(ordering.comma_inversion);
        features[12] = f64::from(ordering.agrees_with_prior);
        features[13] = ordering.token_count as f64 / ordering.display_token_count.max(1) as f64;
        features
    }

    fn interaction_features(&self) -> [f64; ORDER_INTERACTION_FEATURE_COUNT] {
        let additive = self.additive_features();
        let mut features = [0.0; ORDER_INTERACTION_FEATURE_COUNT];
        features[..ORDER_ADDITIVE_FEATURE_COUNT].copy_from_slice(&additive);
        let quality = additive[1];
        let margin = additive[2];
        let role = additive[3];
        let reliability = additive[4];
        features[14] = quality * additive[7];
        features[15] = quality * additive[8];
        features[16] = quality * additive[11];
        features[17] = quality * additive[12];
        features[18] = quality * margin;
        features[19] = quality * reliability;
        features[20] = role * additive[12];
        features[21] = margin * reliability;
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
    const ORDERING: [Self; 3] = [Self::Additive, Self::Interaction, Self::RerankedInteraction];

    fn as_str(self) -> &'static str {
        match self {
            Self::Additive => "additive_ordering",
            Self::Interaction => "interaction_ordering",
            Self::RerankedInteraction => "reranked_interaction",
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
enum OrderingModel {
    Additive(LogisticModel<ORDER_ADDITIVE_FEATURE_COUNT>),
    Interaction(LogisticModel<ORDER_INTERACTION_FEATURE_COUNT>),
}

impl OrderingModel {
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
struct OrderingPolicy {
    model: OrderingModel,
    threshold: f64,
}

impl OrderingPolicy {
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
struct OrderingOperatingPoint {
    policy: OrderingPolicy,
    metrics: EmissionMetrics,
}

#[derive(Clone)]
struct OrderingFold {
    held_out: Population,
    variant: CalibrationVariant,
    target: f64,
    ranking: RankingConfig,
    policy: OrderingPolicy,
    training_metrics: EmissionMetrics,
    held_out_metrics: EmissionMetrics,
}

#[derive(Clone)]
struct CrossValidatedOrderingPoint {
    variant: CalibrationVariant,
    target: f64,
    metrics: EmissionMetrics,
}

#[derive(Clone)]
struct FullDevelopmentVariant {
    variant: CalibrationVariant,
    ranking: RankingConfig,
    model: OrderingModel,
    frontier: Vec<OrderingOperatingPoint>,
}

#[derive(Clone)]
struct SelectedOrderingPoint {
    target: f64,
    variant: CalibrationVariant,
    full_development: OrderingOperatingPoint,
    ranking: RankingConfig,
}

#[derive(Clone)]
struct BaselineSelection {
    target: f64,
    family: super::Family,
    full_development: super::OperatingPoint,
}

pub(crate) fn run_ordering_diagnostic(
    output: &Path,
    corpus: &impl EvidenceSource,
    holdouts: Vec<FrozenHoldout>,
    fixtures: &Path,
) -> Result<String> {
    let holdouts = validate_and_order_holdouts(holdouts)?;
    let proxy_rows = build_proxy_ordering_rows(corpus, &holdouts);
    let validation_rows = build_validation_ordering_rows(corpus, fixtures)?;
    assert_ordering_dataset_counts(&proxy_rows)?;

    let baseline_rows = proxy_rows
        .iter()
        .map(|row| row.base.clone())
        .collect::<Vec<_>>();
    super::assert_historical_checkpoints(&baseline_rows)?;
    let baseline_folds = super::logo_frontier(&baseline_rows)?;
    let baseline_best = super::best_cross_validated_families(&baseline_folds);
    let baseline_selections = baseline_full_development(&baseline_rows, &baseline_best)?;

    let configs = ranking_configs();
    let full_ranking = select_ranking_config(&proxy_rows, &configs);
    let grid = ranking_grid(&proxy_rows, &configs);
    let folds = ordering_logo_frontier(&proxy_rows, &configs)?;
    let best_ordering = best_ordering_by_target(&folds);
    let full_variants = full_development_variants(&proxy_rows, full_ranking)?;
    let selections = select_full_development_points(&best_ordering, &full_variants);

    let outputs = build_outputs(
        &holdouts,
        &proxy_rows,
        &validation_rows,
        &baseline_best,
        &baseline_folds,
        &baseline_selections,
        &grid,
        full_ranking,
        &folds,
        &best_ordering,
        &full_variants,
        &selections,
        corpus,
    )?;
    let repeated = build_outputs(
        &holdouts,
        &proxy_rows,
        &validation_rows,
        &baseline_best,
        &baseline_folds,
        &baseline_selections,
        &grid,
        full_ranking,
        &folds,
        &best_ordering,
        &full_variants,
        &selections,
        corpus,
    )?;
    if outputs != repeated {
        return Err("ordering diagnostic serialization is not deterministic".into());
    }
    for (name, bytes) in &outputs {
        fs::write(output.join(name), bytes)?;
    }
    Ok(String::from_utf8(
        outputs
            .get("report.md")
            .ok_or("ordering report missing")?
            .clone(),
    )?)
}

fn build_proxy_ordering_rows(
    corpus: &impl EvidenceSource,
    holdouts: &[FrozenHoldout],
) -> Vec<OrderingRow> {
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
            rows.push(build_ordering_row(
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

fn build_validation_ordering_rows(
    corpus: &impl EvidenceSource,
    fixtures: &Path,
) -> Result<Vec<OrderingRow>> {
    Ok(generate_cases(fixtures, false)?
        .into_iter()
        .filter(|case| case.split == Split::Validation)
        .enumerate()
        .map(|(ordinal, case)| build_ordering_row_from_case(corpus, ordinal, &case))
        .collect())
}

fn build_ordering_row_from_case(
    corpus: &impl EvidenceSource,
    ordinal: usize,
    case: &Case,
) -> OrderingRow {
    build_ordering_row(
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
fn build_ordering_row(
    corpus: &impl EvidenceSource,
    population: Population,
    ordinal: usize,
    display_name: &str,
    expected_greeting: Option<&str>,
    country_hint: Option<&str>,
    locale_hint: Option<&str>,
    category: &str,
) -> OrderingRow {
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
    let display_token_count = canonicalize(display_name).split_whitespace().count();
    let prior = resolve_name_order_prior(country_hint, locale_hint);
    let candidate_ordering = diagnostic
        .candidates
        .iter()
        .map(|candidate| candidate_ordering(display_name, candidate, display_token_count, prior))
        .collect::<Vec<_>>();
    let candidate_matches = diagnostic
        .candidates
        .iter()
        .map(|candidate| greeting_matches(expected_greeting, Some(&candidate.display)))
        .collect();
    OrderingRow {
        base,
        diagnostic,
        candidate_ordering,
        candidate_matches,
        category: category.to_string(),
    }
}

fn candidate_ordering(
    display_name: &str,
    candidate: &CandidateDiagnostic,
    display_token_count: usize,
    prior: NameOrderPrior,
) -> CandidateOrdering {
    let end = candidate
        .start
        .saturating_add(candidate.length)
        .saturating_sub(1);
    let initial = candidate.start == 0;
    let final_candidate = end.saturating_add(1) == display_token_count;
    let position = match (initial, final_candidate) {
        (true, true) => CandidatePosition::Whole,
        (true, false) => CandidatePosition::Initial,
        (false, true) => CandidatePosition::Final,
        (false, false) => CandidatePosition::Interior,
    };
    let comma_present = display_name.contains(',');
    let comma_inversion = comma_inversion_candidate(display_name, candidate);
    let agrees_with_prior = matches!(
        (prior, position),
        (NameOrderPrior::GivenFirst, CandidatePosition::Initial)
            | (NameOrderPrior::SurnameFirst, CandidatePosition::Final)
    );
    let conflicts_with_prior = matches!(
        (prior, position),
        (NameOrderPrior::GivenFirst, CandidatePosition::Final)
            | (NameOrderPrior::SurnameFirst, CandidatePosition::Initial)
    );
    CandidateOrdering {
        start: candidate.start,
        end,
        token_count: candidate.length,
        display_token_count,
        position,
        comma_present,
        comma_inversion,
        prior,
        agrees_with_prior,
        conflicts_with_prior,
    }
}

fn comma_inversion_candidate(display_name: &str, candidate: &CandidateDiagnostic) -> bool {
    let mut commas = display_name.match_indices(',');
    let Some((comma, _)) = commas.next() else {
        return false;
    };
    if commas.next().is_some() {
        return false;
    }
    let (Some(start), Some(end)) = (candidate.byte_start, candidate.byte_end) else {
        return false;
    };
    comma < start
        && !display_name[..comma].trim().is_empty()
        && display_name[comma + 1..start]
            .chars()
            .all(char::is_whitespace)
        && !display_name[start..end].contains(',')
        && candidate.start > 0
}

fn resolve_name_order_prior(
    country_hint: Option<&str>,
    locale_hint: Option<&str>,
) -> NameOrderPrior {
    locale_hint
        .and_then(locale_name_order_prior)
        .or_else(|| country_hint.and_then(region_name_order_prior))
        .unwrap_or(NameOrderPrior::Neutral)
}

fn locale_name_order_prior(locale: &str) -> Option<NameOrderPrior> {
    let subtags = locale
        .split(['-', '_'])
        .filter(|subtag| !subtag.is_empty())
        .collect::<Vec<_>>();
    let language = subtags.first()?.to_ascii_lowercase();
    if language == "x" {
        return None;
    }
    if language == "und" {
        return subtags
            .iter()
            .skip(1)
            .find_map(|subtag| region_name_order_prior(subtag));
    }
    if !(2..=3).contains(&language.len())
        || !language.bytes().all(|byte| byte.is_ascii_alphabetic())
    {
        return None;
    }
    Some(if SURNAME_FIRST_LANGUAGES.contains(&language.as_str()) {
        NameOrderPrior::SurnameFirst
    } else {
        NameOrderPrior::GivenFirst
    })
}

fn region_name_order_prior(region: &str) -> Option<NameOrderPrior> {
    let bytes = region.trim().as_bytes();
    if bytes.len() != 2 || !bytes.iter().all(u8::is_ascii_alphabetic) {
        return None;
    }
    let region = [bytes[0].to_ascii_uppercase(), bytes[1].to_ascii_uppercase()];
    KNOWN_REGIONS.binary_search(&region).ok().map(|_| {
        if SURNAME_FIRST_REGIONS.binary_search(&region).is_ok() {
            NameOrderPrior::SurnameFirst
        } else {
            NameOrderPrior::GivenFirst
        }
    })
}

fn nonempty(value: &str) -> Option<&str> {
    (!value.is_empty()).then_some(value)
}

fn ranking_configs() -> Vec<RankingConfig> {
    let mut configs = vec![RankingConfig::FROZEN];
    for family in [RankingFamily::Flat, RankingFamily::Confirmatory] {
        for generic_weight in GENERIC_WEIGHTS {
            for prior_weight in PRIOR_WEIGHTS {
                for comma_weight in COMMA_WEIGHTS {
                    if generic_weight == 0.0 && prior_weight == 0.0 && comma_weight == 0.0 {
                        continue;
                    }
                    configs.push(RankingConfig {
                        family,
                        generic_weight,
                        prior_weight,
                        comma_weight,
                    });
                }
            }
        }
    }
    configs
}

fn rank_row(row: &OrderingRow, config: RankingConfig) -> RankedRow {
    let mut ranked = row
        .diagnostic
        .candidates
        .iter()
        .zip(&row.candidate_ordering)
        .enumerate()
        .map(|(index, (candidate, ordering))| {
            (
                index,
                candidate.score + config.adjustment(candidate, *ordering),
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
            ordering: None,
            competitor_relation: CandidateRelation::None,
            selected_index: None,
        };
    };
    let second = ranked.get(1).copied();
    let candidate = &row.diagnostic.candidates[selected_index];
    let ordering = row.candidate_ordering[selected_index];
    let competitor_relation = second.map_or(CandidateRelation::None, |(index, _)| {
        candidate_relation(ordering, row.candidate_ordering[index])
    });
    let winner_margin = second.map_or(1.0, |(_, score)| adjusted_score - score);
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
        ordering: Some(ordering),
        competitor_relation,
        selected_index: Some(selected_index),
    }
}

fn candidate_relation(
    selected: CandidateOrdering,
    competitor: CandidateOrdering,
) -> CandidateRelation {
    if selected.end < competitor.start {
        CandidateRelation::Before
    } else if selected.start > competitor.end {
        CandidateRelation::After
    } else {
        CandidateRelation::Overlap
    }
}

fn ranking_metrics(rows: &[OrderingRow], config: RankingConfig) -> RankingMetrics {
    let mut metrics = RankingMetrics::default();
    for row in rows {
        let ranked = rank_row(row, config);
        metrics.rows += 1;
        metrics.expected_greetings += usize::from(row.base.expected_greeting);
        metrics.expected_nulls += usize::from(!row.base.expected_greeting);
        metrics.generation_ceiling += usize::from(
            row.base.expected_greeting && row.candidate_matches.iter().any(|matches| *matches),
        );
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

fn ranking_grid(
    rows: &[OrderingRow],
    configs: &[RankingConfig],
) -> Vec<(RankingConfig, RankingMetrics)> {
    configs
        .iter()
        .copied()
        .map(|config| (config, ranking_metrics(rows, config)))
        .collect()
}

fn select_ranking_config(rows: &[OrderingRow], configs: &[RankingConfig]) -> RankingConfig {
    configs
        .iter()
        .copied()
        .max_by(|left, right| compare_ranking_configs(rows, *left, *right))
        .expect("ranking grid is nonempty")
}

fn compare_ranking_configs(
    rows: &[OrderingRow],
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
        .then_with(|| right.maximum_weight().total_cmp(&left.maximum_weight()))
        .then_with(|| right.complexity().cmp(&left.complexity()))
        .then_with(|| right.parameters().cmp(&left.parameters()))
}

fn assert_ordering_dataset_counts(rows: &[OrderingRow]) -> Result<()> {
    let expected = [
        (Population::V1, 1_957, 1_616, 341),
        (Population::V2, 1_496, 1_217, 279),
        (Population::V3, 1_474, 1_232, 242),
        (Population::V4, 1_441, 1_220, 221),
        (Population::V5, 1_440, 1_193, 247),
    ];
    for (population, total, greetings, nulls) in expected {
        let population_rows = rows
            .iter()
            .filter(|row| row.base.population == population)
            .collect::<Vec<_>>();
        let actual_greetings = population_rows
            .iter()
            .filter(|row| row.base.expected_greeting)
            .count();
        if population_rows.len() != total
            || actual_greetings != greetings
            || population_rows.len() - actual_greetings != nulls
        {
            return Err(format!(
                "{} population mismatch: rows={}, greetings={}, nulls={}",
                population.as_str(),
                population_rows.len(),
                actual_greetings,
                population_rows.len() - actual_greetings,
            )
            .into());
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct WeightedTrainingRow<const N: usize> {
    features: [f64; N],
    label: f64,
    weight: f64,
}

fn ordering_logo_frontier(
    rows: &[OrderingRow],
    configs: &[RankingConfig],
) -> Result<Vec<OrderingFold>> {
    let mut folds = Vec::new();
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
        for variant in CalibrationVariant::ORDERING {
            let training_ranked = ranked_rows_for_variant(&training, variant, ranking);
            let held_out_ranked = ranked_rows_for_variant(&held_out_rows, variant, ranking);
            let model = fit_ordering_model(&training_ranked, variant)?;
            let frontier = ordering_frontier(&training_ranked, &model);
            for target in TARGETS {
                let Some(selected) = select_ordering_point(&frontier, target) else {
                    continue;
                };
                folds.push(OrderingFold {
                    held_out,
                    variant,
                    target,
                    ranking,
                    policy: selected.policy.clone(),
                    training_metrics: selected.metrics,
                    held_out_metrics: evaluate_ordering_policy(
                        held_out_ranked.iter(),
                        &selected.policy,
                    ),
                });
            }
        }
    }
    Ok(folds)
}

fn best_ordering_by_target(folds: &[OrderingFold]) -> Vec<CrossValidatedOrderingPoint> {
    TARGETS
        .into_iter()
        .filter_map(|target| {
            CalibrationVariant::ORDERING
                .into_iter()
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
                    Some(CrossValidatedOrderingPoint {
                        variant,
                        target,
                        metrics,
                    })
                })
                .max_by(compare_cross_validated_ordering_points)
        })
        .collect()
}

fn compare_cross_validated_ordering_points(
    left: &CrossValidatedOrderingPoint,
    right: &CrossValidatedOrderingPoint,
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
    rows: &[OrderingRow],
    ranking: RankingConfig,
) -> Result<Vec<FullDevelopmentVariant>> {
    CalibrationVariant::ORDERING
        .into_iter()
        .map(|variant| {
            let ranked = ranked_rows_for_variant(rows, variant, ranking);
            let model = fit_ordering_model(&ranked, variant)?;
            let frontier = ordering_frontier(&ranked, &model);
            Ok(FullDevelopmentVariant {
                variant,
                ranking: ranking_for_variant(variant, ranking),
                model,
                frontier,
            })
        })
        .collect()
}

fn select_full_development_points(
    best: &[CrossValidatedOrderingPoint],
    variants: &[FullDevelopmentVariant],
) -> Vec<SelectedOrderingPoint> {
    best.iter()
        .filter_map(|point| {
            let variant = variants
                .iter()
                .find(|variant| variant.variant == point.variant)?;
            let full_development = select_ordering_point(&variant.frontier, point.target)?.clone();
            Some(SelectedOrderingPoint {
                target: point.target,
                variant: point.variant,
                full_development,
                ranking: variant.ranking,
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

fn ranked_rows_for_variant(
    rows: &[OrderingRow],
    variant: CalibrationVariant,
    ranking: RankingConfig,
) -> Vec<RankedRow> {
    let ranking = ranking_for_variant(variant, ranking);
    rows.iter()
        .map(|row| {
            if ranking == RankingConfig::FROZEN {
                frozen_ranked_row(row)
            } else {
                rank_row(row, ranking)
            }
        })
        .collect()
}

fn ranking_for_variant(variant: CalibrationVariant, ranking: RankingConfig) -> RankingConfig {
    match variant {
        CalibrationVariant::Additive | CalibrationVariant::Interaction => RankingConfig::FROZEN,
        CalibrationVariant::RerankedInteraction => ranking,
    }
}

fn frozen_ranked_row(row: &OrderingRow) -> RankedRow {
    let selected_index = (!row.diagnostic.candidates.is_empty()).then_some(0);
    let ordering = selected_index.map(|index| row.candidate_ordering[index]);
    let competitor_relation = match (ordering, row.candidate_ordering.get(1).copied()) {
        (Some(selected), Some(competitor)) => candidate_relation(selected, competitor),
        _ => CandidateRelation::None,
    };
    RankedRow {
        features: row.base.clone(),
        ordering,
        competitor_relation,
        selected_index,
    }
}

fn fit_ordering_model(rows: &[RankedRow], variant: CalibrationVariant) -> Result<OrderingModel> {
    match variant {
        CalibrationVariant::Additive => Ok(OrderingModel::Additive(fit_logistic(
            rows,
            RankedRow::additive_features,
        )?)),
        CalibrationVariant::Interaction | CalibrationVariant::RerankedInteraction => Ok(
            OrderingModel::Interaction(fit_logistic(rows, RankedRow::interaction_features)?),
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
            return Err("ordering logistic optimizer line search failed".into());
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
        "ordering logistic optimizer did not converge in {MAX_OPTIMIZER_ITERATIONS} iterations"
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
        return Err("ordering calibration requires at least one proxy generation".into());
    }
    let counts = populations
        .iter()
        .map(|population| {
            let count = rows
                .iter()
                .filter(|row| row.features.population == *population && row.features.eligible())
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

fn ordering_frontier(rows: &[RankedRow], model: &OrderingModel) -> Vec<OrderingOperatingPoint> {
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
            let policy = OrderingPolicy {
                model: model.clone(),
                threshold,
            };
            OrderingOperatingPoint {
                metrics: evaluate_ordering_policy(rows.iter(), &policy),
                policy,
            }
        })
        .collect::<Vec<_>>();
    deduplicate_ordering_points(&mut points, rows);
    points
}

fn evaluate_ordering_policy<'a>(
    rows: impl Iterator<Item = &'a RankedRow>,
    policy: &OrderingPolicy,
) -> EmissionMetrics {
    let mut metrics = EmissionMetrics::default();
    for row in rows {
        metrics.observe(&row.features, policy.emits(row));
    }
    metrics
}

fn deduplicate_ordering_points(points: &mut Vec<OrderingOperatingPoint>, rows: &[RankedRow]) {
    let mut unique = BTreeMap::<Vec<u64>, OrderingOperatingPoint>::new();
    for point in points.drain(..) {
        let signature = ordering_emission_signature(rows, &point.policy);
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

fn ordering_emission_signature(rows: &[RankedRow], policy: &OrderingPolicy) -> Vec<u64> {
    let mut signature = vec![0_u64; rows.len().div_ceil(64)];
    for (index, row) in rows.iter().enumerate() {
        if policy.emits(row) {
            signature[index / 64] |= 1_u64 << (index % 64);
        }
    }
    signature
}

fn select_ordering_point(
    points: &[OrderingOperatingPoint],
    target: f64,
) -> Option<&OrderingOperatingPoint> {
    points
        .iter()
        .filter(|point| {
            point
                .metrics
                .precision()
                .is_some_and(|precision| precision >= target)
        })
        .max_by(|left, right| compare_ordering_points(left, right))
}

fn compare_ordering_points(
    left: &OrderingOperatingPoint,
    right: &OrderingOperatingPoint,
) -> Ordering {
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

#[allow(clippy::too_many_arguments)]
fn build_outputs(
    holdouts: &[FrozenHoldout],
    proxy_rows: &[OrderingRow],
    validation_rows: &[OrderingRow],
    baseline_best: &[super::CrossValidatedPoint],
    baseline_folds: &[super::FoldResult],
    baseline_selections: &[BaselineSelection],
    grid: &[(RankingConfig, RankingMetrics)],
    full_ranking: RankingConfig,
    folds: &[OrderingFold],
    best_ordering: &[CrossValidatedOrderingPoint],
    full_variants: &[FullDevelopmentVariant],
    selections: &[SelectedOrderingPoint],
    corpus: &impl EvidenceSource,
) -> Result<BTreeMap<String, Vec<u8>>> {
    let mut outputs = BTreeMap::new();
    outputs.insert(
        "dataset_provenance.csv".to_string(),
        super::dataset_provenance_csv(holdouts)?,
    );
    outputs.insert(
        "ordering_features.csv".to_string(),
        ordering_features_csv(proxy_rows, validation_rows)?,
    );
    outputs.insert(
        "ordering_correlations.csv".to_string(),
        ordering_correlations_csv(proxy_rows)?,
    );
    outputs.insert("ranking_grid.csv".to_string(), ranking_grid_csv(grid)?);
    outputs.insert(
        "ranking_logo.csv".to_string(),
        ranking_logo_csv(proxy_rows, folds)?,
    );
    outputs.insert(
        "frontier_comparison.csv".to_string(),
        frontier_comparison_csv(baseline_best, folds)?,
    );
    outputs.insert(
        "ordering_logo_results.csv".to_string(),
        ordering_logo_csv(folds)?,
    );
    outputs.insert(
        "per_generation_operating_points.csv".to_string(),
        ordering_logo_csv(folds)?,
    );
    outputs.insert(
        "interaction_coefficients.csv".to_string(),
        interaction_coefficients_csv(full_variants, folds)?,
    );
    outputs.insert(
        "hint_coverage.csv".to_string(),
        hint_coverage_csv(proxy_rows, validation_rows)?,
    );
    outputs.insert(
        "comma_contribution.csv".to_string(),
        feature_contribution_csv(proxy_rows, "comma_inversion")?,
    );
    outputs.insert(
        "generic_position_contribution.csv".to_string(),
        feature_contribution_csv(proxy_rows, "generic_position")?,
    );
    outputs.insert(
        "synthetic_validation.csv".to_string(),
        synthetic_validation_csv(validation_rows, baseline_selections, selections)?,
    );
    outputs.insert(
        "qualitative_examples.csv".to_string(),
        qualitative_examples_csv(corpus, selections)?,
    );
    outputs.insert("complexity.csv".to_string(), complexity_csv()?);
    outputs.insert(
        "report.md".to_string(),
        build_report(
            holdouts,
            proxy_rows,
            validation_rows,
            baseline_best,
            baseline_folds,
            baseline_selections,
            grid,
            full_ranking,
            folds,
            best_ordering,
            selections,
            corpus,
        )?
        .into_bytes(),
    );
    Ok(outputs)
}

fn ordering_features_csv(
    proxy_rows: &[OrderingRow],
    validation_rows: &[OrderingRow],
) -> Result<Vec<u8>> {
    super::csv_bytes(|writer| {
        writer.write_record([
            "population",
            "ordinal",
            "category",
            "expected_outcome",
            "selected_matches_expected",
            "candidate_start_token",
            "candidate_end_token",
            "candidate_token_count",
            "display_token_count",
            "candidate_is_initial",
            "candidate_is_final",
            "selected_candidate_position",
            "strongest_competitor_position",
            "comma_present",
            "comma_inversion_candidate",
            "name_order_prior",
            "candidate_agrees_with_order_prior",
            "candidate_conflicts_with_order_prior",
            "candidate_quality",
            "winner_margin",
            "role_signal",
            "reliability",
            "country_hint_present",
            "locale_hint_present",
            "c4_emits",
        ])?;
        for row in proxy_rows.iter().chain(validation_rows) {
            let ranked = frozen_ranked_row(row);
            let ordering = ranked.ordering;
            writer.write_record([
                row.base.population.as_str().to_string(),
                row.base.ordinal.to_string(),
                row.category.clone(),
                if row.base.expected_greeting {
                    "greeting"
                } else {
                    "null"
                }
                .to_string(),
                row.base.selected_matches.to_string(),
                ordering.map_or_else(String::new, |value| value.start.to_string()),
                ordering.map_or_else(String::new, |value| value.end.to_string()),
                ordering.map_or_else(String::new, |value| value.token_count.to_string()),
                ordering.map_or_else(String::new, |value| value.display_token_count.to_string()),
                ordering
                    .is_some_and(CandidateOrdering::is_initial)
                    .to_string(),
                ordering
                    .is_some_and(CandidateOrdering::is_final)
                    .to_string(),
                ordering
                    .map_or("none", |value| value.position.as_str())
                    .to_string(),
                ranked.competitor_relation.as_str().to_string(),
                ordering
                    .is_some_and(|value| value.comma_present)
                    .to_string(),
                ordering
                    .is_some_and(|value| value.comma_inversion)
                    .to_string(),
                ordering
                    .map_or("neutral", |value| value.prior.as_str())
                    .to_string(),
                ordering
                    .is_some_and(|value| value.agrees_with_prior)
                    .to_string(),
                ordering
                    .is_some_and(|value| value.conflicts_with_prior)
                    .to_string(),
                super::float(row.base.candidate_quality),
                super::float(row.base.winner_margin),
                super::float(row.base.role_signal),
                super::float(row.base.reliability),
                row.base.country_hint_present.to_string(),
                row.base.locale_hint_present.to_string(),
                row.base.c4_emits.to_string(),
            ])?;
        }
        Ok(())
    })
}

fn ordering_correlations_csv(rows: &[OrderingRow]) -> Result<Vec<u8>> {
    super::csv_bytes(|writer| {
        writer.write_record([
            "population",
            "outcome_population",
            "feature",
            "feature_true",
            "population_rows",
            "rate",
        ])?;
        for population in Population::PROXIES
            .into_iter()
            .map(Some)
            .chain(std::iter::once(None))
        {
            let population_name = population.map_or("COMBINED_SPENT", Population::as_str);
            let selected = rows
                .iter()
                .filter(|row| population.is_none_or(|value| row.base.population == value))
                .collect::<Vec<_>>();
            for (outcome_name, outcome) in [
                (
                    "correct_selected_winner",
                    OutcomePopulation::CorrectSelected,
                ),
                ("wrong_selected_winner", OutcomePopulation::WrongSelected),
                (
                    "expected_null_selected_winner",
                    OutcomePopulation::ExpectedNullSelected,
                ),
                (
                    "correct_winner_abstained_by_c4",
                    OutcomePopulation::CorrectC4Abstention,
                ),
            ] {
                let outcome_rows = selected
                    .iter()
                    .copied()
                    .filter(|row| outcome.includes(&row.base))
                    .collect::<Vec<_>>();
                for feature in OrderingFeature::ALL {
                    let count = outcome_rows
                        .iter()
                        .filter(|row| feature.present(&frozen_ranked_row(row)))
                        .count();
                    writer.write_record([
                        population_name.to_string(),
                        outcome_name.to_string(),
                        feature.as_str().to_string(),
                        count.to_string(),
                        outcome_rows.len().to_string(),
                        super::optional_float(ratio(count, outcome_rows.len())),
                    ])?;
                }
            }
        }
        Ok(())
    })
}

#[derive(Clone, Copy)]
enum OutcomePopulation {
    CorrectSelected,
    WrongSelected,
    ExpectedNullSelected,
    CorrectC4Abstention,
}

impl OutcomePopulation {
    fn includes(self, row: &FeatureRow) -> bool {
        match self {
            Self::CorrectSelected => {
                row.expected_greeting && row.winner_present && row.selected_matches
            }
            Self::WrongSelected => {
                row.expected_greeting && row.winner_present && !row.selected_matches
            }
            Self::ExpectedNullSelected => !row.expected_greeting && row.winner_present,
            Self::CorrectC4Abstention => {
                row.expected_greeting && row.selected_matches && row.vetoes_pass && !row.c4_emits
            }
        }
    }
}

#[derive(Clone, Copy)]
enum OrderingFeature {
    Initial,
    Final,
    BeforeCompetitor,
    AfterCompetitor,
    CommaInversion,
    PriorAgreement,
    PriorConflict,
}

impl OrderingFeature {
    const ALL: [Self; 7] = [
        Self::Initial,
        Self::Final,
        Self::BeforeCompetitor,
        Self::AfterCompetitor,
        Self::CommaInversion,
        Self::PriorAgreement,
        Self::PriorConflict,
    ];

    fn as_str(self) -> &'static str {
        match self {
            Self::Initial => "candidate_initial",
            Self::Final => "candidate_final",
            Self::BeforeCompetitor => "candidate_before_strongest_competitor",
            Self::AfterCompetitor => "candidate_after_strongest_competitor",
            Self::CommaInversion => "comma_inversion",
            Self::PriorAgreement => "agrees_with_order_prior",
            Self::PriorConflict => "conflicts_with_order_prior",
        }
    }

    fn present(self, row: &RankedRow) -> bool {
        match self {
            Self::Initial => row.ordering.is_some_and(CandidateOrdering::is_initial),
            Self::Final => row.ordering.is_some_and(CandidateOrdering::is_final),
            Self::BeforeCompetitor => row.competitor_relation == CandidateRelation::Before,
            Self::AfterCompetitor => row.competitor_relation == CandidateRelation::After,
            Self::CommaInversion => row.ordering.is_some_and(|value| value.comma_inversion),
            Self::PriorAgreement => row.ordering.is_some_and(|value| value.agrees_with_prior),
            Self::PriorConflict => row.ordering.is_some_and(|value| value.conflicts_with_prior),
        }
    }
}

fn ranking_grid_csv(grid: &[(RankingConfig, RankingMetrics)]) -> Result<Vec<u8>> {
    super::csv_bytes(|writer| {
        writer.write_record([
            "family",
            "generic_weight",
            "prior_weight",
            "comma_weight",
            "maximum_adjustment",
            "rows",
            "expected_greetings",
            "expected_nulls",
            "winner_present",
            "generation_ceiling",
            "correct_winners",
            "wrong_winners",
            "null_winners",
            "ranking_ceiling",
        ])?;
        for (config, metrics) in grid {
            writer.write_record([
                config.family.as_str().to_string(),
                super::float(config.generic_weight),
                super::float(config.prior_weight),
                super::float(config.comma_weight),
                super::float(MAX_ORDER_ADJUSTMENT),
                metrics.rows.to_string(),
                metrics.expected_greetings.to_string(),
                metrics.expected_nulls.to_string(),
                metrics.winner_present.to_string(),
                metrics.generation_ceiling.to_string(),
                metrics.correct_winners.to_string(),
                metrics.wrong_winners.to_string(),
                metrics.null_winners.to_string(),
                super::optional_float(ratio(metrics.correct_winners, metrics.expected_greetings)),
            ])?;
        }
        Ok(())
    })
}

fn ranking_logo_csv(rows: &[OrderingRow], folds: &[OrderingFold]) -> Result<Vec<u8>> {
    super::csv_bytes(|writer| {
        writer.write_record([
            "held_out_generation",
            "ranking_parameters",
            "scope",
            "rows",
            "expected_greetings",
            "generation_ceiling",
            "correct_winners",
            "wrong_winners",
            "null_winners",
            "ranking_ceiling",
        ])?;
        for held_out in Population::PROXIES {
            let Some(fold) = folds.iter().find(|fold| fold.held_out == held_out) else {
                continue;
            };
            let config = fold.ranking;
            for (scope, selected) in [
                (
                    "training",
                    rows.iter()
                        .filter(|row| row.base.population != held_out)
                        .cloned()
                        .collect::<Vec<_>>(),
                ),
                (
                    "held_out",
                    rows.iter()
                        .filter(|row| row.base.population == held_out)
                        .cloned()
                        .collect::<Vec<_>>(),
                ),
            ] {
                let metrics = ranking_metrics(&selected, config);
                writer.write_record([
                    held_out.as_str().to_string(),
                    config.parameters(),
                    scope.to_string(),
                    metrics.rows.to_string(),
                    metrics.expected_greetings.to_string(),
                    metrics.generation_ceiling.to_string(),
                    metrics.correct_winners.to_string(),
                    metrics.wrong_winners.to_string(),
                    metrics.null_winners.to_string(),
                    super::optional_float(ratio(
                        metrics.correct_winners,
                        metrics.expected_greetings,
                    )),
                ])?;
            }
        }
        Ok(())
    })
}

fn frontier_comparison_csv(
    baseline: &[super::CrossValidatedPoint],
    folds: &[OrderingFold],
) -> Result<Vec<u8>> {
    super::csv_bytes(|writer| {
        super::write_metrics_header(
            writer,
            &[
                "variant",
                "target",
                "target_met_oof",
                "recall_delta_vs_baseline",
            ],
        )?;
        for target in TARGETS {
            let baseline_point = baseline.iter().find(|point| point.target == target);
            if let Some(point) = baseline_point {
                super::write_metrics_record(
                    writer,
                    &[
                        &format!("baseline_{}", point.family.as_str()),
                        &format!("{target:.4}"),
                        &point
                            .metrics
                            .precision()
                            .is_some_and(|precision| precision >= target)
                            .to_string(),
                        &super::float(0.0),
                    ],
                    point.metrics,
                )?;
            }
            for variant in CalibrationVariant::ORDERING {
                let Some(metrics) = aggregate_ordering_fold_metrics(folds, variant, target) else {
                    continue;
                };
                let delta = metrics.recall().unwrap_or(0.0)
                    - baseline_point
                        .and_then(|point| point.metrics.recall())
                        .unwrap_or(0.0);
                super::write_metrics_record(
                    writer,
                    &[
                        variant.as_str(),
                        &format!("{target:.4}"),
                        &metrics
                            .precision()
                            .is_some_and(|precision| precision >= target)
                            .to_string(),
                        &super::float(delta),
                    ],
                    metrics,
                )?;
            }
        }
        Ok(())
    })
}

fn aggregate_ordering_fold_metrics(
    folds: &[OrderingFold],
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

fn ordering_logo_csv(folds: &[OrderingFold]) -> Result<Vec<u8>> {
    super::csv_bytes(|writer| {
        super::write_metrics_header(
            writer,
            &[
                "variant",
                "target",
                "training_generations",
                "held_out_generation",
                "ranking_parameters",
                "policy_parameters",
                "training_precision",
                "training_recall",
            ],
        )?;
        for fold in folds {
            let training = Population::PROXIES
                .into_iter()
                .filter(|population| *population != fold.held_out)
                .map(Population::as_str)
                .collect::<Vec<_>>()
                .join("+");
            super::write_metrics_record(
                writer,
                &[
                    fold.variant.as_str(),
                    &format!("{:.4}", fold.target),
                    &training,
                    fold.held_out.as_str(),
                    &fold.ranking.parameters(),
                    &fold.policy.parameters(),
                    &super::optional_float(fold.training_metrics.precision()),
                    &super::optional_float(fold.training_metrics.recall()),
                ],
                fold.held_out_metrics,
            )?;
        }
        Ok(())
    })
}

const ADDITIVE_FEATURE_NAMES: [&str; ORDER_ADDITIVE_FEATURE_COUNT] = [
    "decision_score",
    "candidate_quality",
    "winner_margin",
    "role_signal",
    "reliability",
    "sole_candidate",
    "native_provenance",
    "candidate_initial",
    "candidate_final",
    "candidate_before_competitor",
    "candidate_after_competitor",
    "comma_inversion",
    "order_prior_agreement",
    "candidate_span_proportion",
];

const INTERACTION_FEATURE_NAMES: [&str; ORDER_INTERACTION_FEATURE_COUNT] = [
    "decision_score",
    "candidate_quality",
    "winner_margin",
    "role_signal",
    "reliability",
    "sole_candidate",
    "native_provenance",
    "candidate_initial",
    "candidate_final",
    "candidate_before_competitor",
    "candidate_after_competitor",
    "comma_inversion",
    "order_prior_agreement",
    "candidate_span_proportion",
    "quality_x_initial",
    "quality_x_final",
    "quality_x_comma",
    "quality_x_prior_agreement",
    "quality_x_margin",
    "quality_x_reliability",
    "role_x_prior_agreement",
    "margin_x_reliability",
];

fn interaction_coefficients_csv(
    variants: &[FullDevelopmentVariant],
    folds: &[OrderingFold],
) -> Result<Vec<u8>> {
    super::csv_bytes(|writer| {
        writer.write_record([
            "fit",
            "held_out_generation",
            "variant",
            "feature",
            "coefficient",
            "iterations",
        ])?;
        for variant in variants {
            write_ordering_model_rows(
                writer,
                "full_development",
                "",
                variant.variant,
                &variant.model,
            )?;
        }
        let mut seen = BTreeSet::new();
        for fold in folds {
            if seen.insert((fold.held_out, fold.variant)) {
                write_ordering_model_rows(
                    writer,
                    "logo",
                    fold.held_out.as_str(),
                    fold.variant,
                    &fold.policy.model,
                )?;
            }
        }
        Ok(())
    })
}

fn write_ordering_model_rows(
    writer: &mut csv::Writer<Vec<u8>>,
    fit: &str,
    held_out: &str,
    variant: CalibrationVariant,
    model: &OrderingModel,
) -> Result<()> {
    match model {
        OrderingModel::Additive(model) => {
            writer.write_record([
                fit.to_string(),
                held_out.to_string(),
                variant.as_str().to_string(),
                "intercept".to_string(),
                super::float(model.intercept),
                model.iterations.to_string(),
            ])?;
            for (feature, coefficient) in ADDITIVE_FEATURE_NAMES.iter().zip(model.coefficients) {
                writer.write_record([
                    fit.to_string(),
                    held_out.to_string(),
                    variant.as_str().to_string(),
                    (*feature).to_string(),
                    super::float(coefficient),
                    model.iterations.to_string(),
                ])?;
            }
        }
        OrderingModel::Interaction(model) => {
            writer.write_record([
                fit.to_string(),
                held_out.to_string(),
                variant.as_str().to_string(),
                "intercept".to_string(),
                super::float(model.intercept),
                model.iterations.to_string(),
            ])?;
            for (feature, coefficient) in INTERACTION_FEATURE_NAMES.iter().zip(model.coefficients) {
                writer.write_record([
                    fit.to_string(),
                    held_out.to_string(),
                    variant.as_str().to_string(),
                    (*feature).to_string(),
                    super::float(coefficient),
                    model.iterations.to_string(),
                ])?;
            }
        }
    }
    Ok(())
}

fn hint_coverage_csv(
    proxy_rows: &[OrderingRow],
    validation_rows: &[OrderingRow],
) -> Result<Vec<u8>> {
    super::csv_bytes(|writer| {
        writer.write_record([
            "population",
            "hint_status",
            "rows",
            "expected_greetings",
            "correct_selected_winners",
            "order_prior_available",
            "order_prior_agreements",
            "order_prior_conflicts",
        ])?;
        for (population, rows) in [
            ("COMBINED_SPENT", proxy_rows),
            ("VALIDATION", validation_rows),
        ] {
            for (hint_status, hinted) in [("hint_absent", false), ("hint_present", true)] {
                let selected = rows
                    .iter()
                    .filter(|row| {
                        (row.base.country_hint_present || row.base.locale_hint_present) == hinted
                    })
                    .collect::<Vec<_>>();
                let orderings = selected
                    .iter()
                    .filter_map(|row| frozen_ranked_row(row).ordering)
                    .collect::<Vec<_>>();
                writer.write_record([
                    population.to_string(),
                    hint_status.to_string(),
                    selected.len().to_string(),
                    selected
                        .iter()
                        .filter(|row| row.base.expected_greeting)
                        .count()
                        .to_string(),
                    selected
                        .iter()
                        .filter(|row| row.base.expected_greeting && row.base.selected_matches)
                        .count()
                        .to_string(),
                    orderings
                        .iter()
                        .filter(|ordering| ordering.prior != NameOrderPrior::Neutral)
                        .count()
                        .to_string(),
                    orderings
                        .iter()
                        .filter(|ordering| ordering.agrees_with_prior)
                        .count()
                        .to_string(),
                    orderings
                        .iter()
                        .filter(|ordering| ordering.conflicts_with_prior)
                        .count()
                        .to_string(),
                ])?;
            }
        }
        Ok(())
    })
}

fn feature_contribution_csv(rows: &[OrderingRow], feature_name: &str) -> Result<Vec<u8>> {
    let feature = match feature_name {
        "comma_inversion" => OrderingFeature::CommaInversion,
        "generic_position" => OrderingFeature::Initial,
        _ => return Err(format!("unknown ordering contribution feature: {feature_name}").into()),
    };
    super::csv_bytes(|writer| {
        writer.write_record([
            "population",
            "feature",
            "feature_state",
            "rows",
            "correct_selected_winners",
            "wrong_selected_winners",
            "expected_null_selected_winners",
            "correct_c4_abstentions",
        ])?;
        for population in Population::PROXIES
            .into_iter()
            .map(Some)
            .chain(std::iter::once(None))
        {
            for state in [false, true] {
                let selected = rows
                    .iter()
                    .filter(|row| population.is_none_or(|value| row.base.population == value))
                    .filter(|row| feature.present(&frozen_ranked_row(row)) == state)
                    .collect::<Vec<_>>();
                writer.write_record([
                    population
                        .map_or("COMBINED_SPENT", Population::as_str)
                        .to_string(),
                    feature_name.to_string(),
                    state.to_string(),
                    selected.len().to_string(),
                    selected
                        .iter()
                        .filter(|row| OutcomePopulation::CorrectSelected.includes(&row.base))
                        .count()
                        .to_string(),
                    selected
                        .iter()
                        .filter(|row| OutcomePopulation::WrongSelected.includes(&row.base))
                        .count()
                        .to_string(),
                    selected
                        .iter()
                        .filter(|row| OutcomePopulation::ExpectedNullSelected.includes(&row.base))
                        .count()
                        .to_string(),
                    selected
                        .iter()
                        .filter(|row| OutcomePopulation::CorrectC4Abstention.includes(&row.base))
                        .count()
                        .to_string(),
                ])?;
            }
        }
        Ok(())
    })
}

fn synthetic_validation_csv(
    rows: &[OrderingRow],
    baseline: &[BaselineSelection],
    selections: &[SelectedOrderingPoint],
) -> Result<Vec<u8>> {
    super::csv_bytes(|writer| {
        super::write_metrics_header(writer, &["variant", "target", "category"])?;
        let base_rows = rows.iter().map(|row| row.base.clone()).collect::<Vec<_>>();
        super::write_metrics_record(
            writer,
            &["C4", "frozen", "ALL"],
            super::evaluate_policy(base_rows.iter(), &super::Policy::C4),
        )?;
        for selection in baseline {
            super::write_metrics_record(
                writer,
                &[
                    &format!("baseline_{}", selection.family.as_str()),
                    &format!("{:.4}", selection.target),
                    "ALL",
                ],
                super::evaluate_policy(base_rows.iter(), &selection.full_development.policy),
            )?;
        }
        let categories = rows
            .iter()
            .map(|row| row.category.as_str())
            .collect::<BTreeSet<_>>();
        for selection in selections {
            let ranked = ranked_rows_for_selection(rows, selection);
            super::write_metrics_record(
                writer,
                &[
                    selection.variant.as_str(),
                    &format!("{:.4}", selection.target),
                    "ALL",
                ],
                evaluate_ordering_policy(ranked.iter(), &selection.full_development.policy),
            )?;
            for category in &categories {
                let selected = ranked
                    .iter()
                    .zip(rows)
                    .filter(|(_, row)| row.category == *category)
                    .map(|(ranked, _)| ranked)
                    .collect::<Vec<_>>();
                super::write_metrics_record(
                    writer,
                    &[
                        selection.variant.as_str(),
                        &format!("{:.4}", selection.target),
                        category,
                    ],
                    evaluate_ordering_policy(
                        selected.into_iter(),
                        &selection.full_development.policy,
                    ),
                )?;
            }
        }
        Ok(())
    })
}

fn ranked_rows_for_selection(
    rows: &[OrderingRow],
    selection: &SelectedOrderingPoint,
) -> Vec<RankedRow> {
    ranked_rows_for_variant(rows, selection.variant, selection.ranking)
}

fn qualitative_examples_csv(
    corpus: &impl EvidenceSource,
    selections: &[SelectedOrderingPoint],
) -> Result<Vec<u8>> {
    super::csv_bytes(|writer| {
        writer.write_record([
            "input",
            "variant",
            "target",
            "selected_before",
            "selected_after",
            "emits",
            "position",
            "order_prior",
            "prior_agreement",
            "comma_inversion",
            "ranking_parameters",
        ])?;
        for input in [
            // Redacted: Name that emits at the 99% candidate point.
            "Olivier REDACTED",
            // Redacted: Name that emits at the 99% candidate point.
            "Baris REDACTED",
            // Redacted: Name that first emits at the 98% candidate point.
            "Ngoc Lam REDACTED",
            // Redacted: Name that emits at the 99.5% candidate point.
            "Alexandre REDACTED",
        ] {
            let row = build_ordering_row(
                corpus,
                Population::Validation,
                0,
                input,
                None,
                None,
                None,
                "qualitative_only",
            );
            let before = row
                .diagnostic
                .candidates
                .first()
                .map_or("", |candidate| candidate.display.as_str());
            for selection in selections {
                let ranked = ranked_rows_for_selection(std::slice::from_ref(&row), selection)
                    .pop()
                    .expect("one qualitative row");
                let after = ranked.selected_index.map_or("", |index| {
                    row.diagnostic.candidates[index].display.as_str()
                });
                let ordering = ranked.ordering;
                writer.write_record([
                    input.to_string(),
                    selection.variant.as_str().to_string(),
                    format!("{:.4}", selection.target),
                    before.to_string(),
                    after.to_string(),
                    selection.full_development.policy.emits(&ranked).to_string(),
                    ordering
                        .map_or("none", |value| value.position.as_str())
                        .to_string(),
                    ordering
                        .map_or("neutral", |value| value.prior.as_str())
                        .to_string(),
                    ordering
                        .is_some_and(|value| value.agrees_with_prior)
                        .to_string(),
                    ordering
                        .is_some_and(|value| value.comma_inversion)
                        .to_string(),
                    selection.ranking.parameters(),
                ])?;
            }
        }
        Ok(())
    })
}

fn complexity_csv() -> Result<Vec<u8>> {
    super::csv_bytes(|writer| {
        writer.write_record(["item", "value", "unit", "notes"])?;
        let known_region_bytes = mem::size_of_val(&KNOWN_REGIONS).to_string();
        writer.write_record([
            "known_region_table",
            &known_region_bytes,
            "bytes",
            "experimental CLDR-derived two-letter region codes",
        ])?;
        let surname_first_region_bytes = mem::size_of_val(&SURNAME_FIRST_REGIONS).to_string();
        writer.write_record([
            "surname_first_region_table",
            &surname_first_region_bytes,
            "bytes",
            "experimental CLDR-derived two-letter region codes",
        ])?;
        writer.write_record([
            "production_runtime_data",
            "0",
            "bytes",
            "benchmark-only experiment; production C4 is unchanged",
        ])?;
        writer.write_record([
            "production_allocations",
            "0",
            "additional allocations",
            "benchmark-only experiment; no production path change",
        ])?;
        Ok(())
    })
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn build_report(
    holdouts: &[FrozenHoldout],
    proxy_rows: &[OrderingRow],
    validation_rows: &[OrderingRow],
    baseline_best: &[super::CrossValidatedPoint],
    baseline_folds: &[super::FoldResult],
    baseline_selections: &[BaselineSelection],
    grid: &[(RankingConfig, RankingMetrics)],
    full_ranking: RankingConfig,
    folds: &[OrderingFold],
    best_ordering: &[CrossValidatedOrderingPoint],
    selections: &[SelectedOrderingPoint],
    corpus: &impl EvidenceSource,
) -> Result<String> {
    let mut report = String::new();
    writeln!(report, "# Ordering and position evidence diagnostic\n")?;
    writeln!(
        report,
        "C4 remains frozen production behavior. This benchmark-only experiment asks whether weak structural position evidence, and especially confirmatory interactions between position and existing candidate evidence, shift the spent-proxy cross-validated frontier outward. It does not implement or freeze C5.\n"
    )?;

    writeln!(report, "## Data and provenance\n")?;
    writeln!(
        report,
        "The diagnostic uses only spent REAL_PROXY_V1 through V5 plus synthetic VALIDATION. It does not create or inspect V6 or TEST. Proxy labels and raw display names remain in ignored local files; aggregate outputs identify rows only by population and ordinal.\n"
    )?;
    writeln!(
        report,
        "| Population | SHA-256 | Evaluable | Greetings | NULL |"
    )?;
    writeln!(report, "|---|---|---:|---:|---:|")?;
    for holdout in holdouts {
        let population = Population::from_digest(&holdout.manifest.holdout_sha256)
            .expect("validated proxy digest");
        writeln!(
            report,
            "| {} | `{}` | {} | {} | {} |",
            population.as_str(),
            holdout.manifest.holdout_sha256,
            holdout.manifest.evaluable_cases,
            holdout.manifest.expected_greetings,
            holdout.manifest.expected_abstentions,
        )?;
    }
    let expected_greetings = proxy_rows
        .iter()
        .filter(|row| row.base.expected_greeting)
        .count();
    writeln!(
        report,
        "\nCombined: {} evaluable rows, {expected_greetings} expected greetings, and {} expected NULL rows.\n",
        proxy_rows.len(),
        proxy_rows.len() - expected_greetings
    )?;

    writeln!(report, "## Ordering features\n")?;
    writeln!(
        report,
        "Each existing candidate span receives token start/end, span length, total display-token count, initial/final position, position relative to the strongest competitor, comma-inversion shape, and agreement/conflict with a small locale order prior. No unsupported candidate is created. Multi-token candidates retain their existing span.\n"
    )?;
    writeln!(
        report,
        "The locale prior is derived from Unicode CLDR 48 [person-name order data](https://www.unicode.org/reports/tr35/tr35-personNames.html#nameOrderLocales_Element) and `likelySubtags`: `hu`, `ja`, `km`, `ko`, `mn`, `si`, `ta`, `te`, `vi`, `yue`, and `zh` are surname-first; CLDR's `und` default is given-first; unknown or malformed hints are neutral. The experimental representation is {} bytes for known regions plus {} bytes for surname-first regions. Production stores none of it.\n",
        mem::size_of_val(&KNOWN_REGIONS),
        mem::size_of_val(&SURNAME_FIRST_REGIONS)
    )?;
    writeln!(
        report,
        "Generic initial/final evidence is tested separately from hint-aware priors. Comma inversion requires one comma, a nonempty prefix, and a candidate beginning after only whitespace following the comma.\n"
    )?;

    writeln!(report, "## Direct feature correlation\n")?;
    writeln!(
        report,
        "| Outcome population | Rows | Initial | Final | Comma inversion | Prior agreement |"
    )?;
    writeln!(report, "|---|---|---:|---:|---:|---:|")?;
    for (label, outcome) in [
        (
            "correct selected winner",
            OutcomePopulation::CorrectSelected,
        ),
        ("wrong selected winner", OutcomePopulation::WrongSelected),
        (
            "expected-NULL selected winner",
            OutcomePopulation::ExpectedNullSelected,
        ),
        (
            "correct winner rejected by C4",
            OutcomePopulation::CorrectC4Abstention,
        ),
    ] {
        let rows = proxy_rows
            .iter()
            .filter(|row| outcome.includes(&row.base))
            .collect::<Vec<_>>();
        writeln!(
            report,
            "| {label} | {} | {} | {} | {} | {} |",
            rows.len(),
            feature_rate(&rows, OrderingFeature::Initial),
            feature_rate(&rows, OrderingFeature::Final),
            feature_rate(&rows, OrderingFeature::CommaInversion),
            feature_rate(&rows, OrderingFeature::PriorAgreement),
        )?;
    }
    writeln!(
        report,
        "\nDetailed per-generation counts are in `ordering_correlations.csv`.\n"
    )?;

    writeln!(report, "## Ranking experiment\n")?;
    writeln!(
        report,
        "The control applies flat bounded position bonuses. The confirmatory family multiplies those same signals by frozen candidate quality, so weak candidates receive little help while already plausible candidates receive stronger support. All total adjustments are clamped to ±{MAX_ORDER_ADJUSTMENT:.2}. The deterministic search covers generic weights `{GENERIC_WEIGHTS:?}`, prior weights `{PRIOR_WEIGHTS:?}`, and comma weights `{COMMA_WEIGHTS:?}`.\n"
    )?;
    let frozen_ranking = grid
        .iter()
        .find(|(config, _)| *config == RankingConfig::FROZEN)
        .map(|(_, metrics)| *metrics)
        .ok_or("frozen ranking grid row missing")?;
    let selected_ranking = ranking_metrics(proxy_rows, full_ranking);
    writeln!(
        report,
        "Selected full-development ranking: `{}`.\n",
        full_ranking.parameters()
    )?;
    writeln!(
        report,
        "| Ranking | Generation ceiling | Correct winner | Wrong winner | NULL winner | Ranking ceiling |"
    )?;
    writeln!(report, "|---|---:|---:|---:|---:|---:|")?;
    write_ranking_report_row(&mut report, "frozen", frozen_ranking)?;
    write_ranking_report_row(&mut report, "selected ordering", selected_ranking)?;
    writeln!(
        report,
        "\nThe candidate-generation ceiling is unchanged by construction. `ranking_grid.csv` contains every flat and confirmatory configuration; `ranking_logo.csv` contains generation-held-out selections.\n"
    )?;

    writeln!(report, "## Out-of-fold precision/recall frontier\n")?;
    writeln!(
        report,
        "The baseline is the frozen C5 calibration-frontier study. Additive ordering adds the seven structural features as independent terms. Interaction ordering adds eight predeclared confirmatory products: quality × initial/final/comma/prior/margin/reliability, role × prior, and margin × reliability. The reranked interaction variant also applies the bounded selected ranking adjustment. All logistic coefficients remain nonnegative with L2 regularization; the logistic link supplies saturation.\n"
    )?;
    writeln!(
        report,
        "| Target | Baseline variant | Baseline precision | Baseline recall | Best ordering variant | Ordering precision | Target met | Ordering Wilson 95% | Ordering recall | Recall delta | Correct-winner rejected: baseline → ordering |"
    )?;
    writeln!(
        report,
        "|---:|---|---:|---:|---|---:|---:|---|---:|---:|---:|"
    )?;
    for target in TARGETS {
        let baseline = baseline_best.iter().find(|point| point.target == target);
        let ordering = best_ordering.iter().find(|point| point.target == target);
        let (Some(baseline), Some(ordering)) = (baseline, ordering) else {
            continue;
        };
        let delta =
            ordering.metrics.recall().unwrap_or(0.0) - baseline.metrics.recall().unwrap_or(0.0);
        let interval = wilson_interval(ordering.metrics.correct, ordering.metrics.emitted);
        writeln!(
            report,
            "| {:.1}% | {} | {} | {} | {} | {} | {} | {}–{} | {} | {:+.2} pp | {} → {} |",
            target * 100.0,
            baseline.family.as_str(),
            percent(baseline.metrics.precision()),
            percent(baseline.metrics.recall()),
            ordering.variant.as_str(),
            percent(ordering.metrics.precision()),
            ordering
                .metrics
                .precision()
                .is_some_and(|precision| precision >= target),
            percent(interval.map(|value| value.lower)),
            percent(interval.map(|value| value.upper)),
            percent(ordering.metrics.recall()),
            delta * 100.0,
            baseline.metrics.winner_correct_but_abstained,
            ordering.metrics.winner_correct_but_abstained,
        )?;
    }
    writeln!(
        report,
        "\nThe full comparison, including correct, wrong, NULL false emissions, false abstentions, and Wilson intervals, is in `frontier_comparison.csv`.\n"
    )?;

    writeln!(report, "## Per-generation stability and LOGO selection\n")?;
    writeln!(
        report,
        "Each fold selects both ranking parameters and calibration weights without the omitted proxy generation.\n"
    )?;
    writeln!(
        report,
        "| Target | Variant | Held out | Precision | Recall | Correct | Wrong | NULL FP | Correct winner rejected |"
    )?;
    writeln!(report, "|---:|---|---|---:|---:|---:|---:|---:|---:|")?;
    for fold in folds.iter().filter(|fold| {
        [0.995, 0.99, 0.98, 0.95]
            .into_iter()
            .any(|target| fold.target == target)
            && best_ordering
                .iter()
                .any(|point| point.target == fold.target && point.variant == fold.variant)
    }) {
        writeln!(
            report,
            "| {:.1}% | {} | {} | {} | {} | {} | {} | {} | {} |",
            fold.target * 100.0,
            fold.variant.as_str(),
            fold.held_out.as_str(),
            percent(fold.held_out_metrics.precision()),
            percent(fold.held_out_metrics.recall()),
            fold.held_out_metrics.correct,
            fold.held_out_metrics.wrong,
            fold.held_out_metrics.null_false_emissions,
            fold.held_out_metrics.winner_correct_but_abstained,
        )?;
    }
    writeln!(
        report,
        "\n`ordering_logo_results.csv` records every variant, target, fitted policy, training result, and held-out result. Baseline LOGO rows remain reproducible ({} rows).\n",
        baseline_folds.len()
    )?;

    writeln!(report, "## Hints, comma structure, and missing hints\n")?;
    let proxy_hints = proxy_rows
        .iter()
        .filter(|row| row.base.country_hint_present || row.base.locale_hint_present)
        .count();
    let validation_hints = validation_rows
        .iter()
        .filter(|row| row.base.country_hint_present || row.base.locale_hint_present)
        .count();
    let proxy_commas = proxy_rows
        .iter()
        .filter(|row| {
            frozen_ranked_row(row)
                .ordering
                .is_some_and(|ordering| ordering.comma_inversion)
        })
        .count();
    writeln!(
        report,
        "Real-proxy rows with a country or locale hint: {proxy_hints}/{}. Synthetic VALIDATION hint-bearing rows: {validation_hints}/{}. Consequently, the proxy frontier primarily evaluates generic position and comma structure, not culture-aware priors; country-sensitive ordering is only a synthetic sanity check and is not real-world validated here. Proxy comma-inversion candidates: {proxy_commas}. Detailed counts are in `hint_coverage.csv`, `comma_contribution.csv`, and `generic_position_contribution.csv`.\n",
        proxy_rows.len(),
        validation_rows.len()
    )?;

    writeln!(report, "## Synthetic VALIDATION regression check\n")?;
    writeln!(
        report,
        "| Target | Variant | Precision | Recall | Correct | Wrong | NULL FP | Correct winner rejected |"
    )?;
    writeln!(report, "|---:|---|---:|---:|---:|---:|---:|---:|")?;
    let validation_base = validation_rows
        .iter()
        .map(|row| row.base.clone())
        .collect::<Vec<_>>();
    for selection in baseline_selections {
        let metrics =
            super::evaluate_policy(validation_base.iter(), &selection.full_development.policy);
        writeln!(
            report,
            "| {:.1}% | baseline_{} | {} | {} | {} | {} | {} | {} |",
            selection.target * 100.0,
            selection.family.as_str(),
            percent(metrics.precision()),
            percent(metrics.recall()),
            metrics.correct,
            metrics.wrong,
            metrics.null_false_emissions,
            metrics.winner_correct_but_abstained,
        )?;
    }
    for selection in selections {
        let ranked = ranked_rows_for_selection(validation_rows, selection);
        let metrics = evaluate_ordering_policy(ranked.iter(), &selection.full_development.policy);
        writeln!(
            report,
            "| {:.1}% | {} | {} | {} | {} | {} | {} | {} |",
            selection.target * 100.0,
            selection.variant.as_str(),
            percent(metrics.precision()),
            percent(metrics.recall()),
            metrics.correct,
            metrics.wrong,
            metrics.null_false_emissions,
            metrics.winner_correct_but_abstained,
        )?;
    }
    writeln!(
        report,
        "\nCategory-level results, including surname-first, given-first, comma-inverted, and multi-token fixtures, are in `synthetic_validation.csv`.\n"
    )?;
    let baseline_995 = baseline_selections
        .iter()
        .find(|selection| selection.target == 0.995)
        .map(|selection| {
            super::evaluate_policy(validation_base.iter(), &selection.full_development.policy)
        });
    let ordering_995 = selections
        .iter()
        .find(|selection| selection.target == 0.995)
        .map(|selection| {
            let ranked = ranked_rows_for_selection(validation_rows, selection);
            evaluate_ordering_policy(ranked.iter(), &selection.full_development.policy)
        });
    writeln!(
        report,
        "The broad proxy gains do not transfer safely to the synthetic structural population: at the 99.5% selection target, matched baseline VALIDATION precision is {} while the ordering-enabled point is {}. Category rows show that generic first-position evidence especially harms family-name-first and surname-given structures. This prevents a broadly positive recommendation despite the strong proxy correlation.\n",
        percent(baseline_995.and_then(EmissionMetrics::precision)),
        percent(ordering_995.and_then(EmissionMetrics::precision)),
    )?;

    writeln!(report, "## Qualitative smoke tests\n")?;
    writeln!(
        report,
        "These four examples were evaluated only after model and operating-point selection and did not influence fitting.\n"
    )?;
    writeln!(
        report,
        "| Input | Target | Variant | Winner before → after | Emits | Evidence |"
    )?;
    writeln!(report, "|---|---:|---|---|---:|---|")?;
    write_qualitative_report_rows(&mut report, corpus, selections)?;

    let recommendation = ordering_recommendation(
        baseline_best,
        best_ordering,
        validation_rows,
        baseline_selections,
        selections,
    );
    writeln!(report, "\n## Recommendation\n")?;
    writeln!(
        report,
        "**{recommendation}.** {}\n",
        recommendation_explanation(recommendation)
    )?;
    writeln!(
        report,
        "This is a feature-value decision only. C4 remains production behavior, no C5 policy is frozen, and no fresh holdout has been consumed. Candidate quality is tested both as an additive term and as confirmatory interactions; position is not allowed to rescue unsupported candidates.\n"
    )?;

    writeln!(report, "## Complexity and limitations\n")?;
    writeln!(
        report,
        "The experiment adds no production data, code path, allocation, or per-inference cost. A future implementation would need a tiny locale-order table plus several scalar feature products. The spent proxy population is machine-consensus-labeled and largely lacks hints, so its observed precision is not worldwide population precision and it cannot validate locale-sensitive behavior. Capitalization, morphology, neural scoring, new candidate generation, corpus changes, and artifact changes are explicitly outside this experiment.\n"
    )?;
    Ok(report)
}

fn feature_rate(rows: &[&OrderingRow], feature: OrderingFeature) -> String {
    let count = rows
        .iter()
        .filter(|row| feature.present(&frozen_ranked_row(row)))
        .count();
    percent(ratio(count, rows.len()))
}

fn write_ranking_report_row(
    report: &mut String,
    label: &str,
    metrics: RankingMetrics,
) -> Result<()> {
    writeln!(
        report,
        "| {label} | {} | {} | {} | {} | {} |",
        metrics.generation_ceiling,
        metrics.correct_winners,
        metrics.wrong_winners,
        metrics.null_winners,
        percent(ratio(metrics.correct_winners, metrics.expected_greetings)),
    )?;
    Ok(())
}

fn write_qualitative_report_rows(
    report: &mut String,
    corpus: &impl EvidenceSource,
    selections: &[SelectedOrderingPoint],
) -> Result<()> {
    for input in [
        // Redacted: Name that emits at the 99% candidate point.
        "Olivier REDACTED",
        // Redacted: Name that emits at the 99% candidate point.
        "Baris REDACTED",
        // Redacted: Name that first emits at the 98% candidate point.
        "Ngoc Lam REDACTED",
        // Redacted: Name that emits at the 99.5% candidate point.
        "Alexandre REDACTED",
    ] {
        let row = build_ordering_row(
            corpus,
            Population::Validation,
            0,
            input,
            None,
            None,
            None,
            "qualitative_only",
        );
        let before = row
            .diagnostic
            .candidates
            .first()
            .map_or("", |candidate| candidate.display.as_str());
        for selection in selections {
            let ranked = ranked_rows_for_selection(std::slice::from_ref(&row), selection)
                .pop()
                .expect("one qualitative row");
            let after = ranked.selected_index.map_or("", |index| {
                row.diagnostic.candidates[index].display.as_str()
            });
            let evidence = ranked.ordering.map_or_else(
                || "none".to_string(),
                |ordering| {
                    format!(
                        "position={}; prior={}; agrees={}; comma={}",
                        ordering.position.as_str(),
                        ordering.prior.as_str(),
                        ordering.agrees_with_prior,
                        ordering.comma_inversion,
                    )
                },
            );
            writeln!(
                report,
                "| {input} | {:.1}% | {} | {} → {} | {} | {} |",
                selection.target * 100.0,
                selection.variant.as_str(),
                before,
                after,
                selection.full_development.policy.emits(&ranked),
                evidence,
            )?;
        }
    }
    Ok(())
}

fn ordering_recommendation(
    baseline: &[super::CrossValidatedPoint],
    ordering: &[CrossValidatedOrderingPoint],
    validation_rows: &[OrderingRow],
    baseline_selections: &[BaselineSelection],
    selections: &[SelectedOrderingPoint],
) -> &'static str {
    let meaningful = [0.995, 0.99, 0.98, 0.95]
        .into_iter()
        .filter(|target| {
            let baseline = baseline.iter().find(|point| point.target == *target);
            let ordering = ordering.iter().find(|point| point.target == *target);
            match (baseline, ordering) {
                (Some(baseline), Some(ordering)) => {
                    ordering.metrics.recall().unwrap_or(0.0)
                        - baseline.metrics.recall().unwrap_or(0.0)
                        >= 0.01
                        && ordering.metrics.precision().unwrap_or(0.0)
                            >= baseline.metrics.precision().unwrap_or(0.0) - 0.001
                }
                _ => false,
            }
        })
        .count();
    let validation_base = validation_rows
        .iter()
        .map(|row| row.base.clone())
        .collect::<Vec<_>>();
    let severe_synthetic_regressions = [0.995, 0.99, 0.98, 0.95]
        .into_iter()
        .filter(|target| {
            let baseline = baseline_selections
                .iter()
                .find(|selection| selection.target == *target);
            let ordering = selections
                .iter()
                .find(|selection| selection.target == *target);
            let (Some(baseline), Some(ordering)) = (baseline, ordering) else {
                return false;
            };
            let baseline_metrics =
                super::evaluate_policy(validation_base.iter(), &baseline.full_development.policy);
            let ranked = ranked_rows_for_selection(validation_rows, ordering);
            let ordering_metrics =
                evaluate_ordering_policy(ranked.iter(), &ordering.full_development.policy);
            ordering_metrics.precision().unwrap_or(0.0)
                < baseline_metrics.precision().unwrap_or(0.0) - 0.02
        })
        .count();
    if meaningful >= 2 && severe_synthetic_regressions == 0 {
        "Strongly useful"
    } else if TARGETS.into_iter().any(|target| {
        let baseline = baseline.iter().find(|point| point.target == target);
        let ordering = ordering.iter().find(|point| point.target == target);
        matches!((baseline, ordering), (Some(baseline), Some(ordering)) if ordering.metrics.correct > baseline.metrics.correct)
    }) {
        "Marginal"
    } else {
        "Harmful / no value"
    }
}

fn recommendation_explanation(recommendation: &str) -> &'static str {
    match recommendation {
        "Strongly useful" => {
            "Ordering improves held-out recall by at least one percentage point at multiple 95–99.5% targets without a material observed precision loss, so retain it for consideration in a future C5 feature set."
        }
        "Marginal" => {
            "Ordering recovers many held-out proxy greetings, but broader operating points materially regress synthetic family-name-first and surname-given structures. Keep it experimental for a future culture-aware interaction study; do not promote generic position evidence now."
        }
        _ => {
            "Ordering does not reliably move the held-out frontier outward. Drop it rather than preserving a semantically appealing feature without evidence."
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cldr_order_prior_is_neutral_when_unknown_and_uses_locale_before_country() {
        assert_eq!(
            resolve_name_order_prior(Some("FR"), Some("en-US")),
            NameOrderPrior::GivenFirst
        );
        assert_eq!(
            resolve_name_order_prior(Some("FR"), Some("ja-JP")),
            NameOrderPrior::SurnameFirst
        );
        assert_eq!(
            resolve_name_order_prior(None, Some("und-CN")),
            NameOrderPrior::SurnameFirst
        );
        assert_eq!(
            resolve_name_order_prior(Some("FR"), Some("invalid-locale")),
            NameOrderPrior::GivenFirst
        );
        assert_eq!(
            resolve_name_order_prior(Some("ZZ"), None),
            NameOrderPrior::Neutral
        );
    }

    #[test]
    fn comma_inversion_requires_one_nonempty_prefix_and_exact_following_span() {
        let candidate = candidate("Jean", 1, 1, 8, 12, 0.8);
        assert!(comma_inversion_candidate("Martin, Jean", &candidate));
        assert!(!comma_inversion_candidate("Martin Jean", &candidate));
        assert!(!comma_inversion_candidate(",       Jean", &candidate));
        assert!(!comma_inversion_candidate("Martin, Jean, Jr", &candidate));
    }

    #[test]
    fn confirmatory_ranking_scales_position_by_existing_candidate_quality() {
        let candidate = candidate("Alexandre", 0, 1, 0, 9, 0.25);
        let ordering = CandidateOrdering {
            start: 0,
            end: 0,
            token_count: 1,
            display_token_count: 2,
            position: CandidatePosition::Initial,
            comma_present: false,
            comma_inversion: false,
            prior: NameOrderPrior::Neutral,
            agrees_with_prior: false,
            conflicts_with_prior: false,
        };
        let flat = RankingConfig {
            family: RankingFamily::Flat,
            generic_weight: 0.03,
            prior_weight: 0.0,
            comma_weight: 0.0,
        };
        let confirmatory = RankingConfig {
            family: RankingFamily::Confirmatory,
            ..flat
        };
        assert_eq!(
            flat.adjustment(&candidate, ordering).to_bits(),
            0.03_f64.to_bits()
        );
        assert_eq!(
            confirmatory.adjustment(&candidate, ordering).to_bits(),
            0.0075_f64.to_bits()
        );
    }

    #[test]
    fn ranking_adjustment_is_bounded_and_candidate_relation_uses_token_spans() {
        let candidate = candidate("Jean", 1, 1, 8, 12, 1.0);
        let ordering = CandidateOrdering {
            start: 1,
            end: 1,
            token_count: 1,
            display_token_count: 2,
            position: CandidatePosition::Final,
            comma_present: true,
            comma_inversion: true,
            prior: NameOrderPrior::SurnameFirst,
            agrees_with_prior: true,
            conflicts_with_prior: false,
        };
        let config = RankingConfig {
            family: RankingFamily::Flat,
            generic_weight: 0.03,
            prior_weight: 0.03,
            comma_weight: 0.06,
        };
        assert_eq!(
            config.adjustment(&candidate, ordering).to_bits(),
            MAX_ORDER_ADJUSTMENT.to_bits()
        );
        let initial = CandidateOrdering {
            start: 0,
            end: 0,
            position: CandidatePosition::Initial,
            ..ordering
        };
        assert_eq!(
            candidate_relation(initial, ordering),
            CandidateRelation::Before
        );
        assert_eq!(
            candidate_relation(ordering, initial),
            CandidateRelation::After
        );
    }

    fn candidate(
        display: &str,
        start: usize,
        length: usize,
        byte_start: usize,
        byte_end: usize,
        score: f64,
    ) -> CandidateDiagnostic {
        CandidateDiagnostic {
            display: display.to_string(),
            start,
            length,
            byte_start: Some(byte_start),
            byte_end: Some(byte_end),
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
