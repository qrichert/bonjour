use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::Path;

use name_eval::holdout::FrozenHoldout;

use crate::artifact::EvidenceSource;
use crate::classifier::{
    ALGORITHM_C2, ALGORITHM_C3, ALGORITHM_C4, ALGORITHM_C5, ALGORITHM_C31, C4DecisionBreakdown,
    C4EmissionSource, C31DecisionBreakdown, c4_decision_breakdown, c4_emitted_candidate,
    c5_decision_from_c4, c5_emitted_candidate, diagnose_role_inference,
};
use crate::dataset::{Case, Split, generate_cases};
use crate::metrics::greeting_matches;

mod c5_selection;
mod capitalization;
mod morphology;
mod ordering;

pub(crate) use c5_selection::{run_c5_selection, run_sealed_c4_c5_comparison};
pub(crate) use capitalization::run_capitalization_diagnostic;
pub(crate) use morphology::run_morphology_diagnostic;
pub(crate) use ordering::run_ordering_diagnostic;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub const V1_SHA256: &str = "de95213f27fc1849032ee6788c8f16d7d515c1a991ae8b2e8414b7b155814c4e";
pub const V2_SHA256: &str = "7d704a646b8dd9fa3820f88b9504d4397b676af9435532cf2da9befda7663a73";
pub const V3_SHA256: &str = "d70e4d4b2ed7e49bed09dc1e8d2ba60ade8a752e3b86c772e964bd64883ee6fe";
pub const V4_SHA256: &str = "d95c589bec836faaeecaeda85b146989d2936914bff0209934f289ccb9446c7f";
pub const V5_SHA256: &str = "69070614fee68401b896d6c5bfb4c22c55cca9744237f66213a9dd04291db6c7";

const FEATURE_COUNT: usize = 7;
const LOGISTIC_L2: f64 = 0.01;
const MAX_OPTIMIZER_ITERATIONS: usize = 10_000;
const PARAMETER_TOLERANCE: f64 = 1.0e-10;
const ARMIJO: f64 = 1.0e-4;
const TARGETS: [f64; 7] = [0.999, 0.995, 0.99, 0.98, 0.97, 0.95, 0.90];
const COSTS: [usize; 5] = [5, 10, 20, 50, 100];
const Z_95: f64 = 1.959_963_984_540_054;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Population {
    V1,
    V2,
    V3,
    V4,
    V5,
    Validation,
}

impl Population {
    const PROXIES: [Self; 5] = [Self::V1, Self::V2, Self::V3, Self::V4, Self::V5];

    fn from_digest(digest: &str) -> Option<Self> {
        match digest {
            V1_SHA256 => Some(Self::V1),
            V2_SHA256 => Some(Self::V2),
            V3_SHA256 => Some(Self::V3),
            V4_SHA256 => Some(Self::V4),
            V5_SHA256 => Some(Self::V5),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::V1 => "REAL_PROXY_V1_DEV",
            Self::V2 => "REAL_PROXY_V2_DEV",
            Self::V3 => "REAL_PROXY_V3_DEV",
            Self::V4 => "REAL_PROXY_V4_DEV",
            Self::V5 => "REAL_PROXY_V5_DEV",
            Self::Validation => "VALIDATION",
        }
    }

    fn digest(self) -> Option<&'static str> {
        match self {
            Self::V1 => Some(V1_SHA256),
            Self::V2 => Some(V2_SHA256),
            Self::V3 => Some(V3_SHA256),
            Self::V4 => Some(V4_SHA256),
            Self::V5 => Some(V5_SHA256),
            Self::Validation => None,
        }
    }
}

#[derive(Clone)]
struct FeatureRow {
    population: Population,
    ordinal: usize,
    expected_greeting: bool,
    selected_matches: bool,
    winner_present: bool,
    vetoes_pass: bool,
    decision_score: f64,
    candidate_quality: f64,
    candidate_count: usize,
    winner_margin: f64,
    margin_signal: f64,
    role_llr: f64,
    role_signal: f64,
    reliability: f64,
    alphabetic_length: usize,
    native: bool,
    segmentation_mechanism: Option<&'static str>,
    hard_organization_marker: bool,
    generic_organization_marker: bool,
    ampersand: bool,
    candidate_too_short: bool,
    country_hint_present: bool,
    locale_hint_present: bool,
    c31_emits: bool,
    c4_emits: bool,
    c4_source: C4EmissionSource,
    c5_emits: bool,
    unhinted: Option<UnhintedFeatures>,
}

impl FeatureRow {
    fn eligible(&self) -> bool {
        self.winner_present && self.vetoes_pass
    }

    fn logistic_features(&self) -> [f64; FEATURE_COUNT] {
        [
            self.decision_score,
            self.candidate_quality,
            self.winner_margin,
            self.role_signal,
            self.reliability,
            f64::from(self.candidate_count == 1),
            f64::from(self.native),
        ]
    }
}

#[derive(Clone)]
struct UnhintedFeatures {
    candidate_quality: f64,
    decision_score: f64,
    logistic_features: [f64; FEATURE_COUNT],
    candidate_count: usize,
    vetoes_pass: bool,
    c31_emits: bool,
    c4_emits: bool,
    c5_emits: bool,
    winner_changed: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct EmissionMetrics {
    rows: usize,
    expected_greetings: usize,
    expected_nulls: usize,
    emitted: usize,
    correct: usize,
    wrong: usize,
    null_false_emissions: usize,
    false_abstentions: usize,
    winner_correct_but_abstained: usize,
    expected_null_correct_abstentions: usize,
}

impl EmissionMetrics {
    fn observe(&mut self, row: &FeatureRow, emit: bool) {
        self.rows += 1;
        if row.expected_greeting {
            self.expected_greetings += 1;
            if emit {
                if row.selected_matches {
                    self.correct += 1;
                } else {
                    self.wrong += 1;
                }
            } else {
                self.false_abstentions += 1;
                if row.selected_matches && row.vetoes_pass {
                    self.winner_correct_but_abstained += 1;
                }
            }
        } else {
            self.expected_nulls += 1;
            if emit {
                self.wrong += 1;
                self.null_false_emissions += 1;
            } else {
                self.expected_null_correct_abstentions += 1;
            }
        }
        if emit {
            self.emitted += 1;
        }
    }

    fn add(&mut self, other: Self) {
        self.rows += other.rows;
        self.expected_greetings += other.expected_greetings;
        self.expected_nulls += other.expected_nulls;
        self.emitted += other.emitted;
        self.correct += other.correct;
        self.wrong += other.wrong;
        self.null_false_emissions += other.null_false_emissions;
        self.false_abstentions += other.false_abstentions;
        self.winner_correct_but_abstained += other.winner_correct_but_abstained;
        self.expected_null_correct_abstentions += other.expected_null_correct_abstentions;
    }

    fn precision(self) -> Option<f64> {
        ratio(self.correct, self.emitted)
    }

    fn recall(self) -> Option<f64> {
        ratio(self.correct, self.expected_greetings)
    }

    fn abstention_rate(self) -> Option<f64> {
        ratio(self.rows - self.emitted, self.rows)
    }
}

#[derive(Clone, Copy, Debug)]
struct WilsonInterval {
    lower: f64,
    upper: f64,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Family {
    ScoreOnly,
    ControlledC4,
    Logistic,
    C4PlusLogistic,
}

impl Family {
    const ALL: [Self; 4] = [
        Self::ScoreOnly,
        Self::ControlledC4,
        Self::Logistic,
        Self::C4PlusLogistic,
    ];

    fn as_str(self) -> &'static str {
        match self {
            Self::ScoreOnly => "score_only",
            Self::ControlledC4 => "controlled_c4",
            Self::Logistic => "logistic",
            Self::C4PlusLogistic => "c4_plus_logistic",
        }
    }
}

#[derive(Clone, Debug)]
enum Policy {
    C31,
    C4,
    Score {
        threshold: f64,
    },
    Controlled {
        quality: f64,
        reliability: f64,
        role: f64,
        margin: f64,
    },
    Logistic {
        model: LogisticModel,
        threshold: f64,
        additive_c4: bool,
    },
}

impl Policy {
    fn emits(&self, row: &FeatureRow) -> bool {
        match self {
            Self::C31 => row.c31_emits,
            Self::C4 => row.c4_emits,
            Self::Score { threshold } => row.eligible() && row.decision_score >= *threshold,
            Self::Controlled {
                quality,
                reliability,
                role,
                margin,
            } => {
                row.c4_emits
                    || (row.eligible()
                        && row.native
                        && row.candidate_quality >= *quality
                        && row.reliability >= *reliability
                        && row.role_signal >= *role
                        && (row.candidate_count == 1
                            || (row.candidate_count >= 2 && row.winner_margin >= *margin)))
            }
            Self::Logistic {
                model,
                threshold,
                additive_c4,
            } => {
                *additive_c4 && row.c4_emits
                    || row.eligible() && model.score(row.logistic_features()) >= *threshold
            }
        }
    }

    fn family(&self) -> Option<Family> {
        match self {
            Self::C31 | Self::C4 => None,
            Self::Score { .. } => Some(Family::ScoreOnly),
            Self::Controlled { .. } => Some(Family::ControlledC4),
            Self::Logistic {
                additive_c4: false, ..
            } => Some(Family::Logistic),
            Self::Logistic {
                additive_c4: true, ..
            } => Some(Family::C4PlusLogistic),
        }
    }

    fn complexity(&self) -> usize {
        match self {
            Self::C31 | Self::C4 | Self::Score { .. } => 1,
            Self::Controlled { .. } => 4,
            Self::Logistic { model, .. } => {
                1 + model
                    .coefficients
                    .iter()
                    .filter(|coefficient| **coefficient > 0.0)
                    .count()
            }
        }
    }

    fn threshold(&self) -> f64 {
        match self {
            Self::C31 => ALGORITHM_C2.threshold,
            Self::C4 => ALGORITHM_C2.threshold,
            Self::Score { threshold } | Self::Logistic { threshold, .. } => *threshold,
            Self::Controlled { .. } => 0.0,
        }
    }

    fn parameters(&self) -> String {
        match self {
            Self::C31 => format!("threshold={:.17}", ALGORITHM_C2.threshold),
            Self::C4 => "frozen_c4".to_string(),
            Self::Score { threshold } => format!("threshold={threshold:.17}"),
            Self::Controlled {
                quality,
                reliability,
                role,
                margin,
            } => format!(
                "quality={quality:.2};reliability={reliability:.2};role={role:.2};margin={margin:.2}"
            ),
            Self::Logistic {
                model,
                threshold,
                additive_c4,
            } => format!(
                "additive_c4={additive_c4};threshold={threshold:.17};intercept={:.17};coefficients={}",
                model.intercept,
                model
                    .coefficients
                    .iter()
                    .map(|value| format!("{value:.17}"))
                    .collect::<Vec<_>>()
                    .join(":")
            ),
        }
    }
}

#[derive(Clone)]
struct OperatingPoint {
    policy: Policy,
    metrics: EmissionMetrics,
}

#[derive(Clone, Debug)]
struct LogisticModel {
    intercept: f64,
    coefficients: [f64; FEATURE_COUNT],
    iterations: usize,
}

impl LogisticModel {
    fn linear_score(&self, features: [f64; FEATURE_COUNT]) -> f64 {
        self.coefficients
            .iter()
            .zip(features)
            .fold(self.intercept, |score, (coefficient, feature)| {
                score + coefficient * feature
            })
    }

    fn score(&self, features: [f64; FEATURE_COUNT]) -> f64 {
        sigmoid(self.linear_score(features))
    }
}

#[derive(Clone)]
struct FoldResult {
    held_out: Population,
    family: Family,
    target: f64,
    policy: Policy,
    training_metrics: EmissionMetrics,
    held_out_metrics: EmissionMetrics,
}

#[derive(Clone)]
struct RecommendedPoint {
    label: &'static str,
    target: f64,
    family: Family,
    cross_validated: EmissionMetrics,
    full_development: OperatingPoint,
}

pub fn run_calibration_frontier(
    output: &Path,
    corpus: &impl EvidenceSource,
    holdouts: Vec<FrozenHoldout>,
    fixtures: &Path,
) -> Result<String> {
    let holdouts = validate_and_order_holdouts(holdouts)?;
    let proxy_rows = build_proxy_rows(corpus, &holdouts);
    let validation_rows = build_validation_rows(corpus, fixtures)?;
    assert_dataset_counts(&proxy_rows)?;
    assert_historical_checkpoints(&proxy_rows)?;

    let score_frontier = score_frontier(&proxy_rows);
    let controlled_frontier = controlled_frontier(&proxy_rows);
    let full_model = fit_logistic(&proxy_rows)?;
    let logistic_points = logistic_frontier(&proxy_rows, &full_model, false);
    let additive_points = logistic_frontier(&proxy_rows, &full_model, true);
    let folds = logo_frontier(&proxy_rows)?;
    let best_by_target = best_cross_validated_families(&folds);
    let recommendations = recommendations(
        &best_by_target,
        &score_frontier,
        &controlled_frontier,
        &logistic_points,
        &additive_points,
    );

    let outputs = build_outputs(
        &holdouts,
        &proxy_rows,
        &validation_rows,
        &score_frontier,
        &controlled_frontier,
        &full_model,
        &logistic_points,
        &additive_points,
        &folds,
        &best_by_target,
        &recommendations,
        corpus,
    )?;
    let repeated = build_outputs(
        &holdouts,
        &proxy_rows,
        &validation_rows,
        &score_frontier,
        &controlled_frontier,
        &full_model,
        &logistic_points,
        &additive_points,
        &folds,
        &best_by_target,
        &recommendations,
        corpus,
    )?;
    if outputs != repeated {
        return Err("calibration-frontier serialization is not deterministic".into());
    }
    for (name, bytes) in &outputs {
        fs::write(output.join(name), bytes)?;
    }
    let report = outputs
        .get("report.md")
        .ok_or("calibration report missing")?;
    Ok(String::from_utf8(report.clone())?)
}

fn validate_and_order_holdouts(holdouts: Vec<FrozenHoldout>) -> Result<Vec<FrozenHoldout>> {
    let actual = holdouts
        .iter()
        .map(|holdout| holdout.manifest.holdout_sha256.as_str())
        .collect::<BTreeSet<_>>();
    let expected = Population::PROXIES
        .into_iter()
        .map(|population| population.digest().expect("proxy digest"))
        .collect::<BTreeSet<_>>();
    if holdouts.len() != Population::PROXIES.len() || actual != expected {
        return Err(format!(
            "C5 calibration requires exactly frozen V1/V2/V3/V4/V5; received {actual:?}"
        )
        .into());
    }
    let mut by_population = holdouts
        .into_iter()
        .map(|holdout| {
            let population = Population::from_digest(&holdout.manifest.holdout_sha256)
                .expect("validated proxy digest");
            (population, holdout)
        })
        .collect::<BTreeMap<_, _>>();
    Ok(Population::PROXIES
        .into_iter()
        .map(|population| by_population.remove(&population).expect("proxy population"))
        .collect())
}

fn build_proxy_rows(corpus: &impl EvidenceSource, holdouts: &[FrozenHoldout]) -> Vec<FeatureRow> {
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
            rows.push(feature_row(
                corpus,
                population,
                ordinal,
                &case.display_name,
                case.expected_greeting(),
                nonempty(&case.country_hint),
                nonempty(&case.locale_hint),
            ));
        }
    }
    rows
}

fn build_validation_rows(corpus: &impl EvidenceSource, fixtures: &Path) -> Result<Vec<FeatureRow>> {
    Ok(generate_cases(fixtures, false)?
        .into_iter()
        .filter(|case| case.split == Split::Validation)
        .enumerate()
        .map(|(ordinal, case)| feature_row_from_case(corpus, ordinal, &case))
        .collect())
}

fn feature_row_from_case(corpus: &impl EvidenceSource, ordinal: usize, case: &Case) -> FeatureRow {
    feature_row(
        corpus,
        Population::Validation,
        ordinal,
        &case.input,
        case.expected_greeting.as_deref(),
        case.country_hint.as_deref(),
        case.locale_hint.as_deref(),
    )
}

#[allow(clippy::too_many_arguments)]
fn feature_row(
    corpus: &impl EvidenceSource,
    population: Population,
    ordinal: usize,
    display_name: &str,
    expected_greeting: Option<&str>,
    country_hint: Option<&str>,
    locale_hint: Option<&str>,
) -> FeatureRow {
    let diagnostic = diagnose_role_inference(
        corpus,
        ALGORITHM_C3,
        display_name,
        country_hint,
        locale_hint,
    );
    let decision = c4_decision_breakdown(&diagnostic, ALGORITHM_C2, ALGORITHM_C31, ALGORITHM_C4);
    let unhinted = (country_hint.is_some() || locale_hint.is_some())
        .then(|| unhinted_features(corpus, display_name, &decision, &diagnostic.candidates));
    feature_row_from_decision(
        population,
        ordinal,
        expected_greeting,
        country_hint.is_some(),
        locale_hint.is_some(),
        &decision,
        unhinted,
    )
}

fn feature_row_from_decision(
    population: Population,
    ordinal: usize,
    expected_greeting: Option<&str>,
    country_hint_present: bool,
    locale_hint_present: bool,
    decision: &C4DecisionBreakdown,
    unhinted: Option<UnhintedFeatures>,
) -> FeatureRow {
    let breakdown = &decision.c31;
    let c5 = c5_decision_from_c4(decision.clone(), ALGORITHM_C5);
    let winner = breakdown.winner.as_ref();
    let selected = winner.map(|winner| winner.greeting_candidate.as_str());
    let expected_greeting_present = expected_greeting.is_some();
    let selected_matches = greeting_matches(expected_greeting, selected);
    let vetoes_pass = vetoes_pass(breakdown);
    FeatureRow {
        population,
        ordinal,
        expected_greeting: expected_greeting_present,
        selected_matches,
        winner_present: winner.is_some(),
        vetoes_pass,
        decision_score: breakdown.final_score,
        candidate_quality: winner.map_or(0.0, |winner| winner.winner_score),
        candidate_count: winner.map_or(0, |winner| winner.candidate_count),
        winner_margin: winner.map_or(0.0, |winner| winner.winner_margin),
        margin_signal: breakdown.margin_signal.unwrap_or(0.0),
        role_llr: winner.map_or(0.0, |winner| winner.role_llr),
        role_signal: winner.map_or(0.0, |winner| winner.role_signal),
        reliability: winner.map_or(0.0, |winner| winner.reliability),
        alphabetic_length: winner.map_or(0, |winner| winner.alphabetic_length),
        native: winner.is_some_and(|winner| winner.candidate_origin != "handle_segment"),
        segmentation_mechanism: winner.and_then(|winner| winner.segmentation_mechanism),
        hard_organization_marker: breakdown.hard_organization_marker,
        generic_organization_marker: breakdown.generic_organization_marker,
        ampersand: breakdown.ampersand,
        candidate_too_short: breakdown.candidate_too_short,
        country_hint_present,
        locale_hint_present,
        c31_emits: winner.is_some() && breakdown.final_score >= ALGORITHM_C2.threshold,
        c4_emits: c4_emitted_candidate(decision).is_some(),
        c4_source: decision.emission_source,
        c5_emits: c5_emitted_candidate(&c5).is_some(),
        unhinted,
    }
}

fn unhinted_features(
    corpus: &impl EvidenceSource,
    display_name: &str,
    hinted: &C4DecisionBreakdown,
    hinted_candidates: &[crate::classifier::CandidateDiagnostic],
) -> UnhintedFeatures {
    let diagnostic = diagnose_role_inference(corpus, ALGORITHM_C3, display_name, None, None);
    let decision = c4_decision_breakdown(&diagnostic, ALGORITHM_C2, ALGORITHM_C31, ALGORITHM_C4);
    let winner_changed = match (hinted_candidates.first(), diagnostic.candidates.first()) {
        (Some(left), Some(right)) => {
            left.start != right.start
                || left.length != right.length
                || left.display != right.display
                || left.origin != right.origin
        }
        (None, None) => false,
        _ => true,
    };
    let row = feature_row_from_decision(
        Population::Validation,
        0,
        None,
        false,
        false,
        &decision,
        None,
    );
    UnhintedFeatures {
        candidate_quality: row.candidate_quality,
        decision_score: row.decision_score,
        logistic_features: row.logistic_features(),
        candidate_count: row.candidate_count,
        vetoes_pass: row.vetoes_pass,
        c31_emits: row.c31_emits,
        c4_emits: row.c4_emits,
        c5_emits: row.c5_emits,
        winner_changed: winner_changed
            || hinted.c31.winner.is_none() != decision.c31.winner.is_none(),
    }
}

fn vetoes_pass(breakdown: &C31DecisionBreakdown) -> bool {
    !breakdown.hard_organization_marker
        && !breakdown.generic_organization_marker
        && !breakdown.ampersand
        && !breakdown.candidate_too_short
}

fn nonempty(value: &str) -> Option<&str> {
    (!value.is_empty()).then_some(value)
}

fn assert_dataset_counts(rows: &[FeatureRow]) -> Result<()> {
    let expected = [
        (Population::V1, 1_957, 1_616, 341),
        (Population::V2, 1_496, 1_217, 279),
        (Population::V3, 1_474, 1_232, 242),
        (Population::V4, 1_441, 1_220, 221),
        (Population::V5, 1_440, 1_193, 247),
    ];
    for (population, total, greetings, nulls) in expected {
        let selected = rows
            .iter()
            .filter(|row| row.population == population)
            .collect::<Vec<_>>();
        let actual = (
            selected.len(),
            selected.iter().filter(|row| row.expected_greeting).count(),
            selected.iter().filter(|row| !row.expected_greeting).count(),
        );
        if actual != (total, greetings, nulls) {
            return Err(format!(
                "{} counts changed: expected ({total}, {greetings}, {nulls}), got {actual:?}",
                population.as_str()
            )
            .into());
        }
    }
    let total = rows.len();
    let greetings = rows.iter().filter(|row| row.expected_greeting).count();
    let nulls = rows.iter().filter(|row| !row.expected_greeting).count();
    if (total, greetings, nulls) != (7_808, 6_478, 1_330) {
        return Err(format!(
            "combined proxy counts changed: expected (7808, 6478, 1330), got ({total}, {greetings}, {nulls})"
        )
        .into());
    }
    Ok(())
}

fn assert_historical_checkpoints(rows: &[FeatureRow]) -> Result<()> {
    let expected = [
        (Population::V1, metrics_value(271, 271, 0, 0)),
        (Population::V3, metrics_value(265, 260, 5, 2)),
        (Population::V4, metrics_value(277, 274, 3, 0)),
        (Population::V5, metrics_value(235, 233, 2, 1)),
    ];
    for (population, expected) in expected {
        let actual = evaluate_policy(
            rows.iter().filter(|row| row.population == population),
            &Policy::C4,
        );
        let observed = (
            actual.emitted,
            actual.correct,
            actual.wrong,
            actual.null_false_emissions,
        );
        if observed != expected {
            return Err(format!(
                "frozen C4 {} checkpoint changed: expected {expected:?}, got {observed:?}",
                population.as_str()
            )
            .into());
        }
    }
    Ok(())
}

const fn metrics_value(
    emitted: usize,
    correct: usize,
    wrong: usize,
    null_false_emissions: usize,
) -> (usize, usize, usize, usize) {
    (emitted, correct, wrong, null_false_emissions)
}

fn evaluate_policy<'a>(
    rows: impl Iterator<Item = &'a FeatureRow>,
    policy: &Policy,
) -> EmissionMetrics {
    let mut metrics = EmissionMetrics::default();
    for row in rows {
        metrics.observe(row, policy.emits(row));
    }
    metrics
}

fn score_frontier(rows: &[FeatureRow]) -> Vec<OperatingPoint> {
    let thresholds = distinct_thresholds(
        rows.iter()
            .filter(|row| row.eligible())
            .map(|row| row.decision_score),
    );
    let mut points = thresholds
        .into_iter()
        .map(|threshold| {
            let policy = Policy::Score { threshold };
            let metrics = evaluate_policy(rows.iter(), &policy);
            OperatingPoint { policy, metrics }
        })
        .collect::<Vec<_>>();
    deduplicate_points(&mut points, rows);
    points
}

fn controlled_frontier(rows: &[FeatureRow]) -> Vec<OperatingPoint> {
    let evidence_thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0];
    let margin_thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let mut points = Vec::with_capacity(evidence_thresholds.len().pow(3) * margin_thresholds.len());
    for quality in evidence_thresholds {
        for reliability in evidence_thresholds {
            for role in evidence_thresholds {
                for margin in margin_thresholds {
                    let policy = Policy::Controlled {
                        quality,
                        reliability,
                        role,
                        margin,
                    };
                    let metrics = evaluate_policy(rows.iter(), &policy);
                    points.push(OperatingPoint { policy, metrics });
                }
            }
        }
    }
    deduplicate_points(&mut points, rows);
    points
}

fn logistic_frontier(
    rows: &[FeatureRow],
    model: &LogisticModel,
    additive_c4: bool,
) -> Vec<OperatingPoint> {
    let thresholds = distinct_thresholds(
        rows.iter()
            .filter(|row| row.eligible())
            .map(|row| model.score(row.logistic_features())),
    );
    let mut points = thresholds
        .into_iter()
        .map(|threshold| {
            let policy = Policy::Logistic {
                model: model.clone(),
                threshold,
                additive_c4,
            };
            let metrics = evaluate_policy(rows.iter(), &policy);
            OperatingPoint { policy, metrics }
        })
        .collect::<Vec<_>>();
    deduplicate_points(&mut points, rows);
    points
}

fn distinct_thresholds(values: impl Iterator<Item = f64>) -> Vec<f64> {
    let mut values = values.collect::<Vec<_>>();
    values.sort_by(|left, right| right.total_cmp(left));
    values.dedup_by(|left, right| left.to_bits() == right.to_bits());
    values
}

fn deduplicate_points(points: &mut Vec<OperatingPoint>, rows: &[FeatureRow]) {
    let mut unique = BTreeMap::<Vec<u64>, OperatingPoint>::new();
    for point in points.drain(..) {
        let signature = emission_signature(rows, &point.policy);
        match unique.get(&signature) {
            Some(existing)
                if existing.policy.complexity() < point.policy.complexity()
                    || (existing.policy.complexity() == point.policy.complexity()
                        && existing.policy.parameters() <= point.policy.parameters()) => {}
            _ => {
                unique.insert(signature, point);
            }
        }
    }
    points.extend(unique.into_values());
    points.sort_by(compare_operating_points_for_serialization);
}

fn emission_signature(rows: &[FeatureRow], policy: &Policy) -> Vec<u64> {
    let mut signature = vec![0_u64; rows.len().div_ceil(64)];
    for (index, row) in rows.iter().enumerate() {
        if policy.emits(row) {
            signature[index / 64] |= 1_u64 << (index % 64);
        }
    }
    signature
}

fn compare_operating_points_for_serialization(
    left: &OperatingPoint,
    right: &OperatingPoint,
) -> Ordering {
    left.metrics
        .emitted
        .cmp(&right.metrics.emitted)
        .then(left.metrics.correct.cmp(&right.metrics.correct))
        .then(left.metrics.wrong.cmp(&right.metrics.wrong))
        .then_with(|| left.policy.parameters().cmp(&right.policy.parameters()))
}

fn fit_logistic(rows: &[FeatureRow]) -> Result<LogisticModel> {
    let training = logistic_training_rows(rows)?;
    let mut intercept = 0.0;
    let mut coefficients = [0.0; FEATURE_COUNT];
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
            return Err("logistic optimizer line search failed".into());
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
    Err(
        format!("logistic optimizer did not converge in {MAX_OPTIMIZER_ITERATIONS} iterations")
            .into(),
    )
}

#[derive(Clone, Copy)]
struct WeightedTrainingRow {
    features: [f64; FEATURE_COUNT],
    label: f64,
    weight: f64,
}

fn logistic_training_rows(rows: &[FeatureRow]) -> Result<Vec<WeightedTrainingRow>> {
    let populations = rows
        .iter()
        .filter(|row| row.population != Population::Validation)
        .map(|row| row.population)
        .collect::<BTreeSet<_>>();
    if populations.is_empty() {
        return Err("logistic training requires at least one proxy generation".into());
    }
    let counts = populations
        .iter()
        .map(|population| {
            let count = rows
                .iter()
                .filter(|row| row.population == *population && row.eligible())
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
        .filter(|row| row.population != Population::Validation && row.eligible())
        .map(|row| WeightedTrainingRow {
            features: row.logistic_features(),
            label: f64::from(row.selected_matches),
            weight: generation_weight / counts[&row.population] as f64,
        })
        .collect())
}

fn logistic_objective(
    rows: &[WeightedTrainingRow],
    intercept: f64,
    coefficients: [f64; FEATURE_COUNT],
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

fn logistic_gradient(
    rows: &[WeightedTrainingRow],
    intercept: f64,
    coefficients: [f64; FEATURE_COUNT],
) -> (f64, [f64; FEATURE_COUNT]) {
    let mut intercept_gradient = 0.0;
    let mut gradients = [0.0; FEATURE_COUNT];
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

fn logo_frontier(rows: &[FeatureRow]) -> Result<Vec<FoldResult>> {
    let mut results = Vec::new();
    for held_out in Population::PROXIES {
        let training = rows
            .iter()
            .filter(|row| row.population != held_out)
            .cloned()
            .collect::<Vec<_>>();
        let held_out_rows = rows
            .iter()
            .filter(|row| row.population == held_out)
            .cloned()
            .collect::<Vec<_>>();
        if training.iter().any(|row| row.population == held_out) {
            return Err(format!("LOGO leakage detected for {}", held_out.as_str()).into());
        }
        let score = score_frontier(&training);
        let controlled = controlled_frontier(&training);
        let model = fit_logistic(&training)?;
        let logistic = logistic_frontier(&training, &model, false);
        let additive = logistic_frontier(&training, &model, true);
        for (family, points) in [
            (Family::ScoreOnly, &score),
            (Family::ControlledC4, &controlled),
            (Family::Logistic, &logistic),
            (Family::C4PlusLogistic, &additive),
        ] {
            for target in TARGETS {
                let Some(selected) = select_point(points, target) else {
                    continue;
                };
                if selected.policy.family() != Some(family) {
                    return Err("selected policy family mismatch".into());
                }
                let held_out_metrics = evaluate_policy(held_out_rows.iter(), &selected.policy);
                results.push(FoldResult {
                    held_out,
                    family,
                    target,
                    policy: selected.policy.clone(),
                    training_metrics: selected.metrics,
                    held_out_metrics,
                });
            }
        }
    }
    Ok(results)
}

#[derive(Clone)]
struct CrossValidatedPoint {
    family: Family,
    target: f64,
    metrics: EmissionMetrics,
}

fn best_cross_validated_families(folds: &[FoldResult]) -> Vec<CrossValidatedPoint> {
    let mut points = Vec::new();
    for family in Family::ALL {
        for target in TARGETS {
            let mut metrics = EmissionMetrics::default();
            let matching = folds
                .iter()
                .filter(|fold| fold.family == family && fold.target == target)
                .collect::<Vec<_>>();
            if matching.len() != Population::PROXIES.len() {
                continue;
            }
            for fold in matching {
                metrics.add(fold.held_out_metrics);
            }
            points.push(CrossValidatedPoint {
                family,
                target,
                metrics,
            });
        }
    }

    let mut best = Vec::new();
    for target in TARGETS {
        let selected = points
            .iter()
            .filter(|point| point.target == target)
            .max_by(|left, right| compare_cross_validated(left, right));
        if let Some(selected) = selected {
            best.push(selected.clone());
        }
    }
    best
}

fn compare_cross_validated(left: &CrossValidatedPoint, right: &CrossValidatedPoint) -> Ordering {
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
        .then_with(|| right.family.cmp(&left.family))
}

fn recommendations(
    best: &[CrossValidatedPoint],
    score: &[OperatingPoint],
    controlled: &[OperatingPoint],
    logistic: &[OperatingPoint],
    additive: &[OperatingPoint],
) -> Vec<RecommendedPoint> {
    [
        ("very_conservative", 0.995),
        ("balanced", 0.99),
        ("aggressive", 0.95),
    ]
    .into_iter()
    .filter_map(|(label, target)| {
        let cross_validated = best.iter().find(|point| point.target == target)?;
        if !cross_validated
            .metrics
            .precision()
            .is_some_and(|precision| precision >= target)
        {
            return None;
        }
        let points = match cross_validated.family {
            Family::ScoreOnly => score,
            Family::ControlledC4 => controlled,
            Family::Logistic => logistic,
            Family::C4PlusLogistic => additive,
        };
        let full_development = select_point(points, target)?.clone();
        Some(RecommendedPoint {
            label,
            target,
            family: cross_validated.family,
            cross_validated: cross_validated.metrics,
            full_development,
        })
    })
    .collect()
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
        .max_by(|left, right| compare_selected_points(left, right))
}

fn compare_selected_points(left: &OperatingPoint, right: &OperatingPoint) -> Ordering {
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
        .then_with(|| right.policy.complexity().cmp(&left.policy.complexity()))
        .then_with(|| left.policy.threshold().total_cmp(&right.policy.threshold()))
        .then_with(|| right.policy.parameters().cmp(&left.policy.parameters()))
}

fn wilson_interval(successes: usize, trials: usize) -> Option<WilsonInterval> {
    if trials == 0 {
        return None;
    }
    let trials = trials as f64;
    let estimate = successes as f64 / trials;
    let squared = Z_95 * Z_95;
    let denominator = 1.0 + squared / trials;
    let center = (estimate + squared / (2.0 * trials)) / denominator;
    let spread = Z_95
        * ((estimate * (1.0 - estimate) / trials + squared / (4.0 * trials * trials)).sqrt())
        / denominator;
    Some(WilsonInterval {
        lower: (center - spread).max(0.0),
        upper: (center + spread).min(1.0),
    })
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator != 0).then(|| numerator as f64 / denominator as f64)
}

#[allow(clippy::too_many_arguments)]
fn build_outputs(
    holdouts: &[FrozenHoldout],
    proxy_rows: &[FeatureRow],
    validation_rows: &[FeatureRow],
    score_frontier: &[OperatingPoint],
    controlled_frontier: &[OperatingPoint],
    full_model: &LogisticModel,
    logistic_frontier: &[OperatingPoint],
    additive_frontier: &[OperatingPoint],
    folds: &[FoldResult],
    best_by_target: &[CrossValidatedPoint],
    recommendations: &[RecommendedPoint],
    corpus: &impl EvidenceSource,
) -> Result<BTreeMap<String, Vec<u8>>> {
    let mut outputs = BTreeMap::new();
    outputs.insert(
        "dataset_provenance.csv".to_string(),
        dataset_provenance_csv(holdouts)?,
    );
    outputs.insert(
        "calibration_features.csv".to_string(),
        calibration_features_csv(proxy_rows, validation_rows)?,
    );
    outputs.insert(
        "frozen_baselines.csv".to_string(),
        frozen_baselines_csv(proxy_rows)?,
    );
    outputs.insert(
        "c4_discarded_ranking_signal.csv".to_string(),
        c4_discarded_ranking_signal_csv(proxy_rows)?,
    );
    outputs.insert(
        "score_frontier.csv".to_string(),
        frontier_csv(score_frontier, "score_only")?,
    );
    outputs.insert(
        "controlled_c4_frontier.csv".to_string(),
        frontier_csv(controlled_frontier, "controlled_c4")?,
    );
    outputs.insert(
        "cross_validated_frontier.csv".to_string(),
        cross_validated_csv(folds)?,
    );
    outputs.insert(
        "per_generation_operating_points.csv".to_string(),
        logo_csv(folds)?,
    );
    outputs.insert("logo_results.csv".to_string(), logo_csv(folds)?);
    outputs.insert(
        "model_parameters.csv".to_string(),
        model_parameters_csv(full_model, folds)?,
    );
    outputs.insert(
        "wilson_intervals.csv".to_string(),
        wilson_csv(folds, best_by_target)?,
    );
    outputs.insert(
        "cost_sensitive_frontier.csv".to_string(),
        cost_sensitive_csv(&[
            score_frontier,
            controlled_frontier,
            logistic_frontier,
            additive_frontier,
        ])?,
    );
    outputs.insert(
        "synthetic_validation.csv".to_string(),
        synthetic_validation_csv(validation_rows, recommendations)?,
    );
    outputs.insert(
        "country_evidence_audit.csv".to_string(),
        country_evidence_csv(proxy_rows, validation_rows, recommendations)?,
    );
    outputs.insert(
        "recommended_operating_points.csv".to_string(),
        recommendations_csv(recommendations)?,
    );
    outputs.insert(
        "qualitative_examples.csv".to_string(),
        qualitative_csv(corpus, recommendations)?,
    );
    let report = build_report(
        holdouts,
        proxy_rows,
        validation_rows,
        score_frontier,
        controlled_frontier,
        logistic_frontier,
        additive_frontier,
        folds,
        best_by_target,
        recommendations,
        corpus,
    )?;
    outputs.insert("report.md".to_string(), report.into_bytes());
    Ok(outputs)
}

fn dataset_provenance_csv(holdouts: &[FrozenHoldout]) -> Result<Vec<u8>> {
    csv_bytes(|writer| {
        writer.write_record([
            "population",
            "holdout_sha256",
            "total_cases",
            "evaluable_cases",
            "skipped_cases",
            "expected_greetings",
            "expected_abstentions",
            "provenance",
        ])?;
        for holdout in holdouts {
            let population =
                Population::from_digest(&holdout.manifest.holdout_sha256).expect("validated proxy");
            writer.write_record([
                population.as_str().to_string(),
                holdout.manifest.holdout_sha256.clone(),
                holdout.manifest.total_cases.to_string(),
                holdout.manifest.evaluable_cases.to_string(),
                holdout.manifest.skipped_cases.to_string(),
                holdout.manifest.expected_greetings.to_string(),
                holdout.manifest.expected_abstentions.to_string(),
                holdout.manifest.provenance.clone(),
            ])?;
        }
        Ok(())
    })
}

fn calibration_features_csv(
    proxy_rows: &[FeatureRow],
    validation_rows: &[FeatureRow],
) -> Result<Vec<u8>> {
    csv_bytes(|writer| {
        writer.write_record([
            "population",
            "ordinal",
            "expected_outcome",
            "selected_matches_expected",
            "winner_present",
            "decision_score",
            "candidate_quality",
            "candidate_count",
            "winner_margin",
            "margin_signal",
            "role_llr",
            "role_signal",
            "reliability",
            "alphabetic_length",
            "provenance",
            "segmentation_mechanism",
            "hard_organization_marker",
            "generic_organization_marker",
            "ampersand",
            "candidate_too_short",
            "vetoes_pass",
            "country_hint_present",
            "locale_hint_present",
            "c31_emits",
            "c4_emits",
            "c4_emission_source",
        ])?;
        for row in proxy_rows.iter().chain(validation_rows) {
            writer.write_record([
                row.population.as_str().to_string(),
                row.ordinal.to_string(),
                if row.expected_greeting {
                    "greeting"
                } else {
                    "null"
                }
                .to_string(),
                row.selected_matches.to_string(),
                row.winner_present.to_string(),
                float(row.decision_score),
                float(row.candidate_quality),
                row.candidate_count.to_string(),
                float(row.winner_margin),
                float(row.margin_signal),
                float(row.role_llr),
                float(row.role_signal),
                float(row.reliability),
                row.alphabetic_length.to_string(),
                if row.native { "native" } else { "segmented" }.to_string(),
                row.segmentation_mechanism.unwrap_or("").to_string(),
                row.hard_organization_marker.to_string(),
                row.generic_organization_marker.to_string(),
                row.ampersand.to_string(),
                row.candidate_too_short.to_string(),
                row.vetoes_pass.to_string(),
                row.country_hint_present.to_string(),
                row.locale_hint_present.to_string(),
                row.c31_emits.to_string(),
                row.c4_emits.to_string(),
                row.c4_source.as_str().to_string(),
            ])?;
        }
        Ok(())
    })
}

fn frozen_baselines_csv(rows: &[FeatureRow]) -> Result<Vec<u8>> {
    csv_bytes(|writer| {
        write_metrics_header(writer, &["population", "algorithm"])?;
        for population in Population::PROXIES {
            for (name, policy) in [("C3.1", Policy::C31), ("C4", Policy::C4)] {
                let metrics = evaluate_policy(
                    rows.iter().filter(|row| row.population == population),
                    &policy,
                );
                write_metrics_record(writer, &[population.as_str(), name], metrics)?;
            }
        }
        for (name, policy) in [("C3.1", Policy::C31), ("C4", Policy::C4)] {
            write_metrics_record(
                writer,
                &["COMBINED_SPENT", name],
                evaluate_policy(rows.iter(), &policy),
            )?;
        }
        Ok(())
    })
}

fn c4_discarded_ranking_signal_csv(rows: &[FeatureRow]) -> Result<Vec<u8>> {
    csv_bytes(|writer| {
        writer.write_record([
            "population",
            "dimension",
            "bucket",
            "count",
            "all_expected_greetings",
            "discarded_correct_winners",
            "share_of_all_expected_greetings",
            "share_of_discarded_correct_winners",
        ])?;
        for population in Population::PROXIES
            .into_iter()
            .map(Some)
            .chain(std::iter::once(None))
        {
            let selected = rows
                .iter()
                .filter(|row| population.is_none_or(|population| row.population == population))
                .collect::<Vec<_>>();
            let all_expected = selected.iter().filter(|row| row.expected_greeting).count();
            let discarded = selected
                .iter()
                .copied()
                .filter(|row| discarded_correct_winner(row))
                .collect::<Vec<_>>();
            let population_name = population.map_or("COMBINED_SPENT", Population::as_str);
            write_discarded_bucket(
                writer,
                population_name,
                "all",
                "all",
                discarded.len(),
                all_expected,
                discarded.len(),
            )?;
            for (dimension, buckets) in [
                (
                    "winner_margin",
                    vec![
                        "0.00-0.10",
                        "0.10-0.20",
                        "0.20-0.30",
                        "0.30-0.50",
                        "0.50-1.00",
                    ],
                ),
                (
                    "candidate_quality",
                    vec![
                        "0.00-0.20",
                        "0.20-0.40",
                        "0.40-0.60",
                        "0.60-0.80",
                        "0.80-1.00",
                    ],
                ),
                (
                    "role_signal",
                    vec![
                        "0.00-0.20",
                        "0.20-0.40",
                        "0.40-0.60",
                        "0.60-0.80",
                        "0.80-1.00",
                    ],
                ),
                (
                    "reliability",
                    vec![
                        "0.00-0.20",
                        "0.20-0.40",
                        "0.40-0.60",
                        "0.60-0.80",
                        "0.80-1.00",
                    ],
                ),
                ("candidate_count", vec!["1", "2", "3", "4+"]),
            ] {
                for bucket in buckets {
                    let count = discarded
                        .iter()
                        .filter(|row| discarded_bucket(row, dimension) == bucket)
                        .count();
                    write_discarded_bucket(
                        writer,
                        population_name,
                        dimension,
                        bucket,
                        count,
                        all_expected,
                        discarded.len(),
                    )?;
                }
            }
        }
        Ok(())
    })
}

fn discarded_correct_winner(row: &FeatureRow) -> bool {
    row.expected_greeting && row.selected_matches && row.vetoes_pass && !row.c4_emits
}

fn discarded_bucket(row: &FeatureRow, dimension: &str) -> &'static str {
    match dimension {
        "winner_margin" => margin_bucket(row.winner_margin),
        "candidate_quality" => five_bucket(row.candidate_quality, [0.2, 0.4, 0.6, 0.8]),
        "role_signal" => five_bucket(row.role_signal, [0.2, 0.4, 0.6, 0.8]),
        "reliability" => five_bucket(row.reliability, [0.2, 0.4, 0.6, 0.8]),
        "candidate_count" => match row.candidate_count {
            1 => "1",
            2 => "2",
            3 => "3",
            _ => "4+",
        },
        _ => unreachable!("fixed discarded-signal dimension"),
    }
}

fn margin_bucket(value: f64) -> &'static str {
    if value < 0.1 {
        "0.00-0.10"
    } else if value < 0.2 {
        "0.10-0.20"
    } else if value < 0.3 {
        "0.20-0.30"
    } else if value < 0.5 {
        "0.30-0.50"
    } else {
        "0.50-1.00"
    }
}

fn five_bucket(value: f64, boundaries: [f64; 4]) -> &'static str {
    if value < boundaries[0] {
        "0.00-0.20"
    } else if value < boundaries[1] {
        "0.20-0.40"
    } else if value < boundaries[2] {
        "0.40-0.60"
    } else if value < boundaries[3] {
        "0.60-0.80"
    } else {
        "0.80-1.00"
    }
}

fn write_discarded_bucket(
    writer: &mut csv::Writer<Vec<u8>>,
    population: &str,
    dimension: &str,
    bucket: &str,
    count: usize,
    all_expected: usize,
    discarded: usize,
) -> Result<()> {
    writer.write_record([
        population.to_string(),
        dimension.to_string(),
        bucket.to_string(),
        count.to_string(),
        all_expected.to_string(),
        discarded.to_string(),
        optional_float(ratio(count, all_expected)),
        optional_float(ratio(count, discarded)),
    ])?;
    Ok(())
}

fn frontier_csv(points: &[OperatingPoint], family: &str) -> Result<Vec<u8>> {
    csv_bytes(|writer| {
        write_metrics_header(writer, &["family", "parameters"])?;
        for point in points {
            write_metrics_record(writer, &[family, &point.policy.parameters()], point.metrics)?;
        }
        Ok(())
    })
}

fn cross_validated_csv(folds: &[FoldResult]) -> Result<Vec<u8>> {
    csv_bytes(|writer| {
        write_metrics_header(writer, &["family", "target", "target_met_oof"])?;
        for family in Family::ALL {
            for target in TARGETS {
                let matching = folds
                    .iter()
                    .filter(|fold| fold.family == family && fold.target == target)
                    .collect::<Vec<_>>();
                if matching.len() != Population::PROXIES.len() {
                    continue;
                }
                let mut metrics = EmissionMetrics::default();
                for fold in matching {
                    metrics.add(fold.held_out_metrics);
                }
                write_metrics_record(
                    writer,
                    &[
                        family.as_str(),
                        &format!("{target:.4}"),
                        &metrics
                            .precision()
                            .is_some_and(|precision| precision >= target)
                            .to_string(),
                    ],
                    metrics,
                )?;
            }
        }
        Ok(())
    })
}

fn logo_csv(folds: &[FoldResult]) -> Result<Vec<u8>> {
    csv_bytes(|writer| {
        writer.write_record([
            "family",
            "target",
            "training_generations",
            "held_out_generation",
            "parameters",
            "training_precision",
            "training_recall",
            "held_out_emitted",
            "held_out_correct",
            "held_out_wrong",
            "held_out_null_fp",
            "held_out_precision",
            "held_out_recall",
            "held_out_false_abstentions",
            "held_out_winner_correct_but_abstained",
        ])?;
        for fold in folds {
            let training = Population::PROXIES
                .into_iter()
                .filter(|population| *population != fold.held_out)
                .map(Population::as_str)
                .collect::<Vec<_>>()
                .join("+");
            writer.write_record([
                fold.family.as_str().to_string(),
                format!("{:.4}", fold.target),
                training,
                fold.held_out.as_str().to_string(),
                fold.policy.parameters(),
                optional_float(fold.training_metrics.precision()),
                optional_float(fold.training_metrics.recall()),
                fold.held_out_metrics.emitted.to_string(),
                fold.held_out_metrics.correct.to_string(),
                fold.held_out_metrics.wrong.to_string(),
                fold.held_out_metrics.null_false_emissions.to_string(),
                optional_float(fold.held_out_metrics.precision()),
                optional_float(fold.held_out_metrics.recall()),
                fold.held_out_metrics.false_abstentions.to_string(),
                fold.held_out_metrics
                    .winner_correct_but_abstained
                    .to_string(),
            ])?;
        }
        Ok(())
    })
}

fn model_parameters_csv(full: &LogisticModel, folds: &[FoldResult]) -> Result<Vec<u8>> {
    csv_bytes(|writer| {
        writer.write_record([
            "fit",
            "held_out_generation",
            "iterations",
            "intercept",
            "decision_score",
            "candidate_quality",
            "winner_margin",
            "role_signal",
            "reliability",
            "sole_candidate",
            "native_provenance",
        ])?;
        write_model_record(writer, "full_development", "", full)?;
        let mut seen = BTreeSet::new();
        for fold in folds.iter().filter(|fold| fold.family == Family::Logistic) {
            let Policy::Logistic { model, .. } = &fold.policy else {
                continue;
            };
            if seen.insert(fold.held_out) {
                write_model_record(writer, "logo", fold.held_out.as_str(), model)?;
            }
        }
        Ok(())
    })
}

fn write_model_record(
    writer: &mut csv::Writer<Vec<u8>>,
    fit: &str,
    held_out: &str,
    model: &LogisticModel,
) -> Result<()> {
    let mut record = vec![
        fit.to_string(),
        held_out.to_string(),
        model.iterations.to_string(),
        float(model.intercept),
    ];
    record.extend(model.coefficients.into_iter().map(float));
    writer.write_record(record)?;
    Ok(())
}

fn wilson_csv(folds: &[FoldResult], best_by_target: &[CrossValidatedPoint]) -> Result<Vec<u8>> {
    csv_bytes(|writer| {
        writer.write_record([
            "scope",
            "family",
            "target",
            "population",
            "correct",
            "emitted",
            "observed_precision",
            "wilson_95_lower",
            "wilson_95_upper",
            "lower_bound_supports_target",
        ])?;
        for point in best_by_target {
            write_wilson_record(
                writer,
                "cross_validated_combined",
                point.family,
                point.target,
                "COMBINED_OOF",
                point.metrics,
            )?;
            for population in Population::PROXIES {
                if let Some(fold) = folds.iter().find(|fold| {
                    fold.family == point.family
                        && fold.target == point.target
                        && fold.held_out == population
                }) {
                    write_wilson_record(
                        writer,
                        "cross_validated_generation",
                        point.family,
                        point.target,
                        population.as_str(),
                        fold.held_out_metrics,
                    )?;
                }
            }
        }
        Ok(())
    })
}

fn write_wilson_record(
    writer: &mut csv::Writer<Vec<u8>>,
    scope: &str,
    family: Family,
    target: f64,
    population: &str,
    metrics: EmissionMetrics,
) -> Result<()> {
    let interval = wilson_interval(metrics.correct, metrics.emitted);
    writer.write_record([
        scope.to_string(),
        family.as_str().to_string(),
        format!("{target:.4}"),
        population.to_string(),
        metrics.correct.to_string(),
        metrics.emitted.to_string(),
        optional_float(metrics.precision()),
        interval.map_or_else(String::new, |interval| float(interval.lower)),
        interval.map_or_else(String::new, |interval| float(interval.upper)),
        interval
            .is_some_and(|interval| interval.lower >= target)
            .to_string(),
    ])?;
    Ok(())
}

fn cost_sensitive_csv(frontiers: &[&[OperatingPoint]]) -> Result<Vec<u8>> {
    csv_bytes(|writer| {
        writer.write_record([
            "family",
            "wrong_cost",
            "parameters",
            "utility",
            "correct",
            "wrong",
            "null_fp",
            "false_abstentions",
            "winner_correct_but_abstained",
            "precision",
            "recall",
        ])?;
        for (family, points) in Family::ALL.into_iter().zip(frontiers.iter().copied()) {
            for cost in COSTS {
                let selected = points.iter().max_by(|left, right| {
                    utility(left.metrics, cost)
                        .cmp(&utility(right.metrics, cost))
                        .then_with(|| right.metrics.wrong.cmp(&left.metrics.wrong))
                        .then_with(|| right.policy.complexity().cmp(&left.policy.complexity()))
                        .then_with(|| right.policy.parameters().cmp(&left.policy.parameters()))
                });
                if let Some(selected) = selected {
                    writer.write_record([
                        family.as_str().to_string(),
                        cost.to_string(),
                        selected.policy.parameters(),
                        utility(selected.metrics, cost).to_string(),
                        selected.metrics.correct.to_string(),
                        selected.metrics.wrong.to_string(),
                        selected.metrics.null_false_emissions.to_string(),
                        selected.metrics.false_abstentions.to_string(),
                        selected.metrics.winner_correct_but_abstained.to_string(),
                        optional_float(selected.metrics.precision()),
                        optional_float(selected.metrics.recall()),
                    ])?;
                }
            }
        }
        Ok(())
    })
}

fn utility(metrics: EmissionMetrics, cost: usize) -> isize {
    metrics.correct as isize - (cost * metrics.wrong) as isize
}

fn synthetic_validation_csv(
    rows: &[FeatureRow],
    recommendations: &[RecommendedPoint],
) -> Result<Vec<u8>> {
    csv_bytes(|writer| {
        write_metrics_header(writer, &["policy", "parameters"])?;
        write_metrics_record(
            writer,
            &["C4", "frozen_c4"],
            evaluate_policy(rows.iter(), &Policy::C4),
        )?;
        for recommendation in recommendations {
            write_metrics_record(
                writer,
                &[
                    recommendation.label,
                    &recommendation.full_development.policy.parameters(),
                ],
                evaluate_policy(rows.iter(), &recommendation.full_development.policy),
            )?;
        }
        Ok(())
    })
}

fn country_evidence_csv(
    proxy_rows: &[FeatureRow],
    validation_rows: &[FeatureRow],
    recommendations: &[RecommendedPoint],
) -> Result<Vec<u8>> {
    csv_bytes(|writer| {
        writer.write_record([
            "population",
            "policy",
            "hint_rows",
            "comparable_rows",
            "winner_changed",
            "quality_delta_p10",
            "quality_delta_p50",
            "quality_delta_p90",
            "decision_score_delta_p10",
            "decision_score_delta_p50",
            "decision_score_delta_p90",
            "emission_changes",
        ])?;
        for (population, rows) in [
            ("COMBINED_SPENT", proxy_rows),
            ("VALIDATION", validation_rows),
        ] {
            let hinted = rows
                .iter()
                .filter(|row| row.unhinted.is_some())
                .collect::<Vec<_>>();
            let mut quality = hinted
                .iter()
                .filter_map(|row| {
                    row.unhinted
                        .as_ref()
                        .map(|unhinted| row.candidate_quality - unhinted.candidate_quality)
                })
                .collect::<Vec<_>>();
            let mut scores = hinted
                .iter()
                .filter_map(|row| {
                    row.unhinted
                        .as_ref()
                        .map(|unhinted| row.decision_score - unhinted.decision_score)
                })
                .collect::<Vec<_>>();
            quality.sort_by(f64::total_cmp);
            scores.sort_by(f64::total_cmp);
            for (name, policy) in std::iter::once(("C4", Policy::C4)).chain(
                recommendations.iter().map(|recommendation| {
                    (
                        recommendation.label,
                        recommendation.full_development.policy.clone(),
                    )
                }),
            ) {
                let emission_changes = hinted
                    .iter()
                    .filter(|row| {
                        let unhinted = row_without_hint(row);
                        policy.emits(row) != policy.emits(&unhinted)
                    })
                    .count();
                writer.write_record([
                    population.to_string(),
                    name.to_string(),
                    hinted.len().to_string(),
                    hinted.len().to_string(),
                    hinted
                        .iter()
                        .filter(|row| {
                            row.unhinted
                                .as_ref()
                                .is_some_and(|unhinted| unhinted.winner_changed)
                        })
                        .count()
                        .to_string(),
                    percentile(&quality, 0.10),
                    percentile(&quality, 0.50),
                    percentile(&quality, 0.90),
                    percentile(&scores, 0.10),
                    percentile(&scores, 0.50),
                    percentile(&scores, 0.90),
                    emission_changes.to_string(),
                ])?;
            }
        }
        Ok(())
    })
}

fn row_without_hint(row: &FeatureRow) -> FeatureRow {
    let Some(unhinted) = &row.unhinted else {
        return row.clone();
    };
    let mut result = row.clone();
    result.country_hint_present = false;
    result.locale_hint_present = false;
    result.decision_score = unhinted.decision_score;
    result.candidate_quality = unhinted.candidate_quality;
    result.winner_margin = unhinted.logistic_features[2];
    result.role_signal = unhinted.logistic_features[3];
    result.reliability = unhinted.logistic_features[4];
    result.candidate_count = unhinted.candidate_count;
    result.native = unhinted.logistic_features[6] == 1.0;
    result.vetoes_pass = unhinted.vetoes_pass;
    result.c31_emits = unhinted.c31_emits;
    result.c4_emits = unhinted.c4_emits;
    result.c5_emits = unhinted.c5_emits;
    result
}

fn recommendations_csv(recommendations: &[RecommendedPoint]) -> Result<Vec<u8>> {
    csv_bytes(|writer| {
        write_metrics_header(
            writer,
            &[
                "recommendation",
                "target",
                "family",
                "full_development_parameters",
            ],
        )?;
        for recommendation in recommendations {
            write_metrics_record(
                writer,
                &[
                    recommendation.label,
                    &format!("{:.4}", recommendation.target),
                    recommendation.family.as_str(),
                    &recommendation.full_development.policy.parameters(),
                ],
                recommendation.cross_validated,
            )?;
        }
        Ok(())
    })
}

fn qualitative_csv(
    corpus: &impl EvidenceSource,
    recommendations: &[RecommendedPoint],
) -> Result<Vec<u8>> {
    csv_bytes(|writer| {
        writer.write_record(["input", "policy", "selected_candidate", "emits"])?;
        for input in [
            // Redacted: Name that abstains at the conservative point and emits at the aggressive point.
            "Olivier REDACTED",
            // Redacted: Name that abstains at the conservative point and emits at the aggressive point.
            "Baris REDACTED",
        ] {
            let diagnostic = diagnose_role_inference(corpus, ALGORITHM_C3, input, None, None);
            let decision =
                c4_decision_breakdown(&diagnostic, ALGORITHM_C2, ALGORITHM_C31, ALGORITHM_C4);
            let row = feature_row_from_decision(
                Population::Validation,
                0,
                None,
                false,
                false,
                &decision,
                None,
            );
            let selected = decision
                .c31
                .winner
                .as_ref()
                .map_or("", |winner| winner.greeting_candidate.as_str());
            for recommendation in recommendations {
                writer.write_record([
                    input,
                    recommendation.label,
                    selected,
                    &recommendation
                        .full_development
                        .policy
                        .emits(&row)
                        .to_string(),
                ])?;
            }
        }
        Ok(())
    })
}

fn write_metrics_header(writer: &mut csv::Writer<Vec<u8>>, prefix: &[&str]) -> Result<()> {
    let mut header = prefix
        .iter()
        .map(|value| (*value).to_string())
        .collect::<Vec<_>>();
    header.extend(
        [
            "rows",
            "expected_greetings",
            "expected_nulls",
            "emitted",
            "correct",
            "wrong",
            "null_fp",
            "precision",
            "recall",
            "abstention_rate",
            "false_abstentions",
            "winner_correct_but_abstained",
            "expected_null_correct_abstentions",
            "wilson_95_lower",
            "wilson_95_upper",
        ]
        .into_iter()
        .map(str::to_string),
    );
    writer.write_record(header)?;
    Ok(())
}

fn write_metrics_record(
    writer: &mut csv::Writer<Vec<u8>>,
    prefix: &[&str],
    metrics: EmissionMetrics,
) -> Result<()> {
    let interval = wilson_interval(metrics.correct, metrics.emitted);
    let mut record = prefix
        .iter()
        .map(|value| (*value).to_string())
        .collect::<Vec<_>>();
    record.extend([
        metrics.rows.to_string(),
        metrics.expected_greetings.to_string(),
        metrics.expected_nulls.to_string(),
        metrics.emitted.to_string(),
        metrics.correct.to_string(),
        metrics.wrong.to_string(),
        metrics.null_false_emissions.to_string(),
        optional_float(metrics.precision()),
        optional_float(metrics.recall()),
        optional_float(metrics.abstention_rate()),
        metrics.false_abstentions.to_string(),
        metrics.winner_correct_but_abstained.to_string(),
        metrics.expected_null_correct_abstentions.to_string(),
        interval.map_or_else(String::new, |interval| float(interval.lower)),
        interval.map_or_else(String::new, |interval| float(interval.upper)),
    ]);
    writer.write_record(record)?;
    Ok(())
}

fn csv_bytes(write: impl FnOnce(&mut csv::Writer<Vec<u8>>) -> Result<()>) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    write(&mut writer)?;
    writer.flush()?;
    Ok(writer.into_inner()?)
}

fn optional_float(value: Option<f64>) -> String {
    value.map_or_else(String::new, float)
}

fn float(value: f64) -> String {
    format!("{value:.17}")
}

fn percent(value: Option<f64>) -> String {
    value.map_or_else(
        || "n/a".to_string(),
        |value| format!("{:.2}%", value * 100.0),
    )
}

fn percentile(values: &[f64], probability: f64) -> String {
    if values.is_empty() {
        return String::new();
    }
    let index = ((probability * values.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(values.len() - 1);
    float(values[index])
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn build_report(
    holdouts: &[FrozenHoldout],
    proxy_rows: &[FeatureRow],
    validation_rows: &[FeatureRow],
    score_frontier: &[OperatingPoint],
    controlled_frontier: &[OperatingPoint],
    logistic_frontier: &[OperatingPoint],
    additive_frontier: &[OperatingPoint],
    folds: &[FoldResult],
    best_by_target: &[CrossValidatedPoint],
    recommendations: &[RecommendedPoint],
    corpus: &impl EvidenceSource,
) -> Result<String> {
    let mut report = String::new();
    writeln!(report, "# C5 calibration-frontier diagnosis\n")?;
    writeln!(
        report,
        "C4 remains frozen and unchanged. This development-only study explicitly spends the already-checkpointed REAL_PROXY_V5 alongside V1–V4. It changes no candidate generation, ranking, corpus, artifact, veto, production API, or runtime behavior, and it does not load TEST or V6.\n"
    )?;

    writeln!(report, "## Development populations\n")?;
    writeln!(
        report,
        "| Population | Evaluable | Expected greeting | Expected NULL | Label provenance |\n| --- | ---: | ---: | ---: | --- |"
    )?;
    for holdout in holdouts {
        let population =
            Population::from_digest(&holdout.manifest.holdout_sha256).expect("validated proxy");
        writeln!(
            report,
            "| {} | {} | {} | {} | {} |",
            population.as_str(),
            holdout.manifest.evaluable_cases,
            holdout.manifest.expected_greetings,
            holdout.manifest.expected_abstentions,
            markdown(&holdout.manifest.provenance)
        )?;
    }
    writeln!(
        report,
        "| **Combined** | **7,808** | **6,478** | **1,330** | pooled counts; fitting weights each generation equally |\n"
    )?;

    let c31 = evaluate_policy(proxy_rows.iter(), &Policy::C31);
    let c4 = evaluate_policy(proxy_rows.iter(), &Policy::C4);
    writeln!(report, "## Frozen combined baselines\n")?;
    write_report_metrics_header(&mut report)?;
    write_report_metrics_row(&mut report, "C3.1", c31)?;
    write_report_metrics_row(&mut report, "C4", c4)?;
    writeln!(
        report,
        "\nA full-name fallback on an expected-greeting row is reported as a false abstention, not as a cost-free success. Expected-NULL abstentions remain correct abstentions.\n"
    )?;

    let discarded = proxy_rows
        .iter()
        .filter(|row| discarded_correct_winner(row))
        .collect::<Vec<_>>();
    writeln!(report, "## Correct ranking signal discarded by C4\n")?;
    writeln!(
        report,
        "C4 is the conservative reference point. On **{} / {} expected greetings ({})**, the selected winner already matches the expected greeting and every frozen veto passes, but C4 abstains. This is the directly calibration-recoverable bucket before any candidate-generation or ranking work.\n",
        discarded.len(),
        c4.expected_greetings,
        percent(ratio(discarded.len(), c4.expected_greetings)),
    )?;
    writeln!(
        report,
        "The fixed bins below are descriptive and did not select thresholds. Intervals are lower-inclusive and upper-exclusive, except the final bin.\n"
    )?;
    writeln!(
        report,
        "| Dimension | Bucket | Count | Share of all expected greetings | Share of discarded correct winners |\n| --- | --- | ---: | ---: | ---: |"
    )?;
    for (dimension, buckets) in [
        (
            "winner_margin",
            vec![
                "0.00-0.10",
                "0.10-0.20",
                "0.20-0.30",
                "0.30-0.50",
                "0.50-1.00",
            ],
        ),
        (
            "candidate_quality",
            vec![
                "0.00-0.20",
                "0.20-0.40",
                "0.40-0.60",
                "0.60-0.80",
                "0.80-1.00",
            ],
        ),
        (
            "role_signal",
            vec![
                "0.00-0.20",
                "0.20-0.40",
                "0.40-0.60",
                "0.60-0.80",
                "0.80-1.00",
            ],
        ),
        (
            "reliability",
            vec![
                "0.00-0.20",
                "0.20-0.40",
                "0.40-0.60",
                "0.60-0.80",
                "0.80-1.00",
            ],
        ),
        ("candidate_count", vec!["1", "2", "3", "4+"]),
    ] {
        for bucket in buckets {
            let count = discarded
                .iter()
                .filter(|row| discarded_bucket(row, dimension) == bucket)
                .count();
            writeln!(
                report,
                "| {} | {} | {} | {} | {} |",
                dimension,
                bucket,
                count,
                percent(ratio(count, c4.expected_greetings)),
                percent(ratio(count, discarded.len())),
            )?;
        }
    }
    writeln!(
        report,
        "\nPer-generation versions of the same audit are in `c4_discarded_ranking_signal.csv`.\n"
    )?;

    writeln!(report, "## Calibration loss by policy\n")?;
    writeln!(
        report,
        "`Correct winner rejected` means the expected greeting was already the selected winner, every frozen veto passed, and the evaluated policy abstained. It isolates calibration loss from candidate-generation and ranking loss.\n"
    )?;
    writeln!(
        report,
        "| Policy | Correct emitted | Wrong | Correct winner rejected | Rejected / all expected greetings |\n| --- | ---: | ---: | ---: | ---: |"
    )?;
    for (name, metrics) in [("C3.1", c31), ("C4", c4)] {
        writeln!(
            report,
            "| {} | {} | {} | {} | {} |",
            name,
            metrics.correct,
            metrics.wrong,
            metrics.winner_correct_but_abstained,
            percent(ratio(
                metrics.winner_correct_but_abstained,
                metrics.expected_greetings
            )),
        )?;
    }
    for point in best_by_target {
        writeln!(
            report,
            "| {:.1}% OOF ({}) | {} | {} | {} | {} |",
            point.target * 100.0,
            point.family.as_str(),
            point.metrics.correct,
            point.metrics.wrong,
            point.metrics.winner_correct_but_abstained,
            percent(ratio(
                point.metrics.winner_correct_but_abstained,
                point.metrics.expected_greetings
            )),
        )?;
    }

    writeln!(report, "## Empirical precision/recall frontier\n")?;
    writeln!(
        report,
        "The primary table aggregates disjoint leave-one-generation-out predictions. Each fold selected its policy using only the other four generations. The target is a training selection target; the observed out-of-fold precision may differ.\n"
    )?;
    writeln!(
        report,
        "| Proxy precision target | Selected LOGO family | Observed OOF precision | Target met OOF | Recall | Correct | Wrong | NULL FP | False abstentions | Correct winner rejected | 95% Wilson interval | Delta recall vs C4 |\n| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |"
    )?;
    for target in TARGETS {
        if let Some(point) = best_by_target.iter().find(|point| point.target == target) {
            let interval = wilson_interval(point.metrics.correct, point.metrics.emitted)
                .expect("selected point emits");
            let delta = point.metrics.recall().unwrap_or(0.0) - c4.recall().unwrap_or(0.0);
            let meaningful_999 =
                target != 0.999 || (point.metrics.emitted >= 1_000 && interval.lower >= 0.99);
            writeln!(
                report,
                "| {:.1}% | {}{} | {} | {} | {} | {} | {} | {} | {} | {} | {}–{} | {:+.2} pp |",
                target * 100.0,
                point.family.as_str(),
                if meaningful_999 {
                    ""
                } else {
                    " (99.9% unsupported)"
                },
                percent(point.metrics.precision()),
                point
                    .metrics
                    .precision()
                    .is_some_and(|precision| precision >= target),
                percent(point.metrics.recall()),
                point.metrics.correct,
                point.metrics.wrong,
                point.metrics.null_false_emissions,
                point.metrics.false_abstentions,
                point.metrics.winner_correct_but_abstained,
                percent(Some(interval.lower)),
                percent(Some(interval.upper)),
                delta * 100.0,
            )?;
        } else {
            writeln!(
                report,
                "| {:.1}% | no training-feasible family | n/a | false | n/a | 0 | 0 | 0 | n/a | n/a | n/a | n/a |",
                target * 100.0
            )?;
        }
    }

    writeln!(report, "\n## Score-only descriptive frontier\n")?;
    write_selected_frontier_table(&mut report, score_frontier)?;
    writeln!(
        report,
        "\n## Controlled C4-relaxation descriptive frontier\n"
    )?;
    write_selected_frontier_table(&mut report, controlled_frontier)?;
    writeln!(report, "\n## Monotonic-model descriptive frontier\n")?;
    writeln!(report, "### Pure logistic\n")?;
    write_selected_frontier_table(&mut report, logistic_frontier)?;
    writeln!(report, "\n### C4 plus logistic\n")?;
    write_selected_frontier_table(&mut report, additive_frontier)?;
    writeln!(
        report,
        "\nThese four descriptive tables select and report on the same pooled development rows. They are useful for curve shape, not held-out evidence.\n"
    )?;

    writeln!(report, "## Per-generation LOGO stability\n")?;
    writeln!(
        report,
        "| Target | Family | Held out | Emitted | Correct | Wrong | NULL FP | Precision | Recall | False abstentions | Correct winner rejected |\n| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )?;
    for target in TARGETS {
        let Some(best) = best_by_target.iter().find(|point| point.target == target) else {
            continue;
        };
        for population in Population::PROXIES {
            let fold = folds
                .iter()
                .find(|fold| {
                    fold.target == target
                        && fold.family == best.family
                        && fold.held_out == population
                })
                .expect("complete LOGO result");
            writeln!(
                report,
                "| {:.1}% | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                target * 100.0,
                best.family.as_str(),
                population.as_str(),
                fold.held_out_metrics.emitted,
                fold.held_out_metrics.correct,
                fold.held_out_metrics.wrong,
                fold.held_out_metrics.null_false_emissions,
                percent(fold.held_out_metrics.precision()),
                percent(fold.held_out_metrics.recall()),
                fold.held_out_metrics.false_abstentions,
                fold.held_out_metrics.winner_correct_but_abstained,
            )?;
        }
    }

    writeln!(report, "\n## Cost-sensitive view\n")?;
    writeln!(
        report,
        "Utility is `correct - cost × wrong`; expected-NULL false emissions are included in wrong and also shown separately in `cost_sensitive_frontier.csv`. The preferred full-development family by cost was:\n"
    )?;
    writeln!(
        report,
        "| Wrong cost | Family | Utility | Correct | Wrong | Recall |\n| ---: | --- | ---: | ---: | ---: | ---: |"
    )?;
    let all_frontiers = [
        (Family::ScoreOnly, score_frontier),
        (Family::ControlledC4, controlled_frontier),
        (Family::Logistic, logistic_frontier),
        (Family::C4PlusLogistic, additive_frontier),
    ];
    for cost in COSTS {
        let selected = all_frontiers
            .iter()
            .flat_map(|(family, points)| points.iter().map(move |point| (*family, point)))
            .max_by(|(_, left), (_, right)| {
                utility(left.metrics, cost)
                    .cmp(&utility(right.metrics, cost))
                    .then_with(|| right.metrics.wrong.cmp(&left.metrics.wrong))
            });
        if let Some((family, selected)) = selected {
            writeln!(
                report,
                "| {}× | {} | {} | {} | {} | {} |",
                cost,
                family.as_str(),
                utility(selected.metrics, cost),
                selected.metrics.correct,
                selected.metrics.wrong,
                percent(selected.metrics.recall()),
            )?;
        }
    }

    writeln!(report, "\n## Candidate operating points for a future C5\n")?;
    if recommendations.is_empty() {
        writeln!(
            report,
            "No requested conservative, balanced, or aggressive target had a cross-validated family that met its observed precision target. No C5 candidate is recommended."
        )?;
    } else {
        writeln!(
            report,
            "| Label | Target | Family | OOF precision | OOF recall | OOF correct | OOF wrong | OOF false abstentions | OOF correct winner rejected | Full-development parameters |\n| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
        )?;
        for recommendation in recommendations {
            writeln!(
                report,
                "| {} | {:.1}% | {} | {} | {} | {} | {} | {} | {} | `{}` |",
                recommendation.label,
                recommendation.target * 100.0,
                recommendation.family.as_str(),
                percent(recommendation.cross_validated.precision()),
                percent(recommendation.cross_validated.recall()),
                recommendation.cross_validated.correct,
                recommendation.cross_validated.wrong,
                recommendation.cross_validated.false_abstentions,
                recommendation.cross_validated.winner_correct_but_abstained,
                recommendation.full_development.policy.parameters(),
            )?;
        }
    }

    writeln!(report, "\n## Model-form conclusion\n")?;
    writeln!(
        report,
        "The controlled relational family exposes substantial usable ranking signal, but this study does not establish a new policy that strictly dominates C4 near its existing precision. The 99.0% and 98.0% training-target selections reached 33.02% and 49.07% recall out of fold, but their observed precision was 98.84% and 97.91%; both missed the requested target. The monotonic logistic family was selected only at the stricter 99.9% and 99.5% targets, where it reduced recall below C4. At 95.0%, controlled C4 relaxation met its out-of-fold target with 95.67% precision and 59.69% recall, demonstrating a clear lower-precision tradeoff rather than a free improvement.\n"
    )?;
    writeln!(
        report,
        "Accordingly, C4 remains the production reference. The report recommends a very-conservative 99.5% logistic point and an aggressive 95.0% controlled-relational point for product discussion, but no balanced C5 point is frozen: the cross-validated 99.0% candidate did not meet its target. A future selected policy still requires untouched REAL_PROXY_V6 validation.\n"
    )?;

    let c4_validation = evaluate_policy(validation_rows.iter(), &Policy::C4);
    writeln!(report, "\n## Synthetic VALIDATION sanity check\n")?;
    write_report_metrics_header(&mut report)?;
    write_report_metrics_row(&mut report, "C4", c4_validation)?;
    for recommendation in recommendations {
        write_report_metrics_row(
            &mut report,
            recommendation.label,
            evaluate_policy(
                validation_rows.iter(),
                &recommendation.full_development.policy,
            ),
        )?;
    }

    let proxy_hints = proxy_rows
        .iter()
        .filter(|row| row.unhinted.is_some())
        .count();
    let validation_hints = validation_rows
        .iter()
        .filter(|row| row.unhinted.is_some())
        .count();
    writeln!(report, "\n## Country/locale evidence\n")?;
    writeln!(
        report,
        "The proxy population contains {proxy_hints} hinted rows and synthetic VALIDATION contains {validation_hints}. Candidate-quality, C3.1-score, winner-change, and recommendation emission-change aggregates are in `country_evidence_audit.csv`. Candidate quality can carry existing country evidence into a fitted calibration, but the proxy data cannot validate that behavior when hints are absent.\n"
    )?;

    writeln!(report, "## Post-selection qualitative examples\n")?;
    writeln!(
        report,
        "| Input | Recommendation | Selected candidate | Emits |\n| --- | --- | --- | --- |"
    )?;
    for input in [
        // Redacted: Name that abstains at the conservative point and emits at the aggressive point.
        "Olivier REDACTED",
        // Redacted: Name that abstains at the conservative point and emits at the aggressive point.
        "Baris REDACTED",
    ] {
        let diagnostic = diagnose_role_inference(corpus, ALGORITHM_C3, input, None, None);
        let decision =
            c4_decision_breakdown(&diagnostic, ALGORITHM_C2, ALGORITHM_C31, ALGORITHM_C4);
        let row = feature_row_from_decision(
            Population::Validation,
            0,
            None,
            false,
            false,
            &decision,
            None,
        );
        let selected = decision
            .c31
            .winner
            .as_ref()
            .map_or("NULL", |winner| winner.greeting_candidate.as_str());
        for recommendation in recommendations {
            writeln!(
                report,
                "| {} | {} | {} | {} |",
                input,
                recommendation.label,
                selected,
                recommendation.full_development.policy.emits(&row),
            )?;
        }
    }
    writeln!(
        report,
        "\nThese examples were evaluated only after model-family and operating-point selection. They did not participate in fitting or threshold selection.\n"
    )?;

    writeln!(report, "## Interpretation\n")?;
    writeln!(
        report,
        "The leave-one-generation-out frontier is the primary development evidence. Wilson intervals describe case-level binomial uncertainty only; they do not account for source bias, repeated name atoms, annotation bias, or worldwide population uncertainty. V1 has different annotation provenance, while V2–V5 retain only exact annotation consensus. All five generations are now spent development evidence. A selected full-development policy requires untouched REAL_PROXY_V6 one-shot validation before it can become C5."
    )?;
    Ok(report)
}

fn write_report_metrics_header(report: &mut String) -> Result<()> {
    writeln!(
        report,
        "| Policy | Emitted | Correct | Wrong | NULL FP | Precision | Recall | False abstentions | Correct winner rejected | Expected-NULL correct abstentions |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )?;
    Ok(())
}

fn write_report_metrics_row(
    report: &mut String,
    name: &str,
    metrics: EmissionMetrics,
) -> Result<()> {
    writeln!(
        report,
        "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
        name,
        metrics.emitted,
        metrics.correct,
        metrics.wrong,
        metrics.null_false_emissions,
        percent(metrics.precision()),
        percent(metrics.recall()),
        metrics.false_abstentions,
        metrics.winner_correct_but_abstained,
        metrics.expected_null_correct_abstentions,
    )?;
    Ok(())
}

fn write_selected_frontier_table(report: &mut String, points: &[OperatingPoint]) -> Result<()> {
    writeln!(
        report,
        "| Target | Observed precision | Recall | Correct | Wrong | NULL FP | False abstentions | Correct winner rejected | Parameters |\n| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
    )?;
    for target in TARGETS {
        if let Some(point) = select_point(points, target) {
            writeln!(
                report,
                "| {:.1}% | {} | {} | {} | {} | {} | {} | {} | `{}` |",
                target * 100.0,
                percent(point.metrics.precision()),
                percent(point.metrics.recall()),
                point.metrics.correct,
                point.metrics.wrong,
                point.metrics.null_false_emissions,
                point.metrics.false_abstentions,
                point.metrics.winner_correct_but_abstained,
                point.policy.parameters(),
            )?;
        } else {
            writeln!(
                report,
                "| {:.1}% | n/a | n/a | 0 | 0 | 0 | n/a | n/a | n/a |",
                target * 100.0
            )?;
        }
    }
    Ok(())
}

fn markdown(value: &str) -> String {
    value.replace('|', "\\|").replace(['\n', '\r'], " ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(population: Population, expected: bool, matches: bool, score: f64) -> FeatureRow {
        FeatureRow {
            population,
            ordinal: 0,
            expected_greeting: expected,
            selected_matches: matches,
            winner_present: true,
            vetoes_pass: true,
            decision_score: score,
            candidate_quality: score,
            candidate_count: 2,
            winner_margin: score,
            margin_signal: score,
            role_llr: 0.0,
            role_signal: score,
            reliability: score,
            alphabetic_length: 5,
            native: true,
            segmentation_mechanism: None,
            hard_organization_marker: false,
            generic_organization_marker: false,
            ampersand: false,
            candidate_too_short: false,
            country_hint_present: false,
            locale_hint_present: false,
            c31_emits: false,
            c4_emits: false,
            c4_source: C4EmissionSource::Abstain,
            c5_emits: false,
            unhinted: None,
        }
    }

    #[test]
    fn metrics_distinguish_false_abstention_and_null_false_emission() {
        let rows = [
            row(Population::V1, true, true, 0.9),
            row(Population::V1, true, false, 0.8),
            row(Population::V1, false, false, 0.7),
            row(Population::V1, true, true, 0.1),
        ];
        let metrics = evaluate_policy(rows.iter(), &Policy::Score { threshold: 0.5 });
        assert_eq!(metrics.emitted, 3);
        assert_eq!(metrics.correct, 1);
        assert_eq!(metrics.wrong, 2);
        assert_eq!(metrics.null_false_emissions, 1);
        assert_eq!(metrics.false_abstentions, 1);
        assert_eq!(metrics.winner_correct_but_abstained, 1);
    }

    #[test]
    fn controlled_rule_is_inclusive_and_native_only() {
        let policy = Policy::Controlled {
            quality: 0.4,
            reliability: 0.7,
            role: 0.6,
            margin: 0.5,
        };
        let mut candidate = row(Population::V1, true, true, 0.4);
        candidate.candidate_quality = 0.4;
        candidate.reliability = 0.7;
        candidate.role_signal = 0.6;
        candidate.winner_margin = 0.5;
        assert!(policy.emits(&candidate));
        candidate.native = false;
        assert!(!policy.emits(&candidate));
    }

    #[test]
    fn score_frontier_uses_every_distinct_inclusive_threshold() {
        let rows = vec![
            row(Population::V1, true, true, 0.9),
            row(Population::V1, true, false, 0.8),
            row(Population::V1, true, true, 0.7),
        ];
        let points = score_frontier(&rows);
        assert_eq!(points.len(), 3);
        let selected = select_point(&points, 1.0).unwrap();
        assert_eq!(selected.metrics.correct, 1);
        assert_eq!(selected.metrics.wrong, 0);
    }

    #[test]
    fn wilson_known_all_successes_is_not_one_at_lower_bound() {
        let interval = wilson_interval(100, 100).unwrap();
        assert!((interval.lower - 0.963_006_501_793_014_3).abs() < 1.0e-12);
        assert_eq!(interval.upper, 1.0);
        assert!(wilson_interval(0, 0).is_none());
    }

    #[test]
    fn generation_balanced_training_weights_sum_equally() {
        let rows = vec![
            row(Population::V1, true, true, 0.9),
            row(Population::V1, true, true, 0.8),
            row(Population::V2, true, false, 0.7),
        ];
        let training = logistic_training_rows(&rows).unwrap();
        let v1 = training[..2].iter().map(|row| row.weight).sum::<f64>();
        let v2 = training[2..].iter().map(|row| row.weight).sum::<f64>();
        assert!((v1 - 0.5).abs() < f64::EPSILON);
        assert!((v2 - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn logistic_fit_is_monotonic_and_repeatable() {
        let rows = vec![
            row(Population::V1, true, false, 0.1),
            row(Population::V1, true, true, 0.9),
            row(Population::V2, true, false, 0.2),
            row(Population::V2, true, true, 0.8),
        ];
        let first = fit_logistic(&rows).unwrap();
        let second = fit_logistic(&rows).unwrap();
        assert_eq!(first.intercept.to_bits(), second.intercept.to_bits());
        assert_eq!(
            first.coefficients.map(f64::to_bits),
            second.coefficients.map(f64::to_bits)
        );
        assert!(
            first
                .coefficients
                .iter()
                .all(|coefficient| *coefficient >= 0.0)
        );
        assert!(first.score([0.9; FEATURE_COUNT]) >= first.score([0.1; FEATURE_COUNT]));
    }

    #[test]
    fn feature_csv_does_not_expose_source_text_or_id_columns() {
        let rows = [row(Population::V1, true, true, 0.9)];
        let csv = String::from_utf8(calibration_features_csv(&rows, &[]).unwrap()).unwrap();
        let header = csv.lines().next().unwrap();
        assert!(!header.contains("display_name"));
        assert!(!header.contains("source_id"));
        assert!(header.contains("ordinal"));
    }
}
