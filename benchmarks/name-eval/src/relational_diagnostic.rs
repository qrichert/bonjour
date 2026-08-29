use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::Path;

use name_eval::holdout::{FrozenHoldout, HoldoutCase};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::artifact::EvidenceSource;
use crate::classifier::{
    ALGORITHM_C1, ALGORITHM_C2, ALGORITHM_C3, ALGORITHM_C4, ALGORITHM_C31, C31DecisionBreakdown,
    C4DecisionBreakdown, C4EmissionConfig, C4EmissionSource, C4RuleBreakdown, CandidateDiagnostic,
    WinnerFeatures, c2_inference_from_diagnostic, c31_decision_breakdown, c4_decision_breakdown,
    c4_decision_from_c31, c4_emitted_candidate, diagnose_role_inference,
};
use crate::dataset::{Case, Split, generate_cases};
use crate::metrics::greeting_matches;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub const V1_SHA256: &str = "de95213f27fc1849032ee6788c8f16d7d515c1a991ae8b2e8414b7b155814c4e";
pub const V3_SHA256: &str = "d70e4d4b2ed7e49bed09dc1e8d2ba60ade8a752e3b86c772e964bd64883ee6fe";
pub const V4_SHA256: &str = "d95c589bec836faaeecaeda85b146989d2936914bff0209934f289ccb9446c7f";

const QUALITY_STEPS: std::ops::RangeInclusive<usize> = 8..=19;
const MARGINS: [f64; 4] = [0.10, 0.20, 0.30, 0.50];
const LARGE_REVIEW_MARGIN: f64 = 0.30;
const REVIEW_LIMIT: usize = 8;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Population {
    V1,
    V3,
    V4,
    CombinedSpent,
    Validation,
}

impl Population {
    const OUTPUTS: [Self; 5] = [
        Self::V1,
        Self::V3,
        Self::V4,
        Self::CombinedSpent,
        Self::Validation,
    ];

    fn from_digest(digest: &str) -> Option<Self> {
        match digest {
            V1_SHA256 => Some(Self::V1),
            V3_SHA256 => Some(Self::V3),
            V4_SHA256 => Some(Self::V4),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::V1 => "REAL_PROXY_V1_DEV",
            Self::V3 => "REAL_PROXY_V3_DEV",
            Self::V4 => "REAL_PROXY_V4_DEV",
            Self::CombinedSpent => "COMBINED_SPENT",
            Self::Validation => "VALIDATION",
        }
    }

    fn is_spent(self) -> bool {
        matches!(self, Self::V1 | Self::V3 | Self::V4)
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Provenance {
    Native,
    HandleSegment,
    None,
}

impl Provenance {
    fn from_winner(winner: Option<&WinnerFeatures>) -> Self {
        match winner.map(|winner| winner.candidate_origin) {
            Some("handle_segment") => Self::HandleSegment,
            Some(_) => Self::Native,
            None => Self::None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::HandleSegment => "handle_segment",
            Self::None => "none",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Topology {
    None,
    Sole,
    Margin00To10,
    Margin10To20,
    Margin20To30,
    Margin30To50,
    Margin50Plus,
}

impl Topology {
    fn from_candidates(candidates: &[CandidateDiagnostic]) -> Self {
        match candidates {
            [] => Self::None,
            [_] => Self::Sole,
            [winner, second, ..] => Self::from_margin(winner.score - second.score),
        }
    }

    fn from_margin(margin: f64) -> Self {
        if margin < 0.10 {
            Self::Margin00To10
        } else if margin < 0.20 {
            Self::Margin10To20
        } else if margin < 0.30 {
            Self::Margin20To30
        } else if margin < 0.50 {
            Self::Margin30To50
        } else {
            Self::Margin50Plus
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::None => "no_viable_candidate",
            Self::Sole => "sole_viable_candidate",
            Self::Margin00To10 => "multiple_margin_0.00_0.10",
            Self::Margin10To20 => "multiple_margin_0.10_0.20",
            Self::Margin20To30 => "multiple_margin_0.20_0.30",
            Self::Margin30To50 => "multiple_margin_0.30_0.50",
            Self::Margin50Plus => "multiple_margin_0.50_plus",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum WinnerOutcome {
    CorrectEmitted,
    CorrectAbstained,
    WrongWinner,
    ExpectedNullWinner,
}

impl WinnerOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::CorrectEmitted => "correct_winner_emitted",
            Self::CorrectAbstained => "correct_winner_abstained",
            Self::WrongWinner => "wrong_winner",
            Self::ExpectedNullWinner => "expected_null_winner",
        }
    }
}

#[derive(Clone)]
struct DiagnosticCase {
    population: Population,
    holdout_digest: Option<String>,
    id: String,
    display_name: String,
    country_hint: Option<String>,
    locale_hint: Option<String>,
    expected_greeting: Option<String>,
    topology: Topology,
    viable_candidates: usize,
    winner: Option<WinnerFeatures>,
    margin_signal: Option<f64>,
    c2_emitted: Option<String>,
    c3_emitted: Option<String>,
    c31_score: f64,
    c31_emitted: Option<String>,
    c4_decision: C4DecisionBreakdown,
    vetoes_pass: bool,
    country_audit: Option<CountryAuditCase>,
}

impl DiagnosticCase {
    fn provenance(&self) -> Provenance {
        Provenance::from_winner(self.winner.as_ref())
    }

    fn winner_outcome(&self) -> Option<WinnerOutcome> {
        let winner = self.winner.as_ref()?;
        match self.expected_greeting.as_deref() {
            Some(expected)
                if greeting_matches(Some(expected), Some(&winner.greeting_candidate)) =>
            {
                if self.c31_emitted.is_some() {
                    Some(WinnerOutcome::CorrectEmitted)
                } else {
                    Some(WinnerOutcome::CorrectAbstained)
                }
            }
            Some(_) => Some(WinnerOutcome::WrongWinner),
            None => Some(WinnerOutcome::ExpectedNullWinner),
        }
    }

    fn hint_present(&self) -> bool {
        self.country_hint.is_some() || self.locale_hint.is_some()
    }
}

#[derive(Clone)]
struct CountryAuditCase {
    comparable: bool,
    winner_changed: bool,
    quality_delta: Option<f64>,
    final_score_delta: f64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct EmissionMetrics {
    emitted: usize,
    correct: usize,
    wrong: usize,
    null_emissions: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct TopologyMetrics {
    rows: usize,
    expected_greetings: usize,
    expected_nulls: usize,
    correct_winners: usize,
    wrong_winners: usize,
    c31_correct: usize,
    c31_wrong: usize,
    c31_null_emissions: usize,
    correct_winners_abstained: usize,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum RuleFamily {
    Sole,
    Dominant,
    Combined,
}

impl RuleFamily {
    const ALL: [Self; 3] = [Self::Sole, Self::Dominant, Self::Combined];

    fn as_str(self) -> &'static str {
        match self {
            Self::Sole => "sole",
            Self::Dominant => "dominant",
            Self::Combined => "combined",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Rule {
    family: RuleFamily,
    quality: f64,
    reliability: f64,
    role: f64,
    margin: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct RuleDelta {
    correct: usize,
    wrong: usize,
    null_emissions: usize,
}

#[derive(Clone, Debug)]
struct RuleEvaluation {
    rule: Rule,
    deltas: BTreeMap<Population, RuleDelta>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum SelectionKind {
    ZeroError,
    OneWrong,
    OneNull,
}

impl SelectionKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::ZeroError => "zero_error",
            Self::OneWrong => "one_new_wrong",
            Self::OneNull => "one_new_null",
        }
    }
}

#[derive(Clone)]
struct SelectedPoint {
    kind: SelectionKind,
    evaluation: RuleEvaluation,
}

#[derive(Clone, Copy)]
enum C4Branch {
    SoleNative,
    DominantWinner,
    Combined,
}

impl C4Branch {
    const ALL: [Self; 3] = [Self::SoleNative, Self::DominantWinner, Self::Combined];

    fn as_str(self) -> &'static str {
        match self {
            Self::SoleNative => "sole_native",
            Self::DominantWinner => "dominant_winner",
            Self::Combined => "combined_unique",
        }
    }

    fn includes(self, source: C4EmissionSource) -> bool {
        match self {
            Self::SoleNative => source == C4EmissionSource::SoleNative,
            Self::DominantWinner => source == C4EmissionSource::DominantWinner,
            Self::Combined => matches!(
                source,
                C4EmissionSource::SoleNative | C4EmissionSource::DominantWinner
            ),
        }
    }
}

#[derive(Serialize)]
struct QualitativeC4Diagnostic {
    input: &'static str,
    selected_candidate: Option<String>,
    decision_score: f64,
    emission_source: &'static str,
    candidate_count: usize,
    candidate_quality: Option<f64>,
    winner_margin: Option<f64>,
    margin_signal: Option<f64>,
    role_llr: Option<f64>,
    role_signal: Option<f64>,
    reliability: Option<f64>,
    alphabetic_length: Option<usize>,
    segmented_candidate: Option<bool>,
    segmentation_mechanism: Option<&'static str>,
    segmented_candidate_penalty: f64,
    vetoes: QualitativeVetoes,
    conditions: QualitativeConditions,
}

#[derive(Serialize)]
struct QualitativeVetoes {
    strong_organization_marker: bool,
    generic_organization_marker: bool,
    ampersand: bool,
    candidate_too_short: bool,
}

#[derive(Serialize)]
struct QualitativeConditions {
    sole_native: QualitativeRuleBreakdown,
    dominant_winner: QualitativeRuleBreakdown,
}

#[derive(Serialize)]
struct QualitativeRuleBreakdown {
    c3_1_abstained: bool,
    native_candidate: bool,
    candidate_count: usize,
    candidate_count_pass: bool,
    candidate_quality: Option<f64>,
    candidate_quality_min: f64,
    candidate_quality_pass: bool,
    winner_margin: Option<f64>,
    winner_margin_min: Option<f64>,
    winner_margin_pass: bool,
    reliability: Option<f64>,
    reliability_min: f64,
    reliability_pass: bool,
    role_signal: Option<f64>,
    role_signal_min: f64,
    role_signal_pass: bool,
    vetoes_pass: bool,
    passed: bool,
}

pub fn run_relational_diagnostic(
    output: &Path,
    corpus: &impl EvidenceSource,
    holdouts: Vec<FrozenHoldout>,
    fixtures: &Path,
) -> Result<String> {
    validate_holdout_set(&holdouts)?;
    let mut cases = build_spent_cases(corpus, &holdouts);
    cases.extend(build_validation_cases(corpus, fixtures)?);
    assert_frozen_checkpoints(&cases)?;

    write_topology_outcomes(output, &cases)?;
    write_feature_percentiles(output, &cases)?;
    write_feature_categories(output, &cases)?;
    write_country_evidence_audit(output, &cases)?;
    let evaluations = evaluate_operating_points(&cases);
    write_operating_points(output, &evaluations)?;
    let selected = select_operating_points(&evaluations);
    write_selected_points(output, &selected)?;
    write_qualitative_review_sample(output, &cases)?;
    build_report(&cases, &evaluations, &selected)
}

pub fn run_c4_development_freeze(
    output: &Path,
    corpus: &impl EvidenceSource,
    holdouts: Vec<FrozenHoldout>,
    fixtures: &Path,
) -> Result<String> {
    validate_holdout_set(&holdouts)?;
    let mut cases = build_spent_cases(corpus, &holdouts);
    cases.extend(build_validation_cases(corpus, fixtures)?);
    assert_frozen_checkpoints(&cases)?;
    assert_c4_development_checkpoints(&cases)?;
    write_c4_development_summary(output, &cases)?;
    let qualitative = qualitative_c4_diagnostics(corpus);
    let mut json = serde_json::to_vec_pretty(&qualitative)?;
    json.push(b'\n');
    fs::write(output.join("c4_qualitative_diagnostics.json"), json)?;
    build_c4_development_report(&cases, &qualitative)
}

fn validate_holdout_set(holdouts: &[FrozenHoldout]) -> Result<()> {
    let actual = holdouts
        .iter()
        .map(|holdout| holdout.manifest.holdout_sha256.as_str())
        .collect::<BTreeSet<_>>();
    let expected = [V1_SHA256, V3_SHA256, V4_SHA256]
        .into_iter()
        .collect::<BTreeSet<_>>();
    if holdouts.len() != 3 || actual != expected {
        return Err(format!(
            "relational diagnosis requires exactly spent V1/V3/V4; received {actual:?}"
        )
        .into());
    }
    Ok(())
}

fn build_spent_cases(
    corpus: &impl EvidenceSource,
    holdouts: &[FrozenHoldout],
) -> Vec<DiagnosticCase> {
    let mut cases = holdouts
        .iter()
        .flat_map(|holdout| {
            let population = Population::from_digest(&holdout.manifest.holdout_sha256)
                .expect("validated spent digest");
            holdout
                .cases
                .iter()
                .filter(|case| case.is_evaluable())
                .map(move |case| diagnostic_case_from_holdout(corpus, population, holdout, case))
        })
        .collect::<Vec<_>>();
    cases.sort_by(|left, right| (left.population, &left.id).cmp(&(right.population, &right.id)));
    cases
}

fn diagnostic_case_from_holdout(
    corpus: &impl EvidenceSource,
    population: Population,
    holdout: &FrozenHoldout,
    case: &HoldoutCase,
) -> DiagnosticCase {
    diagnostic_case(
        corpus,
        population,
        Some(&holdout.manifest.holdout_sha256),
        &case.id,
        &case.display_name,
        case.expected_greeting(),
        nonempty(&case.country_hint),
        nonempty(&case.locale_hint),
    )
}

fn build_validation_cases(
    corpus: &impl EvidenceSource,
    fixtures: &Path,
) -> Result<Vec<DiagnosticCase>> {
    Ok(generate_cases(fixtures, false)?
        .into_iter()
        .filter(|case| case.split == Split::Validation)
        .map(|case| diagnostic_case_from_generated(corpus, &case))
        .collect())
}

fn diagnostic_case_from_generated(corpus: &impl EvidenceSource, case: &Case) -> DiagnosticCase {
    diagnostic_case(
        corpus,
        Population::Validation,
        None,
        &case.id,
        &case.input,
        case.expected_greeting.as_deref(),
        case.country_hint.as_deref(),
        case.locale_hint.as_deref(),
    )
}

#[allow(clippy::too_many_arguments)]
fn diagnostic_case(
    corpus: &impl EvidenceSource,
    population: Population,
    holdout_digest: Option<&str>,
    id: &str,
    display_name: &str,
    expected_greeting: Option<&str>,
    country_hint: Option<&str>,
    locale_hint: Option<&str>,
) -> DiagnosticCase {
    let c2_diagnostic = diagnose_role_inference(
        corpus,
        ALGORITHM_C1,
        display_name,
        country_hint,
        locale_hint,
    );
    let c2 = c2_inference_from_diagnostic(&c2_diagnostic, ALGORITHM_C2);
    let c2_emitted = c2
        .greeting_at(ALGORITHM_C2.threshold)
        .map(str::to_string);
    let diagnostic = diagnose_role_inference(
        corpus,
        ALGORITHM_C3,
        display_name,
        country_hint,
        locale_hint,
    );
    let topology = Topology::from_candidates(&diagnostic.candidates);
    let c3 = c2_inference_from_diagnostic(&diagnostic, ALGORITHM_C2);
    let c3_emitted = c3
        .greeting_at(ALGORITHM_C2.threshold)
        .map(str::to_string);
    let c4_decision = c4_decision_breakdown(
        &diagnostic,
        ALGORITHM_C2,
        ALGORITHM_C31,
        ALGORITHM_C4,
    );
    let breakdown = &c4_decision.c31;
    let winner = breakdown.winner.clone();
    let c31_emitted = winner.as_ref().and_then(|winner| {
        (breakdown.final_score >= ALGORITHM_C2.threshold).then(|| winner.greeting_candidate.clone())
    });
    let vetoes_pass = vetoes_pass(breakdown);
    let country_audit = (country_hint.is_some() || locale_hint.is_some())
        .then(|| country_audit_case(corpus, display_name, &diagnostic.candidates, breakdown));
    DiagnosticCase {
        population,
        holdout_digest: holdout_digest.map(str::to_string),
        id: id.to_string(),
        display_name: display_name.to_string(),
        country_hint: country_hint.map(str::to_string),
        locale_hint: locale_hint.map(str::to_string),
        expected_greeting: expected_greeting.map(str::to_string),
        topology,
        viable_candidates: diagnostic.candidates.len(),
        winner,
        margin_signal: breakdown.margin_signal,
        c2_emitted,
        c3_emitted,
        c31_score: breakdown.final_score,
        c31_emitted,
        c4_decision,
        vetoes_pass,
        country_audit,
    }
}

fn country_audit_case(
    corpus: &impl EvidenceSource,
    display_name: &str,
    hinted_candidates: &[CandidateDiagnostic],
    hinted_breakdown: &C31DecisionBreakdown,
) -> CountryAuditCase {
    let without_hint = diagnose_role_inference(corpus, ALGORITHM_C3, display_name, None, None);
    let without_hint_breakdown = c31_decision_breakdown(&without_hint, ALGORITHM_C2, ALGORITHM_C31);
    let hinted_winner = hinted_breakdown
        .winner
        .as_ref()
        .and_then(|_| hinted_candidates.first());
    let matched = hinted_winner.and_then(|winner| {
        without_hint
            .candidates
            .iter()
            .find(|candidate| same_candidate(winner, candidate))
    });
    let winner_changed = hinted_candidates
        .first()
        .zip(without_hint.candidates.first())
        .is_some_and(|(left, right)| !same_candidate(left, right));
    CountryAuditCase {
        comparable: matched.is_some(),
        winner_changed,
        quality_delta: hinted_winner
            .zip(matched)
            .map(|(hinted, unhinted)| hinted.score - unhinted.score),
        final_score_delta: hinted_breakdown.final_score - without_hint_breakdown.final_score,
    }
}

fn same_candidate(left: &CandidateDiagnostic, right: &CandidateDiagnostic) -> bool {
    left.start == right.start
        && left.length == right.length
        && left.display == right.display
        && left.origin == right.origin
}

fn vetoes_pass(breakdown: &C31DecisionBreakdown) -> bool {
    !breakdown.hard_organization_marker
        && !breakdown.generic_organization_marker
        && !breakdown.ampersand
        && !breakdown.candidate_too_short
}

fn assert_frozen_checkpoints(cases: &[DiagnosticCase]) -> Result<()> {
    for (population, expected_by_algorithm) in frozen_checkpoint_table() {
        for (algorithm, expected) in expected_by_algorithm {
            let actual = emission_metrics(
                cases.iter().filter(|case| case.population == population),
                algorithm,
            );
            if actual != expected {
                return Err(format!(
                    "frozen {} {} checkpoint changed: expected {expected:?}, got {actual:?}",
                    algorithm.as_str(),
                    population.as_str()
                )
                .into());
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum FrozenAlgorithm {
    C2,
    C3,
    C31,
}

impl FrozenAlgorithm {
    fn as_str(self) -> &'static str {
        match self {
            Self::C2 => "C2",
            Self::C3 => "C3",
            Self::C31 => "C3.1",
        }
    }

    fn emitted(self, case: &DiagnosticCase) -> Option<&str> {
        match self {
            Self::C2 => case.c2_emitted.as_deref(),
            Self::C3 => case.c3_emitted.as_deref(),
            Self::C31 => case.c31_emitted.as_deref(),
        }
    }
}

fn frozen_checkpoint_table() -> [(Population, [(FrozenAlgorithm, EmissionMetrics); 3]); 4] {
    [
        (
            Population::V1,
            [
                (FrozenAlgorithm::C2, emission_metrics_value(207, 207, 0, 0)),
                (FrozenAlgorithm::C3, emission_metrics_value(234, 234, 0, 0)),
                (FrozenAlgorithm::C31, emission_metrics_value(226, 226, 0, 0)),
            ],
        ),
        (
            Population::V3,
            [
                (FrozenAlgorithm::C2, emission_metrics_value(205, 200, 5, 2)),
                (FrozenAlgorithm::C3, emission_metrics_value(223, 217, 6, 3)),
                (FrozenAlgorithm::C31, emission_metrics_value(219, 214, 5, 2)),
            ],
        ),
        (
            Population::V4,
            [
                (FrozenAlgorithm::C2, emission_metrics_value(213, 210, 3, 0)),
                (FrozenAlgorithm::C3, emission_metrics_value(237, 233, 4, 0)),
                (FrozenAlgorithm::C31, emission_metrics_value(227, 224, 3, 0)),
            ],
        ),
        (
            Population::Validation,
            [
                (
                    FrozenAlgorithm::C2,
                    emission_metrics_value(14_686, 14_686, 0, 0),
                ),
                (
                    FrozenAlgorithm::C3,
                    emission_metrics_value(14_686, 14_686, 0, 0),
                ),
                (
                    FrozenAlgorithm::C31,
                    emission_metrics_value(14_686, 14_686, 0, 0),
                ),
            ],
        ),
    ]
}

const fn emission_metrics_value(
    emitted: usize,
    correct: usize,
    wrong: usize,
    null_emissions: usize,
) -> EmissionMetrics {
    EmissionMetrics {
        emitted,
        correct,
        wrong,
        null_emissions,
    }
}

fn emission_metrics<'a>(
    cases: impl Iterator<Item = &'a DiagnosticCase>,
    algorithm: FrozenAlgorithm,
) -> EmissionMetrics {
    let mut metrics = EmissionMetrics::default();
    for case in cases {
        let Some(emitted) = algorithm.emitted(case) else {
            continue;
        };
        metrics.emitted += 1;
        if greeting_matches(case.expected_greeting.as_deref(), Some(emitted)) {
            metrics.correct += 1;
        } else {
            metrics.wrong += 1;
            if case.expected_greeting.is_none() {
                metrics.null_emissions += 1;
            }
        }
    }
    metrics
}

fn assert_c4_development_checkpoints(cases: &[DiagnosticCase]) -> Result<()> {
    let expected_config = C4EmissionConfig {
        sole_quality_min: 0.75,
        sole_reliability_min: 0.40,
        sole_role_signal_min: 0.80,
        dominant_quality_min: 0.40,
        dominant_reliability_min: 0.75,
        dominant_role_signal_min: 0.40,
        dominant_winner_margin_min: 0.50,
    };
    if ALGORITHM_C4 != expected_config {
        return Err(format!(
            "frozen C4 configuration changed: expected {expected_config:?}, got {ALGORITHM_C4:?}"
        )
        .into());
    }

    for case in cases {
        let decision = &case.c4_decision;
        if decision.sole_native.passed && decision.dominant_winner.passed {
            return Err(format!(
                "C4 relational branches overlap for {} {}",
                case.population.as_str(),
                case.id
            )
            .into());
        }
        let emitted = c4_emitted_candidate(decision);
        let selected = decision
            .c31
            .winner
            .as_ref()
            .map(|winner| winner.greeting_candidate.as_str());
        match decision.emission_source {
            C4EmissionSource::C31 => {
                if case.c31_emitted.as_deref() != emitted || emitted != selected {
                    return Err(format!(
                        "C4 changed a C3.1 emission for {} {}",
                        case.population.as_str(),
                        case.id
                    )
                    .into());
                }
            }
            C4EmissionSource::SoleNative | C4EmissionSource::DominantWinner => {
                if case.c31_emitted.is_some() || emitted != selected {
                    return Err(format!(
                        "C4 relational emission changed the selected winner for {} {}",
                        case.population.as_str(),
                        case.id
                    )
                    .into());
                }
            }
            C4EmissionSource::Abstain => {
                if emitted.is_some() {
                    return Err(format!(
                        "C4 abstention emitted a winner for {} {}",
                        case.population.as_str(),
                        case.id
                    )
                    .into());
                }
            }
        }
    }

    let stricter_dominant = C4EmissionConfig {
        dominant_quality_min: 0.70,
        ..ALGORITHM_C4
    };
    let floor_ids = c4_relational_ids(cases, ALGORITHM_C4, C4EmissionSource::DominantWinner);
    let stricter_ids =
        c4_relational_ids(cases, stricter_dominant, C4EmissionSource::DominantWinner);
    if floor_ids != stricter_ids {
        return Err(format!(
            "dominant candidate-quality floors 0.40 and 0.70 selected different rows: {} versus {}",
            floor_ids.len(),
            stricter_ids.len()
        )
        .into());
    }

    for (population, branch, expected) in [
        (
            Population::CombinedSpent,
            C4Branch::SoleNative,
            RuleDelta {
                correct: 17,
                wrong: 0,
                null_emissions: 0,
            },
        ),
        (
            Population::Validation,
            C4Branch::SoleNative,
            RuleDelta::default(),
        ),
        (
            Population::CombinedSpent,
            C4Branch::DominantWinner,
            RuleDelta {
                correct: 124,
                wrong: 0,
                null_emissions: 0,
            },
        ),
        (
            Population::Validation,
            C4Branch::DominantWinner,
            RuleDelta {
                correct: 1_609,
                wrong: 0,
                null_emissions: 0,
            },
        ),
        (
            Population::CombinedSpent,
            C4Branch::Combined,
            RuleDelta {
                correct: 141,
                wrong: 0,
                null_emissions: 0,
            },
        ),
        (
            Population::Validation,
            C4Branch::Combined,
            RuleDelta {
                correct: 1_609,
                wrong: 0,
                null_emissions: 0,
            },
        ),
    ] {
        let actual = c4_increment_metrics(cases, population, branch);
        if actual != expected {
            return Err(format!(
                "frozen C4 {} {} increment changed: expected {expected:?}, got {actual:?}",
                branch.as_str(),
                population.as_str()
            )
            .into());
        }
    }
    Ok(())
}

fn c4_relational_ids(
    cases: &[DiagnosticCase],
    config: C4EmissionConfig,
    source: C4EmissionSource,
) -> BTreeSet<(Population, String)> {
    cases
        .iter()
        .filter_map(|case| {
            let decision = c4_decision_from_c31(
                case.c4_decision.c31.clone(),
                ALGORITHM_C2.threshold,
                config,
            );
            (decision.emission_source == source).then(|| (case.population, case.id.clone()))
        })
        .collect()
}

fn c4_increment_metrics(
    cases: &[DiagnosticCase],
    population: Population,
    branch: C4Branch,
) -> RuleDelta {
    let mut delta = RuleDelta::default();
    for case in cases_for_population(cases, population) {
        if branch.includes(case.c4_decision.emission_source) {
            observe_rule_delta(&mut delta, case);
        }
    }
    delta
}

fn write_c4_development_summary(output: &Path, cases: &[DiagnosticCase]) -> Result<()> {
    let mut writer = csv::Writer::from_path(output.join("c4_development_summary.csv"))?;
    writer.write_record([
        "population",
        "branch",
        "additional_correct",
        "additional_wrong",
        "additional_null_emissions",
    ])?;
    for population in Population::OUTPUTS {
        for branch in C4Branch::ALL {
            let delta = c4_increment_metrics(cases, population, branch);
            writer.write_record([
                population.as_str().to_string(),
                branch.as_str().to_string(),
                delta.correct.to_string(),
                delta.wrong.to_string(),
                delta.null_emissions.to_string(),
            ])?;
        }
    }
    writer.flush()?;
    Ok(())
}

fn qualitative_c4_diagnostics(
    corpus: &impl EvidenceSource,
) -> Vec<QualitativeC4Diagnostic> {
    [
        // Redacted: Name that selects its given name but fails both relational paths.
        "Olivier REDACTED",
        // Redacted: Name that selects its given name but fails both relational paths.
        "Baris REDACTED",
    ]
        .into_iter()
        .map(|input| qualitative_c4_diagnostic(corpus, input))
        .collect()
}

fn qualitative_c4_diagnostic(
    corpus: &impl EvidenceSource,
    input: &'static str,
) -> QualitativeC4Diagnostic {
    let diagnostic = diagnose_role_inference(corpus, ALGORITHM_C3, input, None, None);
    let decision = c4_decision_breakdown(
        &diagnostic,
        ALGORITHM_C2,
        ALGORITHM_C31,
        ALGORITHM_C4,
    );
    let winner = decision.c31.winner.as_ref();
    QualitativeC4Diagnostic {
        input,
        selected_candidate: winner.map(|winner| winner.greeting_candidate.clone()),
        decision_score: decision.c31.final_score,
        emission_source: decision.emission_source.as_str(),
        candidate_count: winner.map_or(0, |winner| winner.candidate_count),
        candidate_quality: winner.map(|winner| winner.winner_score),
        winner_margin: winner.map(|winner| winner.winner_margin),
        margin_signal: decision.c31.margin_signal,
        role_llr: winner.map(|winner| winner.role_llr),
        role_signal: winner.map(|winner| winner.role_signal),
        reliability: winner.map(|winner| winner.reliability),
        alphabetic_length: winner.map(|winner| winner.alphabetic_length),
        segmented_candidate: decision.c31.segmented_candidate,
        segmentation_mechanism: winner.and_then(|winner| winner.segmentation_mechanism),
        segmented_candidate_penalty: decision.c31.segmented_candidate_penalty,
        vetoes: QualitativeVetoes {
            strong_organization_marker: decision.c31.hard_organization_marker,
            generic_organization_marker: decision.c31.generic_organization_marker,
            ampersand: decision.c31.ampersand,
            candidate_too_short: decision.c31.candidate_too_short,
        },
        conditions: QualitativeConditions {
            sole_native: qualitative_rule_breakdown(decision.sole_native),
            dominant_winner: qualitative_rule_breakdown(decision.dominant_winner),
        },
    }
}

fn qualitative_rule_breakdown(rule: C4RuleBreakdown) -> QualitativeRuleBreakdown {
    QualitativeRuleBreakdown {
        c3_1_abstained: rule.c31_abstained,
        native_candidate: rule.native_candidate,
        candidate_count: rule.candidate_count,
        candidate_count_pass: rule.candidate_count_pass,
        candidate_quality: rule.candidate_quality,
        candidate_quality_min: rule.candidate_quality_min,
        candidate_quality_pass: rule.candidate_quality_pass,
        winner_margin: rule.winner_margin,
        winner_margin_min: rule.winner_margin_min,
        winner_margin_pass: rule.winner_margin_pass,
        reliability: rule.reliability,
        reliability_min: rule.reliability_min,
        reliability_pass: rule.reliability_pass,
        role_signal: rule.role_signal,
        role_signal_min: rule.role_signal_min,
        role_signal_pass: rule.role_signal_pass,
        vetoes_pass: rule.vetoes_pass,
        passed: rule.passed,
    }
}

fn build_c4_development_report(
    cases: &[DiagnosticCase],
    qualitative: &[QualitativeC4Diagnostic],
) -> Result<String> {
    let mut report = String::new();
    writeln!(report, "# C4 relational-emission development freeze\n")?;
    writeln!(
        report,
        "C4 is an additive emission policy over frozen C3.1. It does not change candidate generation, ranking, the selected winner, the C3.1 decision score or threshold, the segmented-candidate penalty, or any veto. This run used only spent REAL_PROXY_V1/V3/V4 plus synthetic VALIDATION; it did not load TEST or V5.\n"
    )?;
    writeln!(report, "## Frozen relational rules\n")?;
    writeln!(
        report,
        "- Sole native: `candidate_count == 1`, quality `>= 0.75`, reliability `>= 0.40`, role signal `>= 0.80`, native provenance, and all C3.1 vetoes passing.\n- Dominant winner: `candidate_count >= 2`, raw margin `>= 0.50`, quality `>= 0.40`, reliability `>= 0.75`, role signal `>= 0.40`, native provenance, and all C3.1 vetoes passing.\n"
    )?;
    writeln!(
        report,
        "The dominant quality floor is the searched-grid guardrail. Re-evaluation at `0.70` selected exactly the same rows, so candidate quality did not establish independent conditional discrimination for that branch.\n"
    )?;
    writeln!(report, "## Frozen baseline checkpoints\n")?;
    writeln!(
        report,
        "| Population | Algorithm | Emitted | Correct | Wrong | NULL FP |\n| --- | --- | ---: | ---: | ---: | ---: |"
    )?;
    for (population, expected_by_algorithm) in frozen_checkpoint_table() {
        for (algorithm, expected) in expected_by_algorithm {
            writeln!(
                report,
                "| {} | {} | {} | {} | {} | {} |",
                population.as_str(),
                algorithm.as_str(),
                expected.emitted,
                expected.correct,
                expected.wrong,
                expected.null_emissions
            )?;
        }
    }
    writeln!(report, "\n## C4 additions over C3.1\n")?;
    writeln!(
        report,
        "| Population | Branch | Correct | Wrong | NULL FP |\n| --- | --- | ---: | ---: | ---: |"
    )?;
    for population in Population::OUTPUTS {
        for branch in C4Branch::ALL {
            let delta = c4_increment_metrics(cases, population, branch);
            writeln!(
                report,
                "| {} | {} | {} | {} | {} |",
                population.as_str(),
                branch.as_str(),
                delta.correct,
                delta.wrong,
                delta.null_emissions
            )?;
        }
    }
    let spent_baseline = emission_metrics(
        cases_for_population(cases, Population::CombinedSpent),
        FrozenAlgorithm::C31,
    );
    let spent_delta = c4_increment_metrics(cases, Population::CombinedSpent, C4Branch::Combined);
    let validation_baseline = emission_metrics(
        cases_for_population(cases, Population::Validation),
        FrozenAlgorithm::C31,
    );
    let validation_delta = c4_increment_metrics(cases, Population::Validation, C4Branch::Combined);
    writeln!(
        report,
        "\nThe combined spent C4 checkpoint is {} emitted / {} correct / {} wrong / {} NULL FP. The VALIDATION checkpoint is {} emitted / {} correct / {} wrong / {} NULL FP. The sole and dominant additions are disjoint by candidate-count construction and were also checked case by case.\n",
        spent_baseline.emitted + spent_delta.correct + spent_delta.wrong,
        spent_baseline.correct + spent_delta.correct,
        spent_baseline.wrong + spent_delta.wrong,
        spent_baseline.null_emissions + spent_delta.null_emissions,
        validation_baseline.emitted + validation_delta.correct + validation_delta.wrong,
        validation_baseline.correct + validation_delta.correct,
        validation_baseline.wrong + validation_delta.wrong,
        validation_baseline.null_emissions + validation_delta.null_emissions,
    )?;
    writeln!(report, "## Non-selecting qualitative examples\n")?;
    for example in qualitative {
        writeln!(
            report,
            "- `{}` selected `{}` with C3.1 score `{:.9}`; frozen C4 source: `{}`.",
            example.input,
            example.selected_candidate.as_deref().unwrap_or("NULL"),
            example.decision_score,
            example.emission_source
        )?;
    }
    writeln!(
        report,
        "\nThese examples were evaluated only after the thresholds and reproduction assertions were fixed. They did not select or modify any rule. Full condition traces are in `c4_qualitative_diagnostics.json`.\n"
    )?;
    writeln!(report, "## Status and limits\n")?;
    writeln!(
        report,
        "C4 is frozen only as a development candidate. C3.1 remains the leading independently validated classifier until C4 receives one-shot evaluation on untouched REAL_PROXY_V5. Zero observed development errors over machine-generated or machine-consensus labels are not a worldwide precision claim."
    )?;
    Ok(report)
}

fn write_topology_outcomes(output: &Path, cases: &[DiagnosticCase]) -> Result<()> {
    let mut groups = BTreeMap::<(Population, Topology, Provenance), TopologyMetrics>::new();
    for (population, case) in cases_with_combined(cases) {
        let metrics = groups
            .entry((population, case.topology, case.provenance()))
            .or_default();
        observe_topology(metrics, case);
    }
    let mut writer = csv::Writer::from_path(output.join("topology_outcomes.csv"))?;
    writer.write_record([
        "population",
        "topology",
        "provenance",
        "rows",
        "expected_greetings",
        "expected_nulls",
        "correct_selected_winner",
        "incorrect_selected_winner",
        "c31_emitted_correct",
        "c31_emitted_wrong",
        "c31_null_false_emissions",
        "correct_winners_currently_abstained",
    ])?;
    for ((population, topology, provenance), metrics) in groups {
        writer.write_record([
            population.as_str().to_string(),
            topology.as_str().to_string(),
            provenance.as_str().to_string(),
            metrics.rows.to_string(),
            metrics.expected_greetings.to_string(),
            metrics.expected_nulls.to_string(),
            metrics.correct_winners.to_string(),
            metrics.wrong_winners.to_string(),
            metrics.c31_correct.to_string(),
            metrics.c31_wrong.to_string(),
            metrics.c31_null_emissions.to_string(),
            metrics.correct_winners_abstained.to_string(),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn observe_topology(metrics: &mut TopologyMetrics, case: &DiagnosticCase) {
    metrics.rows += 1;
    if case.expected_greeting.is_some() {
        metrics.expected_greetings += 1;
    } else {
        metrics.expected_nulls += 1;
    }
    match case.winner_outcome() {
        Some(WinnerOutcome::CorrectEmitted) => {
            metrics.correct_winners += 1;
            metrics.c31_correct += 1;
        }
        Some(WinnerOutcome::CorrectAbstained) => {
            metrics.correct_winners += 1;
            metrics.correct_winners_abstained += 1;
        }
        Some(WinnerOutcome::WrongWinner) => {
            metrics.wrong_winners += 1;
            if case.c31_emitted.is_some() {
                metrics.c31_wrong += 1;
            }
        }
        Some(WinnerOutcome::ExpectedNullWinner) => {
            if case.c31_emitted.is_some() {
                metrics.c31_null_emissions += 1;
            }
        }
        None => {}
    }
}

fn write_feature_percentiles(output: &Path, cases: &[DiagnosticCase]) -> Result<()> {
    let mut groups =
        BTreeMap::<(Population, WinnerOutcome, Provenance, &'static str), Vec<f64>>::new();
    for (population, case) in cases_with_combined(cases) {
        let (Some(outcome), Some(winner)) = (case.winner_outcome(), case.winner.as_ref()) else {
            continue;
        };
        for (feature, value) in [
            ("candidate_quality", winner.winner_score),
            ("winner_margin", winner.winner_margin),
            ("margin_signal", case.margin_signal.unwrap_or(0.0)),
            ("role_llr", winner.role_llr),
            ("role_signal", winner.role_signal),
            ("reliability", winner.reliability),
            ("alphabetic_length", winner.alphabetic_length as f64),
            ("viable_candidate_count", case.viable_candidates as f64),
        ] {
            groups
                .entry((population, outcome, case.provenance(), feature))
                .or_default()
                .push(value);
        }
    }
    let mut writer = csv::Writer::from_path(output.join("feature_percentiles.csv"))?;
    writer.write_record([
        "population",
        "winner_outcome",
        "provenance",
        "feature",
        "count",
        "p10",
        "p25",
        "median",
        "p75",
        "p90",
    ])?;
    for ((population, outcome, provenance, feature), mut values) in groups {
        values.sort_by(f64::total_cmp);
        writer.write_record([
            population.as_str().to_string(),
            outcome.as_str().to_string(),
            provenance.as_str().to_string(),
            feature.to_string(),
            values.len().to_string(),
            format_float(nearest_rank(&values, 0.10)),
            format_float(nearest_rank(&values, 0.25)),
            format_float(nearest_rank(&values, 0.50)),
            format_float(nearest_rank(&values, 0.75)),
            format_float(nearest_rank(&values, 0.90)),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn nearest_rank(values: &[f64], percentile: f64) -> f64 {
    assert!(!values.is_empty());
    let index = ((percentile * values.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(values.len() - 1);
    values[index]
}

fn write_feature_categories(output: &Path, cases: &[DiagnosticCase]) -> Result<()> {
    let mut groups =
        BTreeMap::<(Population, WinnerOutcome, &'static str, String), (usize, usize)>::new();
    for (population, case) in cases_with_combined(cases) {
        let Some(outcome) = case.winner_outcome() else {
            continue;
        };
        for (dimension, value) in [
            ("provenance", case.provenance().as_str().to_string()),
            (
                "country_or_locale_hint",
                if case.hint_present() {
                    "present"
                } else {
                    "absent"
                }
                .to_string(),
            ),
            (
                "viable_candidate_count",
                if case.viable_candidates >= 5 {
                    "5+".to_string()
                } else {
                    case.viable_candidates.to_string()
                },
            ),
        ] {
            groups
                .entry((population, outcome, dimension, value))
                .or_default()
                .0 += 1;
        }
    }
    let mut totals = BTreeMap::<(Population, WinnerOutcome, &'static str), usize>::new();
    for ((population, outcome, dimension, _), (count, _)) in &groups {
        *totals
            .entry((*population, *outcome, *dimension))
            .or_default() += count;
    }
    let mut writer = csv::Writer::from_path(output.join("feature_categories.csv"))?;
    writer.write_record([
        "population",
        "winner_outcome",
        "dimension",
        "value",
        "count",
        "fraction",
    ])?;
    for ((population, outcome, dimension, value), (count, _)) in groups {
        let total = totals[&(population, outcome, dimension)];
        writer.write_record([
            population.as_str().to_string(),
            outcome.as_str().to_string(),
            dimension.to_string(),
            value,
            count.to_string(),
            format_float(count as f64 / total as f64),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn write_country_evidence_audit(output: &Path, cases: &[DiagnosticCase]) -> Result<()> {
    let mut writer = csv::Writer::from_path(output.join("country_evidence_audit.csv"))?;
    writer.write_record([
        "population",
        "provenance",
        "hint_rows",
        "comparable_rows",
        "winner_changed",
        "quality_abs_delta_ge_0.01",
        "quality_abs_delta_ge_0.05",
        "quality_abs_delta_ge_0.10",
        "correct_abstained_quality_abs_delta_ge_0.05",
        "quality_delta_p10",
        "quality_delta_p25",
        "quality_delta_median",
        "quality_delta_p75",
        "quality_delta_p90",
        "final_score_delta_p10",
        "final_score_delta_p25",
        "final_score_delta_median",
        "final_score_delta_p75",
        "final_score_delta_p90",
    ])?;
    for population in Population::OUTPUTS {
        for provenance in [
            Provenance::Native,
            Provenance::HandleSegment,
            Provenance::None,
        ] {
            let selected = cases_for_population(cases, population)
                .filter(|case| case.provenance() == provenance && case.country_audit.is_some())
                .collect::<Vec<_>>();
            if selected.is_empty() {
                continue;
            }
            let comparable = selected
                .iter()
                .filter(|case| {
                    case.country_audit
                        .as_ref()
                        .is_some_and(|audit| audit.comparable)
                })
                .collect::<Vec<_>>();
            let mut quality = comparable
                .iter()
                .filter_map(|case| case.country_audit.as_ref()?.quality_delta)
                .collect::<Vec<_>>();
            let mut final_scores = selected
                .iter()
                .filter_map(|case| {
                    case.country_audit
                        .as_ref()
                        .map(|audit| audit.final_score_delta)
                })
                .collect::<Vec<_>>();
            quality.sort_by(f64::total_cmp);
            final_scores.sort_by(f64::total_cmp);
            writer.write_record([
                population.as_str().to_string(),
                provenance.as_str().to_string(),
                selected.len().to_string(),
                comparable.len().to_string(),
                selected
                    .iter()
                    .filter(|case| {
                        case.country_audit
                            .as_ref()
                            .is_some_and(|audit| audit.winner_changed)
                    })
                    .count()
                    .to_string(),
                count_absolute_at_least(&quality, 0.01).to_string(),
                count_absolute_at_least(&quality, 0.05).to_string(),
                count_absolute_at_least(&quality, 0.10).to_string(),
                comparable
                    .iter()
                    .filter(|case| {
                        case.winner_outcome() == Some(WinnerOutcome::CorrectAbstained)
                            && case
                                .country_audit
                                .as_ref()
                                .and_then(|audit| audit.quality_delta)
                                .is_some_and(|delta| delta.abs() >= 0.05)
                    })
                    .count()
                    .to_string(),
                format_percentile(&quality, 0.10),
                format_percentile(&quality, 0.25),
                format_percentile(&quality, 0.50),
                format_percentile(&quality, 0.75),
                format_percentile(&quality, 0.90),
                format_percentile(&final_scores, 0.10),
                format_percentile(&final_scores, 0.25),
                format_percentile(&final_scores, 0.50),
                format_percentile(&final_scores, 0.75),
                format_percentile(&final_scores, 0.90),
            ])?;
        }
    }
    writer.flush()?;
    Ok(())
}

fn count_absolute_at_least(values: &[f64], threshold: f64) -> usize {
    values
        .iter()
        .filter(|value| value.abs() >= threshold)
        .count()
}

fn format_percentile(values: &[f64], percentile: f64) -> String {
    if values.is_empty() {
        String::new()
    } else {
        format_float(nearest_rank(values, percentile))
    }
}

fn evaluate_operating_points(cases: &[DiagnosticCase]) -> Vec<RuleEvaluation> {
    let mut evaluations = Vec::new();
    for family in RuleFamily::ALL {
        let margins = if family == RuleFamily::Sole {
            vec![None]
        } else {
            MARGINS.into_iter().map(Some).collect()
        };
        for margin in margins {
            for quality_step in QUALITY_STEPS {
                for reliability_step in QUALITY_STEPS {
                    for role_step in QUALITY_STEPS {
                        let rule = Rule {
                            family,
                            quality: quality_step as f64 * 0.05,
                            reliability: reliability_step as f64 * 0.05,
                            role: role_step as f64 * 0.05,
                            margin,
                        };
                        evaluations.push(evaluate_rule(cases, rule));
                    }
                }
            }
        }
    }
    evaluations
}

fn evaluate_rule(cases: &[DiagnosticCase], rule: Rule) -> RuleEvaluation {
    let mut deltas = Population::OUTPUTS
        .into_iter()
        .map(|population| (population, RuleDelta::default()))
        .collect::<BTreeMap<_, _>>();
    for case in cases {
        if !rule_applies(case, rule) {
            continue;
        }
        observe_rule_delta(deltas.get_mut(&case.population).expect("population"), case);
        if case.population.is_spent() {
            observe_rule_delta(
                deltas
                    .get_mut(&Population::CombinedSpent)
                    .expect("combined population"),
                case,
            );
        }
    }
    RuleEvaluation { rule, deltas }
}

fn rule_applies(case: &DiagnosticCase, rule: Rule) -> bool {
    if case.c31_emitted.is_some() || !case.vetoes_pass || case.provenance() != Provenance::Native {
        return false;
    }
    let Some(winner) = &case.winner else {
        return false;
    };
    if winner.winner_score < rule.quality
        || winner.reliability < rule.reliability
        || winner.role_signal < rule.role
    {
        return false;
    }
    match rule.family {
        RuleFamily::Sole => case.viable_candidates == 1,
        RuleFamily::Dominant => {
            case.viable_candidates >= 2 && winner.winner_margin >= rule.margin.expect("margin")
        }
        RuleFamily::Combined => {
            case.viable_candidates == 1
                || (case.viable_candidates >= 2
                    && winner.winner_margin >= rule.margin.expect("margin"))
        }
    }
}

fn observe_rule_delta(delta: &mut RuleDelta, case: &DiagnosticCase) {
    match case.winner_outcome() {
        Some(WinnerOutcome::CorrectAbstained) => delta.correct += 1,
        Some(WinnerOutcome::WrongWinner) => delta.wrong += 1,
        Some(WinnerOutcome::ExpectedNullWinner) => delta.null_emissions += 1,
        Some(WinnerOutcome::CorrectEmitted) | None => {
            unreachable!("rules only apply to abstained winners")
        }
    }
}

fn write_operating_points(output: &Path, evaluations: &[RuleEvaluation]) -> Result<()> {
    let mut writer = csv::Writer::from_path(output.join("relational_operating_points.csv"))?;
    writer.write_record([
        "family",
        "candidate_quality_min",
        "reliability_min",
        "role_signal_min",
        "winner_margin_min",
        "provenance",
        "population",
        "additional_correct",
        "additional_wrong",
        "additional_null_emissions",
    ])?;
    for evaluation in evaluations {
        for population in Population::OUTPUTS {
            write_rule_record(&mut writer, evaluation, population, None)?;
        }
    }
    writer.flush()?;
    Ok(())
}

fn select_operating_points(evaluations: &[RuleEvaluation]) -> Vec<SelectedPoint> {
    let mut selected = Vec::new();
    for family in RuleFamily::ALL {
        for kind in [
            SelectionKind::ZeroError,
            SelectionKind::OneWrong,
            SelectionKind::OneNull,
        ] {
            if let Some(evaluation) = evaluations
                .iter()
                .filter(|evaluation| {
                    evaluation.rule.family == family && selection_matches(evaluation, kind)
                })
                .max_by(|left, right| compare_operating_points(left, right))
            {
                selected.push(SelectedPoint {
                    kind,
                    evaluation: evaluation.clone(),
                });
            }
        }
    }
    selected
}

fn selection_matches(evaluation: &RuleEvaluation, kind: SelectionKind) -> bool {
    let spent = evaluation.deltas[&Population::CombinedSpent];
    let validation = evaluation.deltas[&Population::Validation];
    if validation.wrong != 0 || validation.null_emissions != 0 {
        return false;
    }
    match kind {
        SelectionKind::ZeroError => spent.wrong == 0 && spent.null_emissions == 0,
        SelectionKind::OneWrong => spent.wrong == 1 && spent.null_emissions == 0,
        SelectionKind::OneNull => spent.wrong == 0 && spent.null_emissions == 1,
    }
}

fn compare_operating_points(left: &RuleEvaluation, right: &RuleEvaluation) -> Ordering {
    let left_spent = left.deltas[&Population::CombinedSpent];
    let right_spent = right.deltas[&Population::CombinedSpent];
    let left_validation = left.deltas[&Population::Validation];
    let right_validation = right.deltas[&Population::Validation];
    left_spent
        .correct
        .cmp(&right_spent.correct)
        .then_with(|| left_validation.correct.cmp(&right_validation.correct))
        .then_with(|| left.rule.quality.total_cmp(&right.rule.quality))
        .then_with(|| left.rule.reliability.total_cmp(&right.rule.reliability))
        .then_with(|| left.rule.role.total_cmp(&right.rule.role))
        .then_with(|| {
            left.rule
                .margin
                .unwrap_or(1.0)
                .total_cmp(&right.rule.margin.unwrap_or(1.0))
        })
        .then_with(|| left.rule.family.cmp(&right.rule.family))
}

fn write_selected_points(output: &Path, selected: &[SelectedPoint]) -> Result<()> {
    let mut writer = csv::Writer::from_path(output.join("relational_selected_points.csv"))?;
    writer.write_record([
        "selection",
        "family",
        "candidate_quality_min",
        "reliability_min",
        "role_signal_min",
        "winner_margin_min",
        "provenance",
        "population",
        "additional_correct",
        "additional_wrong",
        "additional_null_emissions",
    ])?;
    for point in selected {
        for population in Population::OUTPUTS {
            write_rule_record(&mut writer, &point.evaluation, population, Some(point.kind))?;
        }
    }
    writer.flush()?;
    Ok(())
}

fn write_rule_record<W: std::io::Write>(
    writer: &mut csv::Writer<W>,
    evaluation: &RuleEvaluation,
    population: Population,
    selection: Option<SelectionKind>,
) -> Result<()> {
    let delta = evaluation.deltas[&population];
    let mut record = Vec::new();
    if let Some(selection) = selection {
        record.push(selection.as_str().to_string());
    }
    record.extend([
        evaluation.rule.family.as_str().to_string(),
        format_float(evaluation.rule.quality),
        format_float(evaluation.rule.reliability),
        format_float(evaluation.rule.role),
        evaluation.rule.margin.map(format_float).unwrap_or_default(),
        Provenance::Native.as_str().to_string(),
        population.as_str().to_string(),
        delta.correct.to_string(),
        delta.wrong.to_string(),
        delta.null_emissions.to_string(),
    ]);
    writer.write_record(record)?;
    Ok(())
}

fn write_qualitative_review_sample(output: &Path, cases: &[DiagnosticCase]) -> Result<()> {
    let mut selected = Vec::<(&'static str, [u8; 32], &DiagnosticCase)>::new();
    for (stratum, predicate) in [
        (
            "sole_correct_abstention",
            is_sole_correct_abstention as fn(&DiagnosticCase) -> bool,
        ),
        (
            "large_margin_correct_abstention",
            is_large_margin_correct_abstention,
        ),
        ("large_margin_wrong_winner", is_large_margin_wrong_winner),
        (
            "strong_relational_expected_null",
            is_strong_relational_expected_null,
        ),
    ] {
        for provenance in [Provenance::Native, Provenance::HandleSegment] {
            let mut stratum_cases = cases
                .iter()
                .filter(|case| {
                    case.population.is_spent() && case.provenance() == provenance && predicate(case)
                })
                .map(|case| (sample_key(case, stratum), case))
                .collect::<Vec<_>>();
            stratum_cases.sort_by_key(|(key, _)| *key);
            selected.extend(
                stratum_cases
                    .into_iter()
                    .take(REVIEW_LIMIT)
                    .map(|(key, case)| (stratum, key, case)),
            );
        }
    }
    selected.sort_by(|left, right| {
        (left.0, left.2.provenance(), left.1).cmp(&(right.0, right.2.provenance(), right.1))
    });
    let mut writer = csv::Writer::from_path(output.join("qualitative_review_sample.csv"))?;
    writer.write_record([
        "stratum",
        "population",
        "holdout_sha256",
        "id",
        "provenance",
        "display_name",
        "country_hint",
        "locale_hint",
        "expected_greeting",
        "selected_winner",
        "viable_candidates",
        "candidate_quality",
        "winner_margin",
        "role_signal",
        "reliability",
        "c31_score",
    ])?;
    for (stratum, _, case) in selected {
        let winner = case.winner.as_ref().expect("sampled winner");
        writer.write_record([
            stratum.to_string(),
            case.population.as_str().to_string(),
            case.holdout_digest.clone().unwrap_or_default(),
            case.id.clone(),
            case.provenance().as_str().to_string(),
            case.display_name.clone(),
            case.country_hint.clone().unwrap_or_default(),
            case.locale_hint.clone().unwrap_or_default(),
            case.expected_greeting.clone().unwrap_or_default(),
            winner.greeting_candidate.clone(),
            case.viable_candidates.to_string(),
            format_float(winner.winner_score),
            format_float(winner.winner_margin),
            format_float(winner.role_signal),
            format_float(winner.reliability),
            format_float(case.c31_score),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn is_sole_correct_abstention(case: &DiagnosticCase) -> bool {
    case.topology == Topology::Sole
        && case.winner_outcome() == Some(WinnerOutcome::CorrectAbstained)
}

fn is_large_margin_correct_abstention(case: &DiagnosticCase) -> bool {
    case.viable_candidates >= 2
        && case
            .winner
            .as_ref()
            .is_some_and(|winner| winner.winner_margin >= LARGE_REVIEW_MARGIN)
        && case.winner_outcome() == Some(WinnerOutcome::CorrectAbstained)
}

fn is_large_margin_wrong_winner(case: &DiagnosticCase) -> bool {
    case.winner
        .as_ref()
        .is_some_and(|winner| winner.winner_margin >= LARGE_REVIEW_MARGIN)
        && case.winner_outcome() == Some(WinnerOutcome::WrongWinner)
}

fn is_strong_relational_expected_null(case: &DiagnosticCase) -> bool {
    (case.viable_candidates == 1
        || case
            .winner
            .as_ref()
            .is_some_and(|winner| winner.winner_margin >= LARGE_REVIEW_MARGIN))
        && case.winner_outcome() == Some(WinnerOutcome::ExpectedNullWinner)
}

fn sample_key(case: &DiagnosticCase, stratum: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(case.holdout_digest.as_deref().unwrap_or_default());
    hasher.update([0]);
    hasher.update(stratum);
    hasher.update([0]);
    hasher.update(case.provenance().as_str());
    hasher.update([0]);
    hasher.update(&case.id);
    hasher.finalize().into()
}

fn build_report(
    cases: &[DiagnosticCase],
    evaluations: &[RuleEvaluation],
    selected: &[SelectedPoint],
) -> Result<String> {
    let mut report = String::new();
    writeln!(report, "# Relational emission diagnosis before C4\n")?;
    writeln!(
        report,
        "This diagnostic uses only spent REAL_PROXY_V1/V3/V4 plus synthetic VALIDATION. Frozen C3.1 is unchanged; no TEST, V5, or fresh holdout was loaded, and no C4 was implemented.\n"
    )?;
    writeln!(report, "## Frozen C3.1 checkpoints\n")?;
    writeln!(
        report,
        "| Population | Emitted | Correct | Wrong | NULL emissions |"
    )?;
    writeln!(report, "| --- | ---: | ---: | ---: | ---: |")?;
    for population in Population::OUTPUTS {
        let metrics = emission_metrics(
            cases_for_population(cases, population),
            FrozenAlgorithm::C31,
        );
        writeln!(
            report,
            "| {} | {} | {} | {} | {} |",
            population.as_str(),
            metrics.emitted,
            metrics.correct,
            metrics.wrong,
            metrics.null_emissions,
        )?;
    }
    writeln!(report, "\n## Best monotonic operating points\n")?;
    writeln!(
        report,
        "All candidate rules are additive, native-only, and retain every existing C3.1 veto. Handle-segmented winners remain governed solely by frozen C3.1.\n"
    )?;
    writeln!(
        report,
        "| Class | Family | Q | R | L | M | Spent +correct | Spent +wrong | Spent +NULL | Validation +correct |"
    )?;
    writeln!(
        report,
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )?;
    for point in selected {
        let spent = point.evaluation.deltas[&Population::CombinedSpent];
        let validation = point.evaluation.deltas[&Population::Validation];
        writeln!(
            report,
            "| {} | {} | {:.2} | {:.2} | {:.2} | {} | {} | {} | {} | {} |",
            point.kind.as_str(),
            point.evaluation.rule.family.as_str(),
            point.evaluation.rule.quality,
            point.evaluation.rule.reliability,
            point.evaluation.rule.role,
            point
                .evaluation
                .rule
                .margin
                .map_or_else(|| "n/a".to_string(), |margin| format!("{margin:.2}")),
            spent.correct,
            spent.wrong,
            spent.null_emissions,
            validation.correct,
        )?;
    }
    let best_sole = selected_point(selected, SelectionKind::ZeroError, RuleFamily::Sole);
    let best_dominant = selected_point(selected, SelectionKind::ZeroError, RuleFamily::Dominant);
    let best_overall = best_zero_error_point(selected);
    writeln!(report, "\n## Answers\n")?;
    write_answer(
        &mut report,
        1,
        "Sole-candidate status",
        best_sole,
        "sole-candidate",
    )?;
    write_answer(
        &mut report,
        2,
        "Large winner margin",
        best_dominant,
        "dominant-winner",
    )?;
    write_quality_answer(&mut report, best_overall, evaluations)?;
    let spent_country = country_summary(cases, Population::CombinedSpent);
    let validation_country = country_summary(cases, Population::Validation);
    writeln!(
        report,
        "4. **Country-aware evidence:** COMBINED_SPENT supplies {} hint-bearing rows, so the proxy sets cannot measure this effect. On VALIDATION, {} of {} hint-bearing rows are comparable; {} change candidate quality by at least 0.05, including {} currently abstained correct winners. The VALIDATION median final-score delta is {}, showing that country-aware quality can move while the frozen emission score usually does not.",
        spent_country.hint_rows,
        validation_country.comparable,
        validation_country.hint_rows,
        validation_country.material_quality,
        validation_country.material_correct_abstained,
        format_float(validation_country.median_final_delta),
    )?;
    if let Some(point) = best_overall {
        let spent = point.deltas[&Population::CombinedSpent];
        writeln!(
            report,
            "5. **Zero-error relational rule:** the best grid point overall is `{}` and recovers {} spent-proxy correct greetings with zero observed new wrong and NULL emissions while adding {} VALIDATION correct greetings without an observed validation error.",
            point.rule.family.as_str(),
            spent.correct,
            point.deltas[&Population::Validation].correct,
        )?;
        write_c4_candidate(&mut report, point)?;
    } else {
        writeln!(
            report,
            "5. **Zero-error relational rule:** none exists on the documented grid."
        )?;
        writeln!(
            report,
            "6. **C4 development candidate:** none; stop without creating C4."
        )?;
    }
    writeln!(report, "\n## Interpretation limits\n")?;
    writeln!(
        report,
        "V1/V3/V4 labels are machine-generated or machine-consensus proxy evidence, and agreement filtering selects clearer cases. Grid search over spent data can overfit. Zero observed errors is not a worldwide precision claim. Sole-candidate status also means sole candidate under the current corpus and scorer, not under future scoring layers."
    )?;
    Ok(report)
}

fn write_quality_answer(
    report: &mut String,
    best: Option<&RuleEvaluation>,
    evaluations: &[RuleEvaluation],
) -> Result<()> {
    let Some(best) = best else {
        writeln!(
            report,
            "3. **Conditional candidate quality:** no zero-error relational point exists, so the diagnostic establishes no conditional value."
        )?;
        return Ok(());
    };
    let floor = evaluations.iter().find(|evaluation| {
        evaluation.rule.family == best.rule.family
            && (evaluation.rule.quality - 0.40).abs() < f64::EPSILON
            && evaluation.rule.reliability == best.rule.reliability
            && evaluation.rule.role == best.rule.role
            && evaluation.rule.margin == best.rule.margin
    });
    if floor.is_some_and(|floor| floor.deltas == best.deltas) {
        writeln!(
            report,
            "3. **Conditional candidate quality:** lowering Q from {:.2} to the searched floor `0.40` changes no emission or outcome at the selected point. Candidate quality therefore shows no independent conditional value within this grid; the {:.2} floor is only the conservative tie-break.",
            best.rule.quality, best.rule.quality,
        )?;
    } else {
        writeln!(
            report,
            "3. **Conditional candidate quality:** the selected Q floor changes the operating-point outcomes relative to Q `0.40`, so candidate quality contributes conditionally within this grid."
        )?;
    }
    Ok(())
}

fn selected_point(
    selected: &[SelectedPoint],
    kind: SelectionKind,
    family: RuleFamily,
) -> Option<&RuleEvaluation> {
    selected
        .iter()
        .find(|point| point.kind == kind && point.evaluation.rule.family == family)
        .map(|point| &point.evaluation)
}

fn best_zero_error_point(selected: &[SelectedPoint]) -> Option<&RuleEvaluation> {
    selected
        .iter()
        .filter(|point| point.kind == SelectionKind::ZeroError)
        .map(|point| &point.evaluation)
        .max_by(|left, right| compare_operating_points(left, right))
}

fn write_c4_candidate(report: &mut String, point: &RuleEvaluation) -> Result<()> {
    let topology = match point.rule.family {
        RuleFamily::Sole => "candidate_count == 1".to_string(),
        RuleFamily::Dominant => format!(
            "candidate_count >= 2 && margin >= {:.2}",
            point.rule.margin.expect("dominant margin")
        ),
        RuleFamily::Combined => format!(
            "candidate_count == 1 || (candidate_count >= 2 && margin >= {:.2})",
            point.rule.margin.expect("combined margin")
        ),
    };
    writeln!(
        report,
        "6. **C4 development candidate:** native winner; `({topology}) && quality >= {:.2} && reliability >= {:.2} && role_signal >= {:.2}` after all frozen vetoes. This is spent development evidence and requires untouched V5 validation before promotion.",
        point.rule.quality, point.rule.reliability, point.rule.role,
    )?;
    Ok(())
}

fn write_answer(
    report: &mut String,
    number: usize,
    title: &str,
    point: Option<&RuleEvaluation>,
    rule_name: &str,
) -> Result<()> {
    if let Some(point) = point {
        writeln!(
            report,
            "{number}. **{title}:** the best zero-error {rule_name} grid point recovers {} additional correct COMBINED_SPENT greetings and {} VALIDATION greetings.",
            point.deltas[&Population::CombinedSpent].correct,
            point.deltas[&Population::Validation].correct,
        )?;
    } else {
        writeln!(
            report,
            "{number}. **{title}:** no zero-error {rule_name} operating point exists on the documented grid."
        )?;
    }
    Ok(())
}

#[derive(Default)]
struct CountrySummary {
    hint_rows: usize,
    comparable: usize,
    material_quality: usize,
    material_correct_abstained: usize,
    final_deltas: Vec<f64>,
    median_final_delta: f64,
}

fn country_summary(cases: &[DiagnosticCase], population: Population) -> CountrySummary {
    let mut summary = CountrySummary::default();
    for case in cases_for_population(cases, population) {
        let Some(audit) = &case.country_audit else {
            continue;
        };
        summary.hint_rows += 1;
        summary.final_deltas.push(audit.final_score_delta);
        if let Some(delta) = audit.quality_delta {
            summary.comparable += 1;
            if delta.abs() >= 0.05 {
                summary.material_quality += 1;
                if case.winner_outcome() == Some(WinnerOutcome::CorrectAbstained) {
                    summary.material_correct_abstained += 1;
                }
            }
        }
    }
    summary.final_deltas.sort_by(f64::total_cmp);
    if !summary.final_deltas.is_empty() {
        summary.median_final_delta = nearest_rank(&summary.final_deltas, 0.50);
    }
    summary
}

fn cases_with_combined(
    cases: &[DiagnosticCase],
) -> impl Iterator<Item = (Population, &DiagnosticCase)> {
    cases.iter().flat_map(|case| {
        let combined = case
            .population
            .is_spent()
            .then_some((Population::CombinedSpent, case));
        std::iter::once((case.population, case)).chain(combined)
    })
}

fn cases_for_population(
    cases: &[DiagnosticCase],
    population: Population,
) -> impl Iterator<Item = &DiagnosticCase> {
    cases.iter().filter(move |case| {
        if population == Population::CombinedSpent {
            case.population.is_spent()
        } else {
            case.population == population
        }
    })
}

fn format_float(value: f64) -> String {
    format!("{value:.9}")
}

fn nonempty(value: &str) -> Option<&str> {
    (!value.is_empty()).then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn winner(origin: &'static str, candidate_count: usize) -> WinnerFeatures {
        WinnerFeatures {
            greeting_candidate: "Example".to_string(),
            winner_score: 0.8,
            second_score: (candidate_count > 1).then_some(0.4),
            winner_margin: if candidate_count > 1 { 0.4 } else { 1.0 },
            no_competitor: candidate_count == 1,
            role_llr: 2.0,
            role_signal: 0.8,
            reliability: 0.8,
            global_given_count: 1_000,
            global_surname_count: 10,
            candidate_origin: origin,
            segmentation_mechanism: (origin == "handle_segment").then_some("digit"),
            candidate_count,
            alphabetic_length: 7,
            generic_organization_marker: false,
            ampersand_negative_evidence: false,
        }
    }

    fn c4_decision_from_test(
        origin: &'static str,
        candidate_count: usize,
    ) -> C4DecisionBreakdown {
        let winner = winner(origin, candidate_count);
        c4_decision_from_c31(
            C31DecisionBreakdown {
                segmented_candidate: Some(origin == "handle_segment"),
                winner: Some(winner),
                margin_signal: Some(0.8),
                contributions: None,
                pre_veto_score: Some(0.5),
                post_veto_score: 0.5,
                hard_organization_marker: false,
                generic_organization_marker: false,
                ampersand: false,
                candidate_too_short: false,
                segmented_candidate_penalty: 0.0,
                final_score: 0.5,
            },
            ALGORITHM_C2.threshold,
            ALGORITHM_C4,
        )
    }

    fn diagnostic_case_for_test(origin: &'static str, candidate_count: usize) -> DiagnosticCase {
        DiagnosticCase {
            population: Population::V1,
            holdout_digest: Some(V1_SHA256.to_string()),
            id: "case".to_string(),
            display_name: "Example Person".to_string(),
            country_hint: None,
            locale_hint: None,
            expected_greeting: Some("Example".to_string()),
            topology: if candidate_count == 1 {
                Topology::Sole
            } else {
                Topology::Margin30To50
            },
            viable_candidates: candidate_count,
            winner: Some(winner(origin, candidate_count)),
            margin_signal: Some(0.8),
            c2_emitted: None,
            c3_emitted: None,
            c31_score: 0.5,
            c31_emitted: None,
            c4_decision: c4_decision_from_test(origin, candidate_count),
            vetoes_pass: true,
            country_audit: None,
        }
    }

    #[test]
    fn topology_boundaries_are_half_open() {
        assert_eq!(Topology::from_margin(0.099_999), Topology::Margin00To10);
        assert_eq!(Topology::from_margin(0.10), Topology::Margin10To20);
        assert_eq!(Topology::from_margin(0.20), Topology::Margin20To30);
        assert_eq!(Topology::from_margin(0.30), Topology::Margin30To50);
        assert_eq!(Topology::from_margin(0.50), Topology::Margin50Plus);
    }

    #[test]
    fn nearest_rank_is_deterministic() {
        let values = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(nearest_rank(&values, 0.10), 1.0);
        assert_eq!(nearest_rank(&values, 0.50), 2.0);
        assert_eq!(nearest_rank(&values, 0.90), 4.0);
    }

    #[test]
    fn documented_search_grid_has_fixed_endpoints() {
        assert_eq!(*QUALITY_STEPS.start(), 8);
        assert_eq!(*QUALITY_STEPS.end(), 19);
        assert_eq!(MARGINS, [0.10, 0.20, 0.30, 0.50]);
    }

    #[test]
    fn winner_outcomes_distinguish_emission_abstention_wrong_and_null() {
        let mut case = diagnostic_case_for_test("exact", 1);
        assert_eq!(case.winner_outcome(), Some(WinnerOutcome::CorrectAbstained));
        case.c31_emitted = Some("Example".to_string());
        assert_eq!(case.winner_outcome(), Some(WinnerOutcome::CorrectEmitted));
        case.expected_greeting = Some("Other".to_string());
        assert_eq!(case.winner_outcome(), Some(WinnerOutcome::WrongWinner));
        case.expected_greeting = None;
        assert_eq!(
            case.winner_outcome(),
            Some(WinnerOutcome::ExpectedNullWinner)
        );
        case.winner = None;
        assert_eq!(case.winner_outcome(), None);
    }

    #[test]
    fn rule_is_additive_native_only_and_respects_vetoes() {
        let rule = Rule {
            family: RuleFamily::Sole,
            quality: 0.7,
            reliability: 0.7,
            role: 0.7,
            margin: None,
        };
        let native = diagnostic_case_for_test("exact", 1);
        assert!(rule_applies(&native, rule));
        let mut emitted = native.clone();
        emitted.c31_emitted = Some("Example".to_string());
        assert!(!rule_applies(&emitted, rule));
        let mut vetoed = native.clone();
        vetoed.vetoes_pass = false;
        assert!(!rule_applies(&vetoed, rule));
        let segmented = diagnostic_case_for_test("handle_segment", 1);
        assert!(!rule_applies(&segmented, rule));
    }

    #[test]
    fn operating_point_tie_prefers_stricter_thresholds() {
        let cases = [diagnostic_case_for_test("exact", 1)];
        let loose = evaluate_rule(
            &cases,
            Rule {
                family: RuleFamily::Sole,
                quality: 0.6,
                reliability: 0.6,
                role: 0.6,
                margin: None,
            },
        );
        let strict = evaluate_rule(
            &cases,
            Rule {
                family: RuleFamily::Sole,
                quality: 0.7,
                reliability: 0.7,
                role: 0.7,
                margin: None,
            },
        );
        assert_eq!(compare_operating_points(&strict, &loose), Ordering::Greater);
    }

    #[test]
    fn sample_key_depends_on_stratum_and_provenance() {
        let native = diagnostic_case_for_test("exact", 1);
        let segmented = diagnostic_case_for_test("handle_segment", 1);
        assert_ne!(sample_key(&native, "a"), sample_key(&native, "b"));
        assert_ne!(sample_key(&native, "a"), sample_key(&segmented, "a"));
    }
}
