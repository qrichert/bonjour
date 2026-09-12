use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::Path;

use name_eval::holdout::FrozenHoldout;
use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::ordering::{NameOrderPrior, resolve_name_order_prior};
use super::{Population, Result, greeting_matches, validate_and_order_holdouts};
use crate::artifact::EvidenceSource;
use crate::classifier::{
    ALGORITHM_C2, ALGORITHM_C3, ALGORITHM_C4, ALGORITHM_C5, ALGORITHM_C31, C4DecisionBreakdown,
    CandidateDiagnostic, c4_decision_breakdown, c5_decision_from_c4, c5_emitted_candidate,
    canonicalize, diagnose_role_inference, expected_lookup_diagnostic, role_llr,
};
use crate::dataset::{Case, Split, generate_cases};

const QUALITY_THRESHOLDS: [f64; 9] = [0.40, 0.50, 0.55, 0.60, 0.625, 0.65, 0.675, 0.70, 0.75];
const RELIABILITY_THRESHOLDS: [f64; 9] = [0.0, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90];
const ROLE_THRESHOLDS: [f64; 9] = [0.0, 0.20, 0.30, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80];
const ERROR_BUDGETS: [usize; 5] = [0, 1, 5, 10, 25];
const SURNAME_THRESHOLDS: [u64; 10] = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1_000];
const RAW_FILES: usize = 105;
const RAW_ROWS: u64 = 491_655_925;
const GIVEN_TOTAL: u64 = 444_154_759;
const SURNAME_TOTAL: u64 = 489_631_377;
const MPHF_BYTES_PER_KEY: f64 = 777_304.0 / 1_803_175.0;
const BLOOM_FALSE_POSITIVE_RATE: f64 = 0.001;
const STORAGE_ZSTD19_BYTES: [usize; 10] = [1_789, 1_575, 1_259, 1_091, 766, 631, 463, 275, 148, 95];
const FULL_SURNAME_ONLY_KEY_COUNTS: [usize; 10] = [
    35_417_044, 10_271_119, 3_390_945, 1_709_397, 684_930, 320_718, 135_907, 33_385, 8_979, 2_117,
];
const FULL_SURNAME_ONLY_DIRECT_BYTES: [usize; 10] = [
    467_302_147,
    122_230_732,
    37_111_326,
    18_235_331,
    7_177_753,
    3_329_972,
    1_403_589,
    339_910,
    91_447,
    22_358,
];
const FULL_SURNAME_ONLY_ZSTD19_BYTES: [usize; 10] = [
    114_865_997,
    28_703_435,
    8_669_513,
    4_385_703,
    1_820_637,
    888_660,
    399_789,
    110_145,
    32_714,
    8_633,
];

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Outcome {
    CorrectWinner,
    WrongWinner,
    ExpectedNull,
}

impl Outcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::CorrectWinner => "correct_winner",
            Self::WrongWinner => "wrong_winner",
            Self::ExpectedNull => "expected_null",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Position {
    First,
    Second,
    Other,
}

impl Position {
    fn as_str(self) -> &'static str {
        match self {
            Self::First => "first",
            Self::Second => "second",
            Self::Other => "other",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum ComplementClass {
    SurnameObserved,
    Unknown,
    OrganizationOrLexicalNegative,
    GivenObserved,
    NotApplicable,
}

impl ComplementClass {
    fn as_str(self) -> &'static str {
        match self {
            Self::SurnameObserved => "A_surname_observed_no_given_evidence",
            Self::Unknown => "B_unknown_no_surname_evidence",
            Self::OrganizationOrLexicalNegative => "C_organization_or_lexical_negative",
            Self::GivenObserved => "D_given_observed",
            Self::NotApplicable => "not_applicable",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Topology {
    NativeTwoTokenSolePrimary,
    NativeTwoTokenSoleVetoed,
    NativeTwoTokenSolePunctuated,
    NativeTwoTokenMultipleCandidates,
    NativeThreePlusTokens,
    Segmented,
    Other,
}

impl Topology {
    fn as_str(self) -> &'static str {
        match self {
            Self::NativeTwoTokenSolePrimary => "native_two_token_sole_primary",
            Self::NativeTwoTokenSoleVetoed => "native_two_token_sole_vetoed",
            Self::NativeTwoTokenSolePunctuated => "native_two_token_sole_punctuated",
            Self::NativeTwoTokenMultipleCandidates => "native_two_token_multiple_candidates",
            Self::NativeThreePlusTokens => "native_three_plus_tokens",
            Self::Segmented => "segmented_or_handle_derived",
            Self::Other => "other",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum PositionRule {
    Any,
    First,
    Second,
    AgreesWithOrderPrior,
}

impl PositionRule {
    const ALL: [Self; 4] = [
        Self::Any,
        Self::First,
        Self::Second,
        Self::AgreesWithOrderPrior,
    ];

    fn as_str(self) -> &'static str {
        match self {
            Self::Any => "any",
            Self::First => "first",
            Self::Second => "second",
            Self::AgreesWithOrderPrior => "agrees_with_order_prior",
        }
    }

    fn matches(self, row: &ComplementRow) -> bool {
        match self {
            Self::Any => true,
            Self::First => row.position == Position::First,
            Self::Second => row.position == Position::Second,
            Self::AgreesWithOrderPrior => row.agrees_with_order_prior,
        }
    }
}

#[derive(Clone, Debug)]
struct ComplementRow {
    population: Population,
    outcome: Outcome,
    topology: Topology,
    native: bool,
    token_count: usize,
    candidate_count: usize,
    selected_candidate: Option<String>,
    candidate_quality: f64,
    role_signal: f64,
    reliability: f64,
    position: Position,
    order_prior: NameOrderPrior,
    agrees_with_order_prior: bool,
    country_hint_present: bool,
    c5_emits: bool,
    vetoes_pass: bool,
    hard_organization_marker: bool,
    generic_organization_marker: bool,
    ampersand: bool,
    candidate_too_short: bool,
    complement_class: ComplementClass,
    complement_lexically_eligible: Option<bool>,
    complement_alphabetic_length: Option<usize>,
    complement_capitalization: Option<&'static str>,
    complement_given_count: Option<u64>,
    complement_retained_surname_count: Option<u64>,
    complement_role_llr: Option<f64>,
    complement_raw_surname_count: Option<u64>,
    complement_raw_role_llr: Option<f64>,
    complement_normalized: Option<String>,
}

impl ComplementRow {
    fn primary(&self) -> bool {
        self.topology == Topology::NativeTwoTokenSolePrimary && !self.c5_emits
    }

    fn sole_native_winner(&self) -> bool {
        self.native && self.selected_candidate.is_some() && self.candidate_count == 1
    }

    fn attach_raw_surname_count(&mut self, surname_count: u64) {
        self.complement_raw_surname_count = Some(surname_count);
        self.complement_raw_role_llr = Some(role_llr(
            crate::artifact::Evidence {
                global_count: 0,
                country_count: 0,
                effective_count: 0,
                female_count: 0,
                male_count: 0,
                surname_count,
                given_total: GIVEN_TOTAL,
                surname_total: SURNAME_TOTAL,
            },
            ALGORITHM_C3.role_smoothing,
        ));
        if self.vetoes_pass && self.complement_given_count.is_none() {
            self.complement_class = if surname_count > 0 {
                ComplementClass::SurnameObserved
            } else {
                ComplementClass::Unknown
            };
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Gate {
    quality_min: f64,
    reliability_min: f64,
    role_min: f64,
    position: PositionRule,
}

impl Gate {
    fn emits(self, row: &ComplementRow) -> bool {
        row.primary()
            && row.vetoes_pass
            && self.position.matches(row)
            && row.candidate_quality >= self.quality_min
            && row.reliability >= self.reliability_min
            && row.role_signal >= self.role_min
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct GateMetrics {
    emitted: usize,
    correct: usize,
    wrong: usize,
    null_false_emissions: usize,
}

impl GateMetrics {
    fn observe(&mut self, row: &ComplementRow, emit: bool) {
        if !emit {
            return;
        }
        self.emitted += 1;
        match row.outcome {
            Outcome::CorrectWinner => self.correct += 1,
            Outcome::WrongWinner => self.wrong += 1,
            Outcome::ExpectedNull => self.null_false_emissions += 1,
        }
    }

    fn errors(self) -> usize {
        self.wrong + self.null_false_emissions
    }
}

#[derive(Clone, Copy, Debug)]
struct SelectedGate {
    budget: usize,
    gate: Gate,
    metrics: GateMetrics,
}

#[derive(Clone, Copy, Debug)]
struct SurnameGate {
    selected: Gate,
    surname_count_min: u64,
}

impl SurnameGate {
    fn emits(self, row: &ComplementRow) -> bool {
        self.selected.emits(row)
            && row.complement_class == ComplementClass::SurnameObserved
            && row
                .complement_raw_surname_count
                .is_some_and(|count| count >= self.surname_count_min)
    }
}

#[derive(Clone, Copy, Debug)]
struct SelectedSurnameGate {
    budget: usize,
    gate: SurnameGate,
    metrics: GateMetrics,
}

#[derive(Clone, Copy, Debug)]
struct SelectedResidualGate {
    budget: usize,
    reference: Gate,
    surname: SurnameGate,
    metrics: GateMetrics,
    additional_metrics: GateMetrics,
}

#[derive(Clone, Debug)]
struct SurnameScan {
    counts: BTreeMap<String, u64>,
    counts_sha256: String,
    matched_keys: usize,
    matched_observations: u64,
}

struct DiagnosticContext<'a> {
    proxy_rows: &'a [ComplementRow],
    validation_rows: &'a [ComplementRow],
    gates: &'a [Gate],
    surname_gates: &'a [SurnameGate],
    selected: &'a [SelectedGate],
    selected_surname: &'a [SelectedSurnameGate],
    selected_residual: &'a [SelectedResidualGate],
    selected_storage_tradeoff: &'a [SelectedResidualGate],
    probes: &'a [ProbeResult],
    surname_scan: &'a SurnameScan,
}

#[derive(Clone, Copy, Debug)]
struct PercentileSummary {
    count: usize,
    p10: Option<f64>,
    p25: Option<f64>,
    p50: Option<f64>,
    p75: Option<f64>,
    p90: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct ProbeInput {
    case_id: String,
    display_name: String,
    #[serde(default)]
    expected_greeting: String,
    #[serde(default)]
    country_hint: String,
    #[serde(default)]
    locale_hint: String,
    redact_complement: bool,
}

#[derive(Clone, Debug)]
struct ProbeResult {
    label: String,
    row: ComplementRow,
    gate_emissions: Vec<(usize, bool)>,
    surname_gate_emissions: Vec<(usize, bool)>,
}

pub(crate) fn run_complement_diagnostic(
    output: &Path,
    corpus: &impl EvidenceSource,
    holdouts: Vec<FrozenHoldout>,
    fixtures: &Path,
    probes: &Path,
    surname_counts: &Path,
    surname_manifest: &Path,
) -> Result<String> {
    let holdouts = validate_and_order_holdouts(holdouts)?;
    let mut proxy_rows = build_proxy_rows(corpus, &holdouts);
    let mut validation_rows = build_validation_rows(corpus, fixtures)?;
    let gates = gate_grid();
    let selected = ERROR_BUDGETS
        .into_iter()
        .map(|budget| select_gate(&proxy_rows, &gates, budget, None))
        .collect::<Result<Vec<_>>>()?;
    let mut probe_results = evaluate_probes(corpus, probes, &selected)?;
    let expected_keys = expected_surname_keys(&proxy_rows, &probe_results);
    let surname_scan = load_surname_scan(surname_counts, surname_manifest, &expected_keys)?;
    attach_surname_counts(
        &mut proxy_rows,
        &mut validation_rows,
        &mut probe_results,
        &surname_scan,
    );
    let surname_gates = surname_gate_grid(&gates);
    let selected_surname = ERROR_BUDGETS
        .into_iter()
        .map(|budget| select_surname_gate(&proxy_rows, &surname_gates, budget, None))
        .collect::<Result<Vec<_>>>()?;
    let selected_residual = selected
        .iter()
        .map(|reference| select_residual_gate(&proxy_rows, &surname_gates, *reference, None, None))
        .collect::<Result<Vec<_>>>()?;
    let reference_zero = selected
        .iter()
        .find(|point| point.budget == 0)
        .copied()
        .ok_or("zero-error reference gate missing")?;
    let selected_storage_tradeoff = SURNAME_THRESHOLDS
        .into_iter()
        .map(|threshold| {
            select_residual_gate(
                &proxy_rows,
                &surname_gates,
                reference_zero,
                None,
                Some(threshold),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    attach_probe_surname_emissions(&mut probe_results, &selected_surname);
    validate_invariants(&proxy_rows, &validation_rows, &probe_results)?;

    let context = DiagnosticContext {
        proxy_rows: &proxy_rows,
        validation_rows: &validation_rows,
        gates: &gates,
        surname_gates: &surname_gates,
        selected: &selected,
        selected_surname: &selected_surname,
        selected_residual: &selected_residual,
        selected_storage_tradeoff: &selected_storage_tradeoff,
        probes: &probe_results,
        surname_scan: &surname_scan,
    };
    let outputs = build_outputs(&context)?;
    let repeated = build_outputs(&context)?;
    if outputs != repeated {
        return Err("complement diagnostic serialization is not deterministic".into());
    }
    for (name, bytes) in &outputs {
        fs::write(output.join(name), bytes)?;
    }
    let report = outputs
        .get("report.md")
        .ok_or("complement report missing")?;
    Ok(String::from_utf8(report.clone())?)
}

fn expected_surname_keys(proxy_rows: &[ComplementRow], probes: &[ProbeResult]) -> BTreeSet<String> {
    proxy_rows
        .iter()
        .filter(|row| row.primary())
        .filter_map(|row| row.complement_normalized.clone())
        .chain(
            probes
                .iter()
                .filter_map(|probe| probe.row.complement_normalized.clone()),
        )
        .collect()
}

fn load_surname_scan(
    counts_path: &Path,
    manifest_path: &Path,
    expected_keys: &BTreeSet<String>,
) -> Result<SurnameScan> {
    let bytes = fs::read(counts_path)?;
    let counts_sha256 = format!("{:x}", Sha256::digest(&bytes));
    let manifest = load_scan_manifest(manifest_path)?;
    validate_scan_manifest(&manifest, &counts_sha256, expected_keys.len())?;
    let mut reader = csv::Reader::from_reader(bytes.as_slice());
    if reader.headers()?.iter().ne(["name", "surname_count"]) {
        return Err("unexpected complement surname-count header".into());
    }
    let mut counts = BTreeMap::new();
    for result in reader.records() {
        let record = result?;
        let name = record.get(0).ok_or("missing complement surname key")?;
        let count = record
            .get(1)
            .ok_or("missing complement surname count")?
            .parse::<u64>()?;
        if counts.insert(name.to_string(), count).is_some() {
            return Err("duplicate complement surname key".into());
        }
    }
    let actual_keys = counts.keys().cloned().collect::<BTreeSet<_>>();
    if &actual_keys != expected_keys {
        return Err(format!(
            "complement surname-count keys do not match requested set: expected {}, got {}",
            expected_keys.len(),
            actual_keys.len()
        )
        .into());
    }
    let matched_keys = counts.values().filter(|count| **count > 0).count();
    let matched_observations: u64 = counts.values().sum();
    validate_manifest_value(&manifest, "matched_keys", matched_keys.to_string())?;
    validate_manifest_value(
        &manifest,
        "matched_observations",
        matched_observations.to_string(),
    )?;
    Ok(SurnameScan {
        counts,
        counts_sha256,
        matched_keys,
        matched_observations,
    })
}

fn load_scan_manifest(path: &Path) -> Result<BTreeMap<String, String>> {
    let mut reader = csv::Reader::from_path(path)?;
    if reader.headers()?.iter().ne(["key", "value"]) {
        return Err("unexpected complement surname manifest header".into());
    }
    let mut manifest = BTreeMap::new();
    for result in reader.records() {
        let record = result?;
        let key = record.get(0).ok_or("missing surname manifest key")?;
        let value = record.get(1).ok_or("missing surname manifest value")?;
        if manifest
            .insert(key.to_string(), value.to_string())
            .is_some()
        {
            return Err("duplicate complement surname manifest key".into());
        }
    }
    Ok(manifest)
}

fn validate_scan_manifest(
    manifest: &BTreeMap<String, String>,
    counts_sha256: &str,
    target_keys: usize,
) -> Result<()> {
    validate_manifest_value(manifest, "matching", "exact_utf8_byte_equality")?;
    validate_manifest_value(manifest, "raw_files", RAW_FILES.to_string())?;
    validate_manifest_value(manifest, "raw_rows", RAW_ROWS.to_string())?;
    validate_manifest_value(manifest, "nonempty_surnames", SURNAME_TOTAL.to_string())?;
    validate_manifest_value(manifest, "target_keys", target_keys.to_string())?;
    validate_manifest_value(manifest, "counts_sha256", counts_sha256)?;
    Ok(())
}

fn validate_manifest_value(
    manifest: &BTreeMap<String, String>,
    key: &str,
    expected: impl AsRef<str>,
) -> Result<()> {
    let actual = manifest
        .get(key)
        .ok_or_else(|| format!("complement surname manifest is missing {key}"))?;
    if actual != expected.as_ref() {
        return Err(format!(
            "complement surname manifest {key} mismatch: expected {}, got {actual}",
            expected.as_ref()
        )
        .into());
    }
    Ok(())
}

fn attach_surname_counts(
    proxy_rows: &mut [ComplementRow],
    validation_rows: &mut [ComplementRow],
    probes: &mut [ProbeResult],
    scan: &SurnameScan,
) {
    for row in proxy_rows.iter_mut().chain(validation_rows) {
        attach_row_surname_count(row, scan);
    }
    for probe in probes {
        attach_row_surname_count(&mut probe.row, scan);
    }
}

fn attach_row_surname_count(row: &mut ComplementRow, scan: &SurnameScan) {
    let Some(key) = row.complement_normalized.as_deref() else {
        return;
    };
    if let Some(count) = scan.counts.get(key) {
        row.attach_raw_surname_count(*count);
    }
}

fn attach_probe_surname_emissions(probes: &mut [ProbeResult], selected: &[SelectedSurnameGate]) {
    for probe in probes {
        probe.surname_gate_emissions = selected
            .iter()
            .map(|point| (point.budget, point.gate.emits(&probe.row)))
            .collect();
    }
}

fn build_proxy_rows(
    corpus: &impl EvidenceSource,
    holdouts: &[FrozenHoldout],
) -> Vec<ComplementRow> {
    let mut rows = Vec::new();
    for holdout in holdouts {
        let population = Population::from_digest(&holdout.manifest.holdout_sha256)
            .expect("validated proxy digest");
        rows.extend(
            holdout
                .cases
                .iter()
                .filter(|case| case.is_evaluable())
                .map(|case| {
                    build_row(
                        corpus,
                        population,
                        &case.display_name,
                        case.expected_greeting(),
                        nonempty(&case.country_hint),
                        nonempty(&case.locale_hint),
                    )
                }),
        );
    }
    rows
}

fn build_validation_rows(
    corpus: &impl EvidenceSource,
    fixtures: &Path,
) -> Result<Vec<ComplementRow>> {
    Ok(generate_cases(fixtures, false)?
        .into_iter()
        .filter(|case| case.split == Split::Validation)
        .map(|case| build_row_from_case(corpus, &case))
        .collect())
}

fn build_row_from_case(corpus: &impl EvidenceSource, case: &Case) -> ComplementRow {
    build_row(
        corpus,
        Population::Validation,
        &case.input,
        case.expected_greeting.as_deref(),
        case.country_hint.as_deref(),
        case.locale_hint.as_deref(),
    )
}

fn build_row(
    corpus: &impl EvidenceSource,
    population: Population,
    display_name: &str,
    expected_greeting: Option<&str>,
    country_hint: Option<&str>,
    locale_hint: Option<&str>,
) -> ComplementRow {
    let diagnostic = diagnose_role_inference(
        corpus,
        ALGORITHM_C3,
        display_name,
        country_hint,
        locale_hint,
    );
    let decision = c4_decision_breakdown(&diagnostic, ALGORITHM_C2, ALGORITHM_C31, ALGORITHM_C4);
    let c5 = c5_decision_from_c4(decision.clone(), ALGORITHM_C5);
    let winner = decision.c31.winner.as_ref();
    let candidate = diagnostic.candidates.first();
    let selected = winner.map(|winner| winner.greeting_candidate.as_str());
    let outcome = match expected_greeting {
        None => Outcome::ExpectedNull,
        Some(_) if greeting_matches(expected_greeting, selected) => Outcome::CorrectWinner,
        Some(_) => Outcome::WrongWinner,
    };
    let canonical = canonicalize(display_name);
    let tokens = canonical.split_whitespace().collect::<Vec<_>>();
    let native = winner.is_some_and(|winner| winner.candidate_origin != "handle_segment");
    let position = candidate_position(candidate, tokens.len());
    let vetoes_pass = vetoes_pass(&decision);
    let punctuated = canonical
        .chars()
        .any(|character| !character.is_alphabetic() && !character.is_whitespace());
    let complement_index = complement_index(candidate, tokens.len());
    let complement = complement_index.and_then(|index| tokens.get(index).copied());
    let complement_lookup = complement.map(|token| {
        expected_lookup_diagnostic(corpus, ALGORITHM_C3, token, country_hint, locale_hint)
    });
    let complement_class = complement_class(&decision, complement_lookup.as_ref());
    let order_prior = resolve_name_order_prior(country_hint, locale_hint);
    let agrees_with_order_prior = matches!(
        (order_prior, position),
        (NameOrderPrior::GivenFirst, Position::First)
            | (NameOrderPrior::SurnameFirst, Position::Second)
    );
    let topology = topology(
        native,
        tokens.len(),
        winner.map_or(0, |winner| winner.candidate_count),
        candidate,
        punctuated,
        vetoes_pass,
        complement_lookup
            .as_ref()
            .is_some_and(|lookup| lookup.eligible),
    );
    let evidence = complement_lookup
        .as_ref()
        .and_then(|lookup| lookup.evidence);
    ComplementRow {
        population,
        outcome,
        topology,
        native,
        token_count: tokens.len(),
        candidate_count: winner.map_or(0, |winner| winner.candidate_count),
        selected_candidate: selected.map(str::to_string),
        candidate_quality: winner.map_or(0.0, |winner| winner.winner_score),
        role_signal: winner.map_or(0.0, |winner| winner.role_signal),
        reliability: winner.map_or(0.0, |winner| winner.reliability),
        position,
        order_prior,
        agrees_with_order_prior,
        country_hint_present: country_hint.is_some(),
        c5_emits: c5_emitted_candidate(&c5).is_some(),
        vetoes_pass,
        hard_organization_marker: decision.c31.hard_organization_marker,
        generic_organization_marker: decision.c31.generic_organization_marker,
        ampersand: decision.c31.ampersand,
        candidate_too_short: decision.c31.candidate_too_short,
        complement_class,
        complement_lexically_eligible: complement_lookup.as_ref().map(|lookup| lookup.eligible),
        complement_alphabetic_length: complement.map(|token| {
            token
                .chars()
                .filter(|character| character.is_alphabetic())
                .count()
        }),
        complement_capitalization: complement.map(capitalization_pattern),
        complement_given_count: evidence.map(|evidence| evidence.global_count),
        complement_retained_surname_count: evidence.map(|evidence| evidence.surname_count),
        complement_role_llr: complement_lookup.and_then(|lookup| lookup.role_llr),
        complement_raw_surname_count: None,
        complement_raw_role_llr: None,
        complement_normalized: complement.map(canonicalize),
    }
}

fn candidate_position(candidate: Option<&CandidateDiagnostic>, token_count: usize) -> Position {
    match candidate {
        Some(candidate) if candidate.length == 1 && candidate.start == 0 && token_count == 2 => {
            Position::First
        }
        Some(candidate) if candidate.length == 1 && candidate.start == 1 && token_count == 2 => {
            Position::Second
        }
        _ => Position::Other,
    }
}

fn complement_index(candidate: Option<&CandidateDiagnostic>, token_count: usize) -> Option<usize> {
    match candidate_position(candidate, token_count) {
        Position::First => Some(1),
        Position::Second => Some(0),
        Position::Other => None,
    }
}

fn complement_class(
    decision: &C4DecisionBreakdown,
    lookup: Option<&crate::classifier::ExpectedLookupDiagnostic>,
) -> ComplementClass {
    if !vetoes_pass(decision) {
        ComplementClass::OrganizationOrLexicalNegative
    } else if lookup.is_some_and(|lookup| lookup.evidence.is_some()) {
        ComplementClass::GivenObserved
    } else if lookup.is_some_and(|lookup| lookup.eligible) {
        ComplementClass::Unknown
    } else {
        ComplementClass::NotApplicable
    }
}

fn topology(
    native: bool,
    token_count: usize,
    candidate_count: usize,
    candidate: Option<&CandidateDiagnostic>,
    punctuated: bool,
    vetoes_pass: bool,
    complement_eligible: bool,
) -> Topology {
    if !native {
        return if candidate.is_some_and(|candidate| candidate.origin == "handle_segment") {
            Topology::Segmented
        } else {
            Topology::Other
        };
    }
    if token_count >= 3 {
        return Topology::NativeThreePlusTokens;
    }
    if token_count != 2 || candidate_position(candidate, token_count) == Position::Other {
        return Topology::Other;
    }
    if candidate_count != 1 {
        return Topology::NativeTwoTokenMultipleCandidates;
    }
    if punctuated || !complement_eligible {
        return Topology::NativeTwoTokenSolePunctuated;
    }
    if !vetoes_pass {
        return Topology::NativeTwoTokenSoleVetoed;
    }
    Topology::NativeTwoTokenSolePrimary
}

fn vetoes_pass(decision: &C4DecisionBreakdown) -> bool {
    !decision.c31.hard_organization_marker
        && !decision.c31.generic_organization_marker
        && !decision.c31.ampersand
        && !decision.c31.candidate_too_short
}

fn capitalization_pattern(token: &str) -> &'static str {
    let alphabetic = token
        .chars()
        .filter(|character| character.is_alphabetic())
        .collect::<Vec<_>>();
    if alphabetic.is_empty() {
        return "no_letters";
    }
    if alphabetic.iter().all(|character| character.is_uppercase()) {
        return "upper";
    }
    if alphabetic.iter().all(|character| character.is_lowercase()) {
        return "lower";
    }
    if alphabetic[0].is_uppercase()
        && alphabetic[1..]
            .iter()
            .all(|character| character.is_lowercase())
    {
        return "title";
    }
    "mixed"
}

fn gate_grid() -> Vec<Gate> {
    let mut gates = Vec::new();
    for quality_min in QUALITY_THRESHOLDS {
        for reliability_min in RELIABILITY_THRESHOLDS {
            for role_min in ROLE_THRESHOLDS {
                for position in PositionRule::ALL {
                    gates.push(Gate {
                        quality_min,
                        reliability_min,
                        role_min,
                        position,
                    });
                }
            }
        }
    }
    gates
}

fn surname_gate_grid(gates: &[Gate]) -> Vec<SurnameGate> {
    gates
        .iter()
        .flat_map(|gate| {
            SURNAME_THRESHOLDS
                .into_iter()
                .map(|surname_count_min| SurnameGate {
                    selected: *gate,
                    surname_count_min,
                })
        })
        .collect()
}

fn select_gate(
    rows: &[ComplementRow],
    gates: &[Gate],
    budget: usize,
    held_out: Option<Population>,
) -> Result<SelectedGate> {
    gates
        .iter()
        .copied()
        .map(|gate| {
            let metrics = evaluate_gate(rows, gate, |row| {
                row.population != Population::Validation
                    && held_out.is_none_or(|held_out| row.population != held_out)
            });
            SelectedGate {
                budget,
                gate,
                metrics,
            }
        })
        .filter(|point| point.metrics.errors() <= budget)
        .max_by(compare_selected_gates)
        .ok_or_else(|| format!("no reference gate satisfies error budget {budget}").into())
}

fn select_surname_gate(
    rows: &[ComplementRow],
    gates: &[SurnameGate],
    budget: usize,
    held_out: Option<Population>,
) -> Result<SelectedSurnameGate> {
    gates
        .iter()
        .copied()
        .map(|gate| {
            let metrics = evaluate_surname_gate(rows, gate, |row| {
                row.population != Population::Validation
                    && held_out.is_none_or(|held_out| row.population != held_out)
            });
            SelectedSurnameGate {
                budget,
                gate,
                metrics,
            }
        })
        .filter(|point| point.metrics.errors() <= budget)
        .max_by(compare_selected_surname_gates)
        .ok_or_else(|| format!("no surname gate satisfies error budget {budget}").into())
}

fn select_residual_gate(
    rows: &[ComplementRow],
    surname_gates: &[SurnameGate],
    reference: SelectedGate,
    held_out: Option<Population>,
    required_surname_count_min: Option<u64>,
) -> Result<SelectedResidualGate> {
    surname_gates
        .iter()
        .copied()
        .filter(|gate| {
            required_surname_count_min.is_none_or(|required| gate.surname_count_min == required)
        })
        .map(|surname| {
            let (metrics, additional_metrics) =
                evaluate_residual_gate(rows, reference.gate, surname, |row| {
                    row.population != Population::Validation
                        && held_out.is_none_or(|held_out| row.population != held_out)
                });
            SelectedResidualGate {
                budget: reference.budget,
                reference: reference.gate,
                surname,
                metrics,
                additional_metrics,
            }
        })
        .filter(|point| point.metrics.errors() <= reference.budget)
        .max_by(compare_selected_residual_gates)
        .ok_or_else(|| {
            format!(
                "no residual surname gate satisfies error budget {}",
                reference.budget
            )
            .into()
        })
}

fn compare_selected_gates(left: &SelectedGate, right: &SelectedGate) -> Ordering {
    left.metrics
        .correct
        .cmp(&right.metrics.correct)
        .then_with(|| right.metrics.errors().cmp(&left.metrics.errors()))
        .then_with(|| right.metrics.emitted.cmp(&left.metrics.emitted))
        .then_with(|| left.gate.quality_min.total_cmp(&right.gate.quality_min))
        .then_with(|| {
            left.gate
                .reliability_min
                .total_cmp(&right.gate.reliability_min)
        })
        .then_with(|| left.gate.role_min.total_cmp(&right.gate.role_min))
        .then_with(|| right.gate.position.cmp(&left.gate.position))
}

fn compare_selected_surname_gates(
    left: &SelectedSurnameGate,
    right: &SelectedSurnameGate,
) -> Ordering {
    left.metrics
        .correct
        .cmp(&right.metrics.correct)
        .then_with(|| right.metrics.errors().cmp(&left.metrics.errors()))
        .then_with(|| right.metrics.emitted.cmp(&left.metrics.emitted))
        .then_with(|| {
            left.gate
                .surname_count_min
                .cmp(&right.gate.surname_count_min)
        })
        .then_with(|| {
            left.gate
                .selected
                .quality_min
                .total_cmp(&right.gate.selected.quality_min)
        })
        .then_with(|| {
            left.gate
                .selected
                .reliability_min
                .total_cmp(&right.gate.selected.reliability_min)
        })
        .then_with(|| {
            left.gate
                .selected
                .role_min
                .total_cmp(&right.gate.selected.role_min)
        })
        .then_with(|| {
            right
                .gate
                .selected
                .position
                .cmp(&left.gate.selected.position)
        })
}

fn compare_selected_residual_gates(
    left: &SelectedResidualGate,
    right: &SelectedResidualGate,
) -> Ordering {
    left.metrics
        .correct
        .cmp(&right.metrics.correct)
        .then_with(|| right.metrics.errors().cmp(&left.metrics.errors()))
        .then_with(|| {
            left.additional_metrics
                .correct
                .cmp(&right.additional_metrics.correct)
        })
        .then_with(|| {
            left.surname
                .surname_count_min
                .cmp(&right.surname.surname_count_min)
        })
        .then_with(|| {
            left.surname
                .selected
                .quality_min
                .total_cmp(&right.surname.selected.quality_min)
        })
        .then_with(|| {
            left.surname
                .selected
                .reliability_min
                .total_cmp(&right.surname.selected.reliability_min)
        })
        .then_with(|| {
            left.surname
                .selected
                .role_min
                .total_cmp(&right.surname.selected.role_min)
        })
        .then_with(|| {
            right
                .surname
                .selected
                .position
                .cmp(&left.surname.selected.position)
        })
}

fn evaluate_gate(
    rows: &[ComplementRow],
    gate: Gate,
    include: impl Fn(&ComplementRow) -> bool,
) -> GateMetrics {
    let mut metrics = GateMetrics::default();
    for row in rows.iter().filter(|row| include(row)) {
        metrics.observe(row, gate.emits(row));
    }
    metrics
}

fn evaluate_surname_gate(
    rows: &[ComplementRow],
    gate: SurnameGate,
    include: impl Fn(&ComplementRow) -> bool,
) -> GateMetrics {
    let mut metrics = GateMetrics::default();
    for row in rows.iter().filter(|row| include(row)) {
        metrics.observe(row, gate.emits(row));
    }
    metrics
}

fn evaluate_residual_gate(
    rows: &[ComplementRow],
    reference: Gate,
    surname: SurnameGate,
    include: impl Fn(&ComplementRow) -> bool,
) -> (GateMetrics, GateMetrics) {
    let mut metrics = GateMetrics::default();
    let mut additional_metrics = GateMetrics::default();
    for row in rows.iter().filter(|row| include(row)) {
        let reference_emits = reference.emits(row);
        let surname_emits = surname.emits(row);
        metrics.observe(row, reference_emits || surname_emits);
        additional_metrics.observe(row, surname_emits && !reference_emits);
    }
    (metrics, additional_metrics)
}

fn evaluate_probes(
    corpus: &impl EvidenceSource,
    path: &Path,
    selected: &[SelectedGate],
) -> Result<Vec<ProbeResult>> {
    let mut reader = csv::Reader::from_path(path)?;
    let mut results = Vec::new();
    for result in reader.deserialize() {
        let probe: ProbeInput = result?;
        let expected = nonempty(&probe.expected_greeting);
        let row = build_row(
            corpus,
            Population::Validation,
            &probe.display_name,
            expected,
            nonempty(&probe.country_hint),
            nonempty(&probe.locale_hint),
        );
        let label = probe_label(&probe)?;
        let gate_emissions = selected
            .iter()
            .map(|point| (point.budget, point.gate.emits(&row)))
            .collect();
        results.push(ProbeResult {
            label,
            row,
            gate_emissions,
            surname_gate_emissions: Vec::new(),
        });
    }
    if results.is_empty() {
        return Err("complement probe file is empty".into());
    }
    Ok(results)
}

fn probe_label(probe: &ProbeInput) -> Result<String> {
    if !probe.redact_complement {
        return Ok(probe.case_id.clone());
    }
    let expected = nonempty(&probe.expected_greeting)
        .ok_or("a redacted complement probe requires an expected greeting")?;
    Ok(format!("{expected} REDACTED"))
}

fn validate_invariants(
    proxy_rows: &[ComplementRow],
    validation_rows: &[ComplementRow],
    probes: &[ProbeResult],
) -> Result<()> {
    let retained_complements = proxy_rows
        .iter()
        .chain(validation_rows)
        .filter(|row| row.primary() && row.complement_given_count.is_some())
        .count();
    if retained_complements != 0 {
        return Err(format!(
            "primary sole-candidate invariant changed: {retained_complements} complements are present in the retained given-name index"
        )
        .into());
    }
    let missing_raw_counts = proxy_rows
        .iter()
        .filter(|row| row.primary() && row.complement_raw_surname_count.is_none())
        .count();
    if missing_raw_counts != 0 {
        return Err(format!(
            "primary complement surname join is incomplete for {missing_raw_counts} rows"
        )
        .into());
    }
    let safety = probes
        .iter()
        .find(|probe| probe.label == "Motorcycle Club")
        .ok_or("qualitative probes must include the Motorcycle Club safety control")?;
    if safety.row.c5_emits
        || safety.row.vetoes_pass
        || safety.gate_emissions.iter().any(|(_, emits)| *emits)
        || safety
            .surname_gate_emissions
            .iter()
            .any(|(_, emits)| *emits)
    {
        return Err("Motorcycle Club safety control did not remain an abstention".into());
    }
    Ok(())
}

fn build_outputs(context: &DiagnosticContext<'_>) -> Result<BTreeMap<&'static str, Vec<u8>>> {
    let DiagnosticContext {
        proxy_rows,
        validation_rows,
        gates,
        surname_gates,
        selected,
        selected_surname,
        selected_residual,
        selected_storage_tradeoff,
        probes,
        surname_scan,
    } = *context;
    let mut outputs = BTreeMap::new();
    outputs.insert(
        "topology_summary.csv",
        topology_summary(proxy_rows, validation_rows)?,
    );
    outputs.insert(
        "primary_outcome_summary.csv",
        primary_outcome_summary(proxy_rows, validation_rows)?,
    );
    outputs.insert(
        "sole_native_outcome_summary.csv",
        sole_native_outcome_summary(proxy_rows, validation_rows)?,
    );
    outputs.insert(
        "signal_percentiles.csv",
        signal_percentiles(proxy_rows, validation_rows)?,
    );
    outputs.insert(
        "position_summary.csv",
        position_summary(proxy_rows, validation_rows)?,
    );
    outputs.insert(
        "complement_status_summary.csv",
        complement_status_summary(proxy_rows, validation_rows)?,
    );
    outputs.insert(
        "veto_safety_summary.csv",
        veto_safety_summary(proxy_rows, validation_rows)?,
    );
    outputs.insert(
        "reference_frontier.csv",
        reference_frontier(proxy_rows, validation_rows, selected)?,
    );
    outputs.insert("reference_logo.csv", reference_logo(proxy_rows, gates)?);
    outputs.insert(
        "surname_frontier.csv",
        surname_frontier(proxy_rows, validation_rows, selected_surname)?,
    );
    outputs.insert("surname_logo.csv", surname_logo(proxy_rows, surname_gates)?);
    outputs.insert(
        "residual_frontier.csv",
        residual_frontier(proxy_rows, validation_rows, selected_residual)?,
    );
    outputs.insert(
        "residual_logo.csv",
        residual_logo(proxy_rows, gates, surname_gates)?,
    );
    outputs.insert("surname_storage.csv", surname_storage(surname_scan)?);
    outputs.insert(
        "storage_tradeoff.csv",
        storage_tradeoff(selected_storage_tradeoff)?,
    );
    outputs.insert("qualitative_probes.csv", qualitative_csv(probes)?);
    outputs.insert("report.md", report(context).into_bytes());
    Ok(outputs)
}

fn topology_summary(
    proxy_rows: &[ComplementRow],
    validation_rows: &[ComplementRow],
) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record(["population", "topology", "rows"])?;
    for (label, rows) in report_populations(proxy_rows, validation_rows) {
        for topology in [
            Topology::NativeTwoTokenSolePrimary,
            Topology::NativeTwoTokenSoleVetoed,
            Topology::NativeTwoTokenSolePunctuated,
            Topology::NativeTwoTokenMultipleCandidates,
            Topology::NativeThreePlusTokens,
            Topology::Segmented,
            Topology::Other,
        ] {
            writer.write_record([
                label,
                topology.as_str(),
                &rows
                    .iter()
                    .filter(|row| row.topology == topology)
                    .count()
                    .to_string(),
            ])?;
        }
    }
    Ok(writer.into_inner()?)
}

fn primary_outcome_summary(
    proxy_rows: &[ComplementRow],
    validation_rows: &[ComplementRow],
) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "population",
        "outcome",
        "rows",
        "first",
        "second",
        "country_hint",
        "order_prior_available",
        "order_prior_agreement",
    ])?;
    for (label, rows) in report_populations(proxy_rows, validation_rows) {
        for outcome in [
            Outcome::CorrectWinner,
            Outcome::WrongWinner,
            Outcome::ExpectedNull,
        ] {
            let selected = rows
                .iter()
                .copied()
                .filter(|row| row.primary() && row.outcome == outcome)
                .collect::<Vec<_>>();
            writer.write_record([
                label,
                outcome.as_str(),
                &selected.len().to_string(),
                &selected
                    .iter()
                    .filter(|row| row.position == Position::First)
                    .count()
                    .to_string(),
                &selected
                    .iter()
                    .filter(|row| row.position == Position::Second)
                    .count()
                    .to_string(),
                &selected
                    .iter()
                    .filter(|row| row.country_hint_present)
                    .count()
                    .to_string(),
                &selected
                    .iter()
                    .filter(|row| row.order_prior != NameOrderPrior::Neutral)
                    .count()
                    .to_string(),
                &selected
                    .iter()
                    .filter(|row| row.agrees_with_order_prior)
                    .count()
                    .to_string(),
            ])?;
        }
    }
    Ok(writer.into_inner()?)
}

fn sole_native_outcome_summary(
    proxy_rows: &[ComplementRow],
    validation_rows: &[ComplementRow],
) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "population",
        "token_count",
        "topology",
        "c5_emits",
        "vetoes_pass",
        "outcome",
        "rows",
    ])?;
    for (label, rows) in report_populations(proxy_rows, validation_rows) {
        let mut counts = BTreeMap::<(usize, Topology, bool, bool, Outcome), usize>::new();
        for row in rows.into_iter().filter(|row| row.sole_native_winner()) {
            *counts
                .entry((
                    row.token_count,
                    row.topology,
                    row.c5_emits,
                    row.vetoes_pass,
                    row.outcome,
                ))
                .or_default() += 1;
        }
        for ((token_count, topology, c5_emits, vetoes_pass, outcome), count) in counts {
            writer.write_record([
                label,
                &token_count.to_string(),
                topology.as_str(),
                bool_string(c5_emits),
                bool_string(vetoes_pass),
                outcome.as_str(),
                &count.to_string(),
            ])?;
        }
    }
    Ok(writer.into_inner()?)
}

fn signal_percentiles(
    proxy_rows: &[ComplementRow],
    validation_rows: &[ComplementRow],
) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "population",
        "outcome",
        "signal",
        "count",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
    ])?;
    for (label, rows) in report_populations(proxy_rows, validation_rows) {
        for outcome in [
            Outcome::CorrectWinner,
            Outcome::WrongWinner,
            Outcome::ExpectedNull,
        ] {
            let selected = rows
                .iter()
                .copied()
                .filter(|row| row.primary() && row.outcome == outcome)
                .collect::<Vec<_>>();
            for (signal, values) in [
                (
                    "selected_quality",
                    selected
                        .iter()
                        .map(|row| row.candidate_quality)
                        .collect::<Vec<_>>(),
                ),
                (
                    "selected_role_signal",
                    selected
                        .iter()
                        .map(|row| row.role_signal)
                        .collect::<Vec<_>>(),
                ),
                (
                    "selected_reliability",
                    selected
                        .iter()
                        .map(|row| row.reliability)
                        .collect::<Vec<_>>(),
                ),
                (
                    "complement_alphabetic_length",
                    selected
                        .iter()
                        .filter_map(|row| {
                            row.complement_alphabetic_length.map(|value| value as f64)
                        })
                        .collect::<Vec<_>>(),
                ),
                (
                    "complement_given_count",
                    selected
                        .iter()
                        .filter_map(|row| row.complement_given_count.map(|value| value as f64))
                        .collect::<Vec<_>>(),
                ),
                (
                    "complement_surname_count_retained_keys_only",
                    selected
                        .iter()
                        .filter_map(|row| {
                            row.complement_retained_surname_count
                                .map(|value| value as f64)
                        })
                        .collect::<Vec<_>>(),
                ),
                (
                    "complement_role_llr_retained_keys_only",
                    selected
                        .iter()
                        .filter_map(|row| row.complement_role_llr)
                        .collect::<Vec<_>>(),
                ),
                (
                    "complement_raw_surname_count",
                    selected
                        .iter()
                        .filter_map(|row| {
                            row.complement_raw_surname_count.map(|value| value as f64)
                        })
                        .collect::<Vec<_>>(),
                ),
                (
                    "complement_raw_role_llr",
                    selected
                        .iter()
                        .filter_map(|row| row.complement_raw_role_llr)
                        .collect::<Vec<_>>(),
                ),
            ] {
                let summary = percentile_summary(values);
                writer.write_record([
                    label,
                    outcome.as_str(),
                    signal,
                    &summary.count.to_string(),
                    &format_optional(summary.p10),
                    &format_optional(summary.p25),
                    &format_optional(summary.p50),
                    &format_optional(summary.p75),
                    &format_optional(summary.p90),
                ])?;
            }
        }
    }
    Ok(writer.into_inner()?)
}

fn position_summary(
    proxy_rows: &[ComplementRow],
    validation_rows: &[ComplementRow],
) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record(["population", "position", "order_prior", "outcome", "rows"])?;
    for (label, rows) in report_populations(proxy_rows, validation_rows) {
        let mut counts = BTreeMap::<(Position, NameOrderPrior, Outcome), usize>::new();
        for row in rows.into_iter().filter(|row| row.primary()) {
            *counts
                .entry((row.position, row.order_prior, row.outcome))
                .or_default() += 1;
        }
        for ((position, prior, outcome), count) in counts {
            writer.write_record([
                label,
                position.as_str(),
                prior.as_str(),
                outcome.as_str(),
                &count.to_string(),
            ])?;
        }
    }
    Ok(writer.into_inner()?)
}

fn complement_status_summary(
    proxy_rows: &[ComplementRow],
    validation_rows: &[ComplementRow],
) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "population",
        "scope",
        "complement_class",
        "lexically_eligible",
        "capitalization",
        "outcome",
        "rows",
    ])?;
    for (label, rows) in report_populations(proxy_rows, validation_rows) {
        let mut counts = BTreeMap::<(ComplementClass, bool, &'static str, Outcome), usize>::new();
        for row in rows.into_iter().filter(|row| {
            matches!(
                row.topology,
                Topology::NativeTwoTokenSolePrimary
                    | Topology::NativeTwoTokenSoleVetoed
                    | Topology::NativeTwoTokenSolePunctuated
            )
        }) {
            *counts
                .entry((
                    row.complement_class,
                    row.complement_lexically_eligible.unwrap_or(false),
                    row.complement_capitalization.unwrap_or("not_applicable"),
                    row.outcome,
                ))
                .or_default() += 1;
        }
        for ((class, eligible, capitalization, outcome), count) in counts {
            writer.write_record([
                label,
                "native_two_token_sole",
                class.as_str(),
                if eligible { "true" } else { "false" },
                capitalization,
                outcome.as_str(),
                &count.to_string(),
            ])?;
        }
    }
    Ok(writer.into_inner()?)
}

fn veto_safety_summary(
    proxy_rows: &[ComplementRow],
    validation_rows: &[ComplementRow],
) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record(["population", "veto", "outcome", "rows"])?;
    for (label, rows) in report_populations(proxy_rows, validation_rows) {
        for outcome in [
            Outcome::CorrectWinner,
            Outcome::WrongWinner,
            Outcome::ExpectedNull,
        ] {
            for (veto, count) in [
                (
                    "hard_organization_marker",
                    rows.iter()
                        .filter(|row| row.outcome == outcome && row.hard_organization_marker)
                        .count(),
                ),
                (
                    "generic_organization_marker",
                    rows.iter()
                        .filter(|row| row.outcome == outcome && row.generic_organization_marker)
                        .count(),
                ),
                (
                    "ampersand",
                    rows.iter()
                        .filter(|row| row.outcome == outcome && row.ampersand)
                        .count(),
                ),
                (
                    "candidate_too_short",
                    rows.iter()
                        .filter(|row| row.outcome == outcome && row.candidate_too_short)
                        .count(),
                ),
            ] {
                writer.write_record([label, veto, outcome.as_str(), &count.to_string()])?;
            }
        }
    }
    Ok(writer.into_inner()?)
}

fn reference_frontier(
    proxy_rows: &[ComplementRow],
    validation_rows: &[ComplementRow],
    selected: &[SelectedGate],
) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "selected_on",
        "evaluated_on",
        "error_budget",
        "quality_min",
        "reliability_min",
        "role_min",
        "position",
        "emitted",
        "correct",
        "wrong",
        "null_false_emissions",
    ])?;
    for point in selected {
        for (label, rows) in report_populations(proxy_rows, validation_rows) {
            let metrics = evaluate_gate_refs(&rows, point.gate);
            write_gate_record(&mut writer, "all_spent", label, *point, metrics)?;
        }
    }
    Ok(writer.into_inner()?)
}

fn reference_logo(rows: &[ComplementRow], gates: &[Gate]) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "held_out",
        "error_budget",
        "quality_min",
        "reliability_min",
        "role_min",
        "position",
        "training_emitted",
        "training_correct",
        "training_wrong",
        "training_null_false_emissions",
        "held_out_emitted",
        "held_out_correct",
        "held_out_wrong",
        "held_out_null_false_emissions",
    ])?;
    for held_out in Population::PROXIES {
        for budget in ERROR_BUDGETS {
            let selected = select_gate(rows, gates, budget, Some(held_out))?;
            let held_out_metrics =
                evaluate_gate(rows, selected.gate, |row| row.population == held_out);
            writer.write_record([
                held_out.as_str(),
                &budget.to_string(),
                &format!("{:.3}", selected.gate.quality_min),
                &format!("{:.3}", selected.gate.reliability_min),
                &format!("{:.3}", selected.gate.role_min),
                selected.gate.position.as_str(),
                &selected.metrics.emitted.to_string(),
                &selected.metrics.correct.to_string(),
                &selected.metrics.wrong.to_string(),
                &selected.metrics.null_false_emissions.to_string(),
                &held_out_metrics.emitted.to_string(),
                &held_out_metrics.correct.to_string(),
                &held_out_metrics.wrong.to_string(),
                &held_out_metrics.null_false_emissions.to_string(),
            ])?;
        }
    }
    Ok(writer.into_inner()?)
}

fn surname_frontier(
    proxy_rows: &[ComplementRow],
    validation_rows: &[ComplementRow],
    selected: &[SelectedSurnameGate],
) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "selected_on",
        "evaluated_on",
        "error_budget",
        "quality_min",
        "reliability_min",
        "role_min",
        "position",
        "surname_count_min",
        "emitted",
        "correct",
        "wrong",
        "null_false_emissions",
    ])?;
    for point in selected {
        for (label, rows) in report_populations(proxy_rows, validation_rows) {
            let metrics = evaluate_surname_gate_refs(&rows, point.gate);
            write_surname_gate_record(&mut writer, "all_spent", label, *point, metrics)?;
        }
    }
    Ok(writer.into_inner()?)
}

fn surname_logo(rows: &[ComplementRow], gates: &[SurnameGate]) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "held_out",
        "error_budget",
        "quality_min",
        "reliability_min",
        "role_min",
        "position",
        "surname_count_min",
        "training_emitted",
        "training_correct",
        "training_wrong",
        "training_null_false_emissions",
        "held_out_emitted",
        "held_out_correct",
        "held_out_wrong",
        "held_out_null_false_emissions",
    ])?;
    for held_out in Population::PROXIES {
        for budget in ERROR_BUDGETS {
            let selected = select_surname_gate(rows, gates, budget, Some(held_out))?;
            let held_out_metrics =
                evaluate_surname_gate(rows, selected.gate, |row| row.population == held_out);
            writer.write_record([
                held_out.as_str(),
                &budget.to_string(),
                &format!("{:.3}", selected.gate.selected.quality_min),
                &format!("{:.3}", selected.gate.selected.reliability_min),
                &format!("{:.3}", selected.gate.selected.role_min),
                selected.gate.selected.position.as_str(),
                &selected.gate.surname_count_min.to_string(),
                &selected.metrics.emitted.to_string(),
                &selected.metrics.correct.to_string(),
                &selected.metrics.wrong.to_string(),
                &selected.metrics.null_false_emissions.to_string(),
                &held_out_metrics.emitted.to_string(),
                &held_out_metrics.correct.to_string(),
                &held_out_metrics.wrong.to_string(),
                &held_out_metrics.null_false_emissions.to_string(),
            ])?;
        }
    }
    Ok(writer.into_inner()?)
}

fn residual_frontier(
    proxy_rows: &[ComplementRow],
    validation_rows: &[ComplementRow],
    selected: &[SelectedResidualGate],
) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "selected_on",
        "evaluated_on",
        "error_budget",
        "reference_quality_min",
        "reference_reliability_min",
        "reference_role_min",
        "reference_position",
        "surname_quality_min",
        "surname_reliability_min",
        "surname_role_min",
        "surname_position",
        "surname_count_min",
        "emitted",
        "correct",
        "wrong",
        "null_false_emissions",
        "additional_emitted",
        "additional_correct",
        "additional_wrong",
        "additional_null_false_emissions",
    ])?;
    for point in selected {
        for (label, rows) in report_populations(proxy_rows, validation_rows) {
            let (metrics, additional_metrics) =
                evaluate_residual_gate_refs(&rows, point.reference, point.surname);
            write_residual_gate_record(
                &mut writer,
                "all_spent",
                label,
                *point,
                metrics,
                additional_metrics,
            )?;
        }
    }
    Ok(writer.into_inner()?)
}

fn residual_logo(
    rows: &[ComplementRow],
    reference_gates: &[Gate],
    surname_gates: &[SurnameGate],
) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "held_out",
        "error_budget",
        "reference_quality_min",
        "reference_reliability_min",
        "reference_role_min",
        "reference_position",
        "surname_quality_min",
        "surname_reliability_min",
        "surname_role_min",
        "surname_position",
        "surname_count_min",
        "training_correct",
        "training_wrong",
        "training_null_false_emissions",
        "training_additional_correct",
        "training_additional_wrong",
        "training_additional_null_false_emissions",
        "held_out_correct",
        "held_out_wrong",
        "held_out_null_false_emissions",
        "held_out_additional_correct",
        "held_out_additional_wrong",
        "held_out_additional_null_false_emissions",
    ])?;
    for held_out in Population::PROXIES {
        for budget in ERROR_BUDGETS {
            let reference = select_gate(rows, reference_gates, budget, Some(held_out))?;
            let selected =
                select_residual_gate(rows, surname_gates, reference, Some(held_out), None)?;
            let (held_out_metrics, held_out_additional) =
                evaluate_residual_gate(rows, selected.reference, selected.surname, |row| {
                    row.population == held_out
                });
            writer.write_record([
                held_out.as_str(),
                &budget.to_string(),
                &format!("{:.3}", selected.reference.quality_min),
                &format!("{:.3}", selected.reference.reliability_min),
                &format!("{:.3}", selected.reference.role_min),
                selected.reference.position.as_str(),
                &format!("{:.3}", selected.surname.selected.quality_min),
                &format!("{:.3}", selected.surname.selected.reliability_min),
                &format!("{:.3}", selected.surname.selected.role_min),
                selected.surname.selected.position.as_str(),
                &selected.surname.surname_count_min.to_string(),
                &selected.metrics.correct.to_string(),
                &selected.metrics.wrong.to_string(),
                &selected.metrics.null_false_emissions.to_string(),
                &selected.additional_metrics.correct.to_string(),
                &selected.additional_metrics.wrong.to_string(),
                &selected.additional_metrics.null_false_emissions.to_string(),
                &held_out_metrics.correct.to_string(),
                &held_out_metrics.wrong.to_string(),
                &held_out_metrics.null_false_emissions.to_string(),
                &held_out_additional.correct.to_string(),
                &held_out_additional.wrong.to_string(),
                &held_out_additional.null_false_emissions.to_string(),
            ])?;
        }
    }
    Ok(writer.into_inner()?)
}

fn surname_storage(scan: &SurnameScan) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "scope",
        "surname_count_min",
        "key_count",
        "direct_utf8_bytes",
        "zstd19_bytes",
        "estimated_mphf_fingerprint_coarse_bytes",
        "estimated_bloom_membership_bytes_at_0_001_fpr",
    ])?;
    for (index, threshold) in SURNAME_THRESHOLDS.into_iter().enumerate() {
        let keys = scan
            .counts
            .iter()
            .filter(|(_, count)| **count >= threshold)
            .map(|(name, _)| name)
            .collect::<Vec<_>>();
        let direct_bytes = keys.iter().map(|name| name.len() + 1).sum::<usize>();
        write_storage_record(
            &mut writer,
            "encountered_complements",
            threshold,
            keys.len(),
            direct_bytes,
            STORAGE_ZSTD19_BYTES[index],
        )?;
        write_storage_record(
            &mut writer,
            "full_surname_only_source",
            threshold,
            FULL_SURNAME_ONLY_KEY_COUNTS[index],
            FULL_SURNAME_ONLY_DIRECT_BYTES[index],
            FULL_SURNAME_ONLY_ZSTD19_BYTES[index],
        )?;
    }
    Ok(writer.into_inner()?)
}

fn storage_tradeoff(selected: &[SelectedResidualGate]) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "surname_count_min",
        "full_surname_only_keys",
        "full_direct_utf8_bytes",
        "full_zstd19_bytes",
        "estimated_full_mphf_fingerprint_coarse_bytes",
        "estimated_full_bloom_membership_bytes_at_0_001_fpr",
        "union_correct",
        "union_wrong",
        "union_null_false_emissions",
        "additional_correct",
        "additional_wrong",
        "additional_null_false_emissions",
        "surname_quality_min",
        "surname_reliability_min",
        "surname_role_min",
        "surname_position",
    ])?;
    for (index, point) in selected.iter().enumerate() {
        let threshold = point.surname.surname_count_min;
        writer.write_record([
            &threshold.to_string(),
            &FULL_SURNAME_ONLY_KEY_COUNTS[index].to_string(),
            &FULL_SURNAME_ONLY_DIRECT_BYTES[index].to_string(),
            &FULL_SURNAME_ONLY_ZSTD19_BYTES[index].to_string(),
            &estimated_mphf_bytes(FULL_SURNAME_ONLY_KEY_COUNTS[index]).to_string(),
            &estimated_bloom_bytes(FULL_SURNAME_ONLY_KEY_COUNTS[index]).to_string(),
            &point.metrics.correct.to_string(),
            &point.metrics.wrong.to_string(),
            &point.metrics.null_false_emissions.to_string(),
            &point.additional_metrics.correct.to_string(),
            &point.additional_metrics.wrong.to_string(),
            &point.additional_metrics.null_false_emissions.to_string(),
            &format!("{:.3}", point.surname.selected.quality_min),
            &format!("{:.3}", point.surname.selected.reliability_min),
            &format!("{:.3}", point.surname.selected.role_min),
            point.surname.selected.position.as_str(),
        ])?;
    }
    Ok(writer.into_inner()?)
}

fn write_storage_record(
    writer: &mut csv::Writer<Vec<u8>>,
    scope: &str,
    threshold: u64,
    keys: usize,
    direct_bytes: usize,
    zstd19_bytes: usize,
) -> Result<()> {
    writer.write_record([
        scope,
        &threshold.to_string(),
        &keys.to_string(),
        &direct_bytes.to_string(),
        &zstd19_bytes.to_string(),
        &estimated_mphf_bytes(keys).to_string(),
        &estimated_bloom_bytes(keys).to_string(),
    ])?;
    Ok(())
}

fn estimated_mphf_bytes(keys: usize) -> usize {
    ((MPHF_BYTES_PER_KEY + 4.0 + 1.0) * keys as f64).ceil() as usize
}

fn estimated_bloom_bytes(keys: usize) -> usize {
    let bits_per_key = -BLOOM_FALSE_POSITIVE_RATE.ln() / 2_f64.ln().powi(2);
    (bits_per_key * keys as f64 / 8.0).ceil() as usize
}

fn write_gate_record(
    writer: &mut csv::Writer<Vec<u8>>,
    selected_on: &str,
    evaluated_on: &str,
    point: SelectedGate,
    metrics: GateMetrics,
) -> Result<()> {
    writer.write_record([
        selected_on,
        evaluated_on,
        &point.budget.to_string(),
        &format!("{:.3}", point.gate.quality_min),
        &format!("{:.3}", point.gate.reliability_min),
        &format!("{:.3}", point.gate.role_min),
        point.gate.position.as_str(),
        &metrics.emitted.to_string(),
        &metrics.correct.to_string(),
        &metrics.wrong.to_string(),
        &metrics.null_false_emissions.to_string(),
    ])?;
    Ok(())
}

fn write_surname_gate_record(
    writer: &mut csv::Writer<Vec<u8>>,
    selected_on: &str,
    evaluated_on: &str,
    point: SelectedSurnameGate,
    metrics: GateMetrics,
) -> Result<()> {
    writer.write_record([
        selected_on,
        evaluated_on,
        &point.budget.to_string(),
        &format!("{:.3}", point.gate.selected.quality_min),
        &format!("{:.3}", point.gate.selected.reliability_min),
        &format!("{:.3}", point.gate.selected.role_min),
        point.gate.selected.position.as_str(),
        &point.gate.surname_count_min.to_string(),
        &metrics.emitted.to_string(),
        &metrics.correct.to_string(),
        &metrics.wrong.to_string(),
        &metrics.null_false_emissions.to_string(),
    ])?;
    Ok(())
}

fn write_residual_gate_record(
    writer: &mut csv::Writer<Vec<u8>>,
    selected_on: &str,
    evaluated_on: &str,
    point: SelectedResidualGate,
    metrics: GateMetrics,
    additional_metrics: GateMetrics,
) -> Result<()> {
    writer.write_record([
        selected_on,
        evaluated_on,
        &point.budget.to_string(),
        &format!("{:.3}", point.reference.quality_min),
        &format!("{:.3}", point.reference.reliability_min),
        &format!("{:.3}", point.reference.role_min),
        point.reference.position.as_str(),
        &format!("{:.3}", point.surname.selected.quality_min),
        &format!("{:.3}", point.surname.selected.reliability_min),
        &format!("{:.3}", point.surname.selected.role_min),
        point.surname.selected.position.as_str(),
        &point.surname.surname_count_min.to_string(),
        &metrics.emitted.to_string(),
        &metrics.correct.to_string(),
        &metrics.wrong.to_string(),
        &metrics.null_false_emissions.to_string(),
        &additional_metrics.emitted.to_string(),
        &additional_metrics.correct.to_string(),
        &additional_metrics.wrong.to_string(),
        &additional_metrics.null_false_emissions.to_string(),
    ])?;
    Ok(())
}

fn evaluate_gate_refs(rows: &[&ComplementRow], gate: Gate) -> GateMetrics {
    let mut metrics = GateMetrics::default();
    for row in rows {
        metrics.observe(row, gate.emits(row));
    }
    metrics
}

fn evaluate_surname_gate_refs(rows: &[&ComplementRow], gate: SurnameGate) -> GateMetrics {
    let mut metrics = GateMetrics::default();
    for row in rows {
        metrics.observe(row, gate.emits(row));
    }
    metrics
}

fn evaluate_residual_gate_refs(
    rows: &[&ComplementRow],
    reference: Gate,
    surname: SurnameGate,
) -> (GateMetrics, GateMetrics) {
    let mut metrics = GateMetrics::default();
    let mut additional_metrics = GateMetrics::default();
    for row in rows {
        let reference_emits = reference.emits(row);
        let surname_emits = surname.emits(row);
        metrics.observe(row, reference_emits || surname_emits);
        additional_metrics.observe(row, surname_emits && !reference_emits);
    }
    (metrics, additional_metrics)
}

fn qualitative_csv(probes: &[ProbeResult]) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "probe",
        "selected_candidate",
        "outcome",
        "topology",
        "candidate_count",
        "selected_quality",
        "selected_role_signal",
        "selected_reliability",
        "selected_position",
        "complement_class",
        "complement_given_count",
        "complement_retained_surname_count",
        "complement_retained_role_llr",
        "complement_raw_surname_count",
        "complement_raw_role_llr",
        "vetoes_pass",
        "generic_organization_marker",
        "c5_emits",
        "reference_budget_0_emits",
        "reference_budget_1_emits",
        "reference_budget_5_emits",
        "reference_budget_10_emits",
        "reference_budget_25_emits",
        "surname_budget_0_emits",
        "surname_budget_1_emits",
        "surname_budget_5_emits",
        "surname_budget_10_emits",
        "surname_budget_25_emits",
    ])?;
    for probe in probes {
        let emission = |budget| {
            probe
                .gate_emissions
                .iter()
                .find_map(|(candidate, emits)| (*candidate == budget).then_some(*emits))
                .unwrap_or(false)
        };
        let surname_emission = |budget| {
            probe
                .surname_gate_emissions
                .iter()
                .find_map(|(candidate, emits)| (*candidate == budget).then_some(*emits))
                .unwrap_or(false)
        };
        writer.write_record([
            probe.label.as_str(),
            probe.row.selected_candidate.as_deref().unwrap_or(""),
            probe.row.outcome.as_str(),
            probe.row.topology.as_str(),
            &probe.row.candidate_count.to_string(),
            &format!("{:.6}", probe.row.candidate_quality),
            &format!("{:.6}", probe.row.role_signal),
            &format!("{:.6}", probe.row.reliability),
            probe.row.position.as_str(),
            probe.row.complement_class.as_str(),
            &format_optional_u64(probe.row.complement_given_count),
            &format_optional_u64(probe.row.complement_retained_surname_count),
            &format_optional(probe.row.complement_role_llr),
            &format_optional_u64(probe.row.complement_raw_surname_count),
            &format_optional(probe.row.complement_raw_role_llr),
            bool_string(probe.row.vetoes_pass),
            bool_string(probe.row.generic_organization_marker),
            bool_string(probe.row.c5_emits),
            bool_string(emission(0)),
            bool_string(emission(1)),
            bool_string(emission(5)),
            bool_string(emission(10)),
            bool_string(emission(25)),
            bool_string(surname_emission(0)),
            bool_string(surname_emission(1)),
            bool_string(surname_emission(5)),
            bool_string(surname_emission(10)),
            bool_string(surname_emission(25)),
        ])?;
    }
    Ok(writer.into_inner()?)
}

fn report(context: &DiagnosticContext<'_>) -> String {
    let DiagnosticContext {
        proxy_rows,
        validation_rows,
        gates,
        surname_gates,
        selected,
        selected_surname,
        selected_residual,
        selected_storage_tradeoff,
        probes,
        surname_scan,
    } = *context;
    let primary_proxy = proxy_rows.iter().filter(|row| row.primary()).count();
    let primary_validation = validation_rows.iter().filter(|row| row.primary()).count();
    let proxy_outcomes = aggregate_outcomes(proxy_rows.iter().filter(|row| row.primary()));
    let validation_outcomes =
        aggregate_outcomes(validation_rows.iter().filter(|row| row.primary()));
    let all_sole_proxy_outcomes =
        aggregate_outcomes(proxy_rows.iter().filter(|row| row.sole_native_winner()));
    let first_outcomes = aggregate_outcomes(
        proxy_rows
            .iter()
            .filter(|row| row.primary() && row.position == Position::First),
    );
    let second_outcomes = aggregate_outcomes(
        proxy_rows
            .iter()
            .filter(|row| row.primary() && row.position == Position::Second),
    );
    let distinct_proxy_complements = proxy_rows
        .iter()
        .filter(|row| row.primary())
        .filter_map(|row| row.complement_normalized.as_deref())
        .collect::<BTreeSet<_>>()
        .len();
    let retained_proxy = proxy_rows
        .iter()
        .filter(|row| row.primary() && row.complement_given_count.is_some())
        .count();
    let surname_observed_outcomes =
        aggregate_outcomes(proxy_rows.iter().filter(|row| {
            row.primary() && row.complement_class == ComplementClass::SurnameObserved
        }));
    let unknown_outcomes = aggregate_outcomes(
        proxy_rows
            .iter()
            .filter(|row| row.primary() && row.complement_class == ComplementClass::Unknown),
    );
    let reference_zero = selected
        .iter()
        .find(|point| point.budget == 0)
        .expect("zero-error reference point");
    let surname_zero = selected_surname
        .iter()
        .find(|point| point.budget == 0)
        .expect("zero-error surname point");
    let residual_zero = selected_residual
        .iter()
        .find(|point| point.budget == 0)
        .expect("zero-error residual point");
    let residual_zero_logo = Population::PROXIES
        .into_iter()
        .map(|held_out| {
            let reference =
                select_gate(proxy_rows, gates, 0, Some(held_out)).expect("reference LOGO gate");
            let selected =
                select_residual_gate(proxy_rows, surname_gates, reference, Some(held_out), None)
                    .expect("residual LOGO gate");
            let (_, additional) =
                evaluate_residual_gate(proxy_rows, selected.reference, selected.surname, |row| {
                    row.population == held_out
                });
            (held_out, additional)
        })
        .collect::<Vec<_>>();
    let residual_zero_logo_errors = residual_zero_logo
        .iter()
        .map(|(_, metrics)| metrics.errors())
        .sum::<usize>();
    let storage_count_25 = selected_storage_tradeoff
        .iter()
        .find(|point| point.surname.surname_count_min == 25)
        .expect("count-25 storage point");
    let mut output = String::new();
    writeln!(output, "# Sole-candidate complement-evidence diagnostic\n").unwrap();
    writeln!(output, "## Source and integrity\n").unwrap();
    writeln!(output, "The authoritative raw surname source was scanned remotely using exact UTF-8 byte equality, matching the existing `name-surname-v2` semantics. The scan covered {RAW_FILES} CSV files, {RAW_ROWS} person rows, and {SURNAME_TOTAL} non-empty surnames. It counted only {distinct_proxy_complements} distinct spent-proxy complement keys plus the qualitative/safety probes. The private count join SHA-256 is `{}`; it contains {} matched keys and {} matched surname observations.\n", surname_scan.counts_sha256, surname_scan.matched_keys, surname_scan.matched_observations).unwrap();
    writeln!(output, "The compact artifact still has no surname-only keys. In the primary topology a complement present in the retained given-name index would normally be candidate two; the observed retained-complement count remains {retained_proxy}. Raw surname count zero is therefore kept distinct from missing scan data and from positive surname evidence.\n").unwrap();
    writeln!(output, "## Population\n").unwrap();
    writeln!(output, "The strict primary topology (native, exactly two alphabetic lexical tokens, single-token winner, exactly one candidate, C5 abstention, all vetoes passing) contained {primary_proxy} pooled spent-proxy rows and {primary_validation} synthetic VALIDATION rows.\n").unwrap();
    writeln!(output, "Across every spent-proxy sole-native winner before the strict topology filter, there were {} correct winners, {} wrong winners, and {} expected-NULL winners. Their token-count, topology, veto, and frozen-C5 partitions are in `sole_native_outcome_summary.csv`.\n", all_sole_proxy_outcomes[0], all_sole_proxy_outcomes[1], all_sole_proxy_outcomes[2]).unwrap();
    writeln!(
        output,
        "| Population | Correct winner | Wrong winner | Expected NULL |\n|---|---:|---:|---:|"
    )
    .unwrap();
    writeln!(
        output,
        "| Spent proxies | {} | {} | {} |",
        proxy_outcomes[0], proxy_outcomes[1], proxy_outcomes[2]
    )
    .unwrap();
    writeln!(
        output,
        "| VALIDATION | {} | {} | {} |\n",
        validation_outcomes[0], validation_outcomes[1], validation_outcomes[2]
    )
    .unwrap();
    writeln!(output, "| Complement state | Correct winner | Wrong winner | Expected NULL |\n|---|---:|---:|---:|\n| Positive raw surname evidence (A) | {} | {} | {} |\n| No raw surname evidence (B) | {} | {} | {} |\n", surname_observed_outcomes[0], surname_observed_outcomes[1], surname_observed_outcomes[2], unknown_outcomes[0], unknown_outcomes[1], unknown_outcomes[2]).unwrap();
    writeln!(output, "The no-surname-evidence population contains both correct and unsafe sole winners. Consequently, `complement absent from given corpus` remains inadmissible as positive evidence. Organization/legal/common-word controls remain governed by the frozen vetoes and are reported separately.\n").unwrap();
    writeln!(output, "Position strongly separates this spent-proxy topology: first-token selections contain {} correct winners, {} wrong winners, and {} expected-NULL winners, while second-token selections contain {}, {}, and {}, respectively. None of the strict primary rows has a country or locale hint, so the existing name-order prior is unavailable here; this is a raw position result only.\n", first_outcomes[0], first_outcomes[1], first_outcomes[2], second_outcomes[0], second_outcomes[1], second_outcomes[2]).unwrap();
    writeln!(output, "## Selected-candidate-only reference frontier\n").unwrap();
    writeln!(output, "This frontier is a control, not a proposed classifier. It searches the declared quality/reliability/role/position grid with no complement condition. Counts are additional emissions over frozen C5 abstentions.\n").unwrap();
    writeln!(output, "| Error budget | Gate (Q / R / role / position) | Correct | Wrong | NULL FP | VALIDATION correct | VALIDATION wrong | VALIDATION NULL FP |\n|---:|---|---:|---:|---:|---:|---:|---:|").unwrap();
    for point in selected {
        let validation = evaluate_gate(validation_rows, point.gate, |_| true);
        writeln!(
            output,
            "| {} | {:.3} / {:.3} / {:.3} / {} | {} | {} | {} | {} | {} | {} |",
            point.budget,
            point.gate.quality_min,
            point.gate.reliability_min,
            point.gate.role_min,
            point.gate.position.as_str(),
            point.metrics.correct,
            point.metrics.wrong,
            point.metrics.null_false_emissions,
            validation.correct,
            validation.wrong,
            validation.null_false_emissions,
        )
        .unwrap();
    }
    let zero_error_logo = Population::PROXIES
        .into_iter()
        .map(|held_out| {
            evaluate_gate(proxy_rows, reference_zero.gate, |row| {
                row.population == held_out
            })
        })
        .collect::<Vec<_>>();
    let zero_error_logo_correct = zero_error_logo
        .iter()
        .map(|metrics| metrics.correct.to_string())
        .collect::<Vec<_>>()
        .join("/");
    let zero_error_logo_errors = zero_error_logo
        .iter()
        .map(|metrics| metrics.errors())
        .sum::<usize>();
    writeln!(output, "\nThe pooled zero-error control uses {} position with Q >= {:.3}, reliability >= {:.3}, and role >= {:.3}. Applied unchanged to V1/V2/V3/V4/V5, it recovers {zero_error_logo_correct} correct cases with {zero_error_logo_errors} total observed errors. Details are in `reference_logo.csv`. Synthetic VALIDATION supplies no rows in the exact primary topology.\n", reference_zero.gate.position.as_str(), reference_zero.gate.quality_min, reference_zero.gate.reliability_min, reference_zero.gate.role_min).unwrap();
    writeln!(output, "## Positive-surname complement frontier\n").unwrap();
    writeln!(output, "This is the paired search: the same Q/reliability/role/position grid plus `raw complement surname count >= S`. Only class A (no given evidence and positive exact raw surname evidence) is eligible.\n").unwrap();
    writeln!(output, "| Error budget | Gate (Q / R / role / position / surname count) | Correct | Wrong | NULL FP | Delta correct vs reference |\n|---:|---|---:|---:|---:|---:|").unwrap();
    for (reference, surname) in selected.iter().zip(selected_surname) {
        writeln!(
            output,
            "| {} | {:.3} / {:.3} / {:.3} / {} / {} | {} | {} | {} | {:+} |",
            surname.budget,
            surname.gate.selected.quality_min,
            surname.gate.selected.reliability_min,
            surname.gate.selected.role_min,
            surname.gate.selected.position.as_str(),
            surname.gate.surname_count_min,
            surname.metrics.correct,
            surname.metrics.wrong,
            surname.metrics.null_false_emissions,
            surname.metrics.correct as isize - reference.metrics.correct as isize
        )
        .unwrap();
    }
    writeln!(output, "\nLeave-one-generation-out results for the surname-aware family are in `surname_logo.csv`.\n").unwrap();
    writeln!(
        output,
        "## Residual surname value beyond the reference gate\n"
    )
    .unwrap();
    writeln!(output, "This final comparison holds the selected-candidate-only gate fixed at each error budget, then searches for a surname-conditioned relaxed gate whose emissions are unioned with it. `Additional` counts include only rows not already emitted by the reference gate.\n").unwrap();
    writeln!(output, "| Error budget | Reference correct | Union correct | Additional correct | Additional wrong | Additional NULL FP | Surname gate (Q / R / role / position / count) |\n|---:|---:|---:|---:|---:|---:|---|").unwrap();
    for (reference, residual) in selected.iter().zip(selected_residual) {
        writeln!(
            output,
            "| {} | {} | {} | {} | {} | {} | {:.3} / {:.3} / {:.3} / {} / {} |",
            residual.budget,
            reference.metrics.correct,
            residual.metrics.correct,
            residual.additional_metrics.correct,
            residual.additional_metrics.wrong,
            residual.additional_metrics.null_false_emissions,
            residual.surname.selected.quality_min,
            residual.surname.selected.reliability_min,
            residual.surname.selected.role_min,
            residual.surname.selected.position.as_str(),
            residual.surname.surname_count_min,
        )
        .unwrap();
    }
    writeln!(output, "\n`residual_logo.csv` repeats the paired selection while holding out each proxy generation. At the zero-error training budget the held-out additional outcomes are:\n").unwrap();
    writeln!(output, "| Held-out population | Additional correct | Additional wrong | Additional NULL FP |\n|---|---:|---:|---:|").unwrap();
    for (population, metrics) in &residual_zero_logo {
        writeln!(
            output,
            "| {} | {} | {} | {} |",
            population.as_str(),
            metrics.correct,
            metrics.wrong,
            metrics.null_false_emissions,
        )
        .unwrap();
    }
    writeln!(output, "\nThis is the direct test of whether positive complement-surname evidence permits a safe relaxation unavailable to the existing selected-candidate signals. It adds correct held-out cases in all five generations, but incurs {residual_zero_logo_errors} held-out error across the five independently selected zero-training-error gates.\n").unwrap();
    writeln!(output, "## Qualitative probes\n").unwrap();
    writeln!(output, "The probes were run only after the grid and pooled operating points were frozen. Raw person-name strings were used locally for inference; only redacted labels are serialized.\n").unwrap();
    writeln!(output, "| Probe | Selected | Q | Role | Reliability | Raw surname count | Complement state | Frozen C5 | Zero-error reference | Zero-error surname gate |\n|---|---|---:|---:|---:|---:|---|---|---|---|").unwrap();
    for probe in probes {
        let zero_error = probe
            .gate_emissions
            .iter()
            .find_map(|(budget, emits)| (*budget == 0).then_some(*emits))
            .unwrap_or(false);
        let surname_zero_error = probe
            .surname_gate_emissions
            .iter()
            .find_map(|(budget, emits)| (*budget == 0).then_some(*emits))
            .unwrap_or(false);
        writeln!(
            output,
            "| {} | {} | {:.3} | {:.3} | {:.3} | {} | {} | {} | {} | {} |",
            probe.label,
            probe.row.selected_candidate.as_deref().unwrap_or("none"),
            probe.row.candidate_quality,
            probe.row.role_signal,
            probe.row.reliability,
            format_optional_u64(probe.row.complement_raw_surname_count),
            probe.row.complement_class.as_str(),
            yes_no(probe.row.c5_emits),
            yes_no(zero_error),
            yes_no(surname_zero_error),
        )
        .unwrap();
    }
    writeln!(output, "\n`Motorcycle Club` remains an abstention under frozen C5 and every searched gate because all existing vetoes are inherited unchanged, even though its complement has a non-zero raw surname count.\n").unwrap();
    writeln!(output, "## Storage feasibility\n").unwrap();
    writeln!(output, "`surname_storage.csv` reports both the encountered complement keys and a full exact-source scan of surname-observed keys absent from the 1,803,175 retained given-name keys (which contain {GIVEN_TOTAL} observations). Sizes are actual sorted UTF-8 and zstd-19 measurements; MPHF + 32-bit fingerprint + one-byte evidence and Bloom membership are estimates. At the zero-error surname threshold (count >= {}), {} of the 607 queried keys qualify.\n", surname_zero.gate.surname_count_min, surname_scan.counts.values().filter(|count| **count >= surname_zero.gate.surname_count_min).count()).unwrap();
    writeln!(output, "| Minimum raw surname count | Full surname-only keys | Direct bytes | zstd-19 bytes | Estimated MPHF + fingerprint + evidence | Zero-error additional correct |\n|---:|---:|---:|---:|---:|---:|").unwrap();
    for (index, point) in selected_storage_tradeoff.iter().enumerate() {
        writeln!(
            output,
            "| {} | {} | {} | {} | {} | {} |",
            point.surname.surname_count_min,
            FULL_SURNAME_ONLY_KEY_COUNTS[index],
            FULL_SURNAME_ONLY_DIRECT_BYTES[index],
            FULL_SURNAME_ONLY_ZSTD19_BYTES[index],
            estimated_mphf_bytes(FULL_SURNAME_ONLY_KEY_COUNTS[index]),
            point.additional_metrics.correct,
        )
        .unwrap();
    }
    writeln!(output, "\n`storage_tradeoff.csv` joins this full-source size curve to the best zero-error residual gate at each fixed surname threshold.\n").unwrap();
    writeln!(output, "## Explicit answers\n").unwrap();
    writeln!(output, "- **Is low given-name plausibility of the complement useful?** Absence/low evidence alone is unsafe. Its value must be conditioned on positive surname evidence and existing personhood controls.").unwrap();
    writeln!(output, "- **Is absence from the given-name corpus useful?** Unsafe by itself: class B includes wrong and expected-NULL winners as well as correct winners.").unwrap();
    writeln!(output, "- **Does positive surname evidence materially improve safety?** Yes, conditionally on spent proxies. As a standalone gate it recovers {} correct versus {} for the reference search at zero pooled errors ({:+}), but as an additive residual gate it contributes {} correct, {} wrong, and {} NULL false emissions beyond the reference. Zero-error LOGO selection adds correct cases in every generation but incurs {residual_zero_logo_errors} held-out error.", surname_zero.metrics.correct, reference_zero.metrics.correct, surname_zero.metrics.correct as isize - reference_zero.metrics.correct as isize, residual_zero.additional_metrics.correct, residual_zero.additional_metrics.wrong, residual_zero.additional_metrics.null_false_emissions).unwrap();
    writeln!(output, "- **Does position materially separate outcomes?** Yes on spent proxies: first-token winners are overwhelmingly safer than second-token winners, and the selected zero-error first-position gate survives every proxy generation holdout. This remains proxy-only because VALIDATION has no eligible rows and no order hints are present.").unwrap();
    writeln!(output, "- **Is there a complement-based zero/low-error rule?** Yes on pooled spent evidence: the additive count >= 1 rule recovers 66 beyond the zero-error reference. It is not yet a safe production rule because one LOGO fold has a NULL false emission and VALIDATION has no eligible rows.").unwrap();
    writeln!(output, "- **What would production support cost?** Count >= 1 would require an estimated {} bytes for MPHF + fingerprint + evidence. A count >= 25 compromise requires {} bytes and retains {} zero-error pooled additions. Exact threshold curves are in `surname_storage.csv` and `storage_tradeoff.csv`.", estimated_mphf_bytes(FULL_SURNAME_ONLY_KEY_COUNTS[0]), estimated_mphf_bytes(FULL_SURNAME_ONLY_KEY_COUNTS[4]), storage_count_25.additional_metrics.correct).unwrap();
    writeln!(output, "- **Do the named person probes fall inside the selected region?** Martin REDACTED and Olivier REDACTED fall inside both zero-error regions. Baris REDACTED is a two-candidate case and falls outside this experiment. The probes did not select thresholds.").unwrap();
    writeln!(output, "\n## Recommendation\n").unwrap();
    writeln!(output, "Keep positive complement-surname evidence as a promising experimental feature, but do not integrate or build a production index yet. It has substantial residual value on spent proxies (+{} correct at zero pooled errors), including gains in every generation holdout, but the zero-training-error LOGO selection produces {residual_zero_logo_errors} held-out error and synthetic VALIDATION has no eligible rows. A fresh preregistered holdout is required before choosing between the high-coverage count >= 1 representation and a smaller thresholded index. This experiment stops at diagnosis; no production rule or artifact is changed.", residual_zero.additional_metrics.correct).unwrap();
    output
}

fn report_populations<'a>(
    proxy_rows: &'a [ComplementRow],
    validation_rows: &'a [ComplementRow],
) -> Vec<(&'static str, Vec<&'a ComplementRow>)> {
    let mut populations = Population::PROXIES
        .into_iter()
        .map(|population| {
            (
                population.as_str(),
                proxy_rows
                    .iter()
                    .filter(|row| row.population == population)
                    .collect(),
            )
        })
        .collect::<Vec<_>>();
    populations.push(("SPENT_POOLED", proxy_rows.iter().collect()));
    populations.push(("VALIDATION", validation_rows.iter().collect()));
    populations
}

fn aggregate_outcomes<'a>(rows: impl Iterator<Item = &'a ComplementRow>) -> [usize; 3] {
    let mut counts = [0; 3];
    for row in rows {
        match row.outcome {
            Outcome::CorrectWinner => counts[0] += 1,
            Outcome::WrongWinner => counts[1] += 1,
            Outcome::ExpectedNull => counts[2] += 1,
        }
    }
    counts
}

fn percentile_summary(mut values: Vec<f64>) -> PercentileSummary {
    values.sort_by(f64::total_cmp);
    PercentileSummary {
        count: values.len(),
        p10: percentile(&values, 0.10),
        p25: percentile(&values, 0.25),
        p50: percentile(&values, 0.50),
        p75: percentile(&values, 0.75),
        p90: percentile(&values, 0.90),
    }
}

fn percentile(values: &[f64], fraction: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let index = ((values.len() - 1) as f64 * fraction).round() as usize;
    values.get(index).copied()
}

fn format_optional(value: Option<f64>) -> String {
    value.map_or_else(String::new, |value| format!("{value:.6}"))
}

fn format_optional_u64(value: Option<u64>) -> String {
    value.map_or_else(String::new, |value| value.to_string())
}

fn bool_string(value: bool) -> &'static str {
    if value { "true" } else { "false" }
}

fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

fn nonempty(value: &str) -> Option<&str> {
    (!value.is_empty()).then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complement_index_requires_single_token_winner_in_two_tokens() {
        let first = candidate(0, 1);
        let second = candidate(1, 1);
        let whole = candidate(0, 2);
        assert_eq!(complement_index(Some(&first), 2), Some(1));
        assert_eq!(complement_index(Some(&second), 2), Some(0));
        assert_eq!(complement_index(Some(&whole), 2), None);
        assert_eq!(complement_index(Some(&first), 3), None);
    }

    #[test]
    fn reference_gate_inherits_primary_shape_and_vetoes() {
        let gate = Gate {
            quality_min: 0.4,
            reliability_min: 0.4,
            role_min: 0.2,
            position: PositionRule::Any,
        };
        let mut row = row();
        assert!(gate.emits(&row));
        row.vetoes_pass = false;
        assert!(!gate.emits(&row));
        row.vetoes_pass = true;
        row.topology = Topology::NativeTwoTokenSoleVetoed;
        assert!(!gate.emits(&row));
    }

    #[test]
    fn stricter_thresholds_cannot_add_emissions() {
        let row = row();
        let loose = Gate {
            quality_min: 0.4,
            reliability_min: 0.4,
            role_min: 0.2,
            position: PositionRule::Any,
        };
        let strict = Gate {
            quality_min: 0.7,
            reliability_min: 0.8,
            role_min: 0.6,
            position: PositionRule::Any,
        };
        assert!(loose.emits(&row));
        assert!(!strict.emits(&row));
    }

    #[test]
    fn surname_gate_requires_positive_surname_evidence() {
        let selected = Gate {
            quality_min: 0.4,
            reliability_min: 0.4,
            role_min: 0.2,
            position: PositionRule::Any,
        };
        let gate = SurnameGate {
            selected,
            surname_count_min: 5,
        };
        let mut row = row();
        assert!(!gate.emits(&row));
        row.complement_class = ComplementClass::SurnameObserved;
        assert!(gate.emits(&row));
        row.complement_raw_surname_count = Some(4);
        assert!(!gate.emits(&row));
    }

    #[test]
    fn raw_surname_zero_stays_distinct_from_positive_evidence() {
        let mut absent = row();
        absent.attach_raw_surname_count(0);
        assert_eq!(absent.complement_class, ComplementClass::Unknown);
        assert_eq!(absent.complement_raw_surname_count, Some(0));

        let mut observed = row();
        observed.attach_raw_surname_count(1);
        assert_eq!(observed.complement_class, ComplementClass::SurnameObserved);
    }

    #[test]
    fn redacted_probe_label_never_uses_display_name() {
        let probe = ProbeInput {
            case_id: "person_1".to_string(),
            display_name: "Example Private".to_string(),
            expected_greeting: "Example".to_string(),
            country_hint: String::new(),
            locale_hint: String::new(),
            redact_complement: true,
        };
        assert_eq!(probe_label(&probe).unwrap(), "Example REDACTED");
    }

    fn candidate(start: usize, length: usize) -> CandidateDiagnostic {
        CandidateDiagnostic {
            display: "Example".to_string(),
            start,
            length,
            byte_start: None,
            byte_end: None,
            global_given_count: 1,
            country_given_count: 0,
            effective_given_count: 1,
            female_given_count: 0,
            male_given_count: 0,
            global_surname_count: 0,
            role_llr: 0.0,
            role_signal: 0.5,
            reliability: 0.5,
            country_support: 0.0,
            compound_evidence: 0.0,
            compositional_evidence: 0.0,
            remainder_evidence: 0.0,
            origin: "native",
            segmentation_mechanism: None,
            lookup_query: None,
            lookup_mode: None,
            left_lookup_mode: None,
            right_lookup_mode: None,
            score: 0.5,
            algorithm_a_score: 0.5,
            algorithm_b_score: 0.5,
        }
    }

    fn row() -> ComplementRow {
        ComplementRow {
            population: Population::V1,
            outcome: Outcome::CorrectWinner,
            topology: Topology::NativeTwoTokenSolePrimary,
            native: true,
            token_count: 2,
            candidate_count: 1,
            selected_candidate: Some("Example".to_string()),
            candidate_quality: 0.6,
            role_signal: 0.5,
            reliability: 0.7,
            position: Position::First,
            order_prior: NameOrderPrior::GivenFirst,
            agrees_with_order_prior: true,
            country_hint_present: false,
            c5_emits: false,
            vetoes_pass: true,
            hard_organization_marker: false,
            generic_organization_marker: false,
            ampersand: false,
            candidate_too_short: false,
            complement_class: ComplementClass::Unknown,
            complement_lexically_eligible: Some(true),
            complement_alphabetic_length: Some(7),
            complement_capitalization: Some("title"),
            complement_given_count: None,
            complement_retained_surname_count: None,
            complement_role_llr: None,
            complement_raw_surname_count: Some(10),
            complement_raw_role_llr: Some(-1.0),
            complement_normalized: Some("Private".to_string()),
        }
    }
}
