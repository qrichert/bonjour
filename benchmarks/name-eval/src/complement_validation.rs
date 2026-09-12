use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::Path;

use name_eval::holdout::{FrozenHoldout, SealedMetrics, evaluate_explicit_emissions};
use sha2::{Digest, Sha256};

use crate::artifact::EvidenceSource;
use crate::classifier::{
    ALGORITHM_C2, ALGORITHM_C3, ALGORITHM_C4, ALGORITHM_C5, ALGORITHM_C31, C4DecisionBreakdown,
    CandidateDiagnostic, c4_decision_breakdown, c5_decision_from_c4, c5_emitted_candidate,
    canonicalize, diagnose_role_inference, expected_lookup_diagnostic,
};
use crate::metrics::greeting_matches;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const RAW_FILES: usize = 105;
const RAW_ROWS: u64 = 491_655_925;
const NONEMPTY_SURNAMES: u64 = 489_631_377;
const REFERENCE_QUALITY_MIN: f64 = 0.50;
const REFERENCE_RELIABILITY_MIN: f64 = 0.40;
const REFERENCE_ROLE_MIN: f64 = 0.20;
const SURNAME_QUALITY_MIN: f64 = 0.40;
const SURNAME_RELIABILITY_MIN: f64 = 0.00;
const SURNAME_ROLE_MIN: f64 = 0.30;
const SURNAME_COUNT_MIN: u64 = 1;
const SUBSTANTIAL_CORRECT_MIN: usize = 5;
const MIXED_ERROR_MAX: usize = 1;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum ComplementClass {
    SurnameObserved,
    Unknown,
    OrganizationOrLexicalNegative,
    GivenObserved,
}

impl ComplementClass {
    const ALL: [Self; 4] = [
        Self::SurnameObserved,
        Self::Unknown,
        Self::OrganizationOrLexicalNegative,
        Self::GivenObserved,
    ];

    fn as_str(self) -> &'static str {
        match self {
            Self::SurnameObserved => "A_surname_observed_no_given_evidence",
            Self::Unknown => "B_unknown_no_surname_evidence",
            Self::OrganizationOrLexicalNegative => "C_organization_or_lexical_negative",
            Self::GivenObserved => "D_given_observed",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WinnerOutcome {
    Correct,
    Wrong,
    ExpectedNull,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct AdditionMetrics {
    emitted: usize,
    correct: usize,
    wrong: usize,
    null_false_emissions: usize,
}

impl AdditionMetrics {
    fn from_sealed(metrics: SealedMetrics) -> Self {
        Self {
            emitted: metrics.emitted_greetings,
            correct: metrics.correct_greetings,
            wrong: metrics.wrong_greetings,
            null_false_emissions: metrics.false_emissions_on_expected_abstentions,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct ClassMetrics {
    rows: usize,
    correct_winners: usize,
    wrong_winners: usize,
    expected_null_winners: usize,
    reference_emissions: usize,
    surname_residual_emissions: usize,
}

impl ClassMetrics {
    fn observe(&mut self, outcome: WinnerOutcome, reference: bool, surname: bool) {
        self.rows += 1;
        match outcome {
            WinnerOutcome::Correct => self.correct_winners += 1,
            WinnerOutcome::Wrong => self.wrong_winners += 1,
            WinnerOutcome::ExpectedNull => self.expected_null_winners += 1,
        }
        self.reference_emissions += usize::from(reference);
        self.surname_residual_emissions += usize::from(surname);
    }
}

#[derive(Clone, Debug)]
struct ValidationRow {
    selected_candidate: Option<String>,
    c5_emission: Option<String>,
    strict_topology: bool,
    selected_first: bool,
    candidate_quality: f64,
    reliability: f64,
    role_signal: f64,
    complement_class: Option<ComplementClass>,
    complement_surname_count: Option<u64>,
}

impl ValidationRow {
    fn reference_emission(&self) -> Option<String> {
        (self.strict_topology
            && self.selected_first
            && self.candidate_quality >= REFERENCE_QUALITY_MIN
            && self.reliability >= REFERENCE_RELIABILITY_MIN
            && self.role_signal >= REFERENCE_ROLE_MIN)
            .then(|| self.selected_candidate.clone())
            .flatten()
    }

    fn surname_emission(&self) -> Option<String> {
        (self.strict_topology
            && self.selected_first
            && self.candidate_quality >= SURNAME_QUALITY_MIN
            && self.reliability >= SURNAME_RELIABILITY_MIN
            && self.role_signal >= SURNAME_ROLE_MIN
            && self.complement_class == Some(ComplementClass::SurnameObserved)
            && self
                .complement_surname_count
                .is_some_and(|count| count >= SURNAME_COUNT_MIN))
        .then(|| self.selected_candidate.clone())
        .flatten()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SurnameScan {
    counts: BTreeMap<String, u64>,
    counts_sha256: String,
    matched_keys: usize,
    matched_observations: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ValidationResult {
    c5: SealedMetrics,
    reference_policy: SealedMetrics,
    combined_policy: SealedMetrics,
    reference_additions: AdditionMetrics,
    surname_additions: AdditionMetrics,
    combined_additions: AdditionMetrics,
    class_metrics: BTreeMap<ComplementClass, ClassMetrics>,
    strict_topology_rows: usize,
    scan: SurnameScan,
}

pub(crate) fn prepare_complement_lookup(output: &Path, holdout: &FrozenHoldout) -> Result<String> {
    let keys = expected_lookup_keys(holdout);
    let key_bytes = serialize_lookup_keys(&keys)?;
    let key_sha256 = sha256_hex(&key_bytes);
    fs::write(output.join("lookup_keys.csv"), &key_bytes)?;
    fs::write(
        output.join("lookup_manifest.csv"),
        lookup_manifest_csv(holdout, keys.len(), &key_sha256)?,
    )?;

    let mut report = String::new();
    writeln!(report, "# Private V7 complement lookup preparation\n").unwrap();
    writeln!(
        report,
        "The sealed holdout `{}` was checksum-verified before preparing a classifier-blind lexical key superset. No classifier artifact was opened, no inference was run, and labels were not used to select keys.\n",
        holdout.manifest.holdout_sha256
    )
    .unwrap();
    writeln!(report, "- Source rows: {}", holdout.manifest.total_cases).unwrap();
    writeln!(report, "- Distinct lookup keys: {}", keys.len()).unwrap();
    writeln!(report, "- Lookup-key SHA-256: `{key_sha256}`\n").unwrap();
    writeln!(report, "`lookup_keys.csv` is temporary unredacted material and must remain under the ignored work directory, then be deleted after the aggregate validation.").unwrap();
    Ok(report)
}

pub(crate) fn run_complement_validation(
    output: &Path,
    corpus: &impl EvidenceSource,
    holdout: &FrozenHoldout,
    surname_counts: &Path,
    surname_manifest: &Path,
) -> Result<String> {
    let expected_keys = expected_lookup_keys(holdout);
    let scan = load_surname_scan(
        surname_counts,
        surname_manifest,
        &expected_keys,
        &holdout.manifest.holdout_sha256,
    )?;
    validate_safety_control(corpus, &scan.counts)?;
    let result = evaluate_validation(corpus, holdout, scan)?;
    let outputs = build_outputs(holdout, &result)?;
    let repeated = build_outputs(holdout, &result)?;
    if outputs != repeated {
        return Err("V7 complement validation serialization is not deterministic".into());
    }
    for (name, bytes) in &outputs {
        fs::write(output.join(name), bytes)?;
    }
    Ok(String::from_utf8(
        outputs
            .get("validation_report.md")
            .ok_or("V7 validation report missing")?
            .clone(),
    )?)
}

fn evaluate_validation(
    corpus: &impl EvidenceSource,
    holdout: &FrozenHoldout,
    scan: SurnameScan,
) -> Result<ValidationResult> {
    let mut c5_emissions = Vec::with_capacity(holdout.cases.len());
    let mut reference_policy_emissions = Vec::with_capacity(holdout.cases.len());
    let mut combined_policy_emissions = Vec::with_capacity(holdout.cases.len());
    let mut reference_only_emissions = Vec::with_capacity(holdout.cases.len());
    let mut surname_only_emissions = Vec::with_capacity(holdout.cases.len());
    let mut combined_only_emissions = Vec::with_capacity(holdout.cases.len());
    let mut class_metrics = BTreeMap::new();
    let mut strict_topology_rows = 0;

    for case in &holdout.cases {
        if !case.is_evaluable() {
            c5_emissions.push(None);
            reference_policy_emissions.push(None);
            combined_policy_emissions.push(None);
            reference_only_emissions.push(None);
            surname_only_emissions.push(None);
            combined_only_emissions.push(None);
            continue;
        }
        let row = build_validation_row(
            corpus,
            &scan.counts,
            &case.display_name,
            nonempty(&case.country_hint),
            nonempty(&case.locale_hint),
        )?;
        let reference = row.reference_emission();
        let surname = reference
            .is_none()
            .then(|| row.surname_emission())
            .flatten();
        let reference_policy = row.c5_emission.clone().or_else(|| reference.clone());
        let combined_policy = reference_policy.clone().or_else(|| surname.clone());

        strict_topology_rows += usize::from(row.strict_topology);
        if let Some(class) = row.complement_class {
            class_metrics
                .entry(class)
                .or_insert_with(ClassMetrics::default)
                .observe(
                    winner_outcome(case.expected_greeting(), row.selected_candidate.as_deref()),
                    reference.is_some(),
                    surname.is_some(),
                );
        }
        c5_emissions.push(row.c5_emission);
        reference_policy_emissions.push(reference_policy);
        combined_policy_emissions.push(combined_policy);
        reference_only_emissions.push(reference.clone());
        surname_only_emissions.push(surname.clone());
        combined_only_emissions.push(reference.or(surname));
    }

    let c5 = evaluate_explicit_emissions(holdout, &c5_emissions)?;
    let reference_policy = evaluate_explicit_emissions(holdout, &reference_policy_emissions)?;
    let combined_policy = evaluate_explicit_emissions(holdout, &combined_policy_emissions)?;
    let reference_additions = AdditionMetrics::from_sealed(evaluate_explicit_emissions(
        holdout,
        &reference_only_emissions,
    )?);
    let surname_additions = AdditionMetrics::from_sealed(evaluate_explicit_emissions(
        holdout,
        &surname_only_emissions,
    )?);
    let combined_additions = AdditionMetrics::from_sealed(evaluate_explicit_emissions(
        holdout,
        &combined_only_emissions,
    )?);
    validate_additive_metrics(
        c5,
        reference_policy,
        combined_policy,
        reference_additions,
        surname_additions,
        combined_additions,
    )?;
    Ok(ValidationResult {
        c5,
        reference_policy,
        combined_policy,
        reference_additions,
        surname_additions,
        combined_additions,
        class_metrics,
        strict_topology_rows,
        scan,
    })
}

fn build_validation_row(
    corpus: &impl EvidenceSource,
    surname_counts: &BTreeMap<String, u64>,
    display_name: &str,
    country_hint: Option<&str>,
    locale_hint: Option<&str>,
) -> Result<ValidationRow> {
    let diagnostic = diagnose_role_inference(
        corpus,
        ALGORITHM_C3,
        display_name,
        country_hint,
        locale_hint,
    );
    let decision = c4_decision_breakdown(&diagnostic, ALGORITHM_C2, ALGORITHM_C31, ALGORITHM_C4);
    let c5 = c5_decision_from_c4(decision.clone(), ALGORITHM_C5);
    let c5_emission = c5_emitted_candidate(&c5).map(str::to_string);
    let winner = decision.c31.winner.as_ref();
    let candidate = diagnostic.candidates.first();
    let selected_candidate = winner.map(|winner| winner.greeting_candidate.clone());
    let canonical = canonicalize(display_name);
    let tokens = canonical.split_whitespace().collect::<Vec<_>>();
    let position = candidate_position(candidate, tokens.len());
    let complement = complement_index(position).and_then(|index| tokens.get(index).copied());
    let lookup = complement.map(|token| {
        expected_lookup_diagnostic(corpus, ALGORITHM_C3, token, country_hint, locale_hint)
    });
    let complement_surname_count = complement
        .filter(|_| tokens_are_plain_alphabetic(&tokens))
        .map(canonicalize)
        .and_then(|key| surname_counts.get(&key).copied());
    let vetoes_pass = vetoes_pass(&decision);
    let complement_class = complement.map(|_| {
        if !vetoes_pass || lookup.as_ref().is_some_and(|lookup| !lookup.eligible) {
            ComplementClass::OrganizationOrLexicalNegative
        } else if lookup
            .as_ref()
            .is_some_and(|lookup| lookup.evidence.is_some())
        {
            ComplementClass::GivenObserved
        } else if complement_surname_count.is_some_and(|count| count > 0) {
            ComplementClass::SurnameObserved
        } else {
            ComplementClass::Unknown
        }
    });
    let native = winner.is_some_and(|winner| winner.candidate_origin != "handle_segment");
    let candidate_count = winner.map_or(0, |winner| winner.candidate_count);
    let strict_topology = native
        && tokens.len() == 2
        && tokens_are_plain_alphabetic(&tokens)
        && position.is_some()
        && candidate_count == 1
        && c5_emission.is_none()
        && vetoes_pass
        && lookup.as_ref().is_some_and(|lookup| lookup.eligible);
    if strict_topology && complement_surname_count.is_none() {
        return Err("surname scan is missing a strict-topology complement key".into());
    }
    Ok(ValidationRow {
        selected_candidate,
        c5_emission,
        strict_topology,
        selected_first: position == Some(0),
        candidate_quality: winner.map_or(0.0, |winner| winner.winner_score),
        reliability: winner.map_or(0.0, |winner| winner.reliability),
        role_signal: winner.map_or(0.0, |winner| winner.role_signal),
        complement_class,
        complement_surname_count,
    })
}

fn validate_safety_control(
    corpus: &impl EvidenceSource,
    surname_counts: &BTreeMap<String, u64>,
) -> Result<()> {
    let row = build_validation_row(corpus, surname_counts, "Motorcycle Club", None, None)?;
    if row.c5_emission.is_some()
        || row.reference_emission().is_some()
        || row.surname_emission().is_some()
        || row.complement_class != Some(ComplementClass::OrganizationOrLexicalNegative)
    {
        return Err("organization safety control did not remain an abstention".into());
    }
    Ok(())
}

fn validate_additive_metrics(
    c5: SealedMetrics,
    reference_policy: SealedMetrics,
    combined_policy: SealedMetrics,
    reference: AdditionMetrics,
    surname: AdditionMetrics,
    combined: AdditionMetrics,
) -> Result<()> {
    if combined.emitted != reference.emitted + surname.emitted
        || combined.correct != reference.correct + surname.correct
        || combined.wrong != reference.wrong + surname.wrong
        || combined.null_false_emissions
            != reference.null_false_emissions + surname.null_false_emissions
    {
        return Err("V7 frozen gate additions are not mutually exclusive".into());
    }
    validate_policy_delta(c5, reference_policy, reference)?;
    validate_policy_delta(c5, combined_policy, combined)?;
    Ok(())
}

fn validate_policy_delta(
    baseline: SealedMetrics,
    policy: SealedMetrics,
    delta: AdditionMetrics,
) -> Result<()> {
    if policy.emitted_greetings != baseline.emitted_greetings + delta.emitted
        || policy.correct_greetings != baseline.correct_greetings + delta.correct
        || policy.wrong_greetings != baseline.wrong_greetings + delta.wrong
        || policy.false_emissions_on_expected_abstentions
            != baseline.false_emissions_on_expected_abstentions + delta.null_false_emissions
    {
        return Err("V7 policy metrics are not the exact additive C5 delta".into());
    }
    Ok(())
}

fn expected_lookup_keys(holdout: &FrozenHoldout) -> BTreeSet<String> {
    let mut keys = BTreeSet::new();
    for case in &holdout.cases {
        let canonical = canonicalize(&case.display_name);
        let tokens = canonical.split_whitespace().collect::<Vec<_>>();
        if tokens.len() == 2 && tokens_are_plain_alphabetic(&tokens) {
            keys.extend(tokens.into_iter().map(canonicalize));
        }
    }
    keys
}

fn tokens_are_plain_alphabetic(tokens: &[&str]) -> bool {
    tokens
        .iter()
        .all(|token| !token.is_empty() && token.chars().all(char::is_alphabetic))
}

fn candidate_position(
    candidate: Option<&CandidateDiagnostic>,
    token_count: usize,
) -> Option<usize> {
    match candidate {
        Some(candidate) if candidate.length == 1 && candidate.start < token_count => {
            Some(candidate.start)
        }
        _ => None,
    }
}

fn complement_index(position: Option<usize>) -> Option<usize> {
    match position {
        Some(0) => Some(1),
        Some(1) => Some(0),
        _ => None,
    }
}

fn vetoes_pass(decision: &C4DecisionBreakdown) -> bool {
    !decision.c31.hard_organization_marker
        && !decision.c31.generic_organization_marker
        && !decision.c31.ampersand
        && !decision.c31.candidate_too_short
}

fn winner_outcome(expected: Option<&str>, selected: Option<&str>) -> WinnerOutcome {
    match expected {
        None => WinnerOutcome::ExpectedNull,
        Some(_) if greeting_matches(expected, selected) => WinnerOutcome::Correct,
        Some(_) => WinnerOutcome::Wrong,
    }
}

fn serialize_lookup_keys(keys: &BTreeSet<String>) -> Result<Vec<u8>> {
    let mut writer = canonical_writer();
    writer.write_record(["name"])?;
    for key in keys {
        writer.write_record([key])?;
    }
    Ok(writer.into_inner()?)
}

fn lookup_manifest_csv(
    holdout: &FrozenHoldout,
    target_keys: usize,
    key_sha256: &str,
) -> Result<Vec<u8>> {
    let mut writer = canonical_writer();
    writer.write_record(["key", "value"])?;
    for (key, value) in [
        ("holdout_sha256", holdout.manifest.holdout_sha256.clone()),
        (
            "selection",
            "all_plain_alphabetic_tokens_from_two_token_rows".to_string(),
        ),
        ("target_keys", target_keys.to_string()),
        ("lookup_keys_sha256", key_sha256.to_string()),
    ] {
        writer.write_record([key, &value])?;
    }
    Ok(writer.into_inner()?)
}

fn load_surname_scan(
    counts_path: &Path,
    manifest_path: &Path,
    expected_keys: &BTreeSet<String>,
    holdout_sha256: &str,
) -> Result<SurnameScan> {
    let bytes = fs::read(counts_path)?;
    let counts_sha256 = sha256_hex(&bytes);
    let manifest = load_key_value_manifest(manifest_path)?;
    validate_manifest_value(&manifest, "matching", "exact_utf8_byte_equality")?;
    validate_manifest_value(&manifest, "raw_files", RAW_FILES.to_string())?;
    validate_manifest_value(&manifest, "raw_rows", RAW_ROWS.to_string())?;
    validate_manifest_value(
        &manifest,
        "nonempty_surnames",
        NONEMPTY_SURNAMES.to_string(),
    )?;
    validate_manifest_value(&manifest, "holdout_sha256", holdout_sha256)?;
    validate_manifest_value(&manifest, "target_keys", expected_keys.len().to_string())?;
    validate_manifest_value(
        &manifest,
        "lookup_keys_sha256",
        sha256_hex(&serialize_lookup_keys(expected_keys)?),
    )?;
    validate_manifest_value(&manifest, "counts_sha256", &counts_sha256)?;

    let mut reader = csv::Reader::from_reader(bytes.as_slice());
    if reader.headers()?.iter().ne(["name", "surname_count"]) {
        return Err("unexpected V7 complement surname-count header".into());
    }
    let mut counts = BTreeMap::new();
    for result in reader.records() {
        let record = result?;
        let name = record.get(0).ok_or("missing V7 surname key")?;
        let count = record
            .get(1)
            .ok_or("missing V7 surname count")?
            .parse::<u64>()?;
        if counts.insert(name.to_string(), count).is_some() {
            return Err("duplicate V7 surname key".into());
        }
    }
    if counts.keys().ne(expected_keys.iter()) {
        return Err("V7 surname-count keys do not match the sealed lexical key set".into());
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

fn load_key_value_manifest(path: &Path) -> Result<BTreeMap<String, String>> {
    let mut reader = csv::Reader::from_path(path)?;
    if reader.headers()?.iter().ne(["key", "value"]) {
        return Err("unexpected V7 surname manifest header".into());
    }
    let mut manifest = BTreeMap::new();
    for result in reader.records() {
        let record = result?;
        let key = record.get(0).ok_or("missing V7 surname manifest key")?;
        let value = record.get(1).ok_or("missing V7 surname manifest value")?;
        if manifest
            .insert(key.to_string(), value.to_string())
            .is_some()
        {
            return Err("duplicate V7 surname manifest key".into());
        }
    }
    Ok(manifest)
}

fn validate_manifest_value(
    manifest: &BTreeMap<String, String>,
    key: &str,
    expected: impl AsRef<str>,
) -> Result<()> {
    let actual = manifest
        .get(key)
        .ok_or_else(|| format!("V7 surname manifest is missing {key}"))?;
    if actual != expected.as_ref() {
        return Err(format!(
            "V7 surname manifest {key} mismatch: expected {}, got {actual}",
            expected.as_ref()
        )
        .into());
    }
    Ok(())
}

fn build_outputs(
    holdout: &FrozenHoldout,
    result: &ValidationResult,
) -> Result<BTreeMap<&'static str, Vec<u8>>> {
    let mut outputs = BTreeMap::new();
    outputs.insert("policy_metrics.csv", policy_metrics_csv(result)?);
    outputs.insert("additive_deltas.csv", additive_deltas_csv(result)?);
    outputs.insert(
        "complement_classes.csv",
        complement_classes_csv(&result.class_metrics)?,
    );
    outputs.insert("run_manifest.csv", run_manifest_csv(holdout, result)?);
    outputs.insert(
        "validation_report.md",
        validation_report(holdout, result).into_bytes(),
    );
    Ok(outputs)
}

fn policy_metrics_csv(result: &ValidationResult) -> Result<Vec<u8>> {
    let mut writer = canonical_writer();
    writer.write_record([
        "policy",
        "evaluable",
        "expected_greetings",
        "expected_nulls",
        "emitted",
        "correct",
        "wrong",
        "missed_greetings",
        "null_false_emissions",
        "precision",
        "recall",
    ])?;
    for (name, metrics) in [
        ("frozen_c5", result.c5),
        ("c5_plus_first_position", result.reference_policy),
        (
            "c5_plus_first_position_plus_complement_surname",
            result.combined_policy,
        ),
    ] {
        writer.write_record([
            name.to_string(),
            metrics.evaluable_cases.to_string(),
            metrics.expected_greetings.to_string(),
            metrics.expected_abstentions.to_string(),
            metrics.emitted_greetings.to_string(),
            metrics.correct_greetings.to_string(),
            metrics.wrong_greetings.to_string(),
            metrics.expected_greetings_missed.to_string(),
            metrics.false_emissions_on_expected_abstentions.to_string(),
            format_ratio(metrics.greeting_precision()),
            format_ratio(metrics.greeting_recall()),
        ])?;
    }
    Ok(writer.into_inner()?)
}

fn additive_deltas_csv(result: &ValidationResult) -> Result<Vec<u8>> {
    let mut writer = canonical_writer();
    writer.write_record([
        "branch",
        "additional_emitted",
        "additional_correct",
        "additional_wrong",
        "additional_null_false_emissions",
        "recall_change",
    ])?;
    for (name, metrics) in [
        ("first_position", result.reference_additions),
        ("complement_surname_residual", result.surname_additions),
        ("combined", result.combined_additions),
    ] {
        writer.write_record([
            name.to_string(),
            metrics.emitted.to_string(),
            metrics.correct.to_string(),
            metrics.wrong.to_string(),
            metrics.null_false_emissions.to_string(),
            format!(
                "{:.6}",
                metrics.correct as f64 / result.c5.expected_greetings as f64
            ),
        ])?;
    }
    Ok(writer.into_inner()?)
}

fn complement_classes_csv(
    class_metrics: &BTreeMap<ComplementClass, ClassMetrics>,
) -> Result<Vec<u8>> {
    let mut writer = canonical_writer();
    writer.write_record([
        "class",
        "rows",
        "correct_winners",
        "wrong_winners",
        "expected_null_winners",
        "first_position_emissions",
        "complement_surname_residual_emissions",
    ])?;
    for class in ComplementClass::ALL {
        let metrics = class_metrics.get(&class).copied().unwrap_or_default();
        writer.write_record([
            class.as_str().to_string(),
            metrics.rows.to_string(),
            metrics.correct_winners.to_string(),
            metrics.wrong_winners.to_string(),
            metrics.expected_null_winners.to_string(),
            metrics.reference_emissions.to_string(),
            metrics.surname_residual_emissions.to_string(),
        ])?;
    }
    Ok(writer.into_inner()?)
}

fn run_manifest_csv(holdout: &FrozenHoldout, result: &ValidationResult) -> Result<Vec<u8>> {
    let mut writer = canonical_writer();
    writer.write_record(["key", "value"])?;
    let rows = [
        ("holdout_sha256", holdout.manifest.holdout_sha256.clone()),
        ("total_cases", holdout.manifest.total_cases.to_string()),
        (
            "evaluable_cases",
            holdout.manifest.evaluable_cases.to_string(),
        ),
        (
            "strict_topology_rows",
            result.strict_topology_rows.to_string(),
        ),
        ("surname_matching", "exact_utf8_byte_equality".to_string()),
        ("surname_count_min", SURNAME_COUNT_MIN.to_string()),
        ("surname_counts_sha256", result.scan.counts_sha256.clone()),
        ("surname_matched_keys", result.scan.matched_keys.to_string()),
        (
            "surname_matched_observations",
            result.scan.matched_observations.to_string(),
        ),
        ("row_level_output", "forbidden".to_string()),
    ];
    for (key, value) in rows {
        writer.write_record([key, &value])?;
    }
    Ok(writer.into_inner()?)
}

fn validation_report(holdout: &FrozenHoldout, result: &ValidationResult) -> String {
    let mut report = String::new();
    let (verdict, explanation) = validation_verdict(result.surname_additions);
    writeln!(
        report,
        "# Frozen complement-evidence validation on REAL_PROXY_V7\n"
    )
    .unwrap();
    writeln!(report, "The fresh holdout was frozen and checksum-verified as `{}` before classifier inference or surname joining. All results are aggregate-only. The two development gates and the exact raw-surname threshold of `>= 1` were evaluated once without tuning.\n", holdout.manifest.holdout_sha256).unwrap();
    writeln!(report, "V7 contains {} rows: {} evaluable and {} skipped, with {} expected greetings and {} expected NULL decisions.\n", holdout.manifest.total_cases, holdout.manifest.evaluable_cases, holdout.manifest.skipped_cases, holdout.manifest.expected_greetings, holdout.manifest.expected_abstentions).unwrap();
    writeln!(report, "| Policy | Emitted | Correct | Wrong | NULL FP | Precision | Recall |\n|---|---:|---:|---:|---:|---:|---:|").unwrap();
    for (name, metrics) in [
        ("Frozen C5", result.c5),
        ("C5 + first-position reference", result.reference_policy),
        ("C5 + reference + surname residual", result.combined_policy),
    ] {
        writeln!(
            report,
            "| {name} | {} | {} | {} | {} | {} | {} |",
            metrics.emitted_greetings,
            metrics.correct_greetings,
            metrics.wrong_greetings,
            metrics.false_emissions_on_expected_abstentions,
            format_percent(metrics.greeting_precision()),
            format_percent(metrics.greeting_recall())
        )
        .unwrap();
    }
    writeln!(report, "\n## Additive results\n").unwrap();
    writeln!(
        report,
        "| Branch | Additional emitted | Correct | Wrong | NULL FP |\n|---|---:|---:|---:|---:|"
    )
    .unwrap();
    for (name, metrics) in [
        ("First-position reference", result.reference_additions),
        (
            "Positive complement-surname residual",
            result.surname_additions,
        ),
        ("Combined", result.combined_additions),
    ] {
        writeln!(
            report,
            "| {name} | {} | {} | {} | {} |",
            metrics.emitted, metrics.correct, metrics.wrong, metrics.null_false_emissions
        )
        .unwrap();
    }
    writeln!(report, "\nThe surname row counts only emissions not already produced by the first-position reference gate. Expected-NULL false emissions are a subset of wrong emissions. The strict topology contained {} evaluable rows.\n", result.strict_topology_rows).unwrap();
    writeln!(report, "## Frozen gates\n").unwrap();
    writeln!(report, "Both additions require a native, non-segmented, exactly-two-token alphabetic input, one viable single-token winner, frozen C5 abstention, and all existing vetoes passing. Both require the selected token to be first.").unwrap();
    writeln!(report, "- Reference: quality `>= {REFERENCE_QUALITY_MIN:.2}`, reliability `>= {REFERENCE_RELIABILITY_MIN:.2}`, role signal `>= {REFERENCE_ROLE_MIN:.2}`.").unwrap();
    writeln!(report, "- Surname residual: quality `>= {SURNAME_QUALITY_MIN:.2}`, reliability `>= {SURNAME_RELIABILITY_MIN:.2}`, role signal `>= {SURNAME_ROLE_MIN:.2}`, no retained given-name evidence for the complement, and exact raw surname count `>= {SURNAME_COUNT_MIN}`.\n").unwrap();
    writeln!(report, "The fixed organization safety control remained an abstention under C5 and both gates. Unknown complements without positive surname evidence were ineligible for the surname branch.\n").unwrap();
    writeln!(report, "## Aggregate verdict\n").unwrap();
    writeln!(report, "**{verdict}.** {explanation}\n").unwrap();
    writeln!(report, "This one-shot machine-consensus proxy result does not promote either rule, choose a production surname-index threshold, or establish worldwide precision. V7 is now spent; no individual failures were inspected.").unwrap();
    report
}

fn validation_verdict(metrics: AdditionMetrics) -> (&'static str, &'static str) {
    if metrics.correct >= SUBSTANTIAL_CORRECT_MIN && metrics.wrong == 0 {
        (
            "Strong validation",
            "The surname residual recovered a substantial unseen set of correct greetings with no observed additional error.",
        )
    } else if metrics.correct >= SUBSTANTIAL_CORRECT_MIN && metrics.wrong <= MIXED_ERROR_MAX {
        (
            "Mixed but real signal",
            "The surname residual recovered useful unseen greetings with a very small observed error count consistent with the development LOGO warning. It must not be promoted before a separate spent-data storage decision and fresh validation.",
        )
    } else {
        (
            "Negative",
            "The surname residual did not provide enough safe incremental recall to justify production surname-index work.",
        )
    }
}

fn canonical_writer() -> csv::Writer<Vec<u8>> {
    csv::WriterBuilder::new()
        .terminator(csv::Terminator::Any(b'\n'))
        .from_writer(Vec::new())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn format_ratio(value: Option<f64>) -> String {
    value.map_or_else(String::new, |value| format!("{value:.6}"))
}

fn format_percent(value: Option<f64>) -> String {
    value.map_or_else(
        || "n/a".to_string(),
        |value| format!("{:.3}%", value * 100.0),
    )
}

fn nonempty(value: &str) -> Option<&str> {
    (!value.is_empty()).then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn addition(correct: usize, wrong: usize) -> AdditionMetrics {
        AdditionMetrics {
            emitted: correct + wrong,
            correct,
            wrong,
            null_false_emissions: 0,
        }
    }

    #[test]
    fn frozen_verdict_boundaries_are_preregistered() {
        assert_eq!(validation_verdict(addition(5, 0)).0, "Strong validation");
        assert_eq!(
            validation_verdict(addition(5, 1)).0,
            "Mixed but real signal"
        );
        assert_eq!(validation_verdict(addition(4, 0)).0, "Negative");
        assert_eq!(validation_verdict(addition(5, 2)).0, "Negative");
    }

    #[test]
    fn reference_and_surname_gates_are_additive() {
        let row = ValidationRow {
            selected_candidate: Some("Example".to_string()),
            c5_emission: None,
            strict_topology: true,
            selected_first: true,
            candidate_quality: 0.50,
            reliability: 0.40,
            role_signal: 0.30,
            complement_class: Some(ComplementClass::SurnameObserved),
            complement_surname_count: Some(1),
        };
        assert_eq!(row.reference_emission().as_deref(), Some("Example"));
        assert_eq!(row.surname_emission().as_deref(), Some("Example"));

        let residual = ValidationRow {
            candidate_quality: 0.40,
            reliability: 0.00,
            ..row
        };
        assert_eq!(residual.reference_emission(), None);
        assert_eq!(residual.surname_emission().as_deref(), Some("Example"));
    }

    #[test]
    fn unknown_and_vetoed_complements_do_not_pass_surname_gate() {
        let row = ValidationRow {
            selected_candidate: Some("Example".to_string()),
            c5_emission: None,
            strict_topology: true,
            selected_first: true,
            candidate_quality: 0.60,
            reliability: 0.60,
            role_signal: 0.60,
            complement_class: Some(ComplementClass::Unknown),
            complement_surname_count: Some(0),
        };
        assert_eq!(row.surname_emission(), None);

        let vetoed = ValidationRow {
            strict_topology: false,
            complement_class: Some(ComplementClass::SurnameObserved),
            complement_surname_count: Some(10),
            ..row
        };
        assert_eq!(vetoed.reference_emission(), None);
        assert_eq!(vetoed.surname_emission(), None);
    }

    #[test]
    fn lookup_key_serialization_is_sorted_and_canonical() {
        let keys = BTreeSet::from(["Zulu".to_string(), "Alpha".to_string()]);
        assert_eq!(
            String::from_utf8(serialize_lookup_keys(&keys).unwrap()).unwrap(),
            "name\nAlpha\nZulu\n"
        );
    }
}
