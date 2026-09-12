use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::Write as FmtWrite;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::Path;

use name_eval::holdout::{FrozenHoldout, SealedMetrics, evaluate_explicit_emissions};
use sha2::{Digest, Sha256};

use crate::artifact::EvidenceSource;
use crate::classifier::{
    ALGORITHM_C2, ALGORITHM_C3, ALGORITHM_C4, ALGORITHM_C5, ALGORITHM_C6, ALGORITHM_C31,
    C6EmissionSource, c6_decision_breakdown, c6_emitted_candidate, canonicalize,
    diagnose_role_inference, expected_lookup_diagnostic,
};
use crate::surname_index_selection::{FrozenSurnameMembership, load_frozen_membership_candidate};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub(super) const V8_SHA256: &str =
    "55fe9ae0efc7e604e55c997f8c26cd2cfc3e97961514a6a3780cd2f0420ae6c9";
const SURNAME_QUALITY_MIN: f64 = 0.40;
const SURNAME_RELIABILITY_MIN: f64 = 0.00;
const SURNAME_ROLE_MIN: f64 = 0.30;
const EXPECTED_MEMBERS: usize = 35_417_044;
const EXPECTED_GIVEN_NEGATIVES: usize = 1_803_175;
const GENERATED_NEGATIVES: usize = 100_000;
const SUBSTANTIAL_CORRECT_MIN: usize = 5;
const MIXED_ERROR_MAX: usize = 1;

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

#[derive(Clone, Copy, Debug, Default)]
#[allow(clippy::struct_excessive_bools)]
struct ResidualGate {
    c6_abstained: bool,
    native_candidate: bool,
    exactly_two_alphabetic_tokens: bool,
    single_token_winner: bool,
    selected_first: bool,
    candidate_count_pass: bool,
    candidate_quality: f64,
    reliability: f64,
    role_signal: f64,
    complement_lookup_eligible: bool,
    complement_absent_from_given_index: bool,
    complement_surname_member: bool,
    vetoes_pass: bool,
}

impl ResidualGate {
    fn passes(self) -> bool {
        self.c6_abstained
            && self.native_candidate
            && self.exactly_two_alphabetic_tokens
            && self.single_token_winner
            && self.selected_first
            && self.candidate_count_pass
            && self.candidate_quality >= SURNAME_QUALITY_MIN
            && self.reliability >= SURNAME_RELIABILITY_MIN
            && self.role_signal >= SURNAME_ROLE_MIN
            && self.complement_lookup_eligible
            && self.complement_absent_from_given_index
            && self.complement_surname_member
            && self.vetoes_pass
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ArtifactValidation {
    receipt_bytes: Vec<u8>,
    receipt_sha256: String,
    member_queries: usize,
    member_misses: usize,
    given_negative_queries: usize,
    given_false_accepts: usize,
    generated_negative_queries: usize,
    generated_false_accepts: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ArtifactCheckCounts {
    member_queries: usize,
    member_misses: usize,
    given_negative_queries: usize,
    given_false_accepts: usize,
    generated_negative_queries: usize,
    generated_false_accepts: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ValidationResult {
    c6: SealedMetrics,
    combined: SealedMetrics,
    surname_additions: AdditionMetrics,
    residual_topology_rows: usize,
    artifact: ArtifactValidation,
}

pub(crate) fn verify_frozen_surname_candidate(
    output: &Path,
    membership_directory: &Path,
    member_keys: &Path,
    name_totals: &Path,
) -> Result<String> {
    let membership = load_frozen_membership_candidate(membership_directory)?;
    let (member_queries, member_misses, source_keys_sha256) =
        verify_all_members(&membership, member_keys)?;
    if member_queries != EXPECTED_MEMBERS
        || member_misses != 0
        || source_keys_sha256 != membership.source_keys_sha256()
    {
        return Err("frozen surname candidate member verification failed".into());
    }
    let (given_negative_queries, given_false_accepts) =
        verify_given_negatives(&membership, name_totals)?;
    if given_negative_queries != EXPECTED_GIVEN_NEGATIVES || given_false_accepts != 0 {
        return Err("frozen surname candidate retained-given verification failed".into());
    }
    let (generated_negative_queries, generated_false_accepts) =
        verify_generated_negatives(&membership);
    if generated_negative_queries != GENERATED_NEGATIVES || generated_false_accepts != 0 {
        return Err("frozen surname candidate generated-negative verification failed".into());
    }

    let counts = ArtifactCheckCounts {
        member_queries,
        member_misses,
        given_negative_queries,
        given_false_accepts,
        generated_negative_queries,
        generated_false_accepts,
    };
    let receipt = artifact_validation_csv(&membership, counts)?;
    let repeated = artifact_validation_csv(&membership, counts)?;
    if receipt != repeated {
        return Err("surname artifact verification serialization is not deterministic".into());
    }
    fs::write(output.join("artifact_validation.csv"), &receipt)?;
    let report = artifact_verification_report(&membership, &receipt, counts);
    fs::write(output.join("verification_report.md"), report.as_bytes())?;
    Ok(report)
}

pub(crate) fn run_surname_complement_validation(
    output: &Path,
    corpus: &impl EvidenceSource,
    holdout: &FrozenHoldout,
    membership_directory: &Path,
    artifact_receipt: &Path,
) -> Result<String> {
    if holdout.manifest.holdout_sha256 != V8_SHA256 {
        return Err(format!(
            "surname-complement validation requires frozen REAL_PROXY_V8 {}; received {}",
            V8_SHA256, holdout.manifest.holdout_sha256
        )
        .into());
    }
    let membership = load_frozen_membership_candidate(membership_directory)?;
    let artifact = load_artifact_validation(artifact_receipt, &membership)?;
    validate_safety_control(corpus, &membership)?;
    let result = evaluate_validation(corpus, holdout, &membership, artifact)?;
    let outputs = build_outputs(holdout, &membership, &result)?;
    let repeated = build_outputs(holdout, &membership, &result)?;
    if outputs != repeated {
        return Err("V8 surname-complement serialization is not deterministic".into());
    }
    for (name, bytes) in &outputs {
        fs::write(output.join(name), bytes)?;
    }
    Ok(String::from_utf8(
        outputs
            .get("validation_report.md")
            .ok_or("V8 validation report missing")?
            .clone(),
    )?)
}

fn verify_all_members(
    membership: &FrozenSurnameMembership,
    member_keys: &Path,
) -> Result<(usize, usize, String)> {
    let mut reader = BufReader::new(File::open(member_keys)?);
    let mut digest = Sha256::new();
    let mut line = Vec::new();
    let mut previous = String::new();
    let mut queries = 0;
    let mut misses = 0;
    loop {
        line.clear();
        if reader.read_until(b'\n', &mut line)? == 0 {
            break;
        }
        digest.update(&line);
        if line.pop() != Some(b'\n') || line.last() == Some(&b'\r') {
            return Err("surname member-key stream is not canonical LF-delimited UTF-8".into());
        }
        let key = std::str::from_utf8(&line)?;
        if key.is_empty() || (!previous.is_empty() && previous.as_str() >= key) {
            return Err("surname member-key stream is not a unique sorted set".into());
        }
        queries += 1;
        misses += usize::from(!membership.contains(key));
        previous.clear();
        previous.push_str(key);
    }
    Ok((queries, misses, format!("{:x}", digest.finalize())))
}

fn verify_given_negatives(
    membership: &FrozenSurnameMembership,
    name_totals: &Path,
) -> Result<(usize, usize)> {
    let mut reader = csv::Reader::from_path(name_totals)?;
    if reader
        .headers()?
        .iter()
        .ne(["name", "given_count", "as_surname_count"])
    {
        return Err("unexpected retained given-name totals header".into());
    }
    let mut queries = 0;
    let mut false_accepts = 0;
    for record in reader.records() {
        let record = record?;
        let key = record.get(0).ok_or("missing retained given-name key")?;
        queries += 1;
        false_accepts += usize::from(membership.contains(key));
    }
    Ok((queries, false_accepts))
}

fn verify_generated_negatives(membership: &FrozenSurnameMembership) -> (usize, usize) {
    let false_accepts = (0..GENERATED_NEGATIVES)
        .filter(|index| membership.contains(&format!("definitely-not-a-surname-{index:06}")))
        .count();
    (GENERATED_NEGATIVES, false_accepts)
}

fn artifact_validation_csv(
    membership: &FrozenSurnameMembership,
    counts: ArtifactCheckCounts,
) -> Result<Vec<u8>> {
    let mut writer = canonical_writer();
    writer.write_record(["key", "value"])?;
    for (key, value) in [
        (
            "format",
            "frozen-surname-membership-validation-v1".to_string(),
        ),
        (
            "candidate_manifest_sha256",
            membership.manifest_sha256().to_string(),
        ),
        ("candidate_key_count", membership.key_count().to_string()),
        (
            "candidate_artifact_bytes",
            membership.artifact_bytes().to_string(),
        ),
        (
            "candidate_source_keys_sha256",
            membership.source_keys_sha256().to_string(),
        ),
        (
            "candidate_mphf_sha256",
            membership.mphf_sha256().to_string(),
        ),
        (
            "candidate_fingerprints_sha256",
            membership.fingerprint_sha256().to_string(),
        ),
        ("member_queries", counts.member_queries.to_string()),
        ("member_misses", counts.member_misses.to_string()),
        (
            "given_negative_queries",
            counts.given_negative_queries.to_string(),
        ),
        (
            "given_false_accepts",
            counts.given_false_accepts.to_string(),
        ),
        (
            "generated_negative_queries",
            counts.generated_negative_queries.to_string(),
        ),
        (
            "generated_false_accepts",
            counts.generated_false_accepts.to_string(),
        ),
        (
            "nominal_unknown_false_accept_probability",
            "2^-32".to_string(),
        ),
        ("row_level_output", "forbidden".to_string()),
    ] {
        writer.write_record([key, &value])?;
    }
    Ok(writer.into_inner()?)
}

fn artifact_verification_report(
    membership: &FrozenSurnameMembership,
    receipt: &[u8],
    counts: ArtifactCheckCounts,
) -> String {
    let mut report = String::new();
    writeln!(report, "# Frozen surname candidate verification\n").unwrap();
    writeln!(report, "The already frozen count-at-least-1 MPHF + 32-bit-fingerprint candidate authenticated under manifest SHA-256 `{}`. No candidate constituent was rebuilt or changed.\n", membership.manifest_sha256()).unwrap();
    writeln!(report, "- Keys: {}", membership.key_count()).unwrap();
    writeln!(report, "- Artifact bytes: {}", membership.artifact_bytes()).unwrap();
    writeln!(report, "- Full member lookups: {}", counts.member_queries).unwrap();
    writeln!(report, "- Member misses: {}", counts.member_misses).unwrap();
    writeln!(
        report,
        "- Retained-given negative probes: {}",
        counts.given_negative_queries
    )
    .unwrap();
    writeln!(
        report,
        "- Retained-given false accepts: {}",
        counts.given_false_accepts
    )
    .unwrap();
    writeln!(
        report,
        "- Generated negative probes: {}",
        counts.generated_negative_queries
    )
    .unwrap();
    writeln!(
        report,
        "- Generated false accepts: {}",
        counts.generated_false_accepts
    )
    .unwrap();
    writeln!(report, "- Receipt SHA-256: `{}`\n", sha256_hex(receipt)).unwrap();
    writeln!(report, "The MPHF maps queries to candidate slots; the independent fingerprint performs rejection. Its nominal accidental acceptance probability remains approximately `2^-32` per unrelated lookup, not zero.").unwrap();
    report
}

fn load_artifact_validation(
    path: &Path,
    membership: &FrozenSurnameMembership,
) -> Result<ArtifactValidation> {
    let receipt_bytes = fs::read(path)?;
    let values = parse_key_value_csv(&receipt_bytes)?;
    let expected = [
        ("format", "frozen-surname-membership-validation-v1"),
        ("candidate_manifest_sha256", membership.manifest_sha256()),
        ("candidate_key_count", "35417044"),
        ("candidate_artifact_bytes", "156917446"),
        (
            "candidate_source_keys_sha256",
            membership.source_keys_sha256(),
        ),
        ("candidate_mphf_sha256", membership.mphf_sha256()),
        (
            "candidate_fingerprints_sha256",
            membership.fingerprint_sha256(),
        ),
        ("member_queries", "35417044"),
        ("member_misses", "0"),
        ("given_negative_queries", "1803175"),
        ("given_false_accepts", "0"),
        ("generated_negative_queries", "100000"),
        ("generated_false_accepts", "0"),
        ("nominal_unknown_false_accept_probability", "2^-32"),
        ("row_level_output", "forbidden"),
    ];
    if values.len() != expected.len() {
        return Err("surname artifact verification receipt has unexpected fields".into());
    }
    for (key, expected) in expected {
        if values.get(key).map(String::as_str) != Some(expected) {
            return Err(format!("surname artifact verification receipt mismatch for {key}").into());
        }
    }
    Ok(ArtifactValidation {
        receipt_sha256: sha256_hex(&receipt_bytes),
        receipt_bytes,
        member_queries: parse_receipt_usize(&values, "member_queries")?,
        member_misses: parse_receipt_usize(&values, "member_misses")?,
        given_negative_queries: parse_receipt_usize(&values, "given_negative_queries")?,
        given_false_accepts: parse_receipt_usize(&values, "given_false_accepts")?,
        generated_negative_queries: parse_receipt_usize(&values, "generated_negative_queries")?,
        generated_false_accepts: parse_receipt_usize(&values, "generated_false_accepts")?,
    })
}

fn evaluate_validation(
    corpus: &impl EvidenceSource,
    holdout: &FrozenHoldout,
    membership: &FrozenSurnameMembership,
    artifact: ArtifactValidation,
) -> Result<ValidationResult> {
    let mut c6_emissions = Vec::with_capacity(holdout.cases.len());
    let mut combined_emissions = Vec::with_capacity(holdout.cases.len());
    let mut surname_only_emissions = Vec::with_capacity(holdout.cases.len());
    let mut residual_topology_rows = 0;
    for case in &holdout.cases {
        if !case.is_evaluable() {
            c6_emissions.push(None);
            combined_emissions.push(None);
            surname_only_emissions.push(None);
            continue;
        }
        let (c6, surname, residual_topology) = infer_policies(
            corpus,
            membership,
            &case.display_name,
            nonempty(&case.country_hint),
            nonempty(&case.locale_hint),
        )?;
        residual_topology_rows += usize::from(residual_topology);
        combined_emissions.push(c6.clone().or_else(|| surname.clone()));
        c6_emissions.push(c6);
        surname_only_emissions.push(surname);
    }
    let c6 = evaluate_explicit_emissions(holdout, &c6_emissions)?;
    let combined = evaluate_explicit_emissions(holdout, &combined_emissions)?;
    let surname_additions = AdditionMetrics::from_sealed(evaluate_explicit_emissions(
        holdout,
        &surname_only_emissions,
    )?);
    validate_policy_delta(c6, combined, surname_additions)?;
    Ok(ValidationResult {
        c6,
        combined,
        surname_additions,
        residual_topology_rows,
        artifact,
    })
}

fn infer_policies(
    corpus: &impl EvidenceSource,
    membership: &FrozenSurnameMembership,
    display_name: &str,
    country_hint: Option<&str>,
    locale_hint: Option<&str>,
) -> Result<(Option<String>, Option<String>, bool)> {
    let diagnostic = diagnose_role_inference(
        corpus,
        ALGORITHM_C3,
        display_name,
        country_hint,
        locale_hint,
    );
    let c6 = c6_decision_breakdown(
        &diagnostic,
        display_name,
        ALGORITHM_C2,
        ALGORITHM_C31,
        ALGORITHM_C4,
        ALGORITHM_C5,
        ALGORITHM_C6,
    );
    let baseline = c6_emitted_candidate(&c6).map(str::to_string);
    if c6.emission_source != C6EmissionSource::Abstain {
        return Ok((baseline, None, false));
    }

    let winner = c6.c5.c4.c31.winner.as_ref();
    let canonical = canonicalize(display_name);
    let tokens = canonical.split_whitespace().collect::<Vec<_>>();
    let complement = (tokens.len() == 2).then(|| canonicalize(tokens[1]));
    let complement_lookup = complement.as_deref().map(|token| {
        expected_lookup_diagnostic(corpus, ALGORITHM_C3, token, country_hint, locale_hint)
    });
    let gate = ResidualGate {
        c6_abstained: true,
        native_candidate: c6.first_position.native_candidate,
        exactly_two_alphabetic_tokens: c6.first_position.exactly_two_alphabetic_tokens,
        single_token_winner: c6.first_position.single_token_winner,
        selected_first: c6.first_position.selected_first,
        candidate_count_pass: c6.first_position.candidate_count_pass,
        candidate_quality: winner.map_or(0.0, |winner| winner.winner_score),
        reliability: winner.map_or(0.0, |winner| winner.reliability),
        role_signal: winner.map_or(0.0, |winner| winner.role_signal),
        complement_lookup_eligible: complement_lookup
            .as_ref()
            .is_some_and(|lookup| lookup.eligible),
        complement_absent_from_given_index: complement_lookup
            .as_ref()
            .is_some_and(|lookup| lookup.evidence.is_none()),
        complement_surname_member: complement
            .as_deref()
            .is_some_and(|key| membership.contains(key)),
        vetoes_pass: c6.first_position.vetoes_pass,
    };
    let residual_topology = gate.c6_abstained
        && gate.native_candidate
        && gate.exactly_two_alphabetic_tokens
        && gate.single_token_winner
        && gate.selected_first
        && gate.candidate_count_pass
        && gate.vetoes_pass;
    let surname = gate
        .passes()
        .then(|| winner.map(|winner| winner.greeting_candidate.clone()))
        .flatten();
    Ok((baseline, surname, residual_topology))
}

fn validate_safety_control(
    corpus: &impl EvidenceSource,
    membership: &FrozenSurnameMembership,
) -> Result<()> {
    let (c6, surname, _) = infer_policies(corpus, membership, "Motorcycle Club", None, None)?;
    if c6.is_some() || surname.is_some() {
        return Err("organization safety control did not remain an abstention".into());
    }
    Ok(())
}

fn validate_policy_delta(
    baseline: SealedMetrics,
    combined: SealedMetrics,
    delta: AdditionMetrics,
) -> Result<()> {
    if delta.null_false_emissions > delta.wrong
        || combined.emitted_greetings != baseline.emitted_greetings + delta.emitted
        || combined.correct_greetings != baseline.correct_greetings + delta.correct
        || combined.wrong_greetings != baseline.wrong_greetings + delta.wrong
        || combined.false_emissions_on_expected_abstentions
            != baseline.false_emissions_on_expected_abstentions + delta.null_false_emissions
    {
        return Err("V8 surname policy is not the exact additive C6 delta".into());
    }
    Ok(())
}

fn build_outputs(
    holdout: &FrozenHoldout,
    membership: &FrozenSurnameMembership,
    result: &ValidationResult,
) -> Result<BTreeMap<&'static str, Vec<u8>>> {
    let mut outputs = BTreeMap::new();
    outputs.insert(
        "artifact_validation.csv",
        result.artifact.receipt_bytes.clone(),
    );
    outputs.insert("policy_metrics.csv", policy_metrics_csv(result)?);
    outputs.insert("surname_delta.csv", surname_delta_csv(result)?);
    outputs.insert(
        "run_manifest.csv",
        run_manifest_csv(holdout, membership, result)?,
    );
    outputs.insert(
        "validation_report.md",
        validation_report(holdout, membership, result).into_bytes(),
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
        "null_false_emissions",
        "precision",
        "recall",
        "abstention_rate",
    ])?;
    for (name, metrics) in [
        ("frozen_c6", result.c6),
        ("c6_plus_surname", result.combined),
    ] {
        writer.write_record([
            name.to_string(),
            metrics.evaluable_cases.to_string(),
            metrics.expected_greetings.to_string(),
            metrics.expected_abstentions.to_string(),
            metrics.emitted_greetings.to_string(),
            metrics.correct_greetings.to_string(),
            metrics.wrong_greetings.to_string(),
            metrics.false_emissions_on_expected_abstentions.to_string(),
            format_ratio(metrics.greeting_precision()),
            format_ratio(metrics.greeting_recall()),
            format_ratio(metrics.abstention_rate()),
        ])?;
    }
    Ok(writer.into_inner()?)
}

fn surname_delta_csv(result: &ValidationResult) -> Result<Vec<u8>> {
    let mut writer = canonical_writer();
    writer.write_record([
        "branch",
        "additional_emitted",
        "additional_correct",
        "additional_wrong",
        "additional_null_false_emissions",
        "recall_change",
    ])?;
    let metrics = result.surname_additions;
    writer.write_record([
        "complement_surname_residual".to_string(),
        metrics.emitted.to_string(),
        metrics.correct.to_string(),
        metrics.wrong.to_string(),
        metrics.null_false_emissions.to_string(),
        format!(
            "{:.6}",
            metrics.correct as f64 / result.c6.expected_greetings as f64
        ),
    ])?;
    Ok(writer.into_inner()?)
}

fn run_manifest_csv(
    holdout: &FrozenHoldout,
    membership: &FrozenSurnameMembership,
    result: &ValidationResult,
) -> Result<Vec<u8>> {
    let mut writer = canonical_writer();
    writer.write_record(["key", "value"])?;
    for (key, value) in [
        ("holdout", "REAL_PROXY_V8".to_string()),
        ("holdout_sha256", holdout.manifest.holdout_sha256.clone()),
        ("total_cases", holdout.manifest.total_cases.to_string()),
        (
            "evaluable_cases",
            holdout.manifest.evaluable_cases.to_string(),
        ),
        (
            "residual_topology_rows",
            result.residual_topology_rows.to_string(),
        ),
        ("baseline", "production_c6".to_string()),
        ("surname_count_min", "1".to_string()),
        (
            "surname_candidate_manifest_sha256",
            membership.manifest_sha256().to_string(),
        ),
        (
            "surname_artifact_validation_sha256",
            result.artifact.receipt_sha256.clone(),
        ),
        ("row_level_output", "forbidden".to_string()),
        ("threshold_search", "forbidden".to_string()),
        ("production_integration", "false".to_string()),
        ("v9_created", "false".to_string()),
    ] {
        writer.write_record([key, &value])?;
    }
    Ok(writer.into_inner()?)
}

fn validation_report(
    holdout: &FrozenHoldout,
    membership: &FrozenSurnameMembership,
    result: &ValidationResult,
) -> String {
    let mut report = String::new();
    let (verdict, explanation) = validation_verdict(result.surname_additions);
    writeln!(
        report,
        "# Frozen surname-complement validation on REAL_PROXY_V8\n"
    )
    .unwrap();
    writeln!(report, "The fresh holdout was frozen and checksum-verified as `{}` before any classifier or surname inference. The exact count-at-least-1 MPHF + 32-bit-fingerprint candidate was evaluated once, with no threshold search or row-level output.\n", holdout.manifest.holdout_sha256).unwrap();
    writeln!(report, "V8 contains {} rows: {} evaluable and {} skipped, with {} expected greetings and {} expected NULL decisions.\n", holdout.manifest.total_cases, holdout.manifest.evaluable_cases, holdout.manifest.skipped_cases, holdout.manifest.expected_greetings, holdout.manifest.expected_abstentions).unwrap();
    writeln!(report, "| Policy | Emitted | Correct | Wrong | NULL FP | Precision | Recall | Abstention rate |\n|---|---:|---:|---:|---:|---:|---:|---:|").unwrap();
    for (name, metrics) in [
        ("Frozen C6", result.c6),
        ("C6 + surname residual", result.combined),
    ] {
        writeln!(
            report,
            "| {name} | {} | {} | {} | {} | {} | {} | {} |",
            metrics.emitted_greetings,
            metrics.correct_greetings,
            metrics.wrong_greetings,
            metrics.false_emissions_on_expected_abstentions,
            format_percent(metrics.greeting_precision()),
            format_percent(metrics.greeting_recall()),
            format_percent(metrics.abstention_rate())
        )
        .unwrap();
    }
    let delta = result.surname_additions;
    writeln!(report, "\nThe surname residual added **{} emissions: {} correct, {} wrong, and {} NULL false emissions** beyond C6. Expected-NULL false emissions are a subset of wrong emissions. The frozen residual topology contained {} evaluable rows.\n", delta.emitted, delta.correct, delta.wrong, delta.null_false_emissions, result.residual_topology_rows).unwrap();
    writeln!(report, "## Frozen candidate\n").unwrap();
    writeln!(report, "The candidate contains {} surname-only keys and occupies {} bytes under manifest SHA-256 `{}`. Its full {}-member check produced {} misses; {} retained-given and {} generated negative probes produced {} and {} observed false accepts. The MPHF always maps a query to a candidate slot; the independent 32-bit fingerprint provides rejection with nominal accidental acceptance probability `2^-32` per unrelated lookup.\n", membership.key_count(), membership.artifact_bytes(), membership.manifest_sha256(), result.artifact.member_queries, result.artifact.member_misses, result.artifact.given_negative_queries, result.artifact.generated_negative_queries, result.artifact.given_false_accepts, result.artifact.generated_false_accepts).unwrap();
    writeln!(report, "## Aggregate verdict\n").unwrap();
    writeln!(report, "**{verdict}.** {explanation}\n").unwrap();
    writeln!(report, "Historical V7 remains unchanged: first-position `+55 correct / 0 wrong / 0 NULL FP`; surname residual `+18 correct / 1 wrong / 1 NULL FP`. V8 is now spent. No individual V8 row, failure, or correct addition was inspected. Production remains C6; the surname candidate is validated only and is not loaded by normal production behavior. No V9 was created.").unwrap();
    report
}

fn validation_verdict(metrics: AdditionMetrics) -> (&'static str, &'static str) {
    if metrics.correct >= SUBSTANTIAL_CORRECT_MIN && metrics.wrong == 0 {
        (
            "Strong validation",
            "The frozen surname residual recovered a meaningful unseen set of correct greetings with no observed additional error. This validates the candidate for a separate production-engineering decision but does not enable it.",
        )
    } else if metrics.correct >= SUBSTANTIAL_CORRECT_MIN && metrics.wrong <= MIXED_ERROR_MAX {
        (
            "Mixed validation",
            "The frozen surname residual recovered useful unseen greetings with a very small observed error count consistent with V7. The signal is real, but this result alone does not justify shipping the roughly 150 MiB index.",
        )
    } else {
        (
            "Negative validation",
            "The frozen surname residual did not reproduce enough safe incremental recall to justify continuing the surname-index integration path.",
        )
    }
}

fn parse_key_value_csv(bytes: &[u8]) -> Result<BTreeMap<String, String>> {
    let mut reader = csv::Reader::from_reader(bytes);
    if reader.headers()?.iter().ne(["key", "value"]) {
        return Err("unexpected artifact validation receipt header".into());
    }
    let mut values = BTreeMap::new();
    for record in reader.records() {
        let record = record?;
        let key = record.get(0).ok_or("missing artifact receipt key")?;
        let value = record.get(1).ok_or("missing artifact receipt value")?;
        if values.insert(key.to_string(), value.to_string()).is_some() {
            return Err("duplicate artifact receipt key".into());
        }
    }
    Ok(values)
}

fn parse_receipt_usize(values: &BTreeMap<String, String>, key: &str) -> Result<usize> {
    Ok(values
        .get(key)
        .ok_or_else(|| format!("artifact receipt is missing {key}"))?
        .parse()?)
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

    fn passing_gate() -> ResidualGate {
        ResidualGate {
            c6_abstained: true,
            native_candidate: true,
            exactly_two_alphabetic_tokens: true,
            single_token_winner: true,
            selected_first: true,
            candidate_count_pass: true,
            candidate_quality: SURNAME_QUALITY_MIN,
            reliability: SURNAME_RELIABILITY_MIN,
            role_signal: SURNAME_ROLE_MIN,
            complement_lookup_eligible: true,
            complement_absent_from_given_index: true,
            complement_surname_member: true,
            vetoes_pass: true,
        }
    }

    #[test]
    fn frozen_residual_boundaries_are_inclusive_and_every_condition_is_required() {
        assert!(passing_gate().passes());
        for changed in [
            ResidualGate {
                c6_abstained: false,
                ..passing_gate()
            },
            ResidualGate {
                native_candidate: false,
                ..passing_gate()
            },
            ResidualGate {
                exactly_two_alphabetic_tokens: false,
                ..passing_gate()
            },
            ResidualGate {
                single_token_winner: false,
                ..passing_gate()
            },
            ResidualGate {
                selected_first: false,
                ..passing_gate()
            },
            ResidualGate {
                candidate_count_pass: false,
                ..passing_gate()
            },
            ResidualGate {
                complement_lookup_eligible: false,
                ..passing_gate()
            },
            ResidualGate {
                complement_absent_from_given_index: false,
                ..passing_gate()
            },
            ResidualGate {
                complement_surname_member: false,
                ..passing_gate()
            },
            ResidualGate {
                vetoes_pass: false,
                ..passing_gate()
            },
        ] {
            assert!(!changed.passes());
        }
        assert!(
            !ResidualGate {
                candidate_quality: f64::from_bits(SURNAME_QUALITY_MIN.to_bits() - 1),
                ..passing_gate()
            }
            .passes()
        );
        assert!(
            !ResidualGate {
                reliability: -f64::from_bits(SURNAME_RELIABILITY_MIN.to_bits() + 1),
                ..passing_gate()
            }
            .passes()
        );
        assert!(
            !ResidualGate {
                role_signal: f64::from_bits(SURNAME_ROLE_MIN.to_bits() - 1),
                ..passing_gate()
            }
            .passes()
        );
    }

    #[test]
    fn frozen_verdict_boundaries_are_preregistered() {
        let addition = |correct, wrong| AdditionMetrics {
            emitted: correct + wrong,
            correct,
            wrong,
            null_false_emissions: 0,
        };
        assert_eq!(validation_verdict(addition(5, 0)).0, "Strong validation");
        assert_eq!(validation_verdict(addition(5, 1)).0, "Mixed validation");
        assert_eq!(validation_verdict(addition(4, 0)).0, "Negative validation");
        assert_eq!(validation_verdict(addition(5, 2)).0, "Negative validation");
    }

    #[test]
    fn artifact_receipt_parser_rejects_duplicate_keys() {
        assert!(parse_key_value_csv(b"key,value\na,1\na,2\n").is_err());
    }

    #[test]
    fn policy_delta_counts_null_false_emissions_inside_wrong() {
        let baseline = SealedMetrics {
            emitted_greetings: 10,
            correct_greetings: 9,
            wrong_greetings: 1,
            false_emissions_on_expected_abstentions: 1,
            ..SealedMetrics::default()
        };
        let combined = SealedMetrics {
            emitted_greetings: 13,
            correct_greetings: 11,
            wrong_greetings: 2,
            false_emissions_on_expected_abstentions: 2,
            ..SealedMetrics::default()
        };
        assert!(
            validate_policy_delta(
                baseline,
                combined,
                AdditionMetrics {
                    emitted: 3,
                    correct: 2,
                    wrong: 1,
                    null_false_emissions: 1,
                },
            )
            .is_ok()
        );
    }
}
