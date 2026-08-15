use std::error::Error;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::artifact::EvidenceSource;
use crate::classifier::{
    ALGORITHM_C1, CandidateDiagnostic, ExpectedCompositionDiagnostic, ExpectedLookupDiagnostic,
    RoleInferenceDiagnostic, diagnose_role_inference, expected_composition_diagnostic,
    expected_lookup_diagnostic,
};
use crate::metrics::greeting_matches;
use name_eval::holdout::{
    FrozenHoldout, HoldoutCase, LabelStatus, SealedDecision, SealedEvaluation, evaluate_sealed,
    sealed_confidence_buckets_csv, sealed_summary_csv,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub const DIAGNOSTIC_SAMPLE_SEED: u64 = 0x5245_414c_5052_4f58;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum TerminalReason {
    HardOrganizationAbstention,
    LexicallyIneligible,
    NoCorpusEvidence,
    CandidateNotGenerated,
    CandidateLostRanking,
    BelowThreshold,
    EmittedCorrectly,
}

impl TerminalReason {
    const ALL: [Self; 7] = [
        Self::HardOrganizationAbstention,
        Self::LexicallyIneligible,
        Self::NoCorpusEvidence,
        Self::CandidateNotGenerated,
        Self::CandidateLostRanking,
        Self::BelowThreshold,
        Self::EmittedCorrectly,
    ];

    fn as_str(self) -> &'static str {
        match self {
            Self::HardOrganizationAbstention => "hard_organization_abstention",
            Self::LexicallyIneligible => "lexically_ineligible",
            Self::NoCorpusEvidence => "no_direct_or_composable_corpus_evidence",
            Self::CandidateNotGenerated => "evidence_but_candidate_not_generated",
            Self::CandidateLostRanking => "candidate_lost_ranking",
            Self::BelowThreshold => "correct_candidate_below_threshold",
            Self::EmittedCorrectly => "emitted_correctly",
        }
    }

    fn is_miss(self) -> bool {
        self != Self::EmittedCorrectly
    }
}

struct ExpectedCaseDiagnostic {
    id: String,
    display_name: String,
    country_hint: String,
    locale_hint: String,
    expected: String,
    lookup: ExpectedLookupDiagnostic,
    composition: ExpectedCompositionDiagnostic,
    inference: RoleInferenceDiagnostic,
    expected_candidate: Option<CandidateDiagnostic>,
    strongest_competitor: Option<CandidateDiagnostic>,
    expected_rank: Option<usize>,
    corpus_covered: bool,
    candidate_generated: bool,
    production_reachable: bool,
    ranking_won: bool,
    threshold_correct: bool,
    margin: Option<f64>,
    terminal_reason: TerminalReason,
}

pub fn run_proxy_diagnostic(
    output: &Path,
    corpus: &impl EvidenceSource,
    holdout: FrozenHoldout,
    threshold: f64,
) -> Result<String> {
    let mut decisions = Vec::with_capacity(holdout.cases.len());
    let mut expected_cases = Vec::with_capacity(holdout.manifest.expected_greetings);
    let mut emitted_rows = Vec::new();

    for case in &holdout.cases {
        if !case.is_evaluable() {
            decisions.push(None);
            continue;
        }
        let diagnostic = diagnose_role_inference(
            corpus,
            ALGORITHM_C1,
            &case.display_name,
            nonempty(&case.country_hint),
            nonempty(&case.locale_hint),
        );
        let decision = SealedDecision {
            greeting_candidate: diagnostic.inference.greeting_candidate.clone(),
            confidence: diagnostic.inference.confidence,
        };
        if decision.confidence >= threshold && decision.greeting_candidate.is_some() {
            emitted_rows.push((case.clone(), decision.clone()));
        }
        decisions.push(Some(decision));

        if let Some(expected) = case.expected_greeting() {
            expected_cases.push(analyze_expected_case(
                corpus, case, expected, diagnostic, threshold,
            ));
        }
    }

    let checkpoint = evaluate_sealed(&holdout, &decisions, threshold)?;
    verify_checkpoint_ceiling(&expected_cases, &checkpoint)?;
    fs::write(
        output.join("checkpoint_summary_metrics.csv"),
        sealed_summary_csv(&checkpoint)?,
    )?;
    fs::write(
        output.join("checkpoint_confidence_buckets.csv"),
        sealed_confidence_buckets_csv(&checkpoint)?,
    )?;
    write_funnel(output, &expected_cases)?;
    write_oracle_ceilings(output, &expected_cases)?;
    write_expected_cases(output, &expected_cases)?;
    write_emitted_review(output, &emitted_rows, threshold)?;
    write_miss_review_sample(output, &holdout.manifest.holdout_sha256, &expected_cases)?;
    build_report(&holdout, &checkpoint, &expected_cases)
}

fn analyze_expected_case(
    corpus: &impl EvidenceSource,
    case: &HoldoutCase,
    expected: &str,
    inference: RoleInferenceDiagnostic,
    threshold: f64,
) -> ExpectedCaseDiagnostic {
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
    let expected_rank = inference
        .candidates
        .iter()
        .position(|candidate| greeting_matches(Some(expected), Some(&candidate.display)))
        .map(|index| index + 1);
    let expected_candidate = expected_rank.map(|rank| inference.candidates[rank - 1].clone());
    let strongest_competitor = inference
        .candidates
        .iter()
        .find(|candidate| !greeting_matches(Some(expected), Some(&candidate.display)))
        .cloned();
    let candidate_generated = expected_candidate.is_some();
    let corpus_covered = lookup.evidence.is_some() || composition.supported;
    let production_reachable = candidate_generated && !inference.hard_organization_abstention;
    let ranking_won = greeting_matches(
        Some(expected),
        inference.inference.greeting_candidate.as_deref(),
    );
    let threshold_correct =
        greeting_matches(Some(expected), inference.inference.greeting_at(threshold));
    let margin = expected_candidate.as_ref().and_then(|candidate| {
        strongest_competitor
            .as_ref()
            .map(|competitor| candidate.score - competitor.score)
    });
    let terminal_reason = if inference.hard_organization_abstention {
        TerminalReason::HardOrganizationAbstention
    } else if !lookup.eligible {
        TerminalReason::LexicallyIneligible
    } else if !corpus_covered {
        TerminalReason::NoCorpusEvidence
    } else if !candidate_generated {
        TerminalReason::CandidateNotGenerated
    } else if !ranking_won {
        TerminalReason::CandidateLostRanking
    } else if !threshold_correct {
        TerminalReason::BelowThreshold
    } else {
        TerminalReason::EmittedCorrectly
    };

    ExpectedCaseDiagnostic {
        id: case.id.clone(),
        display_name: case.display_name.clone(),
        country_hint: case.country_hint.clone(),
        locale_hint: case.locale_hint.clone(),
        expected: expected.to_string(),
        lookup,
        composition,
        inference,
        expected_candidate,
        strongest_competitor,
        expected_rank,
        corpus_covered,
        candidate_generated,
        production_reachable,
        ranking_won,
        threshold_correct,
        margin,
        terminal_reason,
    }
}

fn verify_checkpoint_ceiling(
    cases: &[ExpectedCaseDiagnostic],
    checkpoint: &SealedEvaluation,
) -> Result<()> {
    let threshold_correct = cases.iter().filter(|case| case.threshold_correct).count();
    if threshold_correct != checkpoint.metrics.correct_greetings {
        return Err(format!(
            "diagnostic threshold ceiling {threshold_correct} differs from checkpoint correct emissions {}",
            checkpoint.metrics.correct_greetings
        )
        .into());
    }
    Ok(())
}

fn write_funnel(output: &Path, cases: &[ExpectedCaseDiagnostic]) -> Result<()> {
    let total = cases.len();
    let stages = [
        ("expected_greetings", total),
        (
            "lexically_eligible",
            count(cases, |case| case.lookup.eligible),
        ),
        (
            "direct_normalized_lookup",
            count(cases, |case| case.lookup.lookup_mode == Some("normalized")),
        ),
        (
            "direct_accent_folded_lookup",
            count(cases, |case| {
                case.lookup.lookup_mode == Some("accent_folded")
            }),
        ),
        (
            "direct_any_lookup",
            count(cases, |case| case.lookup.evidence.is_some()),
        ),
        (
            "corpus_covered_direct_or_composed",
            count(cases, |case| case.corpus_covered),
        ),
        (
            "matching_candidate_generated_counterfactual",
            count(cases, |case| case.candidate_generated),
        ),
        (
            "matching_candidate_production_reachable",
            count(cases, |case| case.production_reachable),
        ),
        (
            "correct_candidate_ranked_first",
            count(cases, |case| case.ranking_won),
        ),
        (
            "correct_candidate_at_or_above_0_93",
            count(cases, |case| case.threshold_correct),
        ),
    ];
    let mut writer = csv::Writer::from_path(output.join("diagnostic_funnel.csv"))?;
    writer.write_record(["kind", "stage_or_reason", "count", "percent_of_expected"])?;
    for (stage, value) in stages {
        writer.write_record([
            "stage",
            stage,
            &value.to_string(),
            &format_ratio(value, total),
        ])?;
    }
    for reason in TerminalReason::ALL {
        let value = count(cases, |case| case.terminal_reason == reason);
        writer.write_record([
            "terminal_reason",
            reason.as_str(),
            &value.to_string(),
            &format_ratio(value, total),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn write_oracle_ceilings(output: &Path, cases: &[ExpectedCaseDiagnostic]) -> Result<()> {
    let total = cases.len();
    let rows = [
        (
            "corpus_coverage",
            count(cases, |case| case.corpus_covered),
            "direct normalized/folded evidence or a generated C1 composition",
        ),
        (
            "candidate_generation_counterfactual",
            count(cases, |case| case.candidate_generated),
            "matching candidate generated before whole-input hard abstention",
        ),
        (
            "candidate_generation_production_reachable",
            count(cases, |case| case.production_reachable),
            "matching candidate generated and no hard organization abstention",
        ),
        (
            "candidate_ranking",
            count(cases, |case| case.ranking_won),
            "production C1 selects the expected greeting before thresholding",
        ),
        (
            "threshold_0_93",
            count(cases, |case| case.threshold_correct),
            "selected expected greeting reaches frozen threshold 0.93",
        ),
    ];
    let mut writer = csv::Writer::from_path(output.join("oracle_ceilings.csv"))?;
    writer.write_record(["ceiling", "count", "denominator", "rate", "definition"])?;
    for (name, value, definition) in rows {
        writer.write_record([
            name,
            &value.to_string(),
            &total.to_string(),
            &format_ratio(value, total),
            definition,
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn write_expected_cases(output: &Path, cases: &[ExpectedCaseDiagnostic]) -> Result<()> {
    let mut writer = csv::Writer::from_path(output.join("expected_case_diagnostics.csv"))?;
    writer.write_record(expected_header())?;
    for case in cases {
        writer.write_record(expected_row(case))?;
    }
    writer.flush()?;
    Ok(())
}

fn expected_header() -> [&'static str; 50] {
    [
        "id",
        "display_name",
        "country_hint",
        "locale_hint",
        "expected_greeting",
        "terminal_reason",
        "lexically_eligible",
        "direct_lookup_mode",
        "direct_matched_query",
        "direct_global_given_count",
        "direct_country_given_count",
        "direct_global_surname_count",
        "direct_role_llr",
        "composition_shape",
        "composition_supported",
        "composition_left_lookup_mode",
        "composition_right_lookup_mode",
        "composition_left_role_llr",
        "composition_right_role_llr",
        "hard_organization_abstention",
        "generic_organization_marker",
        "ampersand_negative_evidence",
        "corpus_covered",
        "candidate_generated",
        "production_reachable",
        "expected_candidate_rank",
        "expected_candidate_origin",
        "expected_candidate_matched_query",
        "expected_candidate_lookup_mode",
        "expected_candidate_left_lookup_mode",
        "expected_candidate_right_lookup_mode",
        "expected_candidate_global_given_count",
        "expected_candidate_country_given_count",
        "expected_candidate_global_surname_count",
        "expected_candidate_role_llr",
        "expected_candidate_score",
        "strongest_competitor",
        "strongest_competitor_origin",
        "strongest_competitor_global_given_count",
        "strongest_competitor_global_surname_count",
        "strongest_competitor_role_llr",
        "strongest_competitor_score",
        "signed_expected_margin",
        "selected_greeting_prethreshold",
        "final_confidence",
        "ranking_won",
        "threshold_correct",
        "candidate_count",
        "manual_judgment",
        "manual_notes",
    ]
}

fn expected_row(case: &ExpectedCaseDiagnostic) -> Vec<String> {
    let direct = case.lookup.evidence;
    let expected = case.expected_candidate.as_ref();
    let competitor = case.strongest_competitor.as_ref();
    vec![
        case.id.clone(),
        case.display_name.clone(),
        case.country_hint.clone(),
        case.locale_hint.clone(),
        case.expected.clone(),
        case.terminal_reason.as_str().to_string(),
        case.lookup.eligible.to_string(),
        optional(case.lookup.lookup_mode),
        case.lookup.matched_query.clone().unwrap_or_default(),
        direct.map_or_else(String::new, |evidence| evidence.global_count.to_string()),
        direct.map_or_else(String::new, |evidence| evidence.country_count.to_string()),
        direct.map_or_else(String::new, |evidence| evidence.surname_count.to_string()),
        optional_f64(case.lookup.role_llr),
        optional(case.composition.shape),
        case.composition.supported.to_string(),
        optional(case.composition.left_lookup_mode),
        optional(case.composition.right_lookup_mode),
        optional_f64(case.composition.left_role_llr),
        optional_f64(case.composition.right_role_llr),
        case.inference.hard_organization_abstention.to_string(),
        case.inference.generic_organization_marker.to_string(),
        case.inference.ampersand_negative_evidence.to_string(),
        case.corpus_covered.to_string(),
        case.candidate_generated.to_string(),
        case.production_reachable.to_string(),
        case.expected_rank
            .map_or_else(String::new, |rank| rank.to_string()),
        expected.map_or_else(String::new, |candidate| candidate.origin.to_string()),
        expected
            .and_then(|candidate| candidate.lookup_query.clone())
            .unwrap_or_default(),
        expected
            .and_then(|candidate| candidate.lookup_mode)
            .map_or_else(String::new, str::to_string),
        expected
            .and_then(|candidate| candidate.left_lookup_mode)
            .map_or_else(String::new, str::to_string),
        expected
            .and_then(|candidate| candidate.right_lookup_mode)
            .map_or_else(String::new, str::to_string),
        expected.map_or_else(String::new, |candidate| {
            candidate.global_given_count.to_string()
        }),
        expected.map_or_else(String::new, |candidate| {
            candidate.country_given_count.to_string()
        }),
        expected.map_or_else(String::new, |candidate| {
            candidate.global_surname_count.to_string()
        }),
        expected.map_or_else(String::new, |candidate| {
            format!("{:.6}", candidate.role_llr)
        }),
        expected.map_or_else(String::new, |candidate| format!("{:.6}", candidate.score)),
        competitor.map_or_else(String::new, |candidate| candidate.display.clone()),
        competitor.map_or_else(String::new, |candidate| candidate.origin.to_string()),
        competitor.map_or_else(String::new, |candidate| {
            candidate.global_given_count.to_string()
        }),
        competitor.map_or_else(String::new, |candidate| {
            candidate.global_surname_count.to_string()
        }),
        competitor.map_or_else(String::new, |candidate| {
            format!("{:.6}", candidate.role_llr)
        }),
        competitor.map_or_else(String::new, |candidate| format!("{:.6}", candidate.score)),
        optional_f64(case.margin),
        case.inference
            .inference
            .greeting_candidate
            .clone()
            .unwrap_or_default(),
        format!("{:.6}", case.inference.inference.confidence),
        case.ranking_won.to_string(),
        case.threshold_correct.to_string(),
        case.inference.candidates.len().to_string(),
        String::new(),
        String::new(),
    ]
}

fn write_emitted_review(
    output: &Path,
    rows: &[(HoldoutCase, SealedDecision)],
    threshold: f64,
) -> Result<()> {
    let mut writer = csv::Writer::from_path(output.join("emitted_review_queue.csv"))?;
    writer.write_record([
        "id",
        "display_name",
        "country_hint",
        "locale_hint",
        "label_status",
        "expected_greeting",
        "predicted_greeting",
        "confidence",
        "outcome",
        "manual_judgment",
        "manual_notes",
    ])?;
    for (case, decision) in rows {
        let predicted = decision
            .greeting_candidate
            .as_deref()
            .filter(|_| decision.confidence >= threshold);
        let correct = greeting_matches(case.expected_greeting(), predicted);
        writer.write_record([
            case.id.clone(),
            case.display_name.clone(),
            case.country_hint.clone(),
            case.locale_hint.clone(),
            label_status(case.label_status).to_string(),
            case.expected_greeting().unwrap_or("").to_string(),
            predicted.unwrap_or("").to_string(),
            format!("{:.6}", decision.confidence),
            if correct { "correct" } else { "wrong" }.to_string(),
            String::new(),
            String::new(),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn write_miss_review_sample(
    output: &Path,
    holdout_sha256: &str,
    cases: &[ExpectedCaseDiagnostic],
) -> Result<()> {
    let mut writer = csv::Writer::from_path(output.join("miss_review_sample.csv"))?;
    let mut header = vec![
        "sample_seed".to_string(),
        "category_population".to_string(),
        "category_selected".to_string(),
    ];
    header.extend(expected_header().map(str::to_string));
    writer.write_record(header)?;
    for reason in TerminalReason::ALL
        .into_iter()
        .filter(|reason| reason.is_miss())
    {
        let mut category = cases
            .iter()
            .filter(|case| case.terminal_reason == reason)
            .collect::<Vec<_>>();
        category.sort_by_key(|case| sample_key(holdout_sha256, reason, &case.id));
        let population = category.len();
        let selected = population.min(50);
        for case in category.into_iter().take(selected) {
            let mut row = vec![
                format!("0x{DIAGNOSTIC_SAMPLE_SEED:016x}"),
                population.to_string(),
                selected.to_string(),
            ];
            row.extend(expected_row(case));
            writer.write_record(row)?;
        }
    }
    writer.flush()?;
    Ok(())
}

fn sample_key(holdout_sha256: &str, reason: TerminalReason, id: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(DIAGNOSTIC_SAMPLE_SEED.to_le_bytes());
    hasher.update(holdout_sha256.as_bytes());
    hasher.update(reason.as_str().as_bytes());
    hasher.update(id.as_bytes());
    hasher.finalize().into()
}

fn build_report(
    holdout: &FrozenHoldout,
    checkpoint: &SealedEvaluation,
    cases: &[ExpectedCaseDiagnostic],
) -> Result<String> {
    let total = cases.len();
    let mut report = String::new();
    writeln!(report, "# REAL_PROXY_V1 diagnostic funnel\n")?;
    writeln!(
        report,
        "This diagnostic deliberately spends checksum-verified holdout `{}`. The original aggregate C1 checkpoint remains historical evidence, but these inspected rows are now development evidence and cannot validate C2.\n",
        holdout.manifest.holdout_sha256
    )?;
    writeln!(report, "## Preserved C1 checkpoint\n")?;
    writeln!(
        report,
        "| Evaluable | Expected greetings | Emitted | Correct | Wrong | Precision | Recall |"
    )?;
    writeln!(report, "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")?;
    writeln!(
        report,
        "| {} | {} | {} | {} | {} | {} | {} |\n",
        checkpoint.metrics.evaluable_cases,
        checkpoint.metrics.expected_greetings,
        checkpoint.metrics.emitted_greetings,
        checkpoint.metrics.correct_greetings,
        checkpoint.metrics.wrong_greetings,
        percent(checkpoint.metrics.greeting_precision()),
        percent(checkpoint.metrics.greeting_recall()),
    )?;
    writeln!(report, "## Funnel and oracle ceilings\n")?;
    writeln!(report, "| Stage | Count | Expected-greeting share |")?;
    writeln!(report, "| --- | ---: | ---: |")?;
    for (label, value) in [
        (
            "Lexically eligible",
            count(cases, |case| case.lookup.eligible),
        ),
        (
            "Direct normalized lookup",
            count(cases, |case| case.lookup.lookup_mode == Some("normalized")),
        ),
        (
            "Direct accent-folded lookup",
            count(cases, |case| {
                case.lookup.lookup_mode == Some("accent_folded")
            }),
        ),
        (
            "Corpus coverage ceiling",
            count(cases, |case| case.corpus_covered),
        ),
        (
            "Candidate-generation ceiling",
            count(cases, |case| case.candidate_generated),
        ),
        (
            "Production-reachable generation",
            count(cases, |case| case.production_reachable),
        ),
        (
            "Candidate-ranking ceiling",
            count(cases, |case| case.ranking_won),
        ),
        (
            "Threshold ceiling at 0.93",
            count(cases, |case| case.threshold_correct),
        ),
    ] {
        writeln!(
            report,
            "| {label} | {value} | {} |",
            percent_count(value, total)
        )?;
    }
    writeln!(report, "\n## Mutually exclusive terminal reasons\n")?;
    writeln!(report, "| Reason | Count | Expected-greeting share |")?;
    writeln!(report, "| --- | ---: | ---: |")?;
    for reason in TerminalReason::ALL {
        let value = count(cases, |case| case.terminal_reason == reason);
        writeln!(
            report,
            "| `{}` | {value} | {} |",
            reason.as_str(),
            percent_count(value, total)
        )?;
    }
    writeln!(report, "\n## Score diagnostics\n")?;
    writeln!(report, "| Distribution | N | Min | Median | P90 | Max |")?;
    writeln!(report, "| --- | ---: | ---: | ---: | ---: | ---: |")?;
    let distributions = [
        (
            "Generated expected-candidate score",
            cases
                .iter()
                .filter_map(|case| {
                    case.expected_candidate
                        .as_ref()
                        .map(|candidate| candidate.score)
                })
                .collect::<Vec<_>>(),
        ),
        (
            "Correct selected final confidence",
            cases
                .iter()
                .filter(|case| case.ranking_won)
                .map(|case| case.inference.inference.confidence)
                .collect::<Vec<_>>(),
        ),
        (
            "Strongest competitor score",
            cases
                .iter()
                .filter_map(|case| {
                    case.strongest_competitor
                        .as_ref()
                        .map(|candidate| candidate.score)
                })
                .collect::<Vec<_>>(),
        ),
        (
            "Signed expected-candidate margin",
            cases
                .iter()
                .filter_map(|case| case.margin)
                .collect::<Vec<_>>(),
        ),
    ];
    for (label, values) in distributions {
        let summary = summarize(values);
        writeln!(
            report,
            "| {label} | {} | {} | {} | {} | {} |",
            summary.count,
            optional_f64(summary.minimum),
            optional_f64(summary.median),
            optional_f64(summary.p90),
            optional_f64(summary.maximum),
        )?;
    }
    writeln!(report, "\n## Review protocol\n")?;
    writeln!(
        report,
        "`emitted_review_queue.csv` contains every emitted row. `miss_review_sample.csv` contains up to 50 deterministic rows per terminal miss category using seed `0x{DIAGNOSTIC_SAMPLE_SEED:016x}`. Row-level files are local development material and must not be committed. Manual review must not rewrite frozen labels or checkpoint metrics.\n"
    )?;
    writeln!(
        report,
        "The proxy contains no `non_person` labels; its expected abstentions have unknown type, so this sample cannot establish an organization false-positive rate. Confidence values are uncalibrated scores, not probabilities. No threshold sweep was run."
    )?;
    Ok(report)
}

#[derive(Default)]
struct DistributionSummary {
    count: usize,
    minimum: Option<f64>,
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
        median: quantile(&values, 0.5),
        p90: quantile(&values, 0.9),
        maximum: values.last().copied(),
    }
}

fn quantile(values: &[f64], probability: f64) -> Option<f64> {
    (!values.is_empty()).then(|| {
        let index = ((values.len() - 1) as f64 * probability).round() as usize;
        values[index]
    })
}

fn count(
    cases: &[ExpectedCaseDiagnostic],
    predicate: impl Fn(&ExpectedCaseDiagnostic) -> bool,
) -> usize {
    cases.iter().filter(|case| predicate(case)).count()
}

fn nonempty(value: &str) -> Option<&str> {
    (!value.is_empty()).then_some(value)
}

fn optional(value: Option<&str>) -> String {
    value.unwrap_or("").to_string()
}

fn optional_f64(value: Option<f64>) -> String {
    value.map_or_else(String::new, |value| format!("{value:.6}"))
}

fn format_ratio(numerator: usize, denominator: usize) -> String {
    if denominator == 0 {
        String::new()
    } else {
        format!("{:.6}", numerator as f64 / denominator as f64)
    }
}

fn percent_count(numerator: usize, denominator: usize) -> String {
    if denominator == 0 {
        "n/a".to_string()
    } else {
        format!("{:.2}%", 100.0 * numerator as f64 / denominator as f64)
    }
}

fn percent(value: Option<f64>) -> String {
    value.map_or_else(
        || "n/a".to_string(),
        |value| format!("{:.2}%", 100.0 * value),
    )
}

fn label_status(status: LabelStatus) -> &'static str {
    match status {
        LabelStatus::Unlabeled => "unlabeled",
        LabelStatus::Greeting => "greeting",
        LabelStatus::Abstain => "abstain",
        LabelStatus::Skip => "skip",
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::artifact::Evidence;

    use super::*;

    struct FakeCorpus(HashMap<String, Evidence>);

    impl EvidenceSource for FakeCorpus {
        fn lookup(&self, name: &str, _country_hint: Option<[u8; 2]>) -> Option<Evidence> {
            self.0.get(name).copied()
        }
    }

    fn evidence(given: u64, surname: u64) -> Evidence {
        Evidence {
            global_count: given,
            country_count: 0,
            effective_count: given,
            female_count: 0,
            male_count: given,
            surname_count: surname,
            given_total: 444_154_759,
            surname_total: 489_631_377,
        }
    }

    fn greeting_case(id: &str, display_name: &str, expected: &str) -> HoldoutCase {
        HoldoutCase {
            id: id.to_string(),
            display_name: display_name.to_string(),
            country_hint: String::new(),
            locale_hint: String::new(),
            label_status: LabelStatus::Greeting,
            expected_greeting: expected.to_string(),
            span_start: None,
            span_end: None,
            case_kind: name_eval::holdout::CaseKind::Person,
        }
    }

    fn analyze(corpus: &FakeCorpus, case: &HoldoutCase) -> ExpectedCaseDiagnostic {
        let diagnostic =
            diagnose_role_inference(corpus, ALGORITHM_C1, &case.display_name, None, None);
        analyze_expected_case(
            corpus,
            case,
            case.expected_greeting().unwrap(),
            diagnostic,
            0.93,
        )
    }

    #[test]
    fn sample_key_is_deterministic_and_category_sensitive() {
        let left = sample_key("abc", TerminalReason::BelowThreshold, "case-1");
        assert_eq!(
            left,
            sample_key("abc", TerminalReason::BelowThreshold, "case-1")
        );
        assert_ne!(
            left,
            sample_key("abc", TerminalReason::CandidateLostRanking, "case-1")
        );
    }

    #[test]
    fn summary_uses_stable_nearest_rank_indices() {
        let summary = summarize(vec![5.0, 1.0, 4.0, 2.0, 3.0]);
        assert_eq!(summary.count, 5);
        assert_eq!(summary.minimum, Some(1.0));
        assert_eq!(summary.median, Some(3.0));
        assert_eq!(summary.p90, Some(5.0));
        assert_eq!(summary.maximum, Some(5.0));
    }

    #[test]
    fn expected_cases_are_partitioned_into_all_terminal_reasons() {
        let corpus = FakeCorpus(HashMap::from([
            ("Martin".to_string(), evidence(50_000, 1_000)),
            ("Anne Marie Rose".to_string(), evidence(50_000, 100)),
            ("Jean".to_string(), evidence(100, 100_000)),
            ("Winner".to_string(), evidence(1_500_000, 0)),
            ("Robin".to_string(), evidence(5, 5)),
            ("Quentin".to_string(), evidence(1_500_000, 0)),
        ]));
        let cases = [
            greeting_case("hard", "Martin GmbH", "Martin"),
            greeting_case("lexical", "A/B", "A/B"),
            greeting_case("missing", "Unknown", "Unknown"),
            greeting_case("not-generated", "Anne Marie Rose", "Anne Marie Rose"),
            greeting_case("lost", "Jean Winner", "Jean"),
            greeting_case("threshold", "Robin", "Robin"),
            greeting_case("correct", "Quentin", "Quentin"),
        ];
        let reasons = cases
            .iter()
            .map(|case| analyze(&corpus, case).terminal_reason)
            .collect::<Vec<_>>();
        assert_eq!(reasons, TerminalReason::ALL);
    }

    #[test]
    fn diagnostic_report_is_aggregate_but_review_rows_are_explicit() {
        let corpus = FakeCorpus(HashMap::from([(
            "Uniqueprivatevalue".to_string(),
            evidence(1_500_000, 0),
        )]));
        let case = greeting_case("case-private", "Uniqueprivatevalue", "Uniqueprivatevalue");
        let holdout = FrozenHoldout {
            cases: vec![case],
            manifest: name_eval::holdout::HoldoutManifest {
                format_version: 1,
                holdout_sha256: "abc123".to_string(),
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
        let output = std::env::temp_dir().join(format!(
            "bonjour-proxy-diagnostic-test-{}",
            std::process::id()
        ));
        if output.exists() {
            fs::remove_dir_all(&output).unwrap();
        }
        fs::create_dir(&output).unwrap();
        let report = run_proxy_diagnostic(&output, &corpus, holdout, 0.93).unwrap();
        assert!(!report.contains("Uniqueprivatevalue"));
        assert!(
            fs::read_to_string(output.join("emitted_review_queue.csv"))
                .unwrap()
                .contains("Uniqueprivatevalue")
        );
        fs::remove_dir_all(output).unwrap();
    }
}
