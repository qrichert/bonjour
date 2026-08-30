use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::fs;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use unicode_general_category::{GeneralCategory, get_general_category};
use unicode_normalization::UnicodeNormalization;

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub const FORMAT_VERSION: u32 = 1;

const HOLDOUT_HEADER: [&str; 9] = [
    "id",
    "display_name",
    "country_hint",
    "locale_hint",
    "label_status",
    "expected_greeting",
    "span_start",
    "span_end",
    "case_kind",
];
const MANIFEST_HEADER: [&str; 11] = [
    "format_version",
    "holdout_sha256",
    "total_cases",
    "evaluable_cases",
    "skipped_cases",
    "expected_greetings",
    "expected_abstentions",
    "person_cases",
    "non_person_cases",
    "unknown_kind_cases",
    "provenance",
];
const BLIND_ANNOTATION_HEADER: [&str; 6] = [
    "id",
    "display_name",
    "country_hint",
    "locale_hint",
    "decision",
    "expected_greeting",
];
const CONSENSUS_HEADER: [&str; 5] = [
    "total_cases",
    "greeting_agreements",
    "null_agreements",
    "annotator_skip_cases",
    "disagreement_cases",
];

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LabelStatus {
    Unlabeled,
    Greeting,
    Abstain,
    Skip,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CaseKind {
    Person,
    NonPerson,
    Unknown,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct HoldoutCase {
    pub id: String,
    pub display_name: String,
    #[serde(default)]
    pub country_hint: String,
    #[serde(default)]
    pub locale_hint: String,
    pub label_status: LabelStatus,
    #[serde(default)]
    pub expected_greeting: String,
    #[serde(default)]
    pub span_start: Option<usize>,
    #[serde(default)]
    pub span_end: Option<usize>,
    pub case_kind: CaseKind,
}

impl HoldoutCase {
    pub fn is_evaluable(&self) -> bool {
        matches!(
            self.label_status,
            LabelStatus::Greeting | LabelStatus::Abstain
        )
    }

    pub fn expected_greeting(&self) -> Option<&str> {
        (self.label_status == LabelStatus::Greeting).then_some(self.expected_greeting.as_str())
    }

    pub fn select_greeting(&mut self, candidate: &SpanCandidate) -> Result<()> {
        validate_span(
            &self.display_name,
            candidate.start,
            candidate.end,
            &candidate.text,
        )?;
        self.label_status = LabelStatus::Greeting;
        self.expected_greeting.clone_from(&candidate.text);
        self.span_start = Some(candidate.start);
        self.span_end = Some(candidate.end);
        self.case_kind = CaseKind::Person;
        Ok(())
    }

    pub fn select_abstention(&mut self, kind: CaseKind) {
        self.label_status = LabelStatus::Abstain;
        self.expected_greeting.clear();
        self.span_start = None;
        self.span_end = None;
        self.case_kind = kind;
    }

    pub fn select_skip(&mut self) {
        self.label_status = LabelStatus::Skip;
        self.expected_greeting.clear();
        self.span_start = None;
        self.span_end = None;
        self.case_kind = CaseKind::Unknown;
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpanCandidate {
    pub start: usize,
    pub end: usize,
    pub text: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct HoldoutManifest {
    pub format_version: u32,
    pub holdout_sha256: String,
    pub total_cases: usize,
    pub evaluable_cases: usize,
    pub skipped_cases: usize,
    pub expected_greetings: usize,
    pub expected_abstentions: usize,
    pub person_cases: usize,
    pub non_person_cases: usize,
    pub unknown_kind_cases: usize,
    pub provenance: String,
}

#[derive(Clone, Debug)]
pub struct FrozenHoldout {
    pub cases: Vec<HoldoutCase>,
    pub manifest: HoldoutManifest,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct ConsensusSummary {
    pub total_cases: usize,
    pub greeting_agreements: usize,
    pub null_agreements: usize,
    pub annotator_skip_cases: usize,
    pub disagreement_cases: usize,
}

#[derive(Clone, Debug)]
pub struct SealedDecision {
    pub greeting_candidate: Option<String>,
    pub confidence: f64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SealedMetrics {
    pub total_labeled_cases: usize,
    pub evaluable_cases: usize,
    pub skipped_cases: usize,
    pub emitted_greetings: usize,
    pub correct_greetings: usize,
    pub wrong_greetings: usize,
    pub expected_greetings: usize,
    pub expected_greetings_missed: usize,
    pub expected_abstentions: usize,
    pub false_emissions_on_expected_abstentions: usize,
    pub abstentions: usize,
    pub non_person_cases: usize,
    pub non_person_false_positives: usize,
}

impl SealedMetrics {
    pub fn greeting_precision(self) -> Option<f64> {
        ratio(self.correct_greetings, self.emitted_greetings)
    }

    pub fn greeting_recall(self) -> Option<f64> {
        ratio(self.correct_greetings, self.expected_greetings)
    }

    pub fn abstention_rate(self) -> Option<f64> {
        ratio(self.abstentions, self.evaluable_cases)
    }

    pub fn non_person_false_positive_rate(self) -> Option<f64> {
        ratio(self.non_person_false_positives, self.non_person_cases)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConfidenceBucket {
    pub label: &'static str,
    pub emitted: usize,
    pub correct: usize,
    pub wrong: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConfidenceBucketSpec {
    pub label: &'static str,
    pub lower: f64,
    pub upper: f64,
}

#[derive(Clone, Debug)]
pub struct SealedEvaluation {
    pub threshold: f64,
    pub metrics: SealedMetrics,
    pub confidence_buckets: [ConfidenceBucket; 4],
}

#[derive(Deserialize)]
struct SourceRow {
    display_name: String,
    #[serde(default)]
    country_hint: String,
    #[serde(default)]
    locale_hint: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct BlindAnnotationRow {
    id: String,
    display_name: String,
    country_hint: String,
    locale_hint: String,
    decision: String,
    expected_greeting: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum AnnotationDecision {
    Greeting(String),
    Abstain,
    Skip,
}

pub fn load_source(path: &Path) -> Result<Vec<HoldoutCase>> {
    let mut reader = csv::Reader::from_path(path)?;
    let headers = reader.headers()?.clone();
    let mut seen = HashSet::new();
    for header in &headers {
        if !matches!(header, "display_name" | "country_hint" | "locale_hint") {
            return Err(format!("unsupported holdout source column {header:?}").into());
        }
        if !seen.insert(header) {
            return Err(format!("duplicate holdout source column {header:?}").into());
        }
    }
    if !seen.contains("display_name") {
        return Err("holdout source is missing display_name".into());
    }
    let mut rows = reader
        .deserialize::<SourceRow>()
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if rows.iter().any(|row| row.display_name.trim().is_empty()) {
        return Err("holdout source contains an empty display_name".into());
    }
    rows.sort_by(|left, right| {
        (&left.display_name, &left.country_hint, &left.locale_hint).cmp(&(
            &right.display_name,
            &right.country_hint,
            &right.locale_hint,
        ))
    });
    let mut cases = Vec::with_capacity(rows.len());
    for (index, row) in rows.into_iter().enumerate() {
        cases.push(HoldoutCase {
            id: format!("case-{index:08}"),
            display_name: row.display_name,
            country_hint: row.country_hint,
            locale_hint: row.locale_hint,
            label_status: LabelStatus::Unlabeled,
            expected_greeting: String::new(),
            span_start: None,
            span_end: None,
            case_kind: CaseKind::Unknown,
        });
    }
    cases.sort_by(|left, right| left.id.cmp(&right.id));
    validate_cases(&cases, true)?;
    Ok(cases)
}

pub fn load_or_initialize_draft(source: &Path, draft: &Path) -> Result<Vec<HoldoutCase>> {
    let source_cases = load_source(source)?;
    if !draft.exists() {
        write_cases_atomic(draft, &source_cases, false)?;
        return Ok(source_cases);
    }
    let draft_cases = load_cases(draft, true)?;
    if source_identity(&source_cases) != source_identity(&draft_cases) {
        return Err("existing draft does not match the supplied holdout source".into());
    }
    Ok(draft_cases)
}

pub fn export_blind_annotation_template(source: &Path, destination: &Path) -> Result<()> {
    let cases = load_source(source)?;
    let bytes = serialize_blind_annotations(
        &cases
            .iter()
            .map(|case| BlindAnnotationRow {
                id: case.id.clone(),
                display_name: case.display_name.clone(),
                country_hint: case.country_hint.clone(),
                locale_hint: case.locale_hint.clone(),
                decision: String::new(),
                expected_greeting: String::new(),
            })
            .collect::<Vec<_>>(),
    )?;
    write_new_file(destination, &bytes)
}

pub fn merge_blind_annotations(
    source: &Path,
    annotation_a: &Path,
    annotation_b: &Path,
    draft: &Path,
    summary_path: &Path,
) -> Result<ConsensusSummary> {
    if draft.exists() || summary_path.exists() {
        return Err("refusing to overwrite an existing consensus output".into());
    }
    let mut cases = load_source(source)?;
    let decisions_a = load_blind_annotations(annotation_a, &cases)?;
    let decisions_b = load_blind_annotations(annotation_b, &cases)?;
    let mut summary = ConsensusSummary {
        total_cases: cases.len(),
        ..ConsensusSummary::default()
    };

    for case in &mut cases {
        let decision_a = decisions_a
            .get(&case.id)
            .ok_or_else(|| format!("annotation A is missing {}", case.id))?;
        let decision_b = decisions_b
            .get(&case.id)
            .ok_or_else(|| format!("annotation B is missing {}", case.id))?;
        match (decision_a, decision_b) {
            (AnnotationDecision::Greeting(left), AnnotationDecision::Greeting(right))
                if left == right =>
            {
                let candidate = exact_annotation_span(&case.display_name, left)?;
                case.select_greeting(&candidate)?;
                summary.greeting_agreements += 1;
            }
            (AnnotationDecision::Abstain, AnnotationDecision::Abstain) => {
                case.select_abstention(CaseKind::Unknown);
                summary.null_agreements += 1;
            }
            (AnnotationDecision::Skip, _) | (_, AnnotationDecision::Skip) => {
                case.select_skip();
                summary.annotator_skip_cases += 1;
            }
            _ => {
                case.select_skip();
                summary.disagreement_cases += 1;
            }
        }
    }

    let draft_bytes = serialize_cases(&cases)?;
    let summary_bytes = serialize_consensus_summary(summary)?;
    write_new_pair(draft, &draft_bytes, summary_path, &summary_bytes)?;
    Ok(summary)
}

pub fn save_draft(path: &Path, cases: &[HoldoutCase]) -> Result<()> {
    validate_cases(cases, true)?;
    write_cases_atomic(path, cases, true)
}

pub fn load_cases(path: &Path, allow_unlabeled: bool) -> Result<Vec<HoldoutCase>> {
    let bytes = fs::read(path)?;
    parse_cases(&bytes, allow_unlabeled)
}

pub fn freeze(
    draft: &Path,
    sealed: &Path,
    manifest_path: &Path,
    provenance: &str,
) -> Result<HoldoutManifest> {
    if provenance.trim().is_empty() {
        return Err("holdout provenance must not be empty".into());
    }
    if sealed.exists() || manifest_path.exists() {
        return Err("refusing to overwrite an existing sealed holdout or manifest".into());
    }
    let cases = load_cases(draft, false)?;
    let sealed_bytes = serialize_cases(&cases)?;
    let manifest = manifest_for(&cases, &sealed_bytes, provenance);
    let manifest_bytes = serialize_manifest(&manifest)?;
    write_new_pair(sealed, &sealed_bytes, manifest_path, &manifest_bytes)?;
    Ok(manifest)
}

pub fn load_frozen(sealed: &Path, manifest_path: &Path) -> Result<FrozenHoldout> {
    let sealed_bytes = fs::read(sealed)?;
    let manifest = load_manifest(manifest_path)?;
    if manifest.format_version != FORMAT_VERSION {
        return Err(format!(
            "unsupported holdout manifest version {}",
            manifest.format_version
        )
        .into());
    }
    let actual_sha256 = sha256_hex(&sealed_bytes);
    if actual_sha256 != manifest.holdout_sha256 {
        return Err(format!(
            "sealed holdout checksum changed: expected {}, got {actual_sha256}",
            manifest.holdout_sha256
        )
        .into());
    }
    let cases = parse_cases(&sealed_bytes, false)?;
    if serialize_cases(&cases)? != sealed_bytes {
        return Err("sealed holdout serialization is not canonical".into());
    }
    let actual = manifest_for(&cases, &sealed_bytes, &manifest.provenance);
    if actual != manifest {
        return Err("sealed holdout counts differ from the frozen manifest".into());
    }
    Ok(FrozenHoldout { cases, manifest })
}

pub fn span_candidates(display_name: &str) -> Vec<SpanCandidate> {
    let mut token_ranges = Vec::<(usize, usize)>::new();
    let mut token_start = None;
    for (index, character) in display_name.char_indices() {
        if is_name_span_character(character) {
            if token_start.is_none() {
                token_start = Some(index);
            }
        } else if let Some(start) = token_start.take() {
            token_ranges.push((start, index));
        }
    }
    if let Some(start) = token_start {
        token_ranges.push((start, display_name.len()));
    }

    let mut candidates = Vec::new();
    for length in 1..=token_ranges.len() {
        for start_index in 0..=token_ranges.len() - length {
            let start = token_ranges[start_index].0;
            let end = token_ranges[start_index + length - 1].1;
            candidates.push(SpanCandidate {
                start,
                end,
                text: display_name[start..end].to_string(),
            });
        }
    }
    candidates
}

fn is_name_span_character(character: char) -> bool {
    character.is_alphabetic()
        || matches!(
            get_general_category(character),
            GeneralCategory::NonspacingMark
                | GeneralCategory::SpacingMark
                | GeneralCategory::EnclosingMark
        )
        || matches!(
            character,
            '\'' | '’' | 'ʼ' | 'ʻ' | '-' | '‐' | '‑' | '‒' | '–' | '—'
        )
}

pub fn render_label_prompt(case: &HoldoutCase) -> String {
    let mut output = String::new();
    output.push_str("\nDisplay name: ");
    output.push_str(&terminal_safe(&case.display_name));
    output.push('\n');
    if !case.country_hint.is_empty() {
        output.push_str("Country hint: ");
        output.push_str(&terminal_safe(&case.country_hint));
        output.push('\n');
    }
    if !case.locale_hint.is_empty() {
        output.push_str("Locale hint: ");
        output.push_str(&terminal_safe(&case.locale_hint));
        output.push('\n');
    }
    output.push_str("\nChoose the exact greeting span:\n");
    for (index, candidate) in span_candidates(&case.display_name).iter().enumerate() {
        output.push_str(&format!(
            "  [{}] {}\n",
            index + 1,
            terminal_safe(&candidate.text)
        ));
    }
    output.push_str(
        "  [N] NULL / no safe greeting\n  [S] SKIP / undecidable\n  [Q] save and quit\n> ",
    );
    output
}

pub fn evaluate_sealed(
    holdout: &FrozenHoldout,
    decisions: &[Option<SealedDecision>],
    threshold: f64,
) -> Result<SealedEvaluation> {
    evaluate_sealed_with_buckets(
        holdout,
        decisions,
        threshold,
        [
            ConfidenceBucketSpec {
                label: "0.93–0.95",
                lower: 0.93,
                upper: 0.95,
            },
            ConfidenceBucketSpec {
                label: "0.95–0.97",
                lower: 0.95,
                upper: 0.97,
            },
            ConfidenceBucketSpec {
                label: "0.97–0.99",
                lower: 0.97,
                upper: 0.99,
            },
            ConfidenceBucketSpec {
                label: "0.99–1.00",
                lower: 0.99,
                upper: 1.00,
            },
        ],
    )
}

pub fn evaluate_sealed_with_buckets(
    holdout: &FrozenHoldout,
    decisions: &[Option<SealedDecision>],
    threshold: f64,
    bucket_specs: [ConfidenceBucketSpec; 4],
) -> Result<SealedEvaluation> {
    if decisions.len() != holdout.cases.len() {
        return Err("sealed decision count does not match holdout case count".into());
    }
    validate_bucket_specs(threshold, bucket_specs)?;
    let mut metrics = SealedMetrics {
        total_labeled_cases: holdout.manifest.total_cases,
        evaluable_cases: holdout.manifest.evaluable_cases,
        skipped_cases: holdout.manifest.skipped_cases,
        expected_greetings: holdout.manifest.expected_greetings,
        expected_abstentions: holdout.manifest.expected_abstentions,
        non_person_cases: holdout.manifest.non_person_cases,
        ..SealedMetrics::default()
    };
    let mut buckets = bucket_specs.map(|spec| ConfidenceBucket {
        label: spec.label,
        emitted: 0,
        correct: 0,
        wrong: 0,
    });

    for (case, decision) in holdout.cases.iter().zip(decisions) {
        if case.label_status == LabelStatus::Skip {
            if decision.is_some() {
                return Err("skipped sealed case unexpectedly has a classifier decision".into());
            }
            continue;
        }
        let decision = decision
            .as_ref()
            .ok_or("evaluable sealed case is missing a classifier decision")?;
        let emitted = decision
            .greeting_candidate
            .as_deref()
            .filter(|_| decision.confidence >= threshold);
        let correct = greeting_matches(case.expected_greeting(), emitted);
        if emitted.is_some() {
            metrics.emitted_greetings += 1;
            if correct {
                metrics.correct_greetings += 1;
            } else {
                metrics.wrong_greetings += 1;
            }
            let bucket = confidence_bucket(&mut buckets, bucket_specs, decision.confidence)?;
            bucket.emitted += 1;
            if correct {
                bucket.correct += 1;
            } else {
                bucket.wrong += 1;
            }
        } else {
            metrics.abstentions += 1;
        }
        if case.label_status == LabelStatus::Greeting && !correct {
            metrics.expected_greetings_missed += 1;
        }
        if case.label_status == LabelStatus::Abstain && emitted.is_some() {
            metrics.false_emissions_on_expected_abstentions += 1;
        }
        if case.case_kind == CaseKind::NonPerson && emitted.is_some() {
            metrics.non_person_false_positives += 1;
        }
    }

    Ok(SealedEvaluation {
        threshold,
        metrics,
        confidence_buckets: buckets,
    })
}

pub fn evaluate_explicit_emissions(
    holdout: &FrozenHoldout,
    emissions: &[Option<String>],
) -> Result<SealedMetrics> {
    if emissions.len() != holdout.cases.len() {
        return Err("sealed emission count does not match holdout case count".into());
    }
    let mut metrics = SealedMetrics {
        total_labeled_cases: holdout.manifest.total_cases,
        evaluable_cases: holdout.manifest.evaluable_cases,
        skipped_cases: holdout.manifest.skipped_cases,
        expected_greetings: holdout.manifest.expected_greetings,
        expected_abstentions: holdout.manifest.expected_abstentions,
        non_person_cases: holdout.manifest.non_person_cases,
        ..SealedMetrics::default()
    };

    for (case, emission) in holdout.cases.iter().zip(emissions) {
        if case.label_status == LabelStatus::Skip {
            if emission.is_some() {
                return Err("skipped sealed case unexpectedly has an emission".into());
            }
            continue;
        }
        let emitted = emission.as_deref();
        let correct = greeting_matches(case.expected_greeting(), emitted);
        if emitted.is_some() {
            metrics.emitted_greetings += 1;
            if correct {
                metrics.correct_greetings += 1;
            } else {
                metrics.wrong_greetings += 1;
            }
        } else {
            metrics.abstentions += 1;
        }
        if case.label_status == LabelStatus::Greeting && !correct {
            metrics.expected_greetings_missed += 1;
        }
        if case.label_status == LabelStatus::Abstain && emitted.is_some() {
            metrics.false_emissions_on_expected_abstentions += 1;
        }
        if case.case_kind == CaseKind::NonPerson && emitted.is_some() {
            metrics.non_person_false_positives += 1;
        }
    }

    Ok(metrics)
}

pub fn sealed_summary_csv(evaluation: &SealedEvaluation) -> Result<Vec<u8>> {
    let metrics = evaluation.metrics;
    let mut writer = canonical_writer();
    writer.write_record([
        "threshold",
        "total_labeled_cases",
        "evaluable_cases",
        "skipped_cases",
        "emitted_greetings",
        "correct_greetings",
        "wrong_greetings",
        "expected_greetings",
        "expected_greetings_missed",
        "expected_abstentions",
        "false_emissions_on_expected_abstentions",
        "abstentions",
        "greeting_precision",
        "greeting_recall",
        "abstention_rate",
        "non_person_cases",
        "non_person_false_positives",
        "non_person_false_positive_rate",
    ])?;
    writer.write_record([
        format!("{:.6}", evaluation.threshold),
        metrics.total_labeled_cases.to_string(),
        metrics.evaluable_cases.to_string(),
        metrics.skipped_cases.to_string(),
        metrics.emitted_greetings.to_string(),
        metrics.correct_greetings.to_string(),
        metrics.wrong_greetings.to_string(),
        metrics.expected_greetings.to_string(),
        metrics.expected_greetings_missed.to_string(),
        metrics.expected_abstentions.to_string(),
        metrics.false_emissions_on_expected_abstentions.to_string(),
        metrics.abstentions.to_string(),
        format_ratio(metrics.greeting_precision()),
        format_ratio(metrics.greeting_recall()),
        format_ratio(metrics.abstention_rate()),
        metrics.non_person_cases.to_string(),
        metrics.non_person_false_positives.to_string(),
        format_ratio(metrics.non_person_false_positive_rate()),
    ])?;
    Ok(writer.into_inner()?)
}

pub fn sealed_confidence_buckets_csv(evaluation: &SealedEvaluation) -> Result<Vec<u8>> {
    let mut writer = canonical_writer();
    writer.write_record(["confidence_bucket", "emitted", "correct", "wrong"])?;
    for bucket in evaluation.confidence_buckets {
        writer.write_record([
            bucket.label.to_string(),
            bucket.emitted.to_string(),
            bucket.correct.to_string(),
            bucket.wrong.to_string(),
        ])?;
    }
    Ok(writer.into_inner()?)
}

fn source_identity(cases: &[HoldoutCase]) -> Vec<(&str, &str, &str, &str)> {
    cases
        .iter()
        .map(|case| {
            (
                case.id.as_str(),
                case.display_name.as_str(),
                case.country_hint.as_str(),
                case.locale_hint.as_str(),
            )
        })
        .collect()
}

fn serialize_blind_annotations(rows: &[BlindAnnotationRow]) -> Result<Vec<u8>> {
    let mut sorted = rows.to_vec();
    sorted.sort_by(|left, right| left.id.cmp(&right.id));
    let mut writer = canonical_writer();
    writer.write_record(BLIND_ANNOTATION_HEADER)?;
    for row in sorted {
        writer.serialize(row)?;
    }
    Ok(writer.into_inner()?)
}

fn load_blind_annotations(
    path: &Path,
    source_cases: &[HoldoutCase],
) -> Result<BTreeMap<String, AnnotationDecision>> {
    let mut reader = csv::Reader::from_path(path)?;
    if reader.headers()?.iter().ne(BLIND_ANNOTATION_HEADER) {
        return Err("blind annotation CSV header does not match the exchange format".into());
    }
    let rows = reader
        .deserialize::<BlindAnnotationRow>()
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if rows.len() != source_cases.len() {
        return Err(format!(
            "blind annotation row count differs from source: expected {}, got {}",
            source_cases.len(),
            rows.len()
        )
        .into());
    }
    let source_by_id = source_cases
        .iter()
        .map(|case| (case.id.as_str(), case))
        .collect::<BTreeMap<_, _>>();
    let mut decisions = BTreeMap::new();
    for row in rows {
        let Some(source) = source_by_id.get(row.id.as_str()) else {
            return Err(format!("blind annotation contains unknown ID {}", row.id).into());
        };
        if row.display_name != source.display_name
            || row.country_hint != source.country_hint
            || row.locale_hint != source.locale_hint
        {
            return Err(format!("blind annotation mutated source fields for {}", row.id).into());
        }
        let decision = parse_annotation_decision(&row, &source.display_name)?;
        if decisions.insert(row.id.clone(), decision).is_some() {
            return Err(format!("blind annotation contains duplicate ID {}", row.id).into());
        }
    }
    if decisions.len() != source_cases.len() {
        return Err("blind annotation does not cover every source case".into());
    }
    Ok(decisions)
}

fn parse_annotation_decision(
    row: &BlindAnnotationRow,
    display_name: &str,
) -> Result<AnnotationDecision> {
    match row.decision.as_str() {
        "GREETING" => {
            if row.expected_greeting.is_empty() {
                return Err(format!("{} has an empty GREETING label", row.id).into());
            }
            exact_annotation_span(display_name, &row.expected_greeting)?;
            Ok(AnnotationDecision::Greeting(row.expected_greeting.clone()))
        }
        "NULL" => {
            if !row.expected_greeting.is_empty() {
                return Err(format!("{} has text attached to a NULL label", row.id).into());
            }
            Ok(AnnotationDecision::Abstain)
        }
        "SKIP" => {
            if !row.expected_greeting.is_empty() {
                return Err(format!("{} has text attached to a SKIP label", row.id).into());
            }
            Ok(AnnotationDecision::Skip)
        }
        "" => Err(format!("{} has not been annotated", row.id).into()),
        other => Err(format!("{} has unsupported decision {other:?}", row.id).into()),
    }
}

fn exact_annotation_span(display_name: &str, expected: &str) -> Result<SpanCandidate> {
    let Some(start) = display_name.find(expected) else {
        return Err("annotation greeting is not an exact span of display_name".into());
    };
    let end = start + expected.len();
    validate_span(display_name, start, end, expected)?;
    Ok(SpanCandidate {
        start,
        end,
        text: expected.to_string(),
    })
}

fn serialize_consensus_summary(summary: ConsensusSummary) -> Result<Vec<u8>> {
    let mut writer = canonical_writer();
    writer.write_record(CONSENSUS_HEADER)?;
    writer.serialize(summary)?;
    Ok(writer.into_inner()?)
}

fn validate_cases(cases: &[HoldoutCase], allow_unlabeled: bool) -> Result<()> {
    let mut ids = HashSet::new();
    let mut previous_id = None::<&str>;
    for case in cases {
        if case.id.is_empty() || !ids.insert(case.id.as_str()) {
            return Err(format!("empty or duplicate holdout ID {:?}", case.id).into());
        }
        if previous_id.is_some_and(|previous| previous >= case.id.as_str()) {
            return Err("holdout cases are not sorted by opaque ID".into());
        }
        previous_id = Some(&case.id);
        match case.label_status {
            LabelStatus::Unlabeled => {
                if !allow_unlabeled {
                    return Err(format!("{} has not been labeled", case.id).into());
                }
                validate_empty_label(case)?;
            }
            LabelStatus::Greeting => {
                if case.case_kind != CaseKind::Person {
                    return Err(
                        format!("{} has a greeting but is not labeled person", case.id).into(),
                    );
                }
                let (Some(start), Some(end)) = (case.span_start, case.span_end) else {
                    return Err(format!("{} is missing greeting span offsets", case.id).into());
                };
                validate_span(&case.display_name, start, end, &case.expected_greeting)?;
            }
            LabelStatus::Abstain => validate_empty_label(case)?,
            LabelStatus::Skip => {
                validate_empty_label(case)?;
                if case.case_kind != CaseKind::Unknown {
                    return Err(format!("{} is skipped but has a case kind", case.id).into());
                }
            }
        }
    }
    Ok(())
}

fn validate_empty_label(case: &HoldoutCase) -> Result<()> {
    if !case.expected_greeting.is_empty() || case.span_start.is_some() || case.span_end.is_some() {
        return Err(format!(
            "{} has label text or offsets without a greeting label",
            case.id
        )
        .into());
    }
    Ok(())
}

fn validate_span(display_name: &str, start: usize, end: usize, expected: &str) -> Result<()> {
    if start >= end
        || !display_name.is_char_boundary(start)
        || !display_name.is_char_boundary(end)
        || display_name.get(start..end) != Some(expected)
    {
        return Err("greeting label is not an exact UTF-8 span of display_name".into());
    }
    Ok(())
}

fn parse_cases(bytes: &[u8], allow_unlabeled: bool) -> Result<Vec<HoldoutCase>> {
    let mut reader = csv::Reader::from_reader(bytes);
    if reader.headers()?.iter().ne(HOLDOUT_HEADER) {
        return Err("holdout CSV header does not match the frozen format".into());
    }
    let cases = reader
        .deserialize::<HoldoutCase>()
        .collect::<std::result::Result<Vec<_>, _>>()?;
    validate_cases(&cases, allow_unlabeled)?;
    Ok(cases)
}

fn serialize_cases(cases: &[HoldoutCase]) -> Result<Vec<u8>> {
    let mut sorted = cases.to_vec();
    sorted.sort_by(|left, right| left.id.cmp(&right.id));
    validate_cases(&sorted, true)?;
    let mut writer = canonical_writer();
    writer.write_record(HOLDOUT_HEADER)?;
    for case in sorted {
        writer.serialize(case)?;
    }
    Ok(writer.into_inner()?)
}

fn manifest_for(cases: &[HoldoutCase], sealed_bytes: &[u8], provenance: &str) -> HoldoutManifest {
    let evaluable = cases.iter().filter(|case| case.is_evaluable()).count();
    HoldoutManifest {
        format_version: FORMAT_VERSION,
        holdout_sha256: sha256_hex(sealed_bytes),
        total_cases: cases.len(),
        evaluable_cases: evaluable,
        skipped_cases: cases.len() - evaluable,
        expected_greetings: cases
            .iter()
            .filter(|case| case.label_status == LabelStatus::Greeting)
            .count(),
        expected_abstentions: cases
            .iter()
            .filter(|case| case.label_status == LabelStatus::Abstain)
            .count(),
        person_cases: cases
            .iter()
            .filter(|case| case.is_evaluable() && case.case_kind == CaseKind::Person)
            .count(),
        non_person_cases: cases
            .iter()
            .filter(|case| case.is_evaluable() && case.case_kind == CaseKind::NonPerson)
            .count(),
        unknown_kind_cases: cases
            .iter()
            .filter(|case| case.is_evaluable() && case.case_kind == CaseKind::Unknown)
            .count(),
        provenance: provenance.trim().to_string(),
    }
}

fn serialize_manifest(manifest: &HoldoutManifest) -> Result<Vec<u8>> {
    let mut writer = canonical_writer();
    writer.write_record(MANIFEST_HEADER)?;
    writer.serialize(manifest)?;
    Ok(writer.into_inner()?)
}

fn load_manifest(path: &Path) -> Result<HoldoutManifest> {
    let mut reader = csv::Reader::from_path(path)?;
    if reader.headers()?.iter().ne(MANIFEST_HEADER) {
        return Err("holdout manifest header does not match the frozen format".into());
    }
    let rows = reader
        .deserialize::<HoldoutManifest>()
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let [manifest] = rows.as_slice() else {
        return Err("holdout manifest must contain exactly one data row".into());
    };
    Ok(manifest.clone())
}

fn canonical_writer() -> csv::Writer<Vec<u8>> {
    csv::WriterBuilder::new()
        .has_headers(false)
        .terminator(csv::Terminator::Any(b'\n'))
        .from_writer(Vec::new())
}

fn write_cases_atomic(path: &Path, cases: &[HoldoutCase], replace: bool) -> Result<()> {
    let bytes = serialize_cases(cases)?;
    write_atomic(path, &bytes, replace)
}

fn write_atomic(path: &Path, bytes: &[u8], replace: bool) -> Result<()> {
    if !replace {
        return write_new_file(path, bytes);
    }
    let temporary = temporary_path(path);
    let result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn write_new_pair(
    left_path: &Path,
    left_bytes: &[u8],
    right_path: &Path,
    right_bytes: &[u8],
) -> Result<()> {
    let mut left_created = false;
    let mut right_created = false;
    let result = (|| -> Result<()> {
        write_new_file(left_path, left_bytes)?;
        left_created = true;
        write_new_file(right_path, right_bytes)?;
        right_created = true;
        Ok(())
    })();
    if result.is_err() {
        if left_created {
            let _ = fs::remove_file(left_path);
        }
        if right_created {
            let _ = fs::remove_file(right_path);
        }
    }
    result
}

fn write_new_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn temporary_path(path: &Path) -> PathBuf {
    let file_name = path.file_name().map_or_else(
        || "holdout".to_string(),
        |name| name.to_string_lossy().into_owned(),
    );
    path.with_file_name(format!(".{file_name}.tmp-{}", std::process::id()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn terminal_safe(value: &str) -> String {
    value
        .chars()
        .flat_map(|character| {
            if character.is_control() {
                character.escape_default().collect::<Vec<_>>()
            } else {
                vec![character]
            }
        })
        .collect()
}

fn greeting_matches(expected: Option<&str>, actual: Option<&str>) -> bool {
    match (expected, actual) {
        (Some(expected), Some(actual)) => {
            normalize_greeting(expected) == normalize_greeting(actual)
        }
        (None, None) => true,
        _ => false,
    }
}

fn normalize_greeting(value: &str) -> String {
    value
        .nfc()
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn confidence_bucket(
    buckets: &mut [ConfidenceBucket; 4],
    specs: [ConfidenceBucketSpec; 4],
    confidence: f64,
) -> Result<&mut ConfidenceBucket> {
    let index = specs
        .iter()
        .enumerate()
        .find(|(index, spec)| {
            confidence >= spec.lower
                && if *index + 1 == specs.len() {
                    confidence <= spec.upper
                } else {
                    confidence < spec.upper
                }
        })
        .map(|(index, _)| index)
        .ok_or_else(|| format!("emitted confidence {confidence} is outside sealed buckets"))?;
    Ok(&mut buckets[index])
}

fn validate_bucket_specs(threshold: f64, specs: [ConfidenceBucketSpec; 4]) -> Result<()> {
    if specs[0].lower != threshold {
        return Err("first sealed confidence bucket must begin at the emission threshold".into());
    }
    for (index, spec) in specs.iter().enumerate() {
        if spec.label.is_empty()
            || !spec.lower.is_finite()
            || !spec.upper.is_finite()
            || spec.lower >= spec.upper
            || index > 0 && specs[index - 1].upper != spec.lower
        {
            return Err("sealed confidence buckets must be labeled, finite, and contiguous".into());
        }
    }
    if specs[3].upper != 1.0 {
        return Err("last sealed confidence bucket must end at 1.0".into());
    }
    Ok(())
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator != 0).then(|| numerator as f64 / denominator as f64)
}

fn format_ratio(value: Option<f64>) -> String {
    value.map_or_else(String::new, |value| format!("{value:.6}"))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static TEMPORARY_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temporary_directory() -> PathBuf {
        let counter = TEMPORARY_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "name-eval-holdout-test-{}-{counter}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        path
    }

    fn source_case(display_name: &str) -> HoldoutCase {
        HoldoutCase {
            id: "case-000000000000000000000000".to_string(),
            display_name: display_name.to_string(),
            country_hint: "FR".to_string(),
            locale_hint: String::new(),
            label_status: LabelStatus::Unlabeled,
            expected_greeting: String::new(),
            span_start: None,
            span_end: None,
            case_kind: CaseKind::Unknown,
        }
    }

    fn frozen_cases() -> Vec<HoldoutCase> {
        let mut greeting = source_case("Élodie Durand");
        let candidate = span_candidates(&greeting.display_name)[0].clone();
        greeting.select_greeting(&candidate).unwrap();
        let mut abstain = source_case("Baris Kebab");
        abstain.id = "case-111111111111111111111111".to_string();
        abstain.select_abstention(CaseKind::NonPerson);
        let mut skipped = source_case("undecidable");
        skipped.id = "case-222222222222222222222222".to_string();
        skipped.select_skip();
        vec![greeting, abstain, skipped]
    }

    fn write_source_fixture(path: &Path) -> Vec<HoldoutCase> {
        fs::write(
            path,
            "display_name,country_hint,locale_hint\nAmbiguous,,\nBaris Kebab,,\nÉlodie Durand,FR,fr-FR\nМария Иванова,,\n",
        )
        .unwrap();
        load_source(path).unwrap()
    }

    fn write_annotation_fixture(path: &Path, cases: &[HoldoutCase], decisions: &[(&str, &str)]) {
        assert_eq!(cases.len(), decisions.len());
        let rows = cases
            .iter()
            .zip(decisions)
            .map(
                |(case, &(decision, expected_greeting))| BlindAnnotationRow {
                    id: case.id.clone(),
                    display_name: case.display_name.clone(),
                    country_hint: case.country_hint.clone(),
                    locale_hint: case.locale_hint.clone(),
                    decision: decision.to_string(),
                    expected_greeting: expected_greeting.to_string(),
                },
            )
            .collect::<Vec<_>>();
        fs::write(path, serialize_blind_annotations(&rows).unwrap()).unwrap();
    }

    #[test]
    fn unicode_round_trip_and_exact_span_are_preserved() {
        let cases = frozen_cases();
        let bytes = serialize_cases(&cases).unwrap();
        let decoded = parse_cases(&bytes, false).unwrap();
        assert_eq!(decoded, cases);
        assert_eq!(decoded[0].expected_greeting, "Élodie");
        assert_eq!(
            decoded[0]
                .display_name
                .get(decoded[0].span_start.unwrap()..decoded[0].span_end.unwrap()),
            Some("Élodie")
        );
    }

    #[test]
    fn null_and_skipped_labels_round_trip() {
        let cases = frozen_cases();
        assert_eq!(cases[1].label_status, LabelStatus::Abstain);
        assert_eq!(cases[1].expected_greeting(), None);
        assert_eq!(cases[2].label_status, LabelStatus::Skip);
        assert!(!cases[2].is_evaluable());
        parse_cases(&serialize_cases(&cases).unwrap(), false).unwrap();
    }

    #[test]
    fn invalid_or_modified_span_is_rejected() {
        let mut cases = frozen_cases();
        cases[0].expected_greeting = "Elodie".to_string();
        assert!(serialize_cases(&cases).is_err());
    }

    #[test]
    fn explicit_emissions_count_correct_abstained_and_skipped_cases() {
        let cases = frozen_cases();
        let sealed_bytes = serialize_cases(&cases).unwrap();
        let holdout = FrozenHoldout {
            manifest: manifest_for(&cases, &sealed_bytes, "fixture"),
            cases,
        };
        let metrics =
            evaluate_explicit_emissions(&holdout, &[Some("Élodie".to_string()), None, None])
                .unwrap();

        assert_eq!(metrics.total_labeled_cases, 3);
        assert_eq!(metrics.evaluable_cases, 2);
        assert_eq!(metrics.skipped_cases, 1);
        assert_eq!(metrics.emitted_greetings, 1);
        assert_eq!(metrics.correct_greetings, 1);
        assert_eq!(metrics.wrong_greetings, 0);
        assert_eq!(metrics.expected_greetings_missed, 0);
        assert_eq!(metrics.false_emissions_on_expected_abstentions, 0);
        assert_eq!(metrics.abstentions, 1);
    }

    #[test]
    fn explicit_emissions_count_wrong_greetings_and_null_false_emissions() {
        let cases = frozen_cases();
        let sealed_bytes = serialize_cases(&cases).unwrap();
        let holdout = FrozenHoldout {
            manifest: manifest_for(&cases, &sealed_bytes, "fixture"),
            cases,
        };
        let metrics = evaluate_explicit_emissions(
            &holdout,
            &[Some("Durand".to_string()), Some("Baris".to_string()), None],
        )
        .unwrap();

        assert_eq!(metrics.emitted_greetings, 2);
        assert_eq!(metrics.correct_greetings, 0);
        assert_eq!(metrics.wrong_greetings, 2);
        assert_eq!(metrics.expected_greetings_missed, 1);
        assert_eq!(metrics.false_emissions_on_expected_abstentions, 1);
        assert_eq!(metrics.non_person_false_positives, 1);
        assert_eq!(metrics.abstentions, 0);
    }

    #[test]
    fn explicit_emissions_reject_bad_lengths_and_skipped_emissions() {
        let cases = frozen_cases();
        let sealed_bytes = serialize_cases(&cases).unwrap();
        let holdout = FrozenHoldout {
            manifest: manifest_for(&cases, &sealed_bytes, "fixture"),
            cases,
        };

        assert!(evaluate_explicit_emissions(&holdout, &[None]).is_err());
        assert!(
            evaluate_explicit_emissions(&holdout, &[None, None, Some("undecidable".to_string())],)
                .is_err()
        );
    }

    #[test]
    fn serialization_is_deterministic_and_sorted() {
        let cases = frozen_cases();
        let mut reversed = cases.clone();
        reversed.reverse();
        assert_eq!(
            serialize_cases(&cases).unwrap(),
            serialize_cases(&reversed).unwrap()
        );
    }

    #[test]
    fn freeze_checksum_and_changed_holdout_detection() {
        let directory = temporary_directory();
        let draft = directory.join("draft.csv");
        let sealed = directory.join("sealed.csv");
        let manifest = directory.join("manifest.csv");
        write_cases_atomic(&draft, &frozen_cases(), false).unwrap();
        let frozen_manifest =
            freeze(&draft, &sealed, &manifest, "opt-in product QA sample").unwrap();
        assert_eq!(
            load_frozen(&sealed, &manifest).unwrap().manifest,
            frozen_manifest
        );
        let mut bytes = fs::read(&sealed).unwrap();
        bytes.push(b'\n');
        fs::write(&sealed, bytes).unwrap();
        assert!(
            load_frozen(&sealed, &manifest)
                .unwrap_err()
                .to_string()
                .contains("checksum changed")
        );
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn prompt_contains_no_classifier_evidence() {
        let prompt = render_label_prompt(&source_case("Anne Marie Dupont"));
        let lowercase = prompt.to_lowercase();
        for forbidden in ["confidence", "frequency", "llr", "classifier", "known name"] {
            assert!(
                !lowercase.contains(forbidden),
                "{forbidden:?} leaked into prompt"
            );
        }
        assert!(prompt.contains("Anne Marie"));
        assert!(prompt.contains("NULL"));
        assert!(prompt.contains("SKIP"));
    }

    #[test]
    fn blind_template_contains_only_source_fields_and_empty_labels() {
        let directory = temporary_directory();
        let source = directory.join("source.csv");
        let template = directory.join("annotation.csv");
        let cases = write_source_fixture(&source);
        export_blind_annotation_template(&source, &template).unwrap();

        let text = fs::read_to_string(&template).unwrap();
        assert_eq!(text.lines().count(), cases.len() + 1);
        for forbidden in ["confidence", "frequency", "llr", "classifier", "surname"] {
            assert!(!text.to_lowercase().contains(forbidden));
        }
        let mut reader = csv::Reader::from_path(&template).unwrap();
        let rows = reader
            .deserialize::<BlindAnnotationRow>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        assert!(
            rows.iter()
                .all(|row| row.decision.is_empty() && row.expected_greeting.is_empty())
        );
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn consensus_accepts_exact_agreement_and_skips_uncertainty() {
        let directory = temporary_directory();
        let source = directory.join("source.csv");
        let annotation_a = directory.join("a.csv");
        let annotation_b = directory.join("b.csv");
        let draft = directory.join("draft.csv");
        let summary_path = directory.join("summary.csv");
        let cases = write_source_fixture(&source);
        write_annotation_fixture(
            &annotation_a,
            &cases,
            &[
                ("GREETING", "Ambiguous"),
                ("NULL", ""),
                ("GREETING", "Élodie"),
                ("SKIP", ""),
            ],
        );
        write_annotation_fixture(
            &annotation_b,
            &cases,
            &[
                ("NULL", ""),
                ("NULL", ""),
                ("GREETING", "Élodie"),
                ("GREETING", "Мария"),
            ],
        );

        let summary =
            merge_blind_annotations(&source, &annotation_a, &annotation_b, &draft, &summary_path)
                .unwrap();
        assert_eq!(
            summary,
            ConsensusSummary {
                total_cases: 4,
                greeting_agreements: 1,
                null_agreements: 1,
                annotator_skip_cases: 1,
                disagreement_cases: 1,
            }
        );
        let consensus = load_cases(&draft, false).unwrap();
        assert_eq!(consensus[0].label_status, LabelStatus::Skip);
        assert_eq!(consensus[1].label_status, LabelStatus::Abstain);
        assert_eq!(consensus[2].expected_greeting(), Some("Élodie"));
        assert_eq!(consensus[2].span_start, Some(0));
        assert_eq!(consensus[2].span_end, Some("Élodie".len()));
        assert_eq!(consensus[3].label_status, LabelStatus::Skip);
        let summary_bytes = fs::read(&summary_path).unwrap();

        let second_draft = directory.join("draft-two.csv");
        let second_summary = directory.join("summary-two.csv");
        merge_blind_annotations(
            &source,
            &annotation_a,
            &annotation_b,
            &second_draft,
            &second_summary,
        )
        .unwrap();
        assert_eq!(fs::read(&draft).unwrap(), fs::read(second_draft).unwrap());
        assert_eq!(summary_bytes, fs::read(second_summary).unwrap());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn consensus_rejects_missing_mutated_duplicate_and_nonspan_annotations() {
        let directory = temporary_directory();
        let source = directory.join("source.csv");
        let cases = write_source_fixture(&source);
        let valid = directory.join("valid.csv");
        write_annotation_fixture(
            &valid,
            &cases,
            &[("NULL", ""), ("NULL", ""), ("NULL", ""), ("NULL", "")],
        );

        let missing = directory.join("missing.csv");
        let mut missing_rows = cases[..3]
            .iter()
            .map(|case| BlindAnnotationRow {
                id: case.id.clone(),
                display_name: case.display_name.clone(),
                country_hint: case.country_hint.clone(),
                locale_hint: case.locale_hint.clone(),
                decision: "NULL".to_string(),
                expected_greeting: String::new(),
            })
            .collect::<Vec<_>>();
        fs::write(
            &missing,
            serialize_blind_annotations(&missing_rows).unwrap(),
        )
        .unwrap();
        assert!(
            load_blind_annotations(&missing, &cases)
                .unwrap_err()
                .to_string()
                .contains("row count differs")
        );

        let mutated = directory.join("mutated.csv");
        missing_rows = cases
            .iter()
            .map(|case| BlindAnnotationRow {
                id: case.id.clone(),
                display_name: case.display_name.clone(),
                country_hint: case.country_hint.clone(),
                locale_hint: case.locale_hint.clone(),
                decision: "NULL".to_string(),
                expected_greeting: String::new(),
            })
            .collect();
        missing_rows[0].display_name.push('!');
        fs::write(
            &mutated,
            serialize_blind_annotations(&missing_rows).unwrap(),
        )
        .unwrap();
        assert!(
            load_blind_annotations(&mutated, &cases)
                .unwrap_err()
                .to_string()
                .contains("mutated source fields")
        );

        let duplicate = directory.join("duplicate.csv");
        missing_rows[0] = missing_rows[1].clone();
        fs::write(
            &duplicate,
            serialize_blind_annotations(&missing_rows).unwrap(),
        )
        .unwrap();
        assert!(load_blind_annotations(&duplicate, &cases).is_err());

        let nonspan = directory.join("nonspan.csv");
        write_annotation_fixture(
            &nonspan,
            &cases,
            &[
                ("GREETING", "Not present"),
                ("NULL", ""),
                ("NULL", ""),
                ("NULL", ""),
            ],
        );
        assert!(
            load_blind_annotations(&nonspan, &cases)
                .unwrap_err()
                .to_string()
                .contains("not an exact span")
        );
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn punctuation_delimits_options_but_name_separators_do_not() {
        let slash = span_candidates("Jean / Sophie");
        assert!(slash.iter().any(|candidate| candidate.text == "Jean"));
        assert!(slash.iter().any(|candidate| candidate.text == "Sophie"));
        let parenthesized = span_candidates("Pierre (Papa)");
        assert!(
            parenthesized
                .iter()
                .any(|candidate| candidate.text == "Pierre")
        );
        assert!(
            parenthesized
                .iter()
                .any(|candidate| candidate.text == "Papa")
        );
        assert_eq!(span_candidates("Jean-Pierre")[0].text, "Jean-Pierre");
        assert_eq!(span_candidates("O'Connor")[0].text, "O'Connor");
    }

    #[test]
    fn source_rejects_unnecessary_personal_data_columns() {
        let directory = temporary_directory();
        let source = directory.join("source.csv");
        fs::write(
            &source,
            "display_name,country_hint,email\nExample,FR,not-retained@example.invalid\n",
        )
        .unwrap();
        assert!(
            load_source(&source)
                .unwrap_err()
                .to_string()
                .contains("unsupported holdout source column")
        );
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn sealed_evaluation_is_aggregate_only() {
        let cases = frozen_cases();
        let sealed_bytes = serialize_cases(&cases).unwrap();
        let holdout = FrozenHoldout {
            manifest: manifest_for(&cases, &sealed_bytes, "test fixture"),
            cases,
        };
        let evaluation = evaluate_sealed(
            &holdout,
            &[
                Some(SealedDecision {
                    greeting_candidate: Some("Élodie".to_string()),
                    confidence: 0.94,
                }),
                Some(SealedDecision {
                    greeting_candidate: Some("Baris".to_string()),
                    confidence: 0.98,
                }),
                None,
            ],
            0.93,
        )
        .unwrap();
        assert_eq!(evaluation.metrics.correct_greetings, 1);
        assert_eq!(evaluation.metrics.wrong_greetings, 1);
        assert_eq!(
            evaluation.metrics.false_emissions_on_expected_abstentions,
            1
        );
        let summary = String::from_utf8(sealed_summary_csv(&evaluation).unwrap()).unwrap();
        let buckets =
            String::from_utf8(sealed_confidence_buckets_csv(&evaluation).unwrap()).unwrap();
        for forbidden in ["display_name", "span_start", "case-", "Élodie", "Baris"] {
            assert!(!summary.contains(forbidden));
            assert!(!buckets.contains(forbidden));
        }
    }

    #[test]
    fn sealed_evaluation_supports_distinct_contiguous_score_buckets() {
        let cases = frozen_cases();
        let sealed_bytes = serialize_cases(&cases).unwrap();
        let holdout = FrozenHoldout {
            manifest: manifest_for(&cases, &sealed_bytes, "test fixture"),
            cases,
        };
        let specs = [
            ConfidenceBucketSpec {
                label: "0.70–0.80",
                lower: 0.70,
                upper: 0.80,
            },
            ConfidenceBucketSpec {
                label: "0.80–0.90",
                lower: 0.80,
                upper: 0.90,
            },
            ConfidenceBucketSpec {
                label: "0.90–0.95",
                lower: 0.90,
                upper: 0.95,
            },
            ConfidenceBucketSpec {
                label: "0.95–1.00",
                lower: 0.95,
                upper: 1.00,
            },
        ];
        let evaluation = evaluate_sealed_with_buckets(
            &holdout,
            &[
                Some(SealedDecision {
                    greeting_candidate: Some("Élodie".to_string()),
                    confidence: 0.70,
                }),
                Some(SealedDecision {
                    greeting_candidate: Some("Baris".to_string()),
                    confidence: 1.00,
                }),
                None,
            ],
            0.70,
            specs,
        )
        .unwrap();
        assert_eq!(evaluation.confidence_buckets[0].correct, 1);
        assert_eq!(evaluation.confidence_buckets[3].wrong, 1);

        let mut invalid = specs;
        invalid[1].lower = 0.81;
        assert!(
            evaluate_sealed_with_buckets(&holdout, &[None, None, None], 0.70, invalid).is_err()
        );
    }
}
