use std::collections::HashSet;
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
    if decisions.len() != holdout.cases.len() {
        return Err("sealed decision count does not match holdout case count".into());
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
    let mut buckets = [
        ConfidenceBucket {
            label: "0.93–0.95",
            emitted: 0,
            correct: 0,
            wrong: 0,
        },
        ConfidenceBucket {
            label: "0.95–0.97",
            emitted: 0,
            correct: 0,
            wrong: 0,
        },
        ConfidenceBucket {
            label: "0.97–0.99",
            emitted: 0,
            correct: 0,
            wrong: 0,
        },
        ConfidenceBucket {
            label: "0.99–1.00",
            emitted: 0,
            correct: 0,
            wrong: 0,
        },
    ];

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
            let bucket = confidence_bucket(&mut buckets, decision.confidence)?;
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
    confidence: f64,
) -> Result<&mut ConfidenceBucket> {
    let index = if (0.93..0.95).contains(&confidence) {
        0
    } else if (0.95..0.97).contains(&confidence) {
        1
    } else if (0.97..0.99).contains(&confidence) {
        2
    } else if (0.99..=1.0).contains(&confidence) {
        3
    } else {
        return Err(format!("emitted confidence {confidence} is outside sealed buckets").into());
    };
    Ok(&mut buckets[index])
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
}
