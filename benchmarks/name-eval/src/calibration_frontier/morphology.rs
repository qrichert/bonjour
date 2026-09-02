use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::Path;
use std::time::Instant;

use name_eval::holdout::FrozenHoldout;
use sha2::{Digest, Sha256};
use unicode_casefold::UnicodeCaseFold;
use unicode_general_category::{GeneralCategory, get_general_category};
use unicode_normalization::UnicodeNormalization;
use unicode_script::{Script, UnicodeScript};
use xxhash_rust::xxh3::xxh3_64_with_seed;

use super::{
    EmissionMetrics, EvidenceSource, FeatureRow, Population, Result, c4_decision_breakdown,
    feature_row_from_decision, greeting_matches, validate_and_order_holdouts, wilson_interval,
};
use crate::classifier::{
    ALGORITHM_C2, ALGORITHM_C3, ALGORITHM_C4, ALGORITHM_C31, C4EmissionSource, CandidateDiagnostic,
    RoleInferenceDiagnostic, canonicalize, diagnose_role_inference,
};
use crate::dataset::{Case, Split, generate_cases};
use crate::lexical::candidate_is_eligible;

const TOTALS_SHA256: &str = "e43e8661261b2762d3d4f2581ebb803af94abb7505409873f46041be1470ff62";
const EXPECTED_KEYS: usize = 1_803_175;
const GIVEN_TOTAL: u64 = 444_154_759;
const SURNAME_OVERLAP_TOTAL: u64 = 364_386_816;
const SURNAME_TOTAL: u64 = 489_631_377;
const ROLE_SMOOTHING: f64 = 0.5;
const MIN_ROLE_COUNT: u64 = 100;
const MIN_ROLE_LLR: f64 = 2.0;
const MIN_ALPHABETIC: usize = 3;
const SPLIT_SEED: u64 = 0x6d6f_7270_682d_7370;
const ORDER_SEED: u64 = 0x6d6f_7270_682d_6f72;
const NGRAM_SEED: u64 = 0x6d6f_7270_682d_6e67;
const NGRAM_COLLISION_SEED: u64 = 0x6d6f_7270_682d_6332;
const START_SENTINEL: u32 = 0x11_0000;
const END_SENTINEL: u32 = 0x11_0001;
const TRAIN_BUCKET_CUTOFF: u64 = 800;
const VALIDATION_BUCKET_CUTOFF: u64 = 900;
const SPLIT_BUCKETS: u64 = 1_000;
const TRAINING_EPOCHS: usize = 5;
const FTRL_BETA: f64 = 1.0;
const FTRL_L1: f64 = 0.0;
const DEFAULT_ALPHA: f64 = 0.10;
const DEFAULT_L2: f64 = 1.0;
const CAPACITIES: [usize; 4] = [16 * 1024, 32 * 1024, 64 * 1024, 128 * 1024];
const NGRAM_MAXIMA: [u8; 3] = [3, 4, 5];
const ALPHAS: [f64; 2] = [0.05, 0.10];
const L2_VALUES: [f64; 2] = [0.1, 1.0];
const RANKING_WEIGHTS: [f64; 5] = [0.0, 0.01, 0.02, 0.03, 0.04];
const TARGETS: [f64; 6] = [0.995, 0.99, 0.98, 0.97, 0.95, 0.90];
const BASE_FEATURES: usize = 7;
const MAIN_FEATURES: usize = 8;
const INTERACTION_FEATURES: usize = 12;
const BASE_FEATURE_NAMES: [&str; BASE_FEATURES] = [
    "decision_score",
    "candidate_quality",
    "winner_margin",
    "role_signal",
    "reliability",
    "sole_candidate",
    "native_candidate",
];
const INTERACTION_FEATURE_NAMES: [&str; 4] = [
    "morphology_x_candidate_quality",
    "morphology_x_role_signal",
    "morphology_x_reliability",
    "morphology_x_winner_margin",
];
const CALIBRATION_L2: f64 = 0.01;
const MAX_CALIBRATION_ITERATIONS: usize = 10_000;
const PARAMETER_TOLERANCE: f64 = 1.0e-10;
const ARMIJO: f64 = 1.0e-4;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum RoleLabel {
    Given,
    Surname,
}

impl RoleLabel {
    fn value(self) -> f64 {
        f64::from(self == Self::Given)
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Given => "given",
            Self::Surname => "surname",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum MorphSplit {
    Train,
    Validation,
    Test,
}

impl MorphSplit {
    const ALL: [Self; 3] = [Self::Train, Self::Validation, Self::Test];

    fn from_group(group: &str) -> Self {
        let bucket = xxh3_64_with_seed(group.as_bytes(), SPLIT_SEED) % SPLIT_BUCKETS;
        if bucket < TRAIN_BUCKET_CUTOFF {
            Self::Train
        } else if bucket < VALIDATION_BUCKET_CUTOFF {
            Self::Validation
        } else {
            Self::Test
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Train => "TRAIN",
            Self::Validation => "VALIDATION",
            Self::Test => "TEST",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
enum ScriptClass {
    Latin,
    Cyrillic,
    Greek,
    Arabic,
    Han,
    Other,
    Mixed,
    #[default]
    Unknown,
}

impl ScriptClass {
    const ALL: [Self; 8] = [
        Self::Latin,
        Self::Cyrillic,
        Self::Greek,
        Self::Arabic,
        Self::Han,
        Self::Other,
        Self::Mixed,
        Self::Unknown,
    ];

    fn as_str(self) -> &'static str {
        match self {
            Self::Latin => "Latin",
            Self::Cyrillic => "Cyrillic",
            Self::Greek => "Greek",
            Self::Arabic => "Arabic",
            Self::Han => "Han",
            Self::Other => "other",
            Self::Mixed => "mixed",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Clone)]
struct LabeledExample {
    normalized: String,
    group: String,
    label: RoleLabel,
    split: MorphSplit,
    script: ScriptClass,
    given_count: u64,
    surname_count: u64,
    role_llr: f64,
    hashes: Vec<(u8, u64)>,
}

#[derive(Clone, Copy, Debug, Default)]
struct LabelStats {
    source_rows: usize,
    lexical_rows: usize,
    raw_given: usize,
    raw_surname: usize,
    conflicting_groups: usize,
    duplicate_group_rows: usize,
}

#[derive(Clone)]
struct TrainingCorpus {
    examples: Vec<LabeledExample>,
    stats: LabelStats,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NgramConfig {
    minimum: u8,
    maximum: u8,
    buckets: usize,
}

impl NgramConfig {
    fn description(self) -> String {
        format!(
            "{}..={}-grams;{}-buckets",
            self.minimum, self.maximum, self.buckets
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct OptimizerConfig {
    alpha: f64,
    beta: f64,
    l1: f64,
    l2: f64,
    epochs: usize,
}

impl OptimizerConfig {
    const DEFAULT: Self = Self {
        alpha: DEFAULT_ALPHA,
        beta: FTRL_BETA,
        l1: FTRL_L1,
        l2: DEFAULT_L2,
        epochs: TRAINING_EPOCHS,
    };
}

#[derive(Clone, Copy, Debug, Default)]
struct BinaryMetrics {
    rows: usize,
    given: usize,
    surname: usize,
    true_given: usize,
    true_surname: usize,
    false_given: usize,
    false_surname: usize,
    accuracy: f64,
    balanced_accuracy: f64,
    roc_auc: f64,
    given_pr_auc: f64,
    surname_pr_auc: f64,
}

#[derive(Clone)]
struct GridResult {
    ngrams: NgramConfig,
    optimizer: OptimizerConfig,
    validation: BinaryMetrics,
    macro_script_auc: f64,
    trained: MorphModel,
}

#[derive(Clone)]
struct MorphModel {
    ngrams: NgramConfig,
    optimizer: OptimizerConfig,
    intercept: f32,
    weights: Vec<f32>,
    occupied: Vec<bool>,
    script_counts: BTreeMap<ScriptClass, usize>,
    maximum_script_count: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct MorphEvidence {
    logit: f64,
    signal: f64,
    reliability: f64,
    bucket_support: f64,
    script_support: f64,
    script: ScriptClass,
}

#[derive(Clone, Copy, Debug, Default)]
struct CollisionStats {
    unique_hashes: usize,
    occupied_buckets: usize,
    bucket_collisions: usize,
    hash64_collisions: usize,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum QuantizationKind {
    F32,
    I16,
    I8,
}

impl QuantizationKind {
    const ALL: [Self; 3] = [Self::F32, Self::I16, Self::I8];

    fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::I16 => "int16",
            Self::I8 => "int8",
        }
    }
}

#[derive(Clone)]
enum QuantizedWeights {
    F32(Vec<f32>),
    I16 { values: Vec<i16>, scale: f32 },
    I8 { values: Vec<i8>, scale: f32 },
}

#[derive(Clone)]
struct QuantizedModel {
    base: MorphModel,
    weights: QuantizedWeights,
}

#[derive(Clone, Copy, Debug, Default)]
struct ErrorSummary {
    p50: f64,
    p95: f64,
    p99: f64,
    maximum: f64,
}

#[derive(Clone)]
struct QuantizationResult {
    kind: QuantizationKind,
    model: QuantizedModel,
    validation: BinaryMetrics,
    test: BinaryMetrics,
    logit_error: ErrorSummary,
    signal_error: ErrorSummary,
    bytes: usize,
}

pub(crate) fn run_morphology_diagnostic(
    output: &Path,
    corpus: &impl EvidenceSource,
    totals: &Path,
    holdouts: Vec<FrozenHoldout>,
    fixtures: &Path,
) -> Result<String> {
    let started = Instant::now();
    let holdouts = validate_and_order_holdouts(holdouts)?;
    let training = load_training_corpus(totals, corpus)?;
    let (grid, selected) = select_morphology_model(&training)?;
    let test_metrics = evaluate_binary(&training.examples, MorphSplit::Test, &selected);
    let collision = collision_stats(&training.examples, &selected);
    let quantization = quantization_study(&training, &selected);
    let proxy_rows = build_proxy_rows(corpus, &holdouts, &selected);
    let validation_rows = build_validation_rows(corpus, fixtures, &selected)?;
    validate_proxy_baseline(&proxy_rows)?;

    let ranking_configs = RANKING_WEIGHTS.map(|weight| RankingConfig { weight });
    let ranking_folds = ranking_logo(&proxy_rows, &ranking_configs);
    let ranking_useful = ranking_is_useful(&ranking_folds);
    let morphology_folds = morphology_logo(&proxy_rows, &ranking_configs, ranking_useful)?;
    let morphology_best = best_by_target(&morphology_folds, ranking_useful);
    let quantized_proxy = quantization
        .iter()
        .map(|quantized| {
            let rows = remap_quantized_rows(&proxy_rows, &quantized.model);
            let ranking_folds = ranking_logo(&rows, &ranking_configs);
            let ranking_useful = ranking_is_useful(&ranking_folds);
            let folds = morphology_logo(&rows, &ranking_configs, ranking_useful)?;
            Ok(QuantizedProxyResult {
                kind: quantized.kind,
                best: best_by_target(&folds, ranking_useful),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let base_rows = proxy_rows
        .iter()
        .map(|row| row.base.clone())
        .collect::<Vec<_>>();
    let baseline_folds = super::logo_frontier(&base_rows)?;
    let baseline_best = super::best_cross_validated_families(&baseline_folds);

    let outputs = build_outputs(
        &training,
        &grid,
        &selected,
        test_metrics,
        collision,
        &quantization,
        &proxy_rows,
        &validation_rows,
        &ranking_folds,
        ranking_useful,
        &morphology_folds,
        &morphology_best,
        &quantized_proxy,
        &baseline_best,
        started.elapsed(),
    )?;
    let repeated = build_outputs(
        &training,
        &grid,
        &selected,
        test_metrics,
        collision,
        &quantization,
        &proxy_rows,
        &validation_rows,
        &ranking_folds,
        ranking_useful,
        &morphology_folds,
        &morphology_best,
        &quantized_proxy,
        &baseline_best,
        started.elapsed(),
    )?;
    for (name, bytes) in &outputs {
        if name != "runtime_observation.txt" && repeated.get(name) != Some(bytes) {
            return Err(format!("morphology output is not deterministic: {name}").into());
        }
        fs::write(output.join(name), bytes)?;
    }
    let report = outputs
        .get("report.md")
        .ok_or("morphology report missing")?;
    Ok(String::from_utf8(report.clone())?)
}

fn load_training_corpus(path: &Path, corpus: &impl EvidenceSource) -> Result<TrainingCorpus> {
    let bytes = fs::read(path)?;
    let digest = format!("{:x}", Sha256::digest(&bytes));
    if digest != TOTALS_SHA256 {
        return Err(format!(
            "morphology totals checksum mismatch: expected {TOTALS_SHA256}, got {digest}"
        )
        .into());
    }
    let mut reader = csv::Reader::from_reader(bytes.as_slice());
    if reader
        .headers()?
        .iter()
        .ne(["name", "given_count", "as_surname_count"])
    {
        return Err("unexpected morphology totals header".into());
    }
    let mut stats = LabelStats::default();
    let mut previous = None::<Vec<u8>>;
    let mut given_sum = 0_u64;
    let mut surname_sum = 0_u64;
    let mut grouped = BTreeMap::<String, Vec<LabeledExample>>::new();
    for result in reader.records() {
        let record = result?;
        stats.source_rows += 1;
        let name = record.get(0).ok_or("missing morphology name")?;
        let name_bytes = name.as_bytes();
        if previous
            .as_ref()
            .is_some_and(|previous| previous.as_slice() >= name_bytes)
        {
            return Err("morphology totals names are not strictly bytewise ordered".into());
        }
        previous = Some(name_bytes.to_vec());
        let given_count = parse_u64(record.get(1), "given_count")?;
        let surname_count = parse_u64(record.get(2), "as_surname_count")?;
        given_sum = given_sum
            .checked_add(given_count)
            .ok_or("given sum overflow")?;
        surname_sum = surname_sum
            .checked_add(surname_count)
            .ok_or("surname overlap sum overflow")?;
        if corpus.lookup(name, None).is_none() {
            return Err(format!(
                "morphology totals key is absent from artifact at row {}",
                stats.source_rows + 1
            )
            .into());
        }
        let Some(normalized) = morphology_normalize(name) else {
            continue;
        };
        stats.lexical_rows += 1;
        let role_llr = role_llr(given_count, surname_count);
        let label = role_label(given_count, surname_count);
        match label {
            Some(RoleLabel::Given) => stats.raw_given += 1,
            Some(RoleLabel::Surname) => stats.raw_surname += 1,
            None => {}
        }
        let Some(label) = label else {
            continue;
        };
        let group = accent_group(&normalized);
        let script = dominant_script(&normalized);
        let hashes = ngram_hashes(&normalized);
        grouped
            .entry(group.clone())
            .or_default()
            .push(LabeledExample {
                normalized,
                group,
                label,
                split: MorphSplit::Train,
                script,
                given_count,
                surname_count,
                role_llr,
                hashes,
            });
    }
    if stats.source_rows != EXPECTED_KEYS
        || given_sum != GIVEN_TOTAL
        || surname_sum != SURNAME_OVERLAP_TOTAL
    {
        return Err(format!(
            "morphology totals invariants changed: rows={}, given={}, surname={}",
            stats.source_rows, given_sum, surname_sum
        )
        .into());
    }

    let mut examples = Vec::new();
    for (group, mut variants) in grouped {
        let labels = variants
            .iter()
            .map(|variant| variant.label)
            .collect::<BTreeSet<_>>();
        if labels.len() != 1 {
            stats.conflicting_groups += 1;
            continue;
        }
        variants.sort_by(|left, right| {
            left.normalized
                .as_bytes()
                .cmp(right.normalized.as_bytes())
                .then_with(|| right.given_count.cmp(&left.given_count))
                .then_with(|| right.surname_count.cmp(&left.surname_count))
        });
        stats.duplicate_group_rows += variants.len().saturating_sub(1);
        let mut selected = variants.remove(0);
        selected.group = group.clone();
        selected.split = MorphSplit::from_group(&group);
        examples.push(selected);
    }
    examples.sort_by(|left, right| {
        xxh3_64_with_seed(left.group.as_bytes(), ORDER_SEED)
            .cmp(&xxh3_64_with_seed(right.group.as_bytes(), ORDER_SEED))
            .then_with(|| left.group.as_bytes().cmp(right.group.as_bytes()))
    });
    for split in MorphSplit::ALL {
        for label in [RoleLabel::Given, RoleLabel::Surname] {
            if !examples
                .iter()
                .any(|example| example.split == split && example.label == label)
            {
                return Err(format!(
                    "{} has no {} morphology examples",
                    split.as_str(),
                    label.as_str()
                )
                .into());
            }
        }
    }
    Ok(TrainingCorpus { examples, stats })
}

fn parse_u64(value: Option<&str>, label: &str) -> Result<u64> {
    value
        .ok_or_else(|| format!("missing {label}").into())
        .and_then(|value| value.parse::<u64>().map_err(Into::into))
}

fn role_llr(given_count: u64, surname_count: u64) -> f64 {
    ((given_count as f64 + ROLE_SMOOTHING) / GIVEN_TOTAL as f64).ln()
        - ((surname_count as f64 + ROLE_SMOOTHING) / SURNAME_TOTAL as f64).ln()
}

fn role_label(given_count: u64, surname_count: u64) -> Option<RoleLabel> {
    let role_llr = role_llr(given_count, surname_count);
    if given_count >= MIN_ROLE_COUNT && role_llr >= MIN_ROLE_LLR {
        Some(RoleLabel::Given)
    } else if surname_count >= MIN_ROLE_COUNT && role_llr <= -MIN_ROLE_LLR {
        Some(RoleLabel::Surname)
    } else {
        None
    }
}

fn morphology_normalize(value: &str) -> Option<String> {
    if value.chars().any(char::is_whitespace) || !candidate_is_eligible(value) {
        return None;
    }
    let canonical = canonicalize(value);
    let folded = canonical
        .case_fold()
        .collect::<String>()
        .nfc()
        .collect::<String>();
    if folded
        .chars()
        .filter(|character| character.is_alphabetic())
        .count()
        < MIN_ALPHABETIC
        || !valid_internal_components(&folded)
    {
        return None;
    }
    Some(folded)
}

fn valid_internal_components(value: &str) -> bool {
    value.split(['-', '\'']).all(|component| {
        !component.is_empty()
            && component.chars().next().is_some_and(char::is_alphabetic)
            && component
                .chars()
                .last()
                .is_some_and(|character| character.is_alphabetic() || is_mark(character))
    })
}

fn accent_group(value: &str) -> String {
    value
        .nfd()
        .filter(|character| !is_mark(*character))
        .nfc()
        .collect()
}

fn is_mark(character: char) -> bool {
    matches!(
        get_general_category(character),
        GeneralCategory::NonspacingMark
            | GeneralCategory::SpacingMark
            | GeneralCategory::EnclosingMark
    )
}

fn dominant_script(value: &str) -> ScriptClass {
    let scripts = value
        .chars()
        .filter(|character| character.is_alphabetic())
        .map(|character| character.script())
        .filter(|script| !matches!(script, Script::Common | Script::Inherited | Script::Unknown))
        .map(script_class)
        .collect::<BTreeSet<_>>();
    match scripts.len() {
        0 => ScriptClass::Unknown,
        1 => *scripts.first().expect("single script"),
        _ => ScriptClass::Mixed,
    }
}

fn script_class(script: Script) -> ScriptClass {
    match script {
        Script::Latin => ScriptClass::Latin,
        Script::Cyrillic => ScriptClass::Cyrillic,
        Script::Greek => ScriptClass::Greek,
        Script::Arabic => ScriptClass::Arabic,
        Script::Han => ScriptClass::Han,
        Script::Common | Script::Inherited | Script::Unknown => ScriptClass::Unknown,
        _ => ScriptClass::Other,
    }
}

fn ngram_hashes(value: &str) -> Vec<(u8, u64)> {
    let mut hashes = ngram_signatures(value)
        .into_iter()
        .map(|(length, primary, _)| (length, primary))
        .collect::<Vec<_>>();
    hashes.sort_unstable();
    hashes.dedup();
    hashes
}

fn ngram_signatures(value: &str) -> Vec<(u8, u64, u64)> {
    let scalars = std::iter::once(START_SENTINEL)
        .chain(value.chars().map(u32::from))
        .chain(std::iter::once(END_SENTINEL))
        .collect::<Vec<_>>();
    let mut signatures = Vec::new();
    for length in 2_u8..=5 {
        if usize::from(length) > scalars.len() {
            continue;
        }
        for window in scalars.windows(usize::from(length)) {
            let mut encoded = [0_u8; 1 + 5 * 4];
            encoded[0] = length;
            for (index, scalar) in window.iter().enumerate() {
                encoded[1 + index * 4..1 + (index + 1) * 4].copy_from_slice(&scalar.to_le_bytes());
            }
            let used = 1 + usize::from(length) * 4;
            signatures.push((
                length,
                xxh3_64_with_seed(&encoded[..used], NGRAM_SEED),
                xxh3_64_with_seed(&encoded[..used], NGRAM_COLLISION_SEED),
            ));
        }
    }
    signatures.sort_unstable();
    signatures.dedup();
    signatures
}

fn sparse_features(example: &LabeledExample, config: NgramConfig) -> Vec<(usize, f64)> {
    sparse_features_from_hashes(&example.hashes, config)
}

fn sparse_features_from_hashes(hashes: &[(u8, u64)], config: NgramConfig) -> Vec<(usize, f64)> {
    let mut features = hashes
        .iter()
        .filter(|(length, _)| *length >= config.minimum && *length <= config.maximum)
        .map(|(_, hash)| {
            let bucket = (*hash as usize) & (config.buckets - 1);
            let sign = if hash >> 63 == 0 { 1_i32 } else { -1_i32 };
            (bucket, sign)
        })
        .collect::<Vec<_>>();
    features.sort_unstable_by_key(|(bucket, _)| *bucket);
    let mut combined = Vec::with_capacity(features.len());
    for (bucket, sign) in features {
        if let Some((last_bucket, total)) = combined.last_mut()
            && *last_bucket == bucket
        {
            *total += sign;
        } else {
            combined.push((bucket, sign));
        }
    }
    combined
        .into_iter()
        .filter_map(|(bucket, total)| (total != 0).then_some((bucket, f64::from(total.signum()))))
        .collect()
}

fn select_morphology_model(training: &TrainingCorpus) -> Result<(Vec<GridResult>, MorphModel)> {
    let mut grid = Vec::new();
    for maximum in NGRAM_MAXIMA {
        for buckets in CAPACITIES {
            let ngrams = NgramConfig {
                minimum: 2,
                maximum,
                buckets,
            };
            let trained =
                train_morphology_model(&training.examples, ngrams, OptimizerConfig::DEFAULT)?;
            let validation = evaluate_binary(&training.examples, MorphSplit::Validation, &trained);
            let macro_script_auc =
                macro_script_auc(&training.examples, MorphSplit::Validation, &trained);
            grid.push(GridResult {
                ngrams,
                optimizer: OptimizerConfig::DEFAULT,
                validation,
                macro_script_auc,
                trained,
            });
        }
    }
    let representation = grid
        .iter()
        .max_by(|left, right| compare_grid_results(left, right))
        .ok_or("empty morphology representation grid")?
        .ngrams;
    for alpha in ALPHAS {
        for l2 in L2_VALUES {
            let optimizer = OptimizerConfig {
                alpha,
                l2,
                ..OptimizerConfig::DEFAULT
            };
            if optimizer == OptimizerConfig::DEFAULT {
                continue;
            }
            let trained = train_morphology_model(&training.examples, representation, optimizer)?;
            let validation = evaluate_binary(&training.examples, MorphSplit::Validation, &trained);
            let macro_script_auc =
                macro_script_auc(&training.examples, MorphSplit::Validation, &trained);
            grid.push(GridResult {
                ngrams: representation,
                optimizer,
                validation,
                macro_script_auc,
                trained,
            });
        }
    }
    let selected = grid
        .iter()
        .max_by(|left, right| compare_grid_results(left, right))
        .ok_or("empty morphology optimizer grid")?
        .trained
        .clone();
    Ok((grid, selected))
}

fn compare_grid_results(left: &GridResult, right: &GridResult) -> Ordering {
    left.macro_script_auc
        .total_cmp(&right.macro_script_auc)
        .then_with(|| left.validation.roc_auc.total_cmp(&right.validation.roc_auc))
        .then_with(|| {
            left.validation
                .balanced_accuracy
                .total_cmp(&right.validation.balanced_accuracy)
        })
        .then_with(|| right.ngrams.buckets.cmp(&left.ngrams.buckets))
        .then_with(|| right.ngrams.maximum.cmp(&left.ngrams.maximum))
        .then_with(|| left.optimizer.l2.total_cmp(&right.optimizer.l2))
        .then_with(|| right.optimizer.alpha.total_cmp(&left.optimizer.alpha))
}

fn train_morphology_model(
    examples: &[LabeledExample],
    ngrams: NgramConfig,
    optimizer: OptimizerConfig,
) -> Result<MorphModel> {
    if !ngrams.buckets.is_power_of_two() {
        return Err("morphology hash buckets must be a power of two".into());
    }
    let training = examples
        .iter()
        .filter(|example| example.split == MorphSplit::Train)
        .collect::<Vec<_>>();
    let given = training
        .iter()
        .filter(|example| example.label == RoleLabel::Given)
        .count();
    let surname = training.len() - given;
    if given == 0 || surname == 0 {
        return Err("morphology training requires both role labels".into());
    }
    let given_weight = training.len() as f64 / (2.0 * given as f64);
    let surname_weight = training.len() as f64 / (2.0 * surname as f64);
    let mut z = vec![0.0_f64; ngrams.buckets];
    let mut n = vec![0.0_f64; ngrams.buckets];
    let mut bias_z = 0.0;
    let mut bias_n = 0.0;
    for _ in 0..optimizer.epochs {
        for example in &training {
            let features = sparse_features(example, ngrams);
            let intercept = ftrl_weight(bias_z, bias_n, optimizer, false);
            let score = features.iter().fold(intercept, |score, (bucket, value)| {
                score + ftrl_weight(z[*bucket], n[*bucket], optimizer, true) * value
            });
            let class_weight = match example.label {
                RoleLabel::Given => given_weight,
                RoleLabel::Surname => surname_weight,
            };
            let residual = (sigmoid(score) - example.label.value()) * class_weight;
            let bias_weight = ftrl_weight(bias_z, bias_n, optimizer, false);
            ftrl_update(&mut bias_z, &mut bias_n, residual, bias_weight, optimizer);
            for (bucket, value) in features {
                let weight = ftrl_weight(z[bucket], n[bucket], optimizer, true);
                ftrl_update(
                    &mut z[bucket],
                    &mut n[bucket],
                    residual * value,
                    weight,
                    optimizer,
                );
            }
        }
    }
    let intercept = ftrl_weight(bias_z, bias_n, optimizer, false) as f32;
    let weights = z
        .iter()
        .zip(&n)
        .map(|(&z, &n)| ftrl_weight(z, n, optimizer, true) as f32)
        .collect::<Vec<_>>();
    let mut occupied = vec![false; ngrams.buckets];
    let mut script_counts = BTreeMap::new();
    for example in training {
        *script_counts.entry(example.script).or_insert(0) += 1;
        for (bucket, _) in sparse_features(example, ngrams) {
            occupied[bucket] = true;
        }
    }
    let maximum_script_count = script_counts.values().copied().max().unwrap_or(0);
    Ok(MorphModel {
        ngrams,
        optimizer,
        intercept,
        weights,
        occupied,
        script_counts,
        maximum_script_count,
    })
}

fn ftrl_weight(z: f64, n: f64, config: OptimizerConfig, regularized: bool) -> f64 {
    let l1 = if regularized { config.l1 } else { 0.0 };
    let l2 = if regularized { config.l2 } else { 0.0 };
    if z.abs() <= l1 {
        0.0
    } else {
        -(z - z.signum() * l1) / ((config.beta + n.sqrt()) / config.alpha + l2)
    }
}

fn ftrl_update(z: &mut f64, n: &mut f64, gradient: f64, weight: f64, config: OptimizerConfig) {
    let next_n = *n + gradient * gradient;
    let sigma = (next_n.sqrt() - n.sqrt()) / config.alpha;
    *z += gradient - sigma * weight;
    *n = next_n;
}

impl MorphModel {
    fn score_token(&self, value: &str) -> MorphEvidence {
        let Some(normalized) = morphology_normalize(value) else {
            return MorphEvidence::default();
        };
        self.score_normalized(&normalized)
    }

    fn score_normalized(&self, normalized: &str) -> MorphEvidence {
        let hashes = ngram_hashes(normalized);
        let features = sparse_features_from_hashes(&hashes, self.ngrams);
        let logit = features
            .iter()
            .fold(f64::from(self.intercept), |score, (bucket, value)| {
                score + f64::from(self.weights[*bucket]) * value
            });
        let occupied = features
            .iter()
            .filter(|(bucket, _)| self.occupied[*bucket])
            .count();
        let bucket_support = ratio_value(occupied, features.len());
        let script = dominant_script(normalized);
        let script_count = self.script_counts.get(&script).copied().unwrap_or(0);
        let script_support = if self.maximum_script_count == 0 || script_count == 0 {
            0.0
        } else {
            (1.0 + script_count as f64).ln() / (1.0 + self.maximum_script_count as f64).ln()
        };
        MorphEvidence {
            logit,
            signal: 2.0 * sigmoid(logit) - 1.0,
            reliability: bucket_support * script_support,
            bucket_support,
            script_support,
            script,
        }
    }

    fn score_candidate(&self, value: &str) -> MorphEvidence {
        let parts = canonicalize(value)
            .split_whitespace()
            .map(|part| self.score_token(part))
            .collect::<Vec<_>>();
        aggregate_morphology(&parts)
    }
}

fn aggregate_morphology(parts: &[MorphEvidence]) -> MorphEvidence {
    if parts.is_empty() {
        return MorphEvidence::default();
    }
    let total_reliability = parts.iter().map(|part| part.reliability).sum::<f64>();
    let (logit, signal) = if total_reliability == 0.0 {
        (0.0, 0.0)
    } else {
        (
            parts
                .iter()
                .map(|part| part.logit * part.reliability)
                .sum::<f64>()
                / total_reliability,
            parts
                .iter()
                .map(|part| part.signal * part.reliability)
                .sum::<f64>()
                / total_reliability,
        )
    };
    let scripts = parts
        .iter()
        .filter(|part| part.reliability > 0.0)
        .map(|part| part.script)
        .collect::<BTreeSet<_>>();
    let script = match scripts.len() {
        0 => ScriptClass::Unknown,
        1 => *scripts.first().expect("single candidate script"),
        _ => ScriptClass::Mixed,
    };
    MorphEvidence {
        logit,
        signal,
        reliability: total_reliability / parts.len() as f64,
        bucket_support: parts.iter().map(|part| part.bucket_support).sum::<f64>()
            / parts.len() as f64,
        script_support: parts.iter().map(|part| part.script_support).sum::<f64>()
            / parts.len() as f64,
        script,
    }
}

fn evaluate_binary(
    examples: &[LabeledExample],
    split: MorphSplit,
    model: &MorphModel,
) -> BinaryMetrics {
    let scored = examples
        .iter()
        .filter(|example| example.split == split)
        .map(|example| {
            (
                example.label,
                model.score_normalized(&example.normalized).logit,
            )
        })
        .collect::<Vec<_>>();
    binary_metrics(&scored)
}

fn binary_metrics(scored: &[(RoleLabel, f64)]) -> BinaryMetrics {
    let mut metrics = BinaryMetrics {
        rows: scored.len(),
        ..BinaryMetrics::default()
    };
    for &(label, score) in scored {
        match (label, score >= 0.0) {
            (RoleLabel::Given, true) => metrics.true_given += 1,
            (RoleLabel::Given, false) => metrics.false_surname += 1,
            (RoleLabel::Surname, false) => metrics.true_surname += 1,
            (RoleLabel::Surname, true) => metrics.false_given += 1,
        }
    }
    metrics.given = metrics.true_given + metrics.false_surname;
    metrics.surname = metrics.true_surname + metrics.false_given;
    metrics.accuracy = ratio_value(metrics.true_given + metrics.true_surname, metrics.rows);
    metrics.balanced_accuracy = 0.5
        * (ratio_value(metrics.true_given, metrics.given)
            + ratio_value(metrics.true_surname, metrics.surname));
    metrics.roc_auc = roc_auc(scored, RoleLabel::Given);
    metrics.given_pr_auc = average_precision(scored, RoleLabel::Given);
    metrics.surname_pr_auc = average_precision(scored, RoleLabel::Surname);
    metrics
}

fn roc_auc(scored: &[(RoleLabel, f64)], positive: RoleLabel) -> f64 {
    let mut ordered = scored.to_vec();
    ordered.sort_by(|left, right| left.1.total_cmp(&right.1));
    if positive == RoleLabel::Surname {
        ordered.reverse();
    }
    let positives = ordered
        .iter()
        .filter(|(label, _)| *label == positive)
        .count();
    let negatives = ordered.len() - positives;
    if positives == 0 || negatives == 0 {
        return 0.0;
    }
    let mut rank_sum = 0.0;
    let mut start = 0;
    while start < ordered.len() {
        let mut end = start + 1;
        while end < ordered.len() && ordered[end].1.to_bits() == ordered[start].1.to_bits() {
            end += 1;
        }
        let average_rank = (start + 1 + end) as f64 / 2.0;
        rank_sum += average_rank
            * ordered[start..end]
                .iter()
                .filter(|(label, _)| *label == positive)
                .count() as f64;
        start = end;
    }
    (rank_sum - positives as f64 * (positives as f64 + 1.0) / 2.0) / (positives * negatives) as f64
}

fn average_precision(scored: &[(RoleLabel, f64)], positive: RoleLabel) -> f64 {
    let mut ordered = scored.to_vec();
    ordered.sort_by(|left, right| {
        let ordering = right.1.total_cmp(&left.1);
        if positive == RoleLabel::Given {
            ordering
        } else {
            ordering.reverse()
        }
    });
    let positives = ordered
        .iter()
        .filter(|(label, _)| *label == positive)
        .count();
    if positives == 0 {
        return 0.0;
    }
    let mut seen_positive = 0;
    let mut total = 0.0;
    for (index, (label, _)) in ordered.iter().enumerate() {
        if *label == positive {
            seen_positive += 1;
            total += seen_positive as f64 / (index + 1) as f64;
        }
    }
    total / positives as f64
}

fn macro_script_auc(examples: &[LabeledExample], split: MorphSplit, model: &MorphModel) -> f64 {
    let aucs = ScriptClass::ALL
        .into_iter()
        .filter_map(|script| {
            let scored = examples
                .iter()
                .filter(|example| example.split == split && example.script == script)
                .map(|example| {
                    (
                        example.label,
                        model.score_normalized(&example.normalized).logit,
                    )
                })
                .collect::<Vec<_>>();
            let given = scored
                .iter()
                .filter(|(label, _)| *label == RoleLabel::Given)
                .count();
            let surname = scored.len() - given;
            (given >= 100 && surname >= 100).then(|| roc_auc(&scored, RoleLabel::Given))
        })
        .collect::<Vec<_>>();
    if aucs.is_empty() {
        0.0
    } else {
        aucs.iter().sum::<f64>() / aucs.len() as f64
    }
}

fn collision_stats(examples: &[LabeledExample], model: &MorphModel) -> CollisionStats {
    let mut hashes = HashSet::new();
    let mut hash_payloads = HashMap::<u64, (u8, u64)>::new();
    let mut hash64_collisions = 0;
    let mut occupied = vec![false; model.ngrams.buckets];
    for example in examples
        .iter()
        .filter(|example| example.split == MorphSplit::Train)
    {
        for (length, hash, secondary) in ngram_signatures(&example.normalized) {
            if length < model.ngrams.minimum || length > model.ngrams.maximum {
                continue;
            }
            hashes.insert(hash);
            let signature = (length, secondary);
            if let Some(previous) = hash_payloads.get(&hash) {
                if previous != &signature {
                    hash64_collisions += 1;
                }
            } else {
                hash_payloads.insert(hash, signature);
            }
            occupied[(hash as usize) & (model.ngrams.buckets - 1)] = true;
        }
    }
    let occupied_buckets = occupied.iter().filter(|occupied| **occupied).count();
    CollisionStats {
        unique_hashes: hashes.len(),
        occupied_buckets,
        bucket_collisions: hashes.len().saturating_sub(occupied_buckets),
        hash64_collisions,
    }
}

fn ratio_value(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
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

fn quantization_study(training: &TrainingCorpus, selected: &MorphModel) -> Vec<QuantizationResult> {
    QuantizationKind::ALL
        .into_iter()
        .map(|kind| {
            let model = quantize_model(selected, kind);
            let validation =
                evaluate_quantized_binary(&training.examples, MorphSplit::Validation, &model);
            let test = evaluate_quantized_binary(&training.examples, MorphSplit::Test, &model);
            let errors = training
                .examples
                .iter()
                .filter(|example| example.split == MorphSplit::Test)
                .map(|example| {
                    let expected = selected.score_normalized(&example.normalized);
                    let actual = model.score_normalized(&example.normalized);
                    (
                        (actual.logit - expected.logit).abs(),
                        (actual.signal - expected.signal).abs(),
                    )
                })
                .collect::<Vec<_>>();
            let bytes = quantized_model_bytes(&model);
            QuantizationResult {
                kind,
                model,
                validation,
                test,
                logit_error: summarize_errors(errors.iter().map(|error| error.0).collect()),
                signal_error: summarize_errors(errors.iter().map(|error| error.1).collect()),
                bytes,
            }
        })
        .collect()
}

fn quantize_model(model: &MorphModel, kind: QuantizationKind) -> QuantizedModel {
    let maximum = model
        .weights
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0_f32, f32::max);
    let weights = match kind {
        QuantizationKind::F32 => QuantizedWeights::F32(model.weights.clone()),
        QuantizationKind::I16 => {
            let scale = if maximum == 0.0 {
                1.0
            } else {
                maximum / f32::from(i16::MAX)
            };
            let values = model
                .weights
                .iter()
                .map(|weight| {
                    (weight / scale)
                        .round()
                        .clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16
                })
                .collect();
            QuantizedWeights::I16 { values, scale }
        }
        QuantizationKind::I8 => {
            let scale = if maximum == 0.0 {
                1.0
            } else {
                maximum / f32::from(i8::MAX)
            };
            let values = model
                .weights
                .iter()
                .map(|weight| {
                    (weight / scale)
                        .round()
                        .clamp(f32::from(i8::MIN), f32::from(i8::MAX)) as i8
                })
                .collect();
            QuantizedWeights::I8 { values, scale }
        }
    };
    QuantizedModel {
        base: model.clone(),
        weights,
    }
}

impl QuantizedModel {
    fn weight(&self, bucket: usize) -> f64 {
        match &self.weights {
            QuantizedWeights::F32(values) => f64::from(values[bucket]),
            QuantizedWeights::I16 { values, scale } => {
                f64::from(values[bucket]) * f64::from(*scale)
            }
            QuantizedWeights::I8 { values, scale } => f64::from(values[bucket]) * f64::from(*scale),
        }
    }

    fn score_normalized(&self, normalized: &str) -> MorphEvidence {
        let hashes = ngram_hashes(normalized);
        let features = sparse_features_from_hashes(&hashes, self.base.ngrams);
        let logit = features
            .iter()
            .fold(f64::from(self.base.intercept), |score, (bucket, value)| {
                score + self.weight(*bucket) * value
            });
        let base = self.base.score_normalized(normalized);
        MorphEvidence {
            logit,
            signal: 2.0 * sigmoid(logit) - 1.0,
            ..base
        }
    }

    fn score_token(&self, value: &str) -> MorphEvidence {
        morphology_normalize(value).map_or_else(MorphEvidence::default, |normalized| {
            self.score_normalized(&normalized)
        })
    }

    fn score_candidate(&self, value: &str) -> MorphEvidence {
        let parts = canonicalize(value)
            .split_whitespace()
            .map(|part| self.score_token(part))
            .collect::<Vec<_>>();
        aggregate_morphology(&parts)
    }
}

fn evaluate_quantized_binary(
    examples: &[LabeledExample],
    split: MorphSplit,
    model: &QuantizedModel,
) -> BinaryMetrics {
    let scored = examples
        .iter()
        .filter(|example| example.split == split)
        .map(|example| {
            (
                example.label,
                model.score_normalized(&example.normalized).logit,
            )
        })
        .collect::<Vec<_>>();
    binary_metrics(&scored)
}

fn quantized_model_bytes(model: &QuantizedModel) -> usize {
    let weight_bytes = match &model.weights {
        QuantizedWeights::F32(values) => values.len() * size_of::<f32>(),
        QuantizedWeights::I16 { values, .. } => values.len() * size_of::<i16>() + size_of::<f32>(),
        QuantizedWeights::I8 { values, .. } => values.len() * size_of::<i8>() + size_of::<f32>(),
    };
    weight_bytes + size_of::<f32>() + model.base.occupied.len().div_ceil(8)
}

fn summarize_errors(mut values: Vec<f64>) -> ErrorSummary {
    values.sort_by(f64::total_cmp);
    ErrorSummary {
        p50: percentile(&values, 50),
        p95: percentile(&values, 95),
        p99: percentile(&values, 99),
        maximum: values.last().copied().unwrap_or(0.0),
    }
}

fn percentile(values: &[f64], percentile: usize) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values[((values.len() - 1) * percentile) / 100]
}

#[derive(Clone)]
struct MorphologyRow {
    base: FeatureRow,
    diagnostic: RoleInferenceDiagnostic,
    candidate_matches: Vec<bool>,
    candidate_morphology: Vec<MorphEvidence>,
    category: String,
}

#[derive(Clone)]
struct RankedRow {
    features: FeatureRow,
    morphology: MorphEvidence,
    selected_index: Option<usize>,
    category: String,
}

impl RankedRow {
    fn base_features(&self) -> [f64; BASE_FEATURES] {
        self.features.logistic_features()
    }

    fn main_features(&self) -> [f64; MAIN_FEATURES] {
        let mut features = [0.0; MAIN_FEATURES];
        features[..BASE_FEATURES].copy_from_slice(&self.base_features());
        features[7] = self.effective_morphology();
        features
    }

    fn interaction_features(&self) -> [f64; INTERACTION_FEATURES] {
        let main = self.main_features();
        let mut features = [0.0; INTERACTION_FEATURES];
        features[..MAIN_FEATURES].copy_from_slice(&main);
        let morphology = main[7];
        features[8] = morphology * main[1];
        features[9] = morphology * main[3];
        features[10] = morphology * main[4];
        features[11] = morphology * main[2];
        features
    }

    fn effective_morphology(&self) -> f64 {
        self.morphology.signal * self.morphology.reliability
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct RankingConfig {
    weight: f64,
}

impl RankingConfig {
    const FROZEN: Self = Self { weight: 0.0 };

    fn adjustment(self, candidate: &CandidateDiagnostic, morphology: MorphEvidence) -> f64 {
        self.weight * morphology.signal * morphology.reliability * candidate.score.clamp(0.0, 1.0)
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

fn build_proxy_rows(
    corpus: &impl EvidenceSource,
    holdouts: &[FrozenHoldout],
    model: &MorphModel,
) -> Vec<MorphologyRow> {
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
            rows.push(morphology_row(
                corpus,
                model,
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
    model: &MorphModel,
) -> Result<Vec<MorphologyRow>> {
    Ok(generate_cases(fixtures, false)?
        .into_iter()
        .filter(|case| case.split == Split::Validation)
        .enumerate()
        .map(|(ordinal, case)| morphology_row_from_case(corpus, model, ordinal, &case))
        .collect())
}

fn morphology_row_from_case(
    corpus: &impl EvidenceSource,
    model: &MorphModel,
    ordinal: usize,
    case: &Case,
) -> MorphologyRow {
    morphology_row(
        corpus,
        model,
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
fn morphology_row(
    corpus: &impl EvidenceSource,
    model: &MorphModel,
    population: Population,
    ordinal: usize,
    display_name: &str,
    expected_greeting: Option<&str>,
    country_hint: Option<&str>,
    locale_hint: Option<&str>,
    category: &str,
) -> MorphologyRow {
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
    let candidate_matches = diagnostic
        .candidates
        .iter()
        .map(|candidate| greeting_matches(expected_greeting, Some(&candidate.display)))
        .collect();
    let candidate_morphology = diagnostic
        .candidates
        .iter()
        .map(|candidate| model.score_candidate(&candidate.display))
        .collect();
    MorphologyRow {
        base,
        diagnostic,
        candidate_matches,
        candidate_morphology,
        category: category.to_string(),
    }
}

fn nonempty(value: &str) -> Option<&str> {
    (!value.is_empty()).then_some(value)
}

fn frozen_ranked_row(row: &MorphologyRow) -> RankedRow {
    RankedRow {
        features: row.base.clone(),
        morphology: row
            .candidate_morphology
            .first()
            .copied()
            .unwrap_or_default(),
        selected_index: (!row.diagnostic.candidates.is_empty()).then_some(0),
        category: row.category.clone(),
    }
}

fn rank_row(row: &MorphologyRow, config: RankingConfig) -> RankedRow {
    if config == RankingConfig::FROZEN {
        return frozen_ranked_row(row);
    }
    let mut ranked = row
        .diagnostic
        .candidates
        .iter()
        .zip(&row.candidate_morphology)
        .enumerate()
        .map(|(index, (candidate, morphology))| {
            (
                index,
                (candidate.score + config.adjustment(candidate, *morphology)).clamp(0.0, 1.0),
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
            morphology: MorphEvidence::default(),
            selected_index: None,
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
    let pre_veto_score = (ALGORITHM_C2.quality_weight * adjusted_score
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
    features.candidate_quality = adjusted_score;
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
        morphology: row.candidate_morphology[selected_index],
        selected_index: Some(selected_index),
        category: row.category.clone(),
    }
}

fn ranking_metrics(rows: &[MorphologyRow], config: RankingConfig) -> RankingMetrics {
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

fn select_ranking_config(rows: &[MorphologyRow], configs: &[RankingConfig]) -> RankingConfig {
    configs
        .iter()
        .copied()
        .max_by(|left, right| {
            let left_metrics = ranking_metrics(rows, *left);
            let right_metrics = ranking_metrics(rows, *right);
            left_metrics
                .correct_winners
                .cmp(&right_metrics.correct_winners)
                .then_with(|| right_metrics.wrong_winners.cmp(&left_metrics.wrong_winners))
                .then_with(|| right_metrics.null_winners.cmp(&left_metrics.null_winners))
                .then_with(|| right.weight.total_cmp(&left.weight))
        })
        .expect("nonempty morphology ranking grid")
}

#[derive(Clone, Copy)]
struct RankingFold {
    held_out: Population,
    config: RankingConfig,
    frozen: RankingMetrics,
    adjusted: RankingMetrics,
}

fn ranking_logo(rows: &[MorphologyRow], configs: &[RankingConfig]) -> Vec<RankingFold> {
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
    let frozen_errors = folds
        .iter()
        .map(|fold| fold.frozen.wrong_winners + fold.frozen.null_winners)
        .sum::<usize>();
    let adjusted_errors = folds
        .iter()
        .map(|fold| fold.adjusted.wrong_winners + fold.adjusted.null_winners)
        .sum::<usize>();
    adjusted_correct > frozen_correct && adjusted_errors <= frozen_errors
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum CalibrationVariant {
    Baseline,
    MorphologyMain,
    MorphologyInteraction,
    RerankedInteraction,
}

impl CalibrationVariant {
    const BASE: [Self; 3] = [
        Self::Baseline,
        Self::MorphologyMain,
        Self::MorphologyInteraction,
    ];

    fn as_str(self) -> &'static str {
        match self {
            Self::Baseline => "baseline_logistic",
            Self::MorphologyMain => "morphology_main",
            Self::MorphologyInteraction => "morphology_interactions",
            Self::RerankedInteraction => "reranked_morphology_interactions",
        }
    }
}

#[derive(Clone, Debug)]
struct CalibrationModel {
    intercept: f64,
    coefficients: Vec<f64>,
    iterations: usize,
    variant: CalibrationVariant,
}

impl CalibrationModel {
    fn features(&self, row: &RankedRow) -> Vec<f64> {
        match self.variant {
            CalibrationVariant::Baseline => row.base_features().to_vec(),
            CalibrationVariant::MorphologyMain => row.main_features().to_vec(),
            CalibrationVariant::MorphologyInteraction | CalibrationVariant::RerankedInteraction => {
                row.interaction_features().to_vec()
            }
        }
    }

    fn score(&self, row: &RankedRow) -> f64 {
        let features = self.features(row);
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
struct CalibrationPolicy {
    model: CalibrationModel,
    threshold: f64,
}

impl CalibrationPolicy {
    fn emits(&self, row: &RankedRow) -> bool {
        row.features.eligible() && self.model.score(row) >= self.threshold
    }
}

#[derive(Clone)]
struct OperatingPoint {
    policy: CalibrationPolicy,
    metrics: EmissionMetrics,
}

#[derive(Clone)]
struct FoldResult {
    held_out: Population,
    variant: CalibrationVariant,
    target: f64,
    ranking: RankingConfig,
    policy: CalibrationPolicy,
    training_metrics: EmissionMetrics,
    held_out_metrics: EmissionMetrics,
}

#[derive(Clone)]
struct CrossValidatedPoint {
    variant: CalibrationVariant,
    target: f64,
    metrics: EmissionMetrics,
}

struct QuantizedProxyResult {
    kind: QuantizationKind,
    best: Vec<CrossValidatedPoint>,
}

fn remap_quantized_rows(rows: &[MorphologyRow], model: &QuantizedModel) -> Vec<MorphologyRow> {
    rows.iter()
        .cloned()
        .map(|mut row| {
            row.candidate_morphology = row
                .diagnostic
                .candidates
                .iter()
                .map(|candidate| model.score_candidate(&candidate.display))
                .collect();
            row
        })
        .collect()
}

fn morphology_logo(
    rows: &[MorphologyRow],
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
            let effective_ranking = if variant == CalibrationVariant::RerankedInteraction {
                ranking
            } else {
                RankingConfig::FROZEN
            };
            let training_ranked = training
                .iter()
                .map(|row| rank_row(row, effective_ranking))
                .collect::<Vec<_>>();
            let held_out_ranked = held_out_rows
                .iter()
                .map(|row| rank_row(row, effective_ranking))
                .collect::<Vec<_>>();
            let model = fit_calibration_model(&training_ranked, variant)?;
            let frontier = calibration_frontier(&training_ranked, &model);
            for target in TARGETS {
                let Some(selected) = select_operating_point(&frontier, target) else {
                    continue;
                };
                results.push(FoldResult {
                    held_out,
                    variant,
                    target,
                    ranking: effective_ranking,
                    policy: selected.policy.clone(),
                    training_metrics: selected.metrics,
                    held_out_metrics: evaluate_calibration(
                        held_out_ranked.iter(),
                        &selected.policy,
                    ),
                });
            }
        }
    }
    Ok(results)
}

fn best_by_target(folds: &[FoldResult], ranking_useful: bool) -> Vec<CrossValidatedPoint> {
    let mut variants = CalibrationVariant::BASE.to_vec();
    if ranking_useful {
        variants.push(CalibrationVariant::RerankedInteraction);
    }
    TARGETS
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
        .then_with(|| right.variant.cmp(&left.variant))
}

fn fit_calibration_model(
    rows: &[RankedRow],
    variant: CalibrationVariant,
) -> Result<CalibrationModel> {
    let training = calibration_training_rows(rows, variant)?;
    let feature_count = training.first().map_or(0, |row| row.features.len());
    let mut intercept = 0.0;
    let mut coefficients = vec![0.0; feature_count];
    let mut objective = calibration_objective(&training, intercept, &coefficients);
    for iteration in 1..=MAX_CALIBRATION_ITERATIONS {
        let (intercept_gradient, gradients) =
            calibration_gradient(&training, intercept, &coefficients);
        let mut step = 1.0;
        let mut accepted = None;
        while step >= f64::EPSILON {
            let next_intercept = intercept - step * intercept_gradient;
            let next_coefficients = coefficients
                .iter()
                .zip(&gradients)
                .map(|(coefficient, gradient)| (coefficient - step * gradient).max(0.0))
                .collect::<Vec<_>>();
            let next_objective =
                calibration_objective(&training, next_intercept, &next_coefficients);
            let directional = intercept_gradient * (next_intercept - intercept)
                + gradients
                    .iter()
                    .zip(next_coefficients.iter().zip(&coefficients))
                    .map(|(gradient, (next, current))| gradient * (next - current))
                    .sum::<f64>();
            if next_objective <= objective + ARMIJO * directional {
                accepted = Some((next_intercept, next_coefficients, next_objective));
                break;
            }
            step *= 0.5;
        }
        let Some((next_intercept, next_coefficients, next_objective)) = accepted else {
            return Err("morphology calibration line search failed".into());
        };
        let change = (next_intercept - intercept).abs().max(
            next_coefficients
                .iter()
                .zip(&coefficients)
                .map(|(next, current)| (next - current).abs())
                .fold(0.0, f64::max),
        );
        intercept = next_intercept;
        coefficients = next_coefficients;
        objective = next_objective;
        if change < PARAMETER_TOLERANCE {
            return Ok(CalibrationModel {
                intercept,
                coefficients,
                iterations: iteration,
                variant,
            });
        }
    }
    Err("morphology calibration optimizer did not converge".into())
}

struct WeightedCalibrationRow {
    features: Vec<f64>,
    label: f64,
    weight: f64,
}

fn calibration_training_rows(
    rows: &[RankedRow],
    variant: CalibrationVariant,
) -> Result<Vec<WeightedCalibrationRow>> {
    let populations = rows
        .iter()
        .map(|row| row.features.population)
        .collect::<BTreeSet<_>>();
    let counts = populations
        .iter()
        .map(|population| {
            (
                *population,
                rows.iter()
                    .filter(|row| row.features.population == *population && row.features.eligible())
                    .count(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    if populations.is_empty() || counts.values().any(|count| *count == 0) {
        return Err("morphology calibration requires eligible rows in every generation".into());
    }
    let generation_weight = 1.0 / populations.len() as f64;
    Ok(rows
        .iter()
        .filter(|row| row.features.eligible())
        .map(|row| {
            let features = match variant {
                CalibrationVariant::Baseline => row.base_features().to_vec(),
                CalibrationVariant::MorphologyMain => row.main_features().to_vec(),
                CalibrationVariant::MorphologyInteraction
                | CalibrationVariant::RerankedInteraction => row.interaction_features().to_vec(),
            };
            WeightedCalibrationRow {
                features,
                label: f64::from(row.features.selected_matches),
                weight: generation_weight / counts[&row.features.population] as f64,
            }
        })
        .collect())
}

fn calibration_objective(
    rows: &[WeightedCalibrationRow],
    intercept: f64,
    coefficients: &[f64],
) -> f64 {
    let loss = rows
        .iter()
        .map(|row| {
            let score = coefficients
                .iter()
                .zip(&row.features)
                .fold(intercept, |score, (coefficient, feature)| {
                    score + coefficient * feature
                });
            row.weight * logistic_loss(score, row.label)
        })
        .sum::<f64>();
    loss + 0.5 * CALIBRATION_L2 * coefficients.iter().map(|value| value * value).sum::<f64>()
}

fn calibration_gradient(
    rows: &[WeightedCalibrationRow],
    intercept: f64,
    coefficients: &[f64],
) -> (f64, Vec<f64>) {
    let mut intercept_gradient = 0.0;
    let mut gradients = vec![0.0; coefficients.len()];
    for row in rows {
        let score = coefficients
            .iter()
            .zip(&row.features)
            .fold(intercept, |score, (coefficient, feature)| {
                score + coefficient * feature
            });
        let residual = row.weight * (sigmoid(score) - row.label);
        intercept_gradient += residual;
        for (gradient, feature) in gradients.iter_mut().zip(&row.features) {
            *gradient += residual * feature;
        }
    }
    for (gradient, coefficient) in gradients.iter_mut().zip(coefficients) {
        *gradient += CALIBRATION_L2 * coefficient;
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

fn calibration_frontier(rows: &[RankedRow], model: &CalibrationModel) -> Vec<OperatingPoint> {
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
            let policy = CalibrationPolicy {
                model: model.clone(),
                threshold,
            };
            OperatingPoint {
                metrics: evaluate_calibration(rows.iter(), &policy),
                policy,
            }
        })
        .collect::<Vec<_>>();
    points.sort_by(|left, right| {
        left.metrics
            .emitted
            .cmp(&right.metrics.emitted)
            .then(left.metrics.correct.cmp(&right.metrics.correct))
            .then(left.policy.threshold.total_cmp(&right.policy.threshold))
    });
    points
}

fn evaluate_calibration<'a>(
    rows: impl Iterator<Item = &'a RankedRow>,
    policy: &CalibrationPolicy,
) -> EmissionMetrics {
    let mut metrics = EmissionMetrics::default();
    for row in rows {
        metrics.observe(&row.features, policy.emits(row));
    }
    metrics
}

fn select_operating_point(points: &[OperatingPoint], target: f64) -> Option<&OperatingPoint> {
    points
        .iter()
        .filter(|point| {
            point
                .metrics
                .precision()
                .is_some_and(|precision| precision >= target)
        })
        .max_by(|left, right| {
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
                .then_with(|| left.policy.threshold.total_cmp(&right.policy.threshold))
        })
}

fn validate_proxy_baseline(rows: &[MorphologyRow]) -> Result<()> {
    let baseline = rows.iter().map(|row| row.base.clone()).collect::<Vec<_>>();
    super::assert_dataset_counts(&baseline)?;
    super::assert_historical_checkpoints(&baseline)?;
    let correct_rejected = baseline
        .iter()
        .filter(|row| {
            row.expected_greeting && row.selected_matches && row.vetoes_pass && !row.c4_emits
        })
        .count();
    if correct_rejected != 3_993 {
        return Err(format!(
            "frozen C4 correct-winner-rejected count changed: expected 3993, got {correct_rejected}"
        )
        .into());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_outputs(
    training: &TrainingCorpus,
    grid: &[GridResult],
    selected: &MorphModel,
    test_metrics: BinaryMetrics,
    collision: CollisionStats,
    quantization: &[QuantizationResult],
    proxy_rows: &[MorphologyRow],
    validation_rows: &[MorphologyRow],
    ranking_folds: &[RankingFold],
    ranking_useful: bool,
    morphology_folds: &[FoldResult],
    morphology_best: &[CrossValidatedPoint],
    quantized_proxy: &[QuantizedProxyResult],
    baseline_best: &[super::CrossValidatedPoint],
    elapsed: std::time::Duration,
) -> Result<BTreeMap<String, Vec<u8>>> {
    let mut outputs = BTreeMap::new();
    outputs.insert(
        "morphology_labels.csv".to_string(),
        label_summary_csv(training)?,
    );
    outputs.insert("morphology_grid.csv".to_string(), grid_csv(grid)?);
    outputs.insert(
        "morphology_standalone.csv".to_string(),
        standalone_csv(training, selected)?,
    );
    outputs.insert(
        "morphology_scripts.csv".to_string(),
        script_metrics_csv(training, selected)?,
    );
    outputs.insert(
        "morphology_collisions.csv".to_string(),
        collision_csv(selected, collision)?,
    );
    outputs.insert(
        "morphology_quantization.csv".to_string(),
        quantization_csv(quantization)?,
    );
    outputs.insert(
        "morphology_proxy_distributions.csv".to_string(),
        proxy_distributions_csv(proxy_rows)?,
    );
    outputs.insert(
        "morphology_ranking_logo.csv".to_string(),
        ranking_csv(ranking_folds)?,
    );
    outputs.insert(
        "morphology_logo_results.csv".to_string(),
        logo_csv(morphology_folds)?,
    );
    outputs.insert(
        "morphology_model_forms.csv".to_string(),
        model_forms_csv(morphology_folds)?,
    );
    outputs.insert(
        "morphology_calibration_coefficients.csv".to_string(),
        calibration_coefficients_csv(morphology_folds)?,
    );
    outputs.insert(
        "morphology_frontier.csv".to_string(),
        frontier_csv(baseline_best, morphology_best, quantized_proxy)?,
    );
    outputs.insert(
        "morphology_synthetic_validation.csv".to_string(),
        synthetic_csv(proxy_rows, validation_rows, morphology_best, ranking_useful)?,
    );
    outputs.insert(
        "morphology_model_f32.bin".to_string(),
        serialize_model(selected),
    );
    outputs.insert(
        "runtime_observation.txt".to_string(),
        runtime_observation(training, selected, elapsed).into_bytes(),
    );
    outputs.insert(
        "report.md".to_string(),
        build_report(
            training,
            grid,
            selected,
            test_metrics,
            collision,
            quantization,
            proxy_rows,
            ranking_folds,
            ranking_useful,
            morphology_folds,
            morphology_best,
            baseline_best,
            validation_rows,
        )?
        .into_bytes(),
    );
    Ok(outputs)
}

fn label_summary_csv(training: &TrainingCorpus) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "split",
        "label",
        "rows",
        "given_observations",
        "surname_observations",
        "role_llr_p10",
        "role_llr_median",
        "role_llr_p90",
    ])?;
    for split in MorphSplit::ALL {
        for label in [RoleLabel::Given, RoleLabel::Surname] {
            let selected = training
                .examples
                .iter()
                .filter(|example| example.split == split && example.label == label)
                .collect::<Vec<_>>();
            let mut role_llrs = selected
                .iter()
                .map(|example| example.role_llr)
                .collect::<Vec<_>>();
            role_llrs.sort_by(f64::total_cmp);
            writer.write_record([
                split.as_str().to_string(),
                label.as_str().to_string(),
                selected.len().to_string(),
                selected
                    .iter()
                    .map(|example| example.given_count)
                    .sum::<u64>()
                    .to_string(),
                selected
                    .iter()
                    .map(|example| example.surname_count)
                    .sum::<u64>()
                    .to_string(),
                float(percentile(&role_llrs, 10)),
                float(percentile(&role_llrs, 50)),
                float(percentile(&role_llrs, 90)),
            ])?;
        }
    }
    Ok(writer.into_inner()?)
}

fn grid_csv(grid: &[GridResult]) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "minimum_ngram",
        "maximum_ngram",
        "buckets",
        "alpha",
        "beta",
        "l1",
        "l2",
        "epochs",
        "validation_rows",
        "validation_accuracy",
        "validation_balanced_accuracy",
        "validation_roc_auc",
        "validation_given_pr_auc",
        "validation_surname_pr_auc",
        "validation_macro_script_auc",
    ])?;
    for row in grid {
        writer.write_record([
            row.ngrams.minimum.to_string(),
            row.ngrams.maximum.to_string(),
            row.ngrams.buckets.to_string(),
            float(row.optimizer.alpha),
            float(row.optimizer.beta),
            float(row.optimizer.l1),
            float(row.optimizer.l2),
            row.optimizer.epochs.to_string(),
            row.validation.rows.to_string(),
            float(row.validation.accuracy),
            float(row.validation.balanced_accuracy),
            float(row.validation.roc_auc),
            float(row.validation.given_pr_auc),
            float(row.validation.surname_pr_auc),
            float(row.macro_script_auc),
        ])?;
    }
    Ok(writer.into_inner()?)
}

fn standalone_csv(training: &TrainingCorpus, model: &MorphModel) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "split",
        "rows",
        "given",
        "surname",
        "true_given",
        "true_surname",
        "false_given",
        "false_surname",
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "given_pr_auc",
        "surname_pr_auc",
    ])?;
    for split in MorphSplit::ALL {
        write_binary_metrics(
            &mut writer,
            split.as_str(),
            evaluate_binary(&training.examples, split, model),
        )?;
    }
    Ok(writer.into_inner()?)
}

fn write_binary_metrics<W: std::io::Write>(
    writer: &mut csv::Writer<W>,
    label: &str,
    metrics: BinaryMetrics,
) -> Result<()> {
    writer.write_record([
        label.to_string(),
        metrics.rows.to_string(),
        metrics.given.to_string(),
        metrics.surname.to_string(),
        metrics.true_given.to_string(),
        metrics.true_surname.to_string(),
        metrics.false_given.to_string(),
        metrics.false_surname.to_string(),
        float(metrics.accuracy),
        float(metrics.balanced_accuracy),
        float(metrics.roc_auc),
        float(metrics.given_pr_auc),
        float(metrics.surname_pr_auc),
    ])?;
    Ok(())
}

fn script_metrics_csv(training: &TrainingCorpus, model: &MorphModel) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "split",
        "script",
        "rows",
        "given",
        "surname",
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
    ])?;
    for split in MorphSplit::ALL {
        for script in ScriptClass::ALL {
            let scored = training
                .examples
                .iter()
                .filter(|example| example.split == split && example.script == script)
                .map(|example| {
                    (
                        example.label,
                        model.score_normalized(&example.normalized).logit,
                    )
                })
                .collect::<Vec<_>>();
            if scored.is_empty() {
                continue;
            }
            let metrics = binary_metrics(&scored);
            writer.write_record([
                split.as_str().to_string(),
                script.as_str().to_string(),
                metrics.rows.to_string(),
                metrics.given.to_string(),
                metrics.surname.to_string(),
                float(metrics.accuracy),
                float(metrics.balanced_accuracy),
                float(metrics.roc_auc),
            ])?;
        }
    }
    Ok(writer.into_inner()?)
}

fn collision_csv(model: &MorphModel, collision: CollisionStats) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "model",
        "unique_ngram_hashes",
        "occupied_buckets",
        "bucket_collisions",
        "bucket_occupancy",
        "bucket_collision_rate",
        "observed_64bit_hash_collisions",
    ])?;
    writer.write_record([
        model.ngrams.description(),
        collision.unique_hashes.to_string(),
        collision.occupied_buckets.to_string(),
        collision.bucket_collisions.to_string(),
        float(ratio_value(
            collision.occupied_buckets,
            model.ngrams.buckets,
        )),
        float(ratio_value(
            collision.bucket_collisions,
            collision.unique_hashes,
        )),
        collision.hash64_collisions.to_string(),
    ])?;
    Ok(writer.into_inner()?)
}

fn quantization_csv(results: &[QuantizationResult]) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "representation",
        "bytes",
        "validation_roc_auc",
        "test_roc_auc",
        "test_balanced_accuracy",
        "logit_error_p50",
        "logit_error_p95",
        "logit_error_p99",
        "logit_error_max",
        "signal_error_p50",
        "signal_error_p95",
        "signal_error_p99",
        "signal_error_max",
    ])?;
    for result in results {
        writer.write_record([
            result.kind.as_str().to_string(),
            result.bytes.to_string(),
            float(result.validation.roc_auc),
            float(result.test.roc_auc),
            float(result.test.balanced_accuracy),
            float(result.logit_error.p50),
            float(result.logit_error.p95),
            float(result.logit_error.p99),
            float(result.logit_error.maximum),
            float(result.signal_error.p50),
            float(result.signal_error.p95),
            float(result.signal_error.p99),
            float(result.signal_error.maximum),
        ])?;
    }
    Ok(writer.into_inner()?)
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum ProxyOutcome {
    CorrectWinner,
    WrongWinner,
    NullWinner,
    CorrectRejected,
}

impl ProxyOutcome {
    const ALL: [Self; 4] = [
        Self::CorrectWinner,
        Self::WrongWinner,
        Self::NullWinner,
        Self::CorrectRejected,
    ];

    fn as_str(self) -> &'static str {
        match self {
            Self::CorrectWinner => "correct_selected_winner",
            Self::WrongWinner => "wrong_selected_winner",
            Self::NullWinner => "expected_null_winner",
            Self::CorrectRejected => "correct_veto_free_winner_rejected_by_c4",
        }
    }

    fn includes(self, row: &MorphologyRow) -> bool {
        match self {
            Self::CorrectWinner => row.base.expected_greeting && row.base.selected_matches,
            Self::WrongWinner => {
                row.base.expected_greeting && row.base.winner_present && !row.base.selected_matches
            }
            Self::NullWinner => !row.base.expected_greeting && row.base.winner_present,
            Self::CorrectRejected => {
                row.base.expected_greeting
                    && row.base.selected_matches
                    && row.base.vetoes_pass
                    && !row.base.c4_emits
            }
        }
    }
}

fn proxy_distributions_csv(rows: &[MorphologyRow]) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "population",
        "outcome",
        "feature",
        "rows",
        "p10",
        "p25",
        "median",
        "p75",
        "p90",
    ])?;
    for population in Population::PROXIES
        .into_iter()
        .map(Some)
        .chain(std::iter::once(None))
    {
        for outcome in ProxyOutcome::ALL {
            let selected = rows
                .iter()
                .filter(|row| population.is_none_or(|value| row.base.population == value))
                .filter(|row| outcome.includes(row))
                .filter_map(|row| row.candidate_morphology.first().copied())
                .collect::<Vec<_>>();
            for (feature, values) in [
                (
                    "morphology_logit",
                    selected.iter().map(|value| value.logit).collect(),
                ),
                (
                    "morphology_signal",
                    selected.iter().map(|value| value.signal).collect(),
                ),
                (
                    "morphology_reliability",
                    selected.iter().map(|value| value.reliability).collect(),
                ),
            ] {
                write_distribution_row(
                    &mut writer,
                    population.map_or("COMBINED", Population::as_str),
                    outcome.as_str(),
                    feature,
                    values,
                )?;
            }
        }
    }
    Ok(writer.into_inner()?)
}

fn write_distribution_row<W: std::io::Write>(
    writer: &mut csv::Writer<W>,
    population: &str,
    outcome: &str,
    feature: &str,
    mut values: Vec<f64>,
) -> Result<()> {
    values.sort_by(f64::total_cmp);
    writer.write_record([
        population.to_string(),
        outcome.to_string(),
        feature.to_string(),
        values.len().to_string(),
        float(percentile(&values, 10)),
        float(percentile(&values, 25)),
        float(percentile(&values, 50)),
        float(percentile(&values, 75)),
        float(percentile(&values, 90)),
    ])?;
    Ok(())
}

fn ranking_csv(folds: &[RankingFold]) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "held_out",
        "selected_weight",
        "variant",
        "generation_ceiling",
        "correct_winners",
        "wrong_winners",
        "null_winners",
        "ranking_ceiling",
    ])?;
    for fold in folds {
        for (variant, metrics) in [("frozen", fold.frozen), ("morphology", fold.adjusted)] {
            writer.write_record([
                fold.held_out.as_str().to_string(),
                float(fold.config.weight),
                variant.to_string(),
                metrics.generation_ceiling.to_string(),
                metrics.correct_winners.to_string(),
                metrics.wrong_winners.to_string(),
                metrics.null_winners.to_string(),
                float(ratio_value(
                    metrics.correct_winners,
                    metrics.expected_greetings,
                )),
            ])?;
        }
    }
    Ok(writer.into_inner()?)
}

fn logo_csv(folds: &[FoldResult]) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "held_out",
        "target",
        "variant",
        "ranking_weight",
        "threshold",
        "iterations",
        "training_emitted",
        "training_precision",
        "held_out_emitted",
        "held_out_correct",
        "held_out_wrong",
        "held_out_null_fp",
        "held_out_precision",
        "held_out_recall",
        "held_out_correct_winner_rejected",
    ])?;
    for fold in folds {
        writer.write_record([
            fold.held_out.as_str().to_string(),
            float(fold.target),
            fold.variant.as_str().to_string(),
            float(fold.ranking.weight),
            float(fold.policy.threshold),
            fold.policy.model.iterations.to_string(),
            fold.training_metrics.emitted.to_string(),
            optional_ratio(fold.training_metrics.precision()),
            fold.held_out_metrics.emitted.to_string(),
            fold.held_out_metrics.correct.to_string(),
            fold.held_out_metrics.wrong.to_string(),
            fold.held_out_metrics.null_false_emissions.to_string(),
            optional_ratio(fold.held_out_metrics.precision()),
            optional_ratio(fold.held_out_metrics.recall()),
            fold.held_out_metrics
                .winner_correct_but_abstained
                .to_string(),
        ])?;
    }
    Ok(writer.into_inner()?)
}

fn model_forms_csv(folds: &[FoldResult]) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "target",
        "variant",
        "emitted",
        "correct",
        "wrong",
        "null_fp",
        "precision",
        "recall",
        "correct_winner_rejected",
    ])?;
    for target in TARGETS {
        for variant in [
            CalibrationVariant::Baseline,
            CalibrationVariant::MorphologyMain,
            CalibrationVariant::MorphologyInteraction,
            CalibrationVariant::RerankedInteraction,
        ] {
            let matching = folds
                .iter()
                .filter(|fold| fold.target == target && fold.variant == variant)
                .collect::<Vec<_>>();
            if matching.len() != Population::PROXIES.len() {
                continue;
            }
            let mut metrics = EmissionMetrics::default();
            for fold in matching {
                metrics.add(fold.held_out_metrics);
            }
            writer.write_record([
                float(target),
                variant.as_str().to_string(),
                metrics.emitted.to_string(),
                metrics.correct.to_string(),
                metrics.wrong.to_string(),
                metrics.null_false_emissions.to_string(),
                optional_ratio(metrics.precision()),
                optional_ratio(metrics.recall()),
                metrics.winner_correct_but_abstained.to_string(),
            ])?;
        }
    }
    Ok(writer.into_inner()?)
}

fn calibration_coefficients_csv(folds: &[FoldResult]) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record(["held_out", "target", "variant", "feature", "coefficient"])?;
    for fold in folds {
        writer.write_record([
            fold.held_out.as_str().to_string(),
            float(fold.target),
            fold.variant.as_str().to_string(),
            "intercept".to_string(),
            float(fold.policy.model.intercept),
        ])?;
        let names = calibration_feature_names(fold.variant);
        if names.len() != fold.policy.model.coefficients.len() {
            return Err("morphology calibration feature-name mismatch".into());
        }
        for (name, coefficient) in names.iter().zip(&fold.policy.model.coefficients) {
            writer.write_record([
                fold.held_out.as_str().to_string(),
                float(fold.target),
                fold.variant.as_str().to_string(),
                (*name).to_string(),
                float(*coefficient),
            ])?;
        }
    }
    Ok(writer.into_inner()?)
}

fn calibration_feature_names(variant: CalibrationVariant) -> Vec<&'static str> {
    let mut names = BASE_FEATURE_NAMES.to_vec();
    if variant != CalibrationVariant::Baseline {
        names.push("morphology_signal_x_reliability");
    }
    if matches!(
        variant,
        CalibrationVariant::MorphologyInteraction | CalibrationVariant::RerankedInteraction
    ) {
        names.extend(INTERACTION_FEATURE_NAMES);
    }
    names
}

fn frontier_csv(
    baseline: &[super::CrossValidatedPoint],
    morphology: &[CrossValidatedPoint],
    quantized: &[QuantizedProxyResult],
) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "target",
        "source",
        "variant",
        "emitted",
        "correct",
        "wrong",
        "null_fp",
        "precision",
        "wilson_lower",
        "wilson_upper",
        "recall",
        "false_abstentions",
        "correct_winner_rejected",
    ])?;
    for target in TARGETS {
        if let Some(point) = baseline.iter().find(|point| point.target == target) {
            write_frontier_row(
                &mut writer,
                target,
                "existing_frontier",
                point.family.as_str(),
                point.metrics,
            )?;
        }
        if let Some(point) = morphology.iter().find(|point| point.target == target) {
            write_frontier_row(
                &mut writer,
                target,
                "morphology_f32",
                point.variant.as_str(),
                point.metrics,
            )?;
        }
        for quantized in quantized {
            if let Some(point) = quantized.best.iter().find(|point| point.target == target) {
                write_frontier_row(
                    &mut writer,
                    target,
                    quantized.kind.as_str(),
                    point.variant.as_str(),
                    point.metrics,
                )?;
            }
        }
    }
    Ok(writer.into_inner()?)
}

fn write_frontier_row<W: std::io::Write>(
    writer: &mut csv::Writer<W>,
    target: f64,
    source: &str,
    variant: &str,
    metrics: EmissionMetrics,
) -> Result<()> {
    let interval = wilson_interval(metrics.correct, metrics.emitted);
    writer.write_record([
        float(target),
        source.to_string(),
        variant.to_string(),
        metrics.emitted.to_string(),
        metrics.correct.to_string(),
        metrics.wrong.to_string(),
        metrics.null_false_emissions.to_string(),
        optional_ratio(metrics.precision()),
        interval.map_or_else(String::new, |interval| float(interval.lower)),
        interval.map_or_else(String::new, |interval| float(interval.upper)),
        optional_ratio(metrics.recall()),
        metrics.false_abstentions.to_string(),
        metrics.winner_correct_but_abstained.to_string(),
    ])?;
    Ok(())
}

fn synthetic_csv(
    proxy_rows: &[MorphologyRow],
    validation_rows: &[MorphologyRow],
    selected: &[CrossValidatedPoint],
    ranking_useful: bool,
) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "target",
        "policy",
        "category",
        "rows",
        "expected_greetings",
        "emitted",
        "correct",
        "wrong",
        "null_fp",
        "precision",
        "recall",
    ])?;
    let categories = validation_rows
        .iter()
        .map(|row| row.category.as_str())
        .collect::<BTreeSet<_>>();
    for target in TARGETS {
        for category in std::iter::once("ALL").chain(categories.iter().copied()) {
            let rows = validation_rows
                .iter()
                .filter(|row| category == "ALL" || row.category == category)
                .collect::<Vec<_>>();
            let mut metrics = EmissionMetrics::default();
            for row in &rows {
                metrics.observe(&row.base, row.base.c4_emits);
            }
            write_synthetic_row(&mut writer, target, "frozen_c4", category, metrics)?;
        }
        let Some(selected) = selected.iter().find(|point| point.target == target) else {
            continue;
        };
        let ranking =
            if selected.variant == CalibrationVariant::RerankedInteraction && ranking_useful {
                select_ranking_config(
                    proxy_rows,
                    &RANKING_WEIGHTS.map(|weight| RankingConfig { weight }),
                )
            } else {
                RankingConfig::FROZEN
            };
        let proxy_ranked = proxy_rows
            .iter()
            .map(|row| rank_row(row, ranking))
            .collect::<Vec<_>>();
        let validation_ranked = validation_rows
            .iter()
            .map(|row| rank_row(row, ranking))
            .collect::<Vec<_>>();
        let model = fit_calibration_model(&proxy_ranked, selected.variant)?;
        let frontier = calibration_frontier(&proxy_ranked, &model);
        let policy = select_operating_point(&frontier, target)
            .ok_or("full-development morphology operating point missing")?
            .policy
            .clone();
        for category in std::iter::once("ALL").chain(categories.iter().copied()) {
            let selected_rows = validation_ranked
                .iter()
                .filter(|row| category == "ALL" || row.category == category);
            let metrics = evaluate_calibration(selected_rows, &policy);
            write_synthetic_row(
                &mut writer,
                target,
                selected.variant.as_str(),
                category,
                metrics,
            )?;
        }
    }
    Ok(writer.into_inner()?)
}

fn write_synthetic_row<W: std::io::Write>(
    writer: &mut csv::Writer<W>,
    target: f64,
    policy: &str,
    category: &str,
    metrics: EmissionMetrics,
) -> Result<()> {
    writer.write_record([
        float(target),
        policy.to_string(),
        category.to_string(),
        metrics.rows.to_string(),
        metrics.expected_greetings.to_string(),
        metrics.emitted.to_string(),
        metrics.correct.to_string(),
        metrics.wrong.to_string(),
        metrics.null_false_emissions.to_string(),
        optional_ratio(metrics.precision()),
        optional_ratio(metrics.recall()),
    ])?;
    Ok(())
}

fn serialize_model(model: &MorphModel) -> Vec<u8> {
    let mut bytes =
        Vec::with_capacity(model.weights.len() * 4 + model.occupied.len().div_ceil(8) + 128);
    bytes.extend_from_slice(b"bonjour-morphology-diagnostic-v1\0");
    bytes.extend_from_slice(&u32::from(model.ngrams.minimum).to_le_bytes());
    bytes.extend_from_slice(&u32::from(model.ngrams.maximum).to_le_bytes());
    bytes.extend_from_slice(&(model.ngrams.buckets as u64).to_le_bytes());
    bytes.extend_from_slice(&model.optimizer.alpha.to_bits().to_le_bytes());
    bytes.extend_from_slice(&model.optimizer.beta.to_bits().to_le_bytes());
    bytes.extend_from_slice(&model.optimizer.l1.to_bits().to_le_bytes());
    bytes.extend_from_slice(&model.optimizer.l2.to_bits().to_le_bytes());
    bytes.extend_from_slice(&(model.optimizer.epochs as u64).to_le_bytes());
    bytes.extend_from_slice(&model.intercept.to_bits().to_le_bytes());
    for weight in &model.weights {
        bytes.extend_from_slice(&weight.to_bits().to_le_bytes());
    }
    for chunk in model.occupied.chunks(8) {
        let mut packed = 0_u8;
        for (offset, occupied) in chunk.iter().enumerate() {
            if *occupied {
                packed |= 1 << offset;
            }
        }
        bytes.push(packed);
    }
    for script in ScriptClass::ALL {
        bytes.extend_from_slice(
            &(model.script_counts.get(&script).copied().unwrap_or(0) as u64).to_le_bytes(),
        );
    }
    bytes
}

fn runtime_observation(
    training: &TrainingCorpus,
    model: &MorphModel,
    total_elapsed: std::time::Duration,
) -> String {
    let sample = training
        .examples
        .iter()
        .filter(|example| example.split == MorphSplit::Test)
        .take(1_000)
        .collect::<Vec<_>>();
    let started = Instant::now();
    let mut checksum = 0.0;
    let repetitions = 100;
    for _ in 0..repetitions {
        for example in &sample {
            checksum += std::hint::black_box(model.score_normalized(&example.normalized).logit);
        }
    }
    let elapsed = started.elapsed();
    let evaluations = sample.len() * repetitions;
    format!(
        "observational only; excluded from deterministic report digest\n\
         total diagnostic elapsed seconds: {:.6}\n\
         morphology evaluations: {evaluations}\n\
         elapsed nanoseconds: {}\n\
         nanoseconds per token: {:.3}\n\
         checksum: {:.17}\n\
         normalization allocation: one temporary normalized String per external token; scoring a pre-normalized token allocates only a short sparse feature Vec\n",
        total_elapsed.as_secs_f64(),
        elapsed.as_nanos(),
        elapsed.as_nanos() as f64 / evaluations.max(1) as f64,
        checksum,
    )
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn build_report(
    training: &TrainingCorpus,
    grid: &[GridResult],
    selected: &MorphModel,
    test_metrics: BinaryMetrics,
    collision: CollisionStats,
    quantization: &[QuantizationResult],
    proxy_rows: &[MorphologyRow],
    ranking_folds: &[RankingFold],
    ranking_useful: bool,
    morphology_folds: &[FoldResult],
    morphology_best: &[CrossValidatedPoint],
    baseline_best: &[super::CrossValidatedPoint],
    validation_rows: &[MorphologyRow],
) -> Result<String> {
    let mut report = String::new();
    writeln!(report, "# Morphological role-evidence diagnostic\n")?;
    writeln!(
        report,
        "This benchmark-only experiment trains a hashed Unicode-scalar character n-gram model to distinguish conservatively labeled given-like spellings from surname-like spellings. It does not model arbitrary non-name text, create candidates, change frozen ranking outside an isolated variant, alter production C4, freeze C5, inspect TEST, or spend REAL_PROXY_V6. Generic position and capitalization are absent.\n"
    )?;
    writeln!(report, "## Exact corpus labels\n")?;
    writeln!(
        report,
        "The exact `name-totals.csv` input contains {} keys and is pinned at SHA-256 `{}`. Exact given counts sum to {} and exact surname-overlap counts sum to {}; role LLR uses the full surname denominator {}. The label rules are `given_count >= {MIN_ROLE_COUNT} && role_llr >= +{MIN_ROLE_LLR:.1}` and `surname_count >= {MIN_ROLE_COUNT} && role_llr <= -{MIN_ROLE_LLR:.1}` after the training-only lexical filter.\n",
        training.stats.source_rows,
        TOTALS_SHA256,
        GIVEN_TOTAL,
        SURNAME_OVERLAP_TOTAL,
        SURNAME_TOTAL
    )?;
    writeln!(
        report,
        "Source rows passing the single-token lexical filter: {}. Raw high-confidence labels: {} given / {} surname. Accent/case grouping discarded {} conflicting groups and collapsed {} duplicate variants.\n",
        training.stats.lexical_rows,
        training.stats.raw_given,
        training.stats.raw_surname,
        training.stats.conflicting_groups,
        training.stats.duplicate_group_rows
    )?;
    writeln!(
        report,
        "| Split | Given | Surname | Total |\n|---|---:|---:|---:|"
    )?;
    for split in MorphSplit::ALL {
        let given = label_count(training, split, RoleLabel::Given);
        let surname = label_count(training, split, RoleLabel::Surname);
        writeln!(
            report,
            "| {} | {} | {} | {} |",
            split.as_str(),
            given,
            surname,
            given + surname
        )?;
    }

    writeln!(report, "\n## Model selection and standalone quality\n")?;
    writeln!(
        report,
        "The deterministic grid evaluated 2-3, 2-4 and 2-5 scalar n-grams at 16K/32K/64K/128K signed-hash buckets, then the four locked alpha/L2 combinations for the best representation. Selection used only corpus-derived VALIDATION. The selected model is `{}` with alpha `{:.2}`, L2 `{:.1}`, and {} epochs.\n",
        selected.ngrams.description(),
        selected.optimizer.alpha,
        selected.optimizer.l2,
        selected.optimizer.epochs
    )?;
    let validation = evaluate_binary(&training.examples, MorphSplit::Validation, selected);
    writeln!(
        report,
        "| Split | Rows | Accuracy | Balanced accuracy | ROC AUC | Given PR AUC | Surname PR AUC |\n|---|---:|---:|---:|---:|---:|---:|"
    )?;
    for (split, metrics) in [
        (
            MorphSplit::Train,
            evaluate_binary(&training.examples, MorphSplit::Train, selected),
        ),
        (MorphSplit::Validation, validation),
        (MorphSplit::Test, test_metrics),
    ] {
        writeln!(
            report,
            "| {} | {} | {:.2}% | {:.2}% | {:.4} | {:.4} | {:.4} |",
            split.as_str(),
            metrics.rows,
            metrics.accuracy * 100.0,
            metrics.balanced_accuracy * 100.0,
            metrics.roc_auc,
            metrics.given_pr_auc,
            metrics.surname_pr_auc
        )?;
    }
    writeln!(
        report,
        "\nThe selected TRAIN vocabulary produced {} unique 64-bit n-gram hashes, {} occupied buckets, {} feature-hash collisions ({:.2}%), and {} observed primary-64-bit collisions under an independent secondary hash.\n",
        collision.unique_hashes,
        collision.occupied_buckets,
        collision.bucket_collisions,
        ratio_value(collision.bucket_collisions, collision.unique_hashes) * 100.0,
        collision.hash64_collisions
    )?;

    writeln!(report, "## Script behavior\n")?;
    writeln!(
        report,
        "| TEST script | Rows | Given | Surname | Balanced accuracy | ROC AUC |\n|---|---:|---:|---:|---:|---:|"
    )?;
    for script in ScriptClass::ALL {
        let scored = training
            .examples
            .iter()
            .filter(|example| example.split == MorphSplit::Test && example.script == script)
            .map(|example| {
                (
                    example.label,
                    selected.score_normalized(&example.normalized).logit,
                )
            })
            .collect::<Vec<_>>();
        if scored.is_empty() {
            continue;
        }
        let metrics = binary_metrics(&scored);
        writeln!(
            report,
            "| {} | {} | {} | {} | {:.2}% | {:.4} |",
            script.as_str(),
            metrics.rows,
            metrics.given,
            metrics.surname,
            metrics.balanced_accuracy * 100.0,
            metrics.roc_auc
        )?;
    }
    writeln!(
        report,
        "\nMorphology reliability multiplies occupied-bucket support by saturating dominant-script TRAIN support. Unknown, mixed, or unrepresented scripts therefore approach neutral evidence rather than inheriting a Latin assumption.\n"
    )?;

    writeln!(report, "## Quantization\n")?;
    writeln!(
        report,
        "| Weights | Bytes | TEST ROC AUC | TEST balanced accuracy | Signal p99 error | Signal max error |\n|---|---:|---:|---:|---:|---:|"
    )?;
    for result in quantization {
        writeln!(
            report,
            "| {} | {} | {:.4} | {:.2}% | {:.6} | {:.6} |",
            result.kind.as_str(),
            result.bytes,
            result.test.roc_auc,
            result.test.balanced_accuracy * 100.0,
            result.signal_error.p99,
            result.signal_error.maximum
        )?;
    }

    let base_ranking = aggregate_ranking(ranking_folds, false);
    let adjusted_ranking = aggregate_ranking(ranking_folds, true);
    writeln!(report, "\n## Proxy ranking\n")?;
    writeln!(
        report,
        "Across 7,808 spent proxy rows, frozen ranking selected {} correct winners, {} wrong positive winners and {} expected-NULL winners; its candidate-generation ceiling was {}. Generation-held-out morphology ranking selected {} correct, {} wrong and {} NULL winners. Ranking morphology is therefore {}.\n",
        base_ranking.correct_winners,
        base_ranking.wrong_winners,
        base_ranking.null_winners,
        base_ranking.generation_ceiling,
        adjusted_ranking.correct_winners,
        adjusted_ranking.wrong_winners,
        adjusted_ranking.null_winners,
        if ranking_useful {
            "eligible for the interaction frontier"
        } else {
            "not independently useful"
        }
    )?;

    writeln!(report, "## Out-of-fold calibration frontier\n")?;
    writeln!(
        report,
        "| Target | Existing precision | Existing recall | Morph precision | Morph recall | Recall delta | Correct | Wrong | NULL FP | Correct winner rejected |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )?;
    for target in TARGETS {
        let baseline = baseline_best
            .iter()
            .find(|point| point.target == target)
            .ok_or("baseline target missing")?;
        let morphology = morphology_best
            .iter()
            .find(|point| point.target == target)
            .ok_or("morphology target missing")?;
        writeln!(
            report,
            "| {:.1}% | {:.2}% | {:.2}% | {:.2}% | {:.2}% | {:+.2} pp | {} | {} | {} | {} |",
            target * 100.0,
            baseline.metrics.precision().unwrap_or(0.0) * 100.0,
            baseline.metrics.recall().unwrap_or(0.0) * 100.0,
            morphology.metrics.precision().unwrap_or(0.0) * 100.0,
            morphology.metrics.recall().unwrap_or(0.0) * 100.0,
            (morphology.metrics.recall().unwrap_or(0.0) - baseline.metrics.recall().unwrap_or(0.0))
                * 100.0,
            morphology.metrics.correct,
            morphology.metrics.wrong,
            morphology.metrics.null_false_emissions,
            morphology.metrics.winner_correct_but_abstained
        )?;
    }
    writeln!(
        report,
        "\nPer-generation model selection, thresholds, precision, recall, wrong counts and NULL false emissions are preserved in `morphology_logo_results.csv`. Wilson intervals and quantized frontiers are in `morphology_frontier.csv`. Proxy rows never train the token model; they only fit generation-held-out emission calibration.\n"
    )?;

    writeln!(report, "## Proxy signal diagnostic\n")?;
    writeln!(
        report,
        "| Population | Rows | Signal p10 | p25 | Median | p75 | p90 |\n|---|---:|---:|---:|---:|---:|---:|"
    )?;
    for outcome in ProxyOutcome::ALL {
        let mut values = proxy_rows
            .iter()
            .filter(|row| outcome.includes(row))
            .filter_map(|row| row.candidate_morphology.first().map(|value| value.signal))
            .collect::<Vec<_>>();
        values.sort_by(f64::total_cmp);
        writeln!(
            report,
            "| {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |",
            outcome.as_str(),
            values.len(),
            percentile(&values, 10),
            percentile(&values, 25),
            percentile(&values, 50),
            percentile(&values, 75),
            percentile(&values, 90)
        )?;
    }

    let recommendation = recommendation(morphology_best, baseline_best, morphology_folds);
    let c4_validation = validation_metrics(validation_rows, |row| row.base.c4_emits);
    writeln!(report, "\n## Synthetic VALIDATION and recommendation\n")?;
    writeln!(
        report,
        "Frozen C4 on synthetic VALIDATION emits {} correct / {} wrong / {} NULL FP at {:.2}% recall. Complete category-specific comparisons for every morphology operating point are in `morphology_synthetic_validation.csv`.\n",
        c4_validation.correct,
        c4_validation.wrong,
        c4_validation.null_false_emissions,
        c4_validation.recall().unwrap_or(0.0) * 100.0
    )?;
    writeln!(
        report,
        "Recommendation: **{}**. {}\n",
        recommendation.0, recommendation.1
    )?;
    writeln!(
        report,
        "The model is a given-versus-surname role-morphology diagnostic, not a calibrated probability and not proof that unknown text is a name. All surname negatives are retained first-name keys, which is appropriate for candidate disambiguation but not arbitrary token admission. The model and all runtime work remain benchmark-only. C4 production and historical classifiers are unchanged; C5 is not frozen and V6 remains untouched.\n"
    )?;
    writeln!(
        report,
        "Qualitative motivating examples are intentionally absent from this deterministic report. They are evaluated only through ignored local inputs after selection, and repository-visible forms use literal `REDACTED`.\n"
    )?;
    writeln!(
        report,
        "The generated f32 model contains {} bytes. External normalization currently allocates one temporary string and sparse feature vector per token; production allocation work was explicitly out of scope. Machine-dependent timing is isolated in `runtime_observation.txt`.\n",
        serialize_model(selected).len()
    )?;
    writeln!(
        report,
        "The full representation/optimizer grid contains {} rows. Generated outputs contain aggregate statistics, hashes, coefficients, and hashed weights only—no corpus names, proxy display names, candidate strings, or personal identifiers.\n",
        grid.len()
    )?;
    Ok(report)
}

fn aggregate_ranking(folds: &[RankingFold], adjusted: bool) -> RankingMetrics {
    let mut result = RankingMetrics::default();
    for fold in folds {
        let metrics = if adjusted { fold.adjusted } else { fold.frozen };
        result.rows += metrics.rows;
        result.expected_greetings += metrics.expected_greetings;
        result.expected_nulls += metrics.expected_nulls;
        result.winner_present += metrics.winner_present;
        result.correct_winners += metrics.correct_winners;
        result.wrong_winners += metrics.wrong_winners;
        result.null_winners += metrics.null_winners;
        result.generation_ceiling += metrics.generation_ceiling;
    }
    result
}

fn validation_metrics(
    rows: &[MorphologyRow],
    emits: impl Fn(&MorphologyRow) -> bool,
) -> EmissionMetrics {
    let mut metrics = EmissionMetrics::default();
    for row in rows {
        metrics.observe(&row.base, emits(row));
    }
    metrics
}

fn recommendation(
    morphology: &[CrossValidatedPoint],
    baseline: &[super::CrossValidatedPoint],
    folds: &[FoldResult],
) -> (&'static str, &'static str) {
    let stable = |target: f64| {
        folds
            .iter()
            .filter(|fold| fold.target == target)
            .all(|fold| fold.held_out_metrics.precision().unwrap_or(0.0) >= target - 0.02)
    };
    let meaningful = [0.99, 0.98].into_iter().any(|target| {
        let Some(morphology) = morphology.iter().find(|point| point.target == target) else {
            return false;
        };
        let Some(baseline) = baseline.iter().find(|point| point.target == target) else {
            return false;
        };
        morphology.metrics.precision().unwrap_or(0.0) >= target
            && morphology.metrics.recall().unwrap_or(0.0)
                >= baseline.metrics.recall().unwrap_or(0.0) + 0.01
            && stable(target)
    });
    if meaningful {
        (
            "strongly useful",
            "Morphology produces a stable, meaningful outward shift near the 98-99% proxy frontier with a small hashed model; retain it for a future explicitly selected C5 candidate.",
        )
    } else if TARGETS.into_iter().any(|target| {
        morphology
            .iter()
            .find(|point| point.target == target)
            .zip(baseline.iter().find(|point| point.target == target))
            .is_some_and(|(morphology, baseline)| {
                morphology.metrics.precision().unwrap_or(0.0) >= target
                    && morphology.metrics.recall().unwrap_or(0.0)
                        > baseline.metrics.recall().unwrap_or(0.0) + 0.005
            })
    }) {
        (
            "marginal",
            "Morphology shows only a small or strict-target-unstable gain; keep the result experimental and do not promote it without stronger evidence.",
        )
    } else {
        (
            "harmful / no value",
            "Morphology does not produce a meaningful stable outward shift over the established frontier; drop it from the future C5 feature set.",
        )
    }
}

fn label_count(training: &TrainingCorpus, split: MorphSplit, label: RoleLabel) -> usize {
    training
        .examples
        .iter()
        .filter(|example| example.split == split && example.label == label)
        .count()
}

fn float(value: f64) -> String {
    format!("{value:.17}")
}

fn optional_ratio(value: Option<f64>) -> String {
    value.map_or_else(String::new, float)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example(value: &str, label: RoleLabel) -> LabeledExample {
        let normalized = morphology_normalize(value).unwrap();
        LabeledExample {
            group: accent_group(&normalized),
            hashes: ngram_hashes(&normalized),
            normalized,
            label,
            split: MorphSplit::Train,
            script: ScriptClass::Latin,
            given_count: 1_000,
            surname_count: 1_000,
            role_llr: 0.0,
        }
    }

    #[test]
    fn normalization_is_unicode_aware_and_token_only() {
        assert_eq!(
            morphology_normalize("E\u{301}LODIE").as_deref(),
            Some("élodie")
        );
        assert_eq!(
            morphology_normalize("Jean-Pierre").as_deref(),
            Some("jean-pierre")
        );
        assert_eq!(
            morphology_normalize("O’Connor").as_deref(),
            Some("o'connor")
        );
        assert_eq!(morphology_normalize("Anne Marie"), None);
        assert_eq!(morphology_normalize("A_Kim"), None);
        assert_eq!(morphology_normalize("--Anne"), None);
    }

    #[test]
    fn role_labels_require_both_count_and_llr_margins() {
        assert_eq!(role_label(100, 0), Some(RoleLabel::Given));
        assert_eq!(role_label(0, 100), Some(RoleLabel::Surname));
        assert_eq!(role_label(99, 0), None);
        assert_eq!(role_label(0, 99), None);
        assert_eq!(role_label(100, 100), None);
    }

    #[test]
    fn accent_variants_share_split_and_script_is_explicit() {
        let accented = morphology_normalize("Élodie").unwrap();
        let plain = morphology_normalize("Elodie").unwrap();
        assert_eq!(accent_group(&accented), accent_group(&plain));
        assert_eq!(
            MorphSplit::from_group(&accent_group(&accented)),
            MorphSplit::from_group(&accent_group(&plain))
        );
        assert_eq!(dominant_script(&accented), ScriptClass::Latin);
        assert_eq!(dominant_script("Мария"), ScriptClass::Cyrillic);
        assert_eq!(dominant_script("محمد"), ScriptClass::Arabic);
        assert_eq!(dominant_script("李"), ScriptClass::Han);
    }

    #[test]
    fn ngrams_use_unicode_scalars_and_boundary_markers() {
        let ascii = ngram_signatures("ab");
        let unicode = ngram_signatures("éa");
        assert_eq!(ascii.len(), 6);
        assert_eq!(unicode.len(), 6);
        assert_eq!(
            ascii
                .iter()
                .map(|(length, _, _)| *length)
                .collect::<Vec<_>>(),
            vec![2, 2, 2, 3, 3, 4]
        );
        assert_ne!(ngram_hashes("ab"), ngram_hashes("ba"));
    }

    #[test]
    fn ftrl_training_and_quantization_are_deterministic() {
        let examples = vec![
            example("Alina", RoleLabel::Given),
            example("Maria", RoleLabel::Given),
            example("Bergson", RoleLabel::Surname),
            example("Markson", RoleLabel::Surname),
        ];
        let ngrams = NgramConfig {
            minimum: 2,
            maximum: 4,
            buckets: 256,
        };
        let left = train_morphology_model(&examples, ngrams, OptimizerConfig::DEFAULT).unwrap();
        let right = train_morphology_model(&examples, ngrams, OptimizerConfig::DEFAULT).unwrap();
        assert_eq!(serialize_model(&left), serialize_model(&right));

        for kind in QuantizationKind::ALL {
            let left = quantize_model(&left, kind);
            let right = quantize_model(&right, kind);
            assert_eq!(
                left.score_token("Alina").logit.to_bits(),
                right.score_token("Alina").logit.to_bits()
            );
            assert!(quantized_model_bytes(&left) > 0);
        }
    }

    #[test]
    fn weak_or_unknown_morphology_degrades_to_neutral_reliability() {
        let model = MorphModel {
            ngrams: NgramConfig {
                minimum: 2,
                maximum: 3,
                buckets: 16,
            },
            optimizer: OptimizerConfig::DEFAULT,
            intercept: 0.0,
            weights: vec![0.0; 16],
            occupied: vec![false; 16],
            script_counts: BTreeMap::new(),
            maximum_script_count: 0,
        };
        let evidence = model.score_token("名前");
        assert_eq!(evidence.reliability, 0.0);
        assert_eq!(evidence.signal, 0.0);
    }

    #[test]
    fn multi_token_evidence_is_reliability_weighted() {
        let evidence = aggregate_morphology(&[
            MorphEvidence {
                logit: 2.0,
                signal: 0.8,
                reliability: 1.0,
                bucket_support: 1.0,
                script_support: 1.0,
                script: ScriptClass::Latin,
            },
            MorphEvidence {
                logit: -2.0,
                signal: -0.8,
                reliability: 0.5,
                bucket_support: 0.5,
                script_support: 0.5,
                script: ScriptClass::Latin,
            },
        ]);
        assert!((evidence.logit - (2.0 / 3.0)).abs() < 1.0e-12);
        assert!((evidence.signal - (0.8 / 3.0)).abs() < 1.0e-12);
        assert_eq!(evidence.reliability, 0.75);
        assert_eq!(evidence.script, ScriptClass::Latin);
    }
}
