use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::error::Error;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::Path;
use std::process::Command;

use bincode::Options;
use boomphf::Mphf;
use name_eval::holdout::FrozenHoldout;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use xxhash_rust::xxh3::xxh3_64_with_seed;

use crate::artifact::EvidenceSource;
use crate::classifier::{
    ALGORITHM_C2, ALGORITHM_C3, ALGORITHM_C4, ALGORITHM_C5, ALGORITHM_C6, ALGORITHM_C31,
    C6EmissionSource, CandidateDiagnostic, c5_emitted_candidate, c6_decision_breakdown,
    canonicalize, diagnose_role_inference, expected_lookup_diagnostic,
};
use crate::metrics::greeting_matches;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub(super) const V1_SHA256: &str =
    "de95213f27fc1849032ee6788c8f16d7d515c1a991ae8b2e8414b7b155814c4e";
pub(super) const V2_SHA256: &str =
    "7d704a646b8dd9fa3820f88b9504d4397b676af9435532cf2da9befda7663a73";
pub(super) const V3_SHA256: &str =
    "d70e4d4b2ed7e49bed09dc1e8d2ba60ade8a752e3b86c772e964bd64883ee6fe";
pub(super) const V4_SHA256: &str =
    "d95c589bec836faaeecaeda85b146989d2936914bff0209934f289ccb9446c7f";
pub(super) const V5_SHA256: &str =
    "69070614fee68401b896d6c5bfb4c22c55cca9744237f66213a9dd04291db6c7";
pub(super) const V6_SHA256: &str =
    "a02d7105ea4f084e9d4ee94b3633e5068eb35e076dd7413b58b6d65549e734b1";
pub(super) const V7_SHA256: &str =
    "901e630f2ed612e9cd40f5c2ecd28f9d47c9768b10ddd762d2136c5a56cd8a6d";
const RAW_FILES: usize = 105;
const RAW_ROWS: u64 = 491_655_925;
const NONEMPTY_SURNAMES: u64 = 489_631_377;
const GIVEN_KEYS: usize = 1_803_175;
const EXISTING_ARTIFACT_BYTES: usize = 36_632_687;
const ROUTING_SEED: u64 = 0x6e61_6d65_2d72_6f75;
const FINGERPRINT_SEED: u64 = 0x6e61_6d65_2d66_7033;
const BLOOM_SEED_A: u64 = 0x7375_726e_2d62_6c31;
const BLOOM_SEED_B: u64 = 0x7375_726e_2d62_6c32;
const MPHF_GAMMA: f64 = 1.7;
const BLOOM_REFERENCE_FPR: f64 = 0.001;
const BLOOM_FINGERPRINT_FPR: f64 = 1.0 / 4_294_967_296.0;
const GENERATED_NEGATIVES: usize = 100_000;
const FROZEN_CANDIDATE_MANIFEST_BYTES: usize = 710;
const FROZEN_CANDIDATE_MANIFEST_SHA256: &str =
    "99d7be0c592eb817eb6ae2c4e59517a12753e86302abed390099293d6c02b675";
const FROZEN_CANDIDATE_SOURCE_KEYS_SHA256: &str =
    "710d491599f2140ccdb85e25cb553d574584bad30ef3543904cdcd7270703013";
const FROZEN_CANDIDATE_MPHF_BYTES: usize = 15_248_560;
const FROZEN_CANDIDATE_MPHF_SHA256: &str =
    "40bcd571685a0fab7f93337b288d32e2ee25937feaa3c7be8e3a0ec86920f635";
const FROZEN_CANDIDATE_FINGERPRINT_BYTES: usize = 141_668_176;
const FROZEN_CANDIDATE_FINGERPRINT_SHA256: &str =
    "a419dfb9d6d792ae08710c1f007cfd58442a0e7224f2630e9d33bd7becdabc7b";
const SURNAME_THRESHOLDS: [u64; 10] = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1_000];
const PRIOR_KEY_COUNTS: [usize; 10] = [
    35_417_044, 10_271_119, 3_390_945, 1_709_397, 684_930, 320_718, 135_907, 33_385, 8_979, 2_117,
];
const PRIOR_RAW_BYTES: [usize; 10] = [
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
const PRIOR_ZSTD19_BYTES: [usize; 10] = [
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
enum Generation {
    V1,
    V2,
    V3,
    V4,
    V5,
    V6,
    V7,
}

impl Generation {
    const ALL: [Self; 7] = [
        Self::V1,
        Self::V2,
        Self::V3,
        Self::V4,
        Self::V5,
        Self::V6,
        Self::V7,
    ];

    fn from_digest(digest: &str) -> Option<Self> {
        match digest {
            V1_SHA256 => Some(Self::V1),
            V2_SHA256 => Some(Self::V2),
            V3_SHA256 => Some(Self::V3),
            V4_SHA256 => Some(Self::V4),
            V5_SHA256 => Some(Self::V5),
            V6_SHA256 => Some(Self::V6),
            V7_SHA256 => Some(Self::V7),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::V1 => "REAL_PROXY_V1",
            Self::V2 => "REAL_PROXY_V2",
            Self::V3 => "REAL_PROXY_V3",
            Self::V4 => "REAL_PROXY_V4",
            Self::V5 => "REAL_PROXY_V5",
            Self::V6 => "REAL_PROXY_V6",
            Self::V7 => "REAL_PROXY_V7",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Metrics {
    emitted: usize,
    correct: usize,
    wrong: usize,
    null_false_emissions: usize,
}

impl Metrics {
    fn observe(&mut self, expected: Option<&str>, selected: Option<&str>) {
        let Some(selected) = selected else {
            return;
        };
        self.emitted += 1;
        if expected.is_some_and(|expected| greeting_matches(Some(expected), Some(selected))) {
            self.correct += 1;
        } else {
            self.wrong += 1;
            self.null_false_emissions += usize::from(expected.is_none());
        }
    }
}

#[derive(Clone, Debug)]
struct SelectionRow {
    generation: Generation,
    expected_greeting: Option<String>,
    selected_candidate: Option<String>,
    first_position_emission: bool,
    residual_topology: bool,
    candidate_quality: f64,
    reliability: f64,
    role_signal: f64,
    complement: Option<String>,
    complement_given_observed: bool,
    complement_surname_count: Option<u64>,
    hard_organization_marker: bool,
    generic_organization_marker: bool,
    ampersand: bool,
    candidate_too_short: bool,
}

impl SelectionRow {
    fn residual_emission(&self, threshold: u64) -> Option<&str> {
        (self.residual_topology
            && !self.complement_given_observed
            && self
                .complement_surname_count
                .is_some_and(|count| count >= threshold))
        .then_some(self.selected_candidate.as_deref())
        .flatten()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ThresholdPoint {
    threshold: u64,
    metrics: Metrics,
    artifact_bytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct InventoryRow {
    threshold: u64,
    key_count: usize,
    raw_utf8_bytes: usize,
    zstd19_bytes: usize,
    raw_sha256: String,
    zstd_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TargetedScan {
    counts: BTreeMap<String, u64>,
    counts_sha256: String,
    inventory_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct MembershipStats {
    mphf_bytes: usize,
    fingerprint_bytes: usize,
    manifest_bytes: usize,
    total_bytes: usize,
    given_negative_queries: usize,
    given_false_accepts: usize,
    generated_negative_queries: usize,
    generated_false_accepts: usize,
    bloom_reference_bytes: usize,
    bloom_reference_false_accepts: usize,
    bloom_fingerprint_bytes: usize,
    bloom_fingerprint_false_accepts: usize,
}

#[derive(Clone, Debug)]
struct SelectionResult {
    rows: Vec<SelectionRow>,
    points: Vec<ThresholdPoint>,
    logo: Vec<(Generation, ThresholdPoint, Metrics)>,
    inventory: Vec<InventoryRow>,
    selected: ThresholdPoint,
    membership: MembershipStats,
    counts_sha256: String,
    selected_key_sha256: String,
    candidate_manifest_sha256: String,
    inventory_sha256: String,
    probes: Vec<ProbeResult>,
}

#[derive(Debug, Deserialize)]
struct ProbeInput {
    label: String,
    display_name: String,
}

#[derive(Clone, Debug)]
struct ProbeResult {
    label: String,
    selected_candidate: Option<String>,
    candidate_quality: f64,
    role_signal: f64,
    reliability: f64,
    complement_surname_count: Option<u64>,
    complement_in_selected_index: bool,
    first_position_emits: bool,
    surname_residual_emits: bool,
    vetoes_pass: bool,
}

struct MembershipIndex {
    mphf: Mphf<u64>,
    fingerprints: Vec<u32>,
}

impl MembershipIndex {
    fn contains(&self, key: &str) -> bool {
        let routing = xxh3_64_with_seed(key.as_bytes(), ROUTING_SEED);
        let Some(slot) = self.mphf.try_hash(&routing) else {
            return false;
        };
        let Ok(slot) = usize::try_from(slot) else {
            return false;
        };
        self.fingerprints.get(slot).copied()
            == Some(xxh3_64_with_seed(key.as_bytes(), FINGERPRINT_SEED) as u32)
    }
}

pub(super) struct FrozenSurnameMembership {
    index: MembershipIndex,
}

impl FrozenSurnameMembership {
    pub(super) fn contains(&self, key: &str) -> bool {
        self.index.contains(key)
    }

    pub(super) const fn key_count(&self) -> usize {
        PRIOR_KEY_COUNTS[0]
    }

    pub(super) const fn artifact_bytes(&self) -> usize {
        FROZEN_CANDIDATE_MPHF_BYTES
            + FROZEN_CANDIDATE_FINGERPRINT_BYTES
            + FROZEN_CANDIDATE_MANIFEST_BYTES
    }

    pub(super) const fn manifest_sha256(&self) -> &'static str {
        FROZEN_CANDIDATE_MANIFEST_SHA256
    }

    pub(super) const fn source_keys_sha256(&self) -> &'static str {
        FROZEN_CANDIDATE_SOURCE_KEYS_SHA256
    }

    pub(super) const fn mphf_sha256(&self) -> &'static str {
        FROZEN_CANDIDATE_MPHF_SHA256
    }

    pub(super) const fn fingerprint_sha256(&self) -> &'static str {
        FROZEN_CANDIDATE_FINGERPRINT_SHA256
    }
}

struct BloomFilter {
    bits: Vec<u8>,
    bit_count: usize,
    hashes: usize,
}

impl BloomFilter {
    fn new(key_count: usize, false_positive_rate: f64) -> Self {
        let bits_per_key = -false_positive_rate.ln() / 2_f64.ln().powi(2);
        let bit_count = (bits_per_key * key_count as f64).ceil().max(8.0) as usize;
        let hashes = ((bit_count as f64 / key_count.max(1) as f64) * 2_f64.ln())
            .round()
            .max(1.0) as usize;
        Self {
            bits: vec![0; bit_count.div_ceil(8)],
            bit_count,
            hashes,
        }
    }

    fn insert(&mut self, key: &str) {
        let bits = self.positions(key).collect::<Vec<_>>();
        for bit in bits {
            self.bits[bit / 8] |= 1 << (bit % 8);
        }
    }

    fn contains(&self, key: &str) -> bool {
        self.positions(key)
            .all(|bit| self.bits[bit / 8] & (1 << (bit % 8)) != 0)
    }

    fn positions<'a>(&'a self, key: &'a str) -> impl Iterator<Item = usize> + 'a {
        let first = xxh3_64_with_seed(key.as_bytes(), BLOOM_SEED_A);
        let second = xxh3_64_with_seed(key.as_bytes(), BLOOM_SEED_B) | 1;
        (0..self.hashes).map(move |index| {
            first.wrapping_add((index as u64).wrapping_mul(second)) as usize % self.bit_count
        })
    }
}

pub(crate) fn prepare_surname_index_selection(
    output: &Path,
    holdouts: Vec<FrozenHoldout>,
    probes_path: &Path,
) -> Result<String> {
    let holdouts = validate_and_order_holdouts(holdouts)?;
    let keys = expected_lookup_keys(&holdouts, Some(probes_path))?;
    let bytes = serialize_keys(&keys)?;
    let digest = sha256_hex(&bytes);
    fs::write(output.join("lookup_keys.csv"), &bytes)?;
    fs::write(
        output.join("lookup_manifest.csv"),
        lookup_manifest_csv(&holdouts, keys.len(), &digest)?,
    )?;
    let mut report = String::new();
    writeln!(report, "# Private spent V1-V7 surname lookup preparation\n").unwrap();
    writeln!(report, "All seven holdouts were checksum-verified before exporting a lexical superset of {} distinct keys. The lookup-key SHA-256 is `{digest}`. This unredacted material must remain under ignored `_wip/`.", keys.len()).unwrap();
    Ok(report)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_surname_index_selection(
    output: &Path,
    corpus: &impl EvidenceSource,
    holdouts: Vec<FrozenHoldout>,
    surname_counts: &Path,
    surname_manifest: &Path,
    inventory_path: &Path,
    key_directory: &Path,
    name_totals: &Path,
    probes_path: &Path,
) -> Result<String> {
    let holdouts = validate_and_order_holdouts(holdouts)?;
    let expected_keys = expected_lookup_keys(&holdouts, Some(probes_path))?;
    let scan = load_targeted_scan(surname_counts, surname_manifest, &expected_keys, &holdouts)?;
    let inventory = load_inventory(inventory_path, &scan.inventory_sha256)?;
    let rows = build_rows(corpus, &holdouts, &scan.counts)?;
    let points = threshold_points(&rows, &inventory);
    let selected = select_point(&points).ok_or("surname threshold grid is empty")?;
    let logo = logo_points(&rows, &inventory)?;
    let (selected_keys, selected_key_sha256) =
        load_selected_keys(key_directory, &inventory, selected.threshold)?;
    let (membership, candidate_manifest_sha256) = build_membership_candidate(
        &output.join("selected-candidate"),
        selected,
        &selected_keys,
        &selected_key_sha256,
        name_totals,
    )?;
    let probes = evaluate_probes(
        corpus,
        probes_path,
        &scan.counts,
        &selected_keys,
        selected.threshold,
    )?;
    v7_error_row(&rows)?;
    validate_safety_probe(&probes)?;
    let result = SelectionResult {
        rows,
        points,
        logo,
        inventory,
        selected,
        membership,
        counts_sha256: scan.counts_sha256,
        selected_key_sha256,
        candidate_manifest_sha256,
        inventory_sha256: scan.inventory_sha256,
        probes,
    };
    let outputs = build_outputs(&result)?;
    let repeated = build_outputs(&result)?;
    if outputs != repeated {
        return Err("surname-index selection serialization is not deterministic".into());
    }
    for (name, bytes) in &outputs {
        fs::write(output.join(name), bytes)?;
    }
    Ok(String::from_utf8(
        outputs
            .get("selection_report.md")
            .ok_or("selection report missing")?
            .clone(),
    )?)
}

fn validate_and_order_holdouts(holdouts: Vec<FrozenHoldout>) -> Result<Vec<FrozenHoldout>> {
    let mut by_generation = BTreeMap::new();
    for holdout in holdouts {
        let generation = Generation::from_digest(&holdout.manifest.holdout_sha256)
            .ok_or("unrecognized spent holdout digest")?;
        if by_generation.insert(generation, holdout).is_some() {
            return Err("duplicate spent holdout generation".into());
        }
    }
    if by_generation.keys().copied().ne(Generation::ALL) {
        return Err("surname-index selection requires exactly REAL_PROXY_V1 through V7".into());
    }
    Ok(Generation::ALL
        .into_iter()
        .map(|generation| {
            by_generation
                .remove(&generation)
                .expect("validated generation")
        })
        .collect())
}

fn expected_lookup_keys(
    holdouts: &[FrozenHoldout],
    probes_path: Option<&Path>,
) -> Result<BTreeSet<String>> {
    let mut keys = BTreeSet::new();
    for holdout in holdouts {
        for case in &holdout.cases {
            extend_lookup_keys(&mut keys, &case.display_name);
        }
    }
    if let Some(path) = probes_path {
        let mut reader = csv::Reader::from_path(path)?;
        for record in reader.deserialize::<ProbeInput>() {
            extend_lookup_keys(&mut keys, &record?.display_name);
        }
    }
    Ok(keys)
}

fn extend_lookup_keys(keys: &mut BTreeSet<String>, display_name: &str) {
    let canonical = canonicalize(display_name);
    let tokens = canonical.split_whitespace().collect::<Vec<_>>();
    if tokens.len() == 2 && tokens_are_plain_alphabetic(&tokens) {
        keys.extend(tokens.into_iter().map(canonicalize));
    }
}

fn serialize_keys(keys: &BTreeSet<String>) -> Result<Vec<u8>> {
    let mut writer = canonical_writer();
    writer.write_record(["name"])?;
    for key in keys {
        writer.write_record([key])?;
    }
    Ok(writer.into_inner()?)
}

fn lookup_manifest_csv(
    holdouts: &[FrozenHoldout],
    key_count: usize,
    key_sha256: &str,
) -> Result<Vec<u8>> {
    let mut writer = canonical_writer();
    writer.write_record(["key", "value"])?;
    writer.write_record([
        "selection",
        "all_plain_alphabetic_tokens_from_v1_through_v7_plus_private_probes",
    ])?;
    writer.write_record(["target_keys", &key_count.to_string()])?;
    writer.write_record(["lookup_keys_sha256", key_sha256])?;
    for (generation, holdout) in Generation::ALL.iter().zip(holdouts) {
        writer.write_record([
            &format!("{}_sha256", generation.as_str().to_ascii_lowercase()),
            &holdout.manifest.holdout_sha256,
        ])?;
    }
    Ok(writer.into_inner()?)
}

fn load_targeted_scan(
    counts_path: &Path,
    manifest_path: &Path,
    expected_keys: &BTreeSet<String>,
    holdouts: &[FrozenHoldout],
) -> Result<TargetedScan> {
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
    validate_manifest_value(&manifest, "given_key_count", GIVEN_KEYS.to_string())?;
    validate_manifest_value(&manifest, "target_keys", expected_keys.len().to_string())?;
    validate_manifest_value(
        &manifest,
        "lookup_keys_sha256",
        sha256_hex(&serialize_keys(expected_keys)?),
    )?;
    validate_manifest_value(&manifest, "counts_sha256", &counts_sha256)?;
    let inventory_sha256 = manifest
        .get("surname_inventory_sha256")
        .ok_or("surname manifest is missing surname_inventory_sha256")?
        .to_string();
    validate_sha256(&inventory_sha256, "surname inventory digest")?;
    for (generation, holdout) in Generation::ALL.iter().zip(holdouts) {
        validate_manifest_value(
            &manifest,
            &format!("{}_sha256", generation.as_str().to_ascii_lowercase()),
            &holdout.manifest.holdout_sha256,
        )?;
    }
    let mut reader = csv::Reader::from_reader(bytes.as_slice());
    if reader.headers()?.iter().ne(["name", "surname_count"]) {
        return Err("unexpected targeted surname-count header".into());
    }
    let mut counts = BTreeMap::new();
    for record in reader.records() {
        let record = record?;
        let name = record.get(0).ok_or("missing targeted surname key")?;
        let count = record
            .get(1)
            .ok_or("missing targeted surname count")?
            .parse::<u64>()?;
        if counts.insert(name.to_string(), count).is_some() {
            return Err("duplicate targeted surname key".into());
        }
    }
    if counts.keys().ne(expected_keys.iter()) {
        return Err("targeted surname-count keys do not match V1-V7 lexical keys".into());
    }
    Ok(TargetedScan {
        counts,
        counts_sha256,
        inventory_sha256,
    })
}

fn load_inventory(path: &Path, expected_sha256: &str) -> Result<Vec<InventoryRow>> {
    let bytes = fs::read(path)?;
    if sha256_hex(&bytes) != expected_sha256 {
        return Err("surname-only inventory failed manifest authentication".into());
    }
    let mut reader = csv::Reader::from_reader(bytes.as_slice());
    if reader.headers()?.iter().ne([
        "surname_count_min",
        "key_count",
        "raw_utf8_bytes",
        "zstd19_bytes",
        "raw_sha256",
        "zstd_sha256",
    ]) {
        return Err("unexpected surname-only inventory header".into());
    }
    let mut rows = Vec::new();
    for record in reader.records() {
        let record = record?;
        rows.push(InventoryRow {
            threshold: parse_field(&record, 0, "inventory threshold")?,
            key_count: parse_field(&record, 1, "inventory key count")?,
            raw_utf8_bytes: parse_field(&record, 2, "inventory raw bytes")?,
            zstd19_bytes: parse_field(&record, 3, "inventory zstd bytes")?,
            raw_sha256: record
                .get(4)
                .ok_or("missing inventory raw digest")?
                .to_string(),
            zstd_sha256: record
                .get(5)
                .ok_or("missing inventory zstd digest")?
                .to_string(),
        });
    }
    if rows.len() != SURNAME_THRESHOLDS.len() {
        return Err("surname-only inventory does not contain the fixed threshold grid".into());
    }
    for (index, row) in rows.iter().enumerate() {
        if row.threshold != SURNAME_THRESHOLDS[index]
            || row.key_count != PRIOR_KEY_COUNTS[index]
            || row.raw_utf8_bytes != PRIOR_RAW_BYTES[index]
            || row.zstd19_bytes != PRIOR_ZSTD19_BYTES[index]
        {
            return Err(format!(
                "surname-only inventory failed prior-curve reproduction at threshold {}",
                SURNAME_THRESHOLDS[index]
            )
            .into());
        }
        validate_sha256(&row.raw_sha256, "inventory raw digest")?;
        validate_sha256(&row.zstd_sha256, "inventory zstd digest")?;
    }
    Ok(rows)
}

fn build_rows(
    corpus: &impl EvidenceSource,
    holdouts: &[FrozenHoldout],
    surname_counts: &BTreeMap<String, u64>,
) -> Result<Vec<SelectionRow>> {
    let mut rows = Vec::new();
    for (generation, holdout) in Generation::ALL.iter().copied().zip(holdouts) {
        for case in &holdout.cases {
            if !case.is_evaluable() {
                continue;
            }
            rows.push(build_row(
                corpus,
                generation,
                case.expected_greeting().map(str::to_string),
                &case.display_name,
                nonempty(&case.country_hint),
                nonempty(&case.locale_hint),
                surname_counts,
            )?);
        }
    }
    Ok(rows)
}

#[allow(clippy::too_many_arguments)]
fn build_row(
    corpus: &impl EvidenceSource,
    generation: Generation,
    expected_greeting: Option<String>,
    display_name: &str,
    country_hint: Option<&str>,
    locale_hint: Option<&str>,
    surname_counts: &BTreeMap<String, u64>,
) -> Result<SelectionRow> {
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
    let c5_emission = c5_emitted_candidate(&c6.c5).map(str::to_string);
    let first_position_emission = c6.emission_source == C6EmissionSource::FirstPosition;
    let winner = c6.c5.c4.c31.winner.as_ref();
    let candidate = diagnostic.candidates.first();
    let selected_candidate = winner.map(|winner| winner.greeting_candidate.clone());
    let canonical = canonicalize(display_name);
    let tokens = canonical.split_whitespace().collect::<Vec<_>>();
    let position = candidate_position(candidate, tokens.len());
    let complement = complement_index(position)
        .and_then(|index| tokens.get(index).copied())
        .map(canonicalize);
    let complement_lookup = complement.as_deref().map(|token| {
        expected_lookup_diagnostic(corpus, ALGORITHM_C3, token, country_hint, locale_hint)
    });
    let complement_surname_count = complement
        .as_ref()
        .and_then(|key| surname_counts.get(key).copied());
    let vetoes_pass = c6.first_position.vetoes_pass;
    let residual_topology = c5_emission.is_none()
        && !first_position_emission
        && c6.first_position.native_candidate
        && c6.first_position.exactly_two_alphabetic_tokens
        && c6.first_position.single_token_winner
        && c6.first_position.selected_first
        && c6.first_position.candidate_count_pass
        && winner.is_some_and(|winner| winner.winner_score >= 0.40)
        && winner.is_some_and(|winner| winner.reliability >= 0.00)
        && winner.is_some_and(|winner| winner.role_signal >= 0.30)
        && vetoes_pass
        && complement_lookup
            .as_ref()
            .is_some_and(|lookup| lookup.eligible);
    if residual_topology && complement_surname_count.is_none() {
        return Err("targeted surname scan is missing a residual complement".into());
    }
    Ok(SelectionRow {
        generation,
        expected_greeting,
        selected_candidate,
        first_position_emission,
        residual_topology,
        candidate_quality: winner.map_or(0.0, |winner| winner.winner_score),
        reliability: winner.map_or(0.0, |winner| winner.reliability),
        role_signal: winner.map_or(0.0, |winner| winner.role_signal),
        complement,
        complement_given_observed: complement_lookup
            .as_ref()
            .is_some_and(|lookup| lookup.evidence.is_some()),
        complement_surname_count,
        hard_organization_marker: c6.c5.c4.c31.hard_organization_marker,
        generic_organization_marker: c6.c5.c4.c31.generic_organization_marker,
        ampersand: c6.c5.c4.c31.ampersand,
        candidate_too_short: c6.c5.c4.c31.candidate_too_short,
    })
}

fn threshold_points(rows: &[SelectionRow], inventory: &[InventoryRow]) -> Vec<ThresholdPoint> {
    SURNAME_THRESHOLDS
        .into_iter()
        .zip(inventory)
        .map(|(threshold, inventory)| ThresholdPoint {
            threshold,
            metrics: evaluate_threshold(rows, threshold, |_| true),
            artifact_bytes: estimated_mphf_membership_bytes(inventory.key_count),
        })
        .collect()
}

fn logo_points(
    rows: &[SelectionRow],
    inventory: &[InventoryRow],
) -> Result<Vec<(Generation, ThresholdPoint, Metrics)>> {
    let mut output = Vec::new();
    for held_out in Generation::ALL {
        let training = SURNAME_THRESHOLDS
            .into_iter()
            .zip(inventory)
            .map(|(threshold, inventory)| ThresholdPoint {
                threshold,
                metrics: evaluate_threshold(rows, threshold, |row| row.generation != held_out),
                artifact_bytes: estimated_mphf_membership_bytes(inventory.key_count),
            })
            .collect::<Vec<_>>();
        let selected = select_point(&training).ok_or("LOGO threshold grid is empty")?;
        let held_out_metrics =
            evaluate_threshold(rows, selected.threshold, |row| row.generation == held_out);
        output.push((held_out, selected, held_out_metrics));
    }
    Ok(output)
}

fn evaluate_threshold(
    rows: &[SelectionRow],
    threshold: u64,
    include: impl Fn(&SelectionRow) -> bool,
) -> Metrics {
    let mut metrics = Metrics::default();
    for row in rows.iter().filter(|row| include(row)) {
        metrics.observe(
            row.expected_greeting.as_deref(),
            row.residual_emission(threshold),
        );
    }
    metrics
}

fn select_point(points: &[ThresholdPoint]) -> Option<ThresholdPoint> {
    points.iter().copied().min_by(compare_points)
}

fn compare_points(left: &ThresholdPoint, right: &ThresholdPoint) -> Ordering {
    left.metrics
        .wrong
        .cmp(&right.metrics.wrong)
        .then_with(|| {
            left.metrics
                .null_false_emissions
                .cmp(&right.metrics.null_false_emissions)
        })
        .then_with(|| right.metrics.correct.cmp(&left.metrics.correct))
        .then_with(|| left.artifact_bytes.cmp(&right.artifact_bytes))
        .then_with(|| right.threshold.cmp(&left.threshold))
}

fn load_selected_keys(
    directory: &Path,
    inventory: &[InventoryRow],
    threshold: u64,
) -> Result<(Vec<String>, String)> {
    let inventory = inventory
        .iter()
        .find(|row| row.threshold == threshold)
        .ok_or("selected threshold is absent from inventory")?;
    let path = directory.join(format!("surname-only-{threshold}.txt.zst"));
    let compressed = fs::read(&path)?;
    if compressed.len() != inventory.zstd19_bytes
        || sha256_hex(&compressed) != inventory.zstd_sha256
    {
        return Err("selected compressed surname-key file failed authentication".into());
    }
    let output = Command::new("zstd")
        .args(["-d", "-q", "-c"])
        .arg(&path)
        .output()?;
    if !output.status.success() {
        return Err(format!("zstd decompression failed with {}", output.status).into());
    }
    let bytes = output.stdout;
    if bytes.len() != inventory.raw_utf8_bytes || sha256_hex(&bytes) != inventory.raw_sha256 {
        return Err("selected raw surname-key file failed authentication".into());
    }
    let text = String::from_utf8(bytes)?;
    let keys = text.lines().map(str::to_string).collect::<Vec<_>>();
    if keys.len() != inventory.key_count
        || keys.windows(2).any(|pair| pair[0] >= pair[1])
        || keys.iter().any(String::is_empty)
    {
        return Err("selected surname keys are not a unique sorted set".into());
    }
    Ok((keys, inventory.raw_sha256.clone()))
}

fn build_membership_candidate(
    directory: &Path,
    selected: ThresholdPoint,
    keys: &[String],
    key_sha256: &str,
    name_totals: &Path,
) -> Result<(MembershipStats, String)> {
    fs::create_dir(directory)?;
    let routing = keys
        .iter()
        .map(|key| xxh3_64_with_seed(key.as_bytes(), ROUTING_SEED))
        .collect::<Vec<_>>();
    let unique_routing = routing.iter().copied().collect::<HashSet<_>>();
    if unique_routing.len() != routing.len() {
        return Err("selected surname keys have a 64-bit routing-hash collision".into());
    }
    let first = build_membership_index(keys, &routing)?;
    let second = build_membership_index(keys, &routing)?;
    if first.1 != second.1 || first.2 != second.2 {
        return Err("selected membership candidate is not byte-deterministic".into());
    }
    let (_, mphf_bytes, fingerprint_bytes) = first;
    fs::write(directory.join("names.mphf"), &mphf_bytes)?;
    fs::write(directory.join("fingerprints.u32"), &fingerprint_bytes)?;
    let index = load_membership_candidate(directory, keys.len())?;
    for key in keys {
        if !index.contains(key) {
            return Err("round-tripped membership candidate missed a member".into());
        }
    }

    let mut bloom_reference = BloomFilter::new(keys.len(), BLOOM_REFERENCE_FPR);
    let mut bloom_fingerprint = BloomFilter::new(keys.len(), BLOOM_FINGERPRINT_FPR);
    for key in keys {
        bloom_reference.insert(key);
        bloom_fingerprint.insert(key);
    }
    let negative_stats = test_negative_membership(
        name_totals,
        keys,
        &index,
        &bloom_reference,
        &bloom_fingerprint,
    )?;
    let manifest = candidate_manifest(
        selected,
        keys.len(),
        key_sha256,
        &mphf_bytes,
        &fingerprint_bytes,
        &negative_stats,
        bloom_reference.bits.len(),
        bloom_fingerprint.bits.len(),
    )?;
    let manifest_sha256 = sha256_hex(&manifest);
    fs::write(directory.join("manifest.csv"), &manifest)?;
    let membership = MembershipStats {
        mphf_bytes: mphf_bytes.len(),
        fingerprint_bytes: fingerprint_bytes.len(),
        manifest_bytes: manifest.len(),
        total_bytes: mphf_bytes.len() + fingerprint_bytes.len() + manifest.len(),
        given_negative_queries: negative_stats.0,
        given_false_accepts: negative_stats.1,
        generated_negative_queries: negative_stats.2,
        generated_false_accepts: negative_stats.3,
        bloom_reference_bytes: bloom_reference.bits.len(),
        bloom_reference_false_accepts: negative_stats.4,
        bloom_fingerprint_bytes: bloom_fingerprint.bits.len(),
        bloom_fingerprint_false_accepts: negative_stats.5,
    };
    Ok((membership, manifest_sha256))
}

pub(super) fn load_frozen_membership_candidate(
    directory: &Path,
) -> Result<FrozenSurnameMembership> {
    let actual_files = fs::read_dir(directory)?
        .map(|entry| {
            entry?
                .file_name()
                .into_string()
                .map_err(|_| "surname candidate contains a non-UTF-8 filename".into())
        })
        .collect::<Result<BTreeSet<_>>>()?;
    let expected_files = ["fingerprints.u32", "manifest.csv", "names.mphf"]
        .into_iter()
        .map(str::to_string)
        .collect::<BTreeSet<_>>();
    if actual_files != expected_files {
        return Err("frozen surname candidate has unexpected constituents".into());
    }

    let manifest_path = directory.join("manifest.csv");
    let manifest_bytes = fs::read(&manifest_path)?;
    if manifest_bytes.len() != FROZEN_CANDIDATE_MANIFEST_BYTES
        || sha256_hex(&manifest_bytes) != FROZEN_CANDIDATE_MANIFEST_SHA256
    {
        return Err("frozen surname candidate manifest failed authentication".into());
    }
    let manifest = load_key_value_manifest(&manifest_path)?;
    let expected_manifest = [
        ("format", "surname-only-membership-candidate-v1"),
        ("surname_count_min", "1"),
        ("key_count", "35417044"),
        ("source_keys_sha256", FROZEN_CANDIDATE_SOURCE_KEYS_SHA256),
        ("routing_seed", "0x6e616d652d726f75"),
        ("fingerprint_seed", "0x6e616d652d667033"),
        ("mphf_gamma", "1.7"),
        ("names_mphf_bytes", "15248560"),
        ("names_mphf_sha256", FROZEN_CANDIDATE_MPHF_SHA256),
        ("fingerprints_bytes", "141668176"),
        ("fingerprints_sha256", FROZEN_CANDIDATE_FINGERPRINT_SHA256),
        ("given_negative_queries", "1803175"),
        ("given_false_accepts", "0"),
        ("generated_negative_queries", "100000"),
        ("generated_false_accepts", "0"),
        ("bloom_0_001_bytes", "63651457"),
        ("bloom_0_001_false_accepts", "1932"),
        ("bloom_2^-32_bytes", "204383975"),
        ("bloom_2^-32_false_accepts", "0"),
    ];
    if manifest.len() != expected_manifest.len() {
        return Err("frozen surname candidate manifest has unexpected fields".into());
    }
    for (key, expected) in expected_manifest {
        validate_manifest_value(&manifest, key, expected)?;
    }

    let mphf_bytes = fs::read(directory.join("names.mphf"))?;
    if mphf_bytes.len() != FROZEN_CANDIDATE_MPHF_BYTES
        || sha256_hex(&mphf_bytes) != FROZEN_CANDIDATE_MPHF_SHA256
    {
        return Err("frozen surname candidate MPHF failed authentication".into());
    }
    let fingerprint_bytes = fs::read(directory.join("fingerprints.u32"))?;
    if fingerprint_bytes.len() != FROZEN_CANDIDATE_FINGERPRINT_BYTES
        || sha256_hex(&fingerprint_bytes) != FROZEN_CANDIDATE_FINGERPRINT_SHA256
    {
        return Err("frozen surname candidate fingerprints failed authentication".into());
    }
    let index = decode_membership_candidate(
        &mphf_bytes,
        &fingerprint_bytes,
        FROZEN_CANDIDATE_FINGERPRINT_BYTES / size_of::<u32>(),
    )?;
    Ok(FrozenSurnameMembership { index })
}

fn load_membership_candidate(directory: &Path, key_count: usize) -> Result<MembershipIndex> {
    let mphf_bytes = fs::read(directory.join("names.mphf"))?;
    let fingerprint_bytes = fs::read(directory.join("fingerprints.u32"))?;
    decode_membership_candidate(&mphf_bytes, &fingerprint_bytes, key_count)
}

fn decode_membership_candidate(
    mphf_bytes: &[u8],
    fingerprint_bytes: &[u8],
    key_count: usize,
) -> Result<MembershipIndex> {
    let mphf = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .deserialize(mphf_bytes)?;
    if fingerprint_bytes.len() != key_count * size_of::<u32>() {
        return Err("round-tripped fingerprint length is invalid".into());
    }
    let fingerprints = fingerprint_bytes
        .chunks_exact(size_of::<u32>())
        .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect::<Vec<_>>();
    Ok(MembershipIndex { mphf, fingerprints })
}

fn build_membership_index(
    keys: &[String],
    routing: &[u64],
) -> Result<(MembershipIndex, Vec<u8>, Vec<u8>)> {
    let mphf = Mphf::new_parallel(MPHF_GAMMA, routing, None);
    let mut fingerprints = vec![0_u32; keys.len()];
    let mut occupied = vec![false; keys.len()];
    for (key, routing) in keys.iter().zip(routing) {
        let slot = usize::try_from(mphf.hash(routing))?;
        if occupied[slot] {
            return Err("MPHF assigned duplicate member slots".into());
        }
        occupied[slot] = true;
        fingerprints[slot] = xxh3_64_with_seed(key.as_bytes(), FINGERPRINT_SEED) as u32;
    }
    let mphf_bytes = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .serialize(&mphf)?;
    let fingerprint_bytes = fingerprints
        .iter()
        .flat_map(|fingerprint| fingerprint.to_le_bytes())
        .collect::<Vec<_>>();
    Ok((
        MembershipIndex { mphf, fingerprints },
        mphf_bytes,
        fingerprint_bytes,
    ))
}

fn test_negative_membership(
    name_totals: &Path,
    selected_keys: &[String],
    index: &MembershipIndex,
    bloom_reference: &BloomFilter,
    bloom_fingerprint: &BloomFilter,
) -> Result<(usize, usize, usize, usize, usize, usize)> {
    let mut reader = csv::Reader::from_path(name_totals)?;
    if reader
        .headers()?
        .iter()
        .ne(["name", "given_count", "as_surname_count"])
    {
        return Err("unexpected name-totals header".into());
    }
    let mut given_queries = 0;
    let mut given_false_accepts = 0;
    let mut bloom_reference_false_accepts = 0;
    let mut bloom_fingerprint_false_accepts = 0;
    for record in reader.records() {
        let record = record?;
        let key = record.get(0).ok_or("missing retained given-name key")?;
        if selected_keys
            .binary_search_by(|candidate| candidate.as_str().cmp(key))
            .is_ok()
        {
            return Err("selected surname-only set duplicates a retained given-name key".into());
        }
        given_queries += 1;
        given_false_accepts += usize::from(index.contains(key));
        bloom_reference_false_accepts += usize::from(bloom_reference.contains(key));
        bloom_fingerprint_false_accepts += usize::from(bloom_fingerprint.contains(key));
    }
    if given_queries != GIVEN_KEYS || given_false_accepts != 0 {
        return Err("retained given-name membership-negative validation failed".into());
    }
    let mut generated_false_accepts = 0;
    for index_value in 0..GENERATED_NEGATIVES {
        let key = format!("definitely-not-a-surname-{index_value:06}");
        generated_false_accepts += usize::from(index.contains(&key));
        bloom_reference_false_accepts += usize::from(bloom_reference.contains(&key));
        bloom_fingerprint_false_accepts += usize::from(bloom_fingerprint.contains(&key));
    }
    if generated_false_accepts != 0 {
        return Err("generated membership-negative validation failed".into());
    }
    Ok((
        given_queries,
        given_false_accepts,
        GENERATED_NEGATIVES,
        generated_false_accepts,
        bloom_reference_false_accepts,
        bloom_fingerprint_false_accepts,
    ))
}

#[allow(clippy::too_many_arguments)]
fn candidate_manifest(
    selected: ThresholdPoint,
    key_count: usize,
    key_sha256: &str,
    mphf_bytes: &[u8],
    fingerprint_bytes: &[u8],
    negative_stats: &(usize, usize, usize, usize, usize, usize),
    bloom_reference_bytes: usize,
    bloom_fingerprint_bytes: usize,
) -> Result<Vec<u8>> {
    let mut writer = canonical_writer();
    writer.write_record(["key", "value"])?;
    for (key, value) in [
        ("format", "surname-only-membership-candidate-v1".to_string()),
        ("surname_count_min", selected.threshold.to_string()),
        ("key_count", key_count.to_string()),
        ("source_keys_sha256", key_sha256.to_string()),
        ("routing_seed", format!("0x{ROUTING_SEED:016x}")),
        ("fingerprint_seed", format!("0x{FINGERPRINT_SEED:016x}")),
        ("mphf_gamma", format!("{MPHF_GAMMA:.1}")),
        ("names_mphf_bytes", mphf_bytes.len().to_string()),
        ("names_mphf_sha256", sha256_hex(mphf_bytes)),
        ("fingerprints_bytes", fingerprint_bytes.len().to_string()),
        ("fingerprints_sha256", sha256_hex(fingerprint_bytes)),
        ("given_negative_queries", negative_stats.0.to_string()),
        ("given_false_accepts", negative_stats.1.to_string()),
        ("generated_negative_queries", negative_stats.2.to_string()),
        ("generated_false_accepts", negative_stats.3.to_string()),
        ("bloom_0_001_bytes", bloom_reference_bytes.to_string()),
        ("bloom_0_001_false_accepts", negative_stats.4.to_string()),
        ("bloom_2^-32_bytes", bloom_fingerprint_bytes.to_string()),
        ("bloom_2^-32_false_accepts", negative_stats.5.to_string()),
    ] {
        writer.write_record([key, &value])?;
    }
    Ok(writer.into_inner()?)
}

fn evaluate_probes(
    corpus: &impl EvidenceSource,
    path: &Path,
    surname_counts: &BTreeMap<String, u64>,
    selected_keys: &[String],
    threshold: u64,
) -> Result<Vec<ProbeResult>> {
    let mut reader = csv::Reader::from_path(path)?;
    let mut probes = Vec::new();
    for record in reader.deserialize::<ProbeInput>() {
        let record = record?;
        let row = build_row(
            corpus,
            Generation::V7,
            None,
            &record.display_name,
            None,
            None,
            surname_counts,
        )?;
        let membership = row.complement.as_ref().is_some_and(|key| {
            selected_keys
                .binary_search_by(|candidate| candidate.as_str().cmp(key))
                .is_ok()
        });
        let expected_membership = !row.complement_given_observed
            && row
                .complement_surname_count
                .is_some_and(|count| count >= threshold);
        if membership != expected_membership {
            return Err(
                "qualitative probe membership disagrees with the selected surname set".into(),
            );
        }
        probes.push(ProbeResult {
            label: record.label,
            selected_candidate: row.selected_candidate,
            candidate_quality: row.candidate_quality,
            role_signal: row.role_signal,
            reliability: row.reliability,
            complement_surname_count: row.complement_surname_count,
            complement_in_selected_index: membership,
            first_position_emits: row.first_position_emission,
            surname_residual_emits: row.residual_topology && membership,
            vetoes_pass: !row.hard_organization_marker
                && !row.generic_organization_marker
                && !row.ampersand
                && !row.candidate_too_short,
        });
    }
    if probes.len() != 4 {
        return Err("qualitative probes do not match the frozen selection inputs".into());
    }
    Ok(probes)
}

fn validate_safety_probe(probes: &[ProbeResult]) -> Result<()> {
    let motorcycle = probes
        .iter()
        .find(|probe| probe.label == "Motorcycle Club")
        .ok_or("Motorcycle Club safety probe is missing")?;
    if motorcycle.first_position_emits || motorcycle.surname_residual_emits {
        return Err("Motorcycle Club did not remain an abstention".into());
    }
    Ok(())
}

fn build_outputs(result: &SelectionResult) -> Result<BTreeMap<&'static str, Vec<u8>>> {
    let mut outputs = BTreeMap::new();
    outputs.insert("thresholds_pooled.csv", thresholds_pooled_csv(result)?);
    outputs.insert(
        "thresholds_by_generation.csv",
        thresholds_by_generation_csv(result)?,
    );
    outputs.insert("threshold_logo.csv", threshold_logo_csv(result)?);
    outputs.insert("selected_candidate.csv", selected_candidate_csv(result)?);
    outputs.insert("v7_error_diagnostic.csv", v7_error_csv(result)?);
    outputs.insert(
        "representation_sizes.csv",
        representation_sizes_csv(result)?,
    );
    outputs.insert("qualitative_probes.csv", qualitative_probes_csv(result)?);
    outputs.insert("run_manifest.csv", run_manifest_csv(result)?);
    outputs.insert("selection_report.md", report(result)?.into_bytes());
    Ok(outputs)
}

fn thresholds_pooled_csv(result: &SelectionResult) -> Result<Vec<u8>> {
    let mut writer = metrics_writer()?;
    for point in &result.points {
        write_metrics(&mut writer, point.threshold.to_string(), point.metrics)?;
    }
    Ok(writer.into_inner()?)
}

fn thresholds_by_generation_csv(result: &SelectionResult) -> Result<Vec<u8>> {
    let mut writer = canonical_writer();
    writer.write_record([
        "generation",
        "surname_count_min",
        "additional_emitted",
        "additional_correct",
        "additional_wrong",
        "additional_null_false_emissions",
    ])?;
    for generation in Generation::ALL {
        for threshold in SURNAME_THRESHOLDS {
            let metrics =
                evaluate_threshold(&result.rows, threshold, |row| row.generation == generation);
            writer.write_record([
                generation.as_str().to_string(),
                threshold.to_string(),
                metrics.emitted.to_string(),
                metrics.correct.to_string(),
                metrics.wrong.to_string(),
                metrics.null_false_emissions.to_string(),
            ])?;
        }
    }
    Ok(writer.into_inner()?)
}

fn threshold_logo_csv(result: &SelectionResult) -> Result<Vec<u8>> {
    let mut writer = canonical_writer();
    writer.write_record([
        "held_out",
        "selected_surname_count_min",
        "training_correct",
        "training_wrong",
        "training_null_false_emissions",
        "held_out_correct",
        "held_out_wrong",
        "held_out_null_false_emissions",
    ])?;
    for (generation, selected, held_out) in &result.logo {
        writer.write_record([
            generation.as_str().to_string(),
            selected.threshold.to_string(),
            selected.metrics.correct.to_string(),
            selected.metrics.wrong.to_string(),
            selected.metrics.null_false_emissions.to_string(),
            held_out.correct.to_string(),
            held_out.wrong.to_string(),
            held_out.null_false_emissions.to_string(),
        ])?;
    }
    Ok(writer.into_inner()?)
}

fn selected_candidate_csv(result: &SelectionResult) -> Result<Vec<u8>> {
    let mut writer = canonical_writer();
    writer.write_record(["key", "value"])?;
    for (key, value) in [
        ("surname_count_min", result.selected.threshold.to_string()),
        (
            "additional_correct",
            result.selected.metrics.correct.to_string(),
        ),
        (
            "additional_wrong",
            result.selected.metrics.wrong.to_string(),
        ),
        (
            "additional_null_false_emissions",
            result.selected.metrics.null_false_emissions.to_string(),
        ),
        (
            "surname_only_keys",
            result
                .inventory
                .iter()
                .find(|row| row.threshold == result.selected.threshold)
                .expect("selected inventory")
                .key_count
                .to_string(),
        ),
        ("candidate_bytes", result.membership.total_bytes.to_string()),
        (
            "combined_name_data_bytes",
            (EXISTING_ARTIFACT_BYTES + result.membership.total_bytes).to_string(),
        ),
        ("source_keys_sha256", result.selected_key_sha256.clone()),
        (
            "candidate_manifest_sha256",
            result.candidate_manifest_sha256.clone(),
        ),
    ] {
        writer.write_record([key, &value])?;
    }
    Ok(writer.into_inner()?)
}

fn v7_error_csv(result: &SelectionResult) -> Result<Vec<u8>> {
    let row = v7_error_row(&result.rows)?;
    let mut writer = canonical_writer();
    writer.write_record([
        "category",
        "raw_surname_count",
        "candidate_quality",
        "role_signal",
        "reliability",
        "selected_position",
        "hard_organization_marker",
        "generic_organization_marker",
        "ampersand",
        "candidate_too_short",
    ])?;
    writer.write_record([
        if row.expected_greeting.is_none() {
            "expected_null"
        } else {
            "wrong_greeting"
        },
        &row.complement_surname_count.unwrap_or(0).to_string(),
        &format!("{:.6}", row.candidate_quality),
        &format!("{:.6}", row.role_signal),
        &format!("{:.6}", row.reliability),
        "first",
        bool_string(row.hard_organization_marker),
        bool_string(row.generic_organization_marker),
        bool_string(row.ampersand),
        bool_string(row.candidate_too_short),
    ])?;
    Ok(writer.into_inner()?)
}

fn representation_sizes_csv(result: &SelectionResult) -> Result<Vec<u8>> {
    let mut writer = canonical_writer();
    writer.write_record([
        "surname_count_min",
        "qualifying_keys",
        "sorted_raw_utf8_bytes",
        "zstd19_bytes",
        "estimated_mphf_fingerprint_bytes",
        "estimated_mphf_fingerprint_evidence_bytes",
        "bloom_0_001_bytes",
        "bloom_2^-32_bytes",
        "actual_selected_mphf_fingerprint_manifest_bytes",
    ])?;
    for row in &result.inventory {
        writer.write_record([
            row.threshold.to_string(),
            row.key_count.to_string(),
            row.raw_utf8_bytes.to_string(),
            row.zstd19_bytes.to_string(),
            estimated_mphf_membership_bytes(row.key_count).to_string(),
            (estimated_mphf_membership_bytes(row.key_count) + row.key_count).to_string(),
            bloom_bytes(row.key_count, BLOOM_REFERENCE_FPR).to_string(),
            bloom_bytes(row.key_count, BLOOM_FINGERPRINT_FPR).to_string(),
            if row.threshold == result.selected.threshold {
                result.membership.total_bytes.to_string()
            } else {
                String::new()
            },
        ])?;
    }
    Ok(writer.into_inner()?)
}

fn qualitative_probes_csv(result: &SelectionResult) -> Result<Vec<u8>> {
    let mut writer = canonical_writer();
    writer.write_record([
        "probe",
        "selected_candidate",
        "candidate_quality",
        "role_signal",
        "reliability",
        "raw_surname_count",
        "complement_in_selected_index",
        "first_position_emits",
        "selected_surname_residual_emits",
        "vetoes_pass",
    ])?;
    for probe in &result.probes {
        writer.write_record([
            probe.label.clone(),
            probe.selected_candidate.clone().unwrap_or_default(),
            format!("{:.6}", probe.candidate_quality),
            format!("{:.6}", probe.role_signal),
            format!("{:.6}", probe.reliability),
            probe
                .complement_surname_count
                .map_or_else(String::new, |count| count.to_string()),
            bool_string(probe.complement_in_selected_index).to_string(),
            bool_string(probe.first_position_emits).to_string(),
            bool_string(probe.surname_residual_emits).to_string(),
            bool_string(probe.vetoes_pass).to_string(),
        ])?;
    }
    Ok(writer.into_inner()?)
}

fn run_manifest_csv(result: &SelectionResult) -> Result<Vec<u8>> {
    let mut writer = canonical_writer();
    writer.write_record(["key", "value"])?;
    for (key, value) in [
        ("spent_generations", "REAL_PROXY_V1-V7".to_string()),
        ("surname_counts_sha256", result.counts_sha256.clone()),
        ("surname_inventory_sha256", result.inventory_sha256.clone()),
        (
            "threshold_grid",
            "1/2/5/10/25/50/100/250/500/1000".to_string(),
        ),
        ("selected_threshold", result.selected.threshold.to_string()),
        ("row_level_output", "forbidden".to_string()),
        ("v8_created", "false".to_string()),
        ("surname_production_integration", "false".to_string()),
    ] {
        writer.write_record([key, &value])?;
    }
    Ok(writer.into_inner()?)
}

fn report(result: &SelectionResult) -> Result<String> {
    let v7_error = v7_error_row(&result.rows)?;
    let mut report = String::new();
    writeln!(report, "# Production-shaped surname-only index selection\n").unwrap();
    writeln!(report, "REAL_PROXY_V1 through V7 are spent development evidence. The selected-candidate, topology, position, and veto conditions are frozen; only the ten declared raw surname-count thresholds were compared. No V8 data was created or inspected.\n").unwrap();
    writeln!(report, "| Minimum surname count | Correct | Wrong | NULL FP | Keys | Estimated membership bytes |\n|---:|---:|---:|---:|---:|---:|").unwrap();
    for (point, inventory) in result.points.iter().zip(&result.inventory) {
        writeln!(
            report,
            "| {} | {} | {} | {} | {} | {} |",
            point.threshold,
            point.metrics.correct,
            point.metrics.wrong,
            point.metrics.null_false_emissions,
            inventory.key_count,
            point.artifact_bytes
        )
        .unwrap();
    }
    writeln!(report, "\nThe declared lexicographic safety objective selected `surname_count >= {}`: **+{} correct / {} wrong / {} NULL FP** beyond production C6.\n", result.selected.threshold, result.selected.metrics.correct, result.selected.metrics.wrong, result.selected.metrics.null_false_emissions).unwrap();
    writeln!(report, "## Per-generation fixed-threshold stability\n").unwrap();
    write!(report, "| Generation |").unwrap();
    for threshold in SURNAME_THRESHOLDS {
        write!(report, " >={threshold} |").unwrap();
    }
    write!(report, "\n|---|").unwrap();
    for _ in SURNAME_THRESHOLDS {
        write!(report, "---:|").unwrap();
    }
    writeln!(report).unwrap();
    for generation in Generation::ALL {
        write!(report, "| {} |", generation.as_str()).unwrap();
        for threshold in SURNAME_THRESHOLDS {
            let metrics =
                evaluate_threshold(&result.rows, threshold, |row| row.generation == generation);
            write!(
                report,
                " {}/{}/{} |",
                metrics.correct, metrics.wrong, metrics.null_false_emissions
            )
            .unwrap();
        }
        writeln!(report).unwrap();
    }
    writeln!(report, "\nEach cell is `correct/wrong/NULL FP`. V1-V6 have no error at any fixed threshold; V7 retains the same single NULL false emission throughout the declared grid.\n").unwrap();
    writeln!(report, "## Leave-one-generation-out stability\n").unwrap();
    writeln!(report, "| Held out | Selected count | Training correct/wrong/NULL | Held-out correct/wrong/NULL |\n|---|---:|---:|---:|").unwrap();
    for (generation, selected, held_out) in &result.logo {
        writeln!(
            report,
            "| {} | {} | {}/{}/{} | {}/{}/{} |",
            generation.as_str(),
            selected.threshold,
            selected.metrics.correct,
            selected.metrics.wrong,
            selected.metrics.null_false_emissions,
            held_out.correct,
            held_out.wrong,
            held_out.null_false_emissions
        )
        .unwrap();
    }
    writeln!(report, "\n## Representation\n").unwrap();
    writeln!(report, "The selected candidate is MPHF + independent 32-bit fingerprint membership with no evidence byte. It contains {} MPHF bytes, {} fingerprint bytes, and {} manifest bytes: **{} bytes total**. Combined with the unchanged {}-byte given-name artifact, the runtime name-data footprint would be **{} bytes** after a future integration.\n", result.membership.mphf_bytes, result.membership.fingerprint_bytes, result.membership.manifest_bytes, result.membership.total_bytes, EXISTING_ARTIFACT_BYTES, EXISTING_ARTIFACT_BYTES + result.membership.total_bytes).unwrap();
    writeln!(report, "The fingerprint's nominal false-accept probability is `2^-32` per unrelated query. All {} retained given-name keys and {} deterministic generated nonmembers produced {} observed MPHF/fingerprint false accepts. Bloom membership at `10^-3` uses {} bytes and produced {} false accepts over the same negative checks; Bloom at `2^-32` uses {} bytes and produced {}. Bloom is not selected.\n", result.membership.given_negative_queries, result.membership.generated_negative_queries, result.membership.given_false_accepts + result.membership.generated_false_accepts, result.membership.bloom_reference_bytes, result.membership.bloom_reference_false_accepts, result.membership.bloom_fingerprint_bytes, result.membership.bloom_fingerprint_false_accepts).unwrap();
    writeln!(report, "The spent V7 false emission has raw complement-surname count {}, candidate quality {:.6}, role signal {:.6}, reliability {:.6}, first-token selection, and no active organization/personhood veto. Its count exceeds the grid maximum, so no declared threshold excludes it. The diagnostic is serialized only in abstract numeric form in `v7_error_diagnostic.csv`; no row string or identifier is written.\n", v7_error.complement_surname_count.unwrap_or(0), v7_error.candidate_quality, v7_error.role_signal, v7_error.reliability).unwrap();
    writeln!(report, "## Qualitative probes\n").unwrap();
    writeln!(report, "| Probe | Selected | Raw surname count | In selected index | First-position emits | Residual emits | Vetoes pass |\n|---|---|---:|---|---|---|---|").unwrap();
    for probe in &result.probes {
        writeln!(
            report,
            "| {} | {} | {} | {} | {} | {} | {} |",
            probe.label,
            probe.selected_candidate.as_deref().unwrap_or(""),
            probe
                .complement_surname_count
                .map_or_else(String::new, |count| count.to_string()),
            bool_string(probe.complement_in_selected_index),
            bool_string(probe.first_position_emits),
            bool_string(probe.surname_residual_emits),
            bool_string(probe.vetoes_pass)
        )
        .unwrap();
    }
    writeln!(report, "\nThe person probes are diagnostic only. Martin REDACTED and Olivier REDACTED already emit through C6 and their complements are members; Baris REDACTED remains outside the sole-candidate topology. `Motorcycle Club` has surname membership at the selected count-1 threshold but remains vetoed and does not emit.\n").unwrap();
    writeln!(report, "## Frozen V8 candidate\n").unwrap();
    writeln!(report, "Validate unchanged production C6 against C6 plus this additive branch: native/non-segmented, exactly two alphabetic tokens, a single-token sole winner selected first, C6 otherwise abstains, quality `>= 0.40`, reliability `>= 0.00`, role signal `>= 0.30`, every existing veto passes, complement absent from the retained given index, and complement present in the separate surname-only membership candidate built from exact raw surname count `>= {}`.\n", result.selected.threshold).unwrap();
    writeln!(report, "The surname branch remains experimental and is not loaded by production. The candidate manifest SHA-256 is `{}` and its source-key SHA-256 is `{}`. Stop here; V8 is the next separate task.", result.candidate_manifest_sha256, result.selected_key_sha256).unwrap();
    Ok(report)
}

fn v7_error_row(rows: &[SelectionRow]) -> Result<&SelectionRow> {
    let errors = rows
        .iter()
        .filter(|row| row.generation == Generation::V7)
        .filter(|row| row.residual_emission(1).is_some())
        .filter(|row| {
            !row.expected_greeting.as_deref().is_some_and(|expected| {
                greeting_matches(Some(expected), row.selected_candidate.as_deref())
            })
        })
        .collect::<Vec<_>>();
    if errors.len() != 1 {
        return Err(format!(
            "expected one spent V7 surname-residual error, found {}",
            errors.len()
        )
        .into());
    }
    Ok(errors[0])
}

fn metrics_writer() -> Result<csv::Writer<Vec<u8>>> {
    let mut writer = canonical_writer();
    writer.write_record([
        "surname_count_min",
        "additional_emitted",
        "additional_correct",
        "additional_wrong",
        "additional_null_false_emissions",
    ])?;
    Ok(writer)
}

fn write_metrics(writer: &mut csv::Writer<Vec<u8>>, label: String, metrics: Metrics) -> Result<()> {
    writer.write_record([
        label,
        metrics.emitted.to_string(),
        metrics.correct.to_string(),
        metrics.wrong.to_string(),
        metrics.null_false_emissions.to_string(),
    ])?;
    Ok(())
}

fn parse_field<T: std::str::FromStr>(
    record: &csv::StringRecord,
    index: usize,
    label: &str,
) -> Result<T>
where
    T::Err: Error + 'static,
{
    Ok(record
        .get(index)
        .ok_or_else(|| format!("missing {label}"))?
        .parse()?)
}

fn load_key_value_manifest(path: &Path) -> Result<BTreeMap<String, String>> {
    let mut reader = csv::Reader::from_path(path)?;
    if reader.headers()?.iter().ne(["key", "value"]) {
        return Err("unexpected surname scan manifest header".into());
    }
    let mut manifest = BTreeMap::new();
    for record in reader.records() {
        let record = record?;
        let key = record.get(0).ok_or("missing surname manifest key")?;
        let value = record.get(1).ok_or("missing surname manifest value")?;
        if manifest
            .insert(key.to_string(), value.to_string())
            .is_some()
        {
            return Err("duplicate surname manifest key".into());
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
        .ok_or_else(|| format!("surname manifest is missing {key}"))?;
    if actual != expected.as_ref() {
        return Err(format!(
            "surname manifest {key} mismatch: expected {}, got {actual}",
            expected.as_ref()
        )
        .into());
    }
    Ok(())
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("{label} must be a SHA-256 digest").into());
    }
    Ok(())
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

fn tokens_are_plain_alphabetic(tokens: &[&str]) -> bool {
    tokens
        .iter()
        .all(|token| !token.is_empty() && token.chars().all(char::is_alphabetic))
}

fn estimated_mphf_membership_bytes(keys: usize) -> usize {
    let mphf_bytes_per_key = 777_304.0 / 1_803_175.0;
    ((mphf_bytes_per_key + 4.0) * keys as f64).ceil() as usize
}

fn bloom_bytes(keys: usize, false_positive_rate: f64) -> usize {
    let bits_per_key = -false_positive_rate.ln() / 2_f64.ln().powi(2);
    (bits_per_key * keys as f64 / 8.0).ceil() as usize
}

fn canonical_writer() -> csv::Writer<Vec<u8>> {
    csv::WriterBuilder::new()
        .terminator(csv::Terminator::Any(b'\n'))
        .from_writer(Vec::new())
}

fn bool_string(value: bool) -> &'static str {
    if value { "true" } else { "false" }
}

fn nonempty(value: &str) -> Option<&str> {
    (!value.is_empty()).then_some(value)
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(
        threshold: u64,
        correct: usize,
        wrong: usize,
        nulls: usize,
        bytes: usize,
    ) -> ThresholdPoint {
        ThresholdPoint {
            threshold,
            metrics: Metrics {
                emitted: correct + wrong,
                correct,
                wrong,
                null_false_emissions: nulls,
            },
            artifact_bytes: bytes,
        }
    }

    #[test]
    fn selection_objective_is_safety_first_then_recall_then_size() {
        let points = [
            point(1, 100, 1, 1, 10),
            point(2, 90, 0, 0, 20),
            point(5, 91, 0, 0, 30),
            point(10, 91, 0, 0, 15),
        ];
        assert_eq!(select_point(&points).unwrap().threshold, 10);
    }

    #[test]
    fn selection_minimizes_errors_when_no_zero_error_point_exists() {
        let points = [point(1, 100, 2, 1, 10), point(2, 80, 1, 1, 20)];
        assert_eq!(select_point(&points).unwrap().threshold, 2);
    }

    #[test]
    fn bloom_filter_has_no_member_misses() {
        let keys = ["Alpha", "Beta", "Gamma"];
        let mut bloom = BloomFilter::new(keys.len(), BLOOM_REFERENCE_FPR);
        for key in keys {
            bloom.insert(key);
        }
        assert!(keys.into_iter().all(|key| bloom.contains(key)));
    }

    #[test]
    fn lookup_key_serialization_is_sorted_and_canonical() {
        let keys = BTreeSet::from(["Zulu".to_string(), "Alpha".to_string()]);
        assert_eq!(
            String::from_utf8(serialize_keys(&keys).unwrap()).unwrap(),
            "name\nAlpha\nZulu\n"
        );
    }

    #[test]
    fn membership_round_trip_preserves_members_and_rejects_malformed_fingerprints() {
        let keys = ["Alpha".to_string(), "Beta".to_string()];
        let routing = keys
            .iter()
            .map(|key| xxh3_64_with_seed(key.as_bytes(), ROUTING_SEED))
            .collect::<Vec<_>>();
        let (_, mphf, fingerprints) = build_membership_index(&keys, &routing).unwrap();
        let decoded = decode_membership_candidate(&mphf, &fingerprints, keys.len()).unwrap();
        assert!(keys.iter().all(|key| decoded.contains(key)));
        assert!(!decoded.contains("DefinitelyAbsent"));
        assert!(decode_membership_candidate(&mphf, &fingerprints[..4], keys.len()).is_err());
    }

    #[test]
    fn frozen_candidate_metadata_is_exact() {
        assert_eq!(PRIOR_KEY_COUNTS[0], 35_417_044);
        assert_eq!(
            FROZEN_CANDIDATE_MPHF_BYTES
                + FROZEN_CANDIDATE_FINGERPRINT_BYTES
                + FROZEN_CANDIDATE_MANIFEST_BYTES,
            156_917_446
        );
        for digest in [
            FROZEN_CANDIDATE_MANIFEST_SHA256,
            FROZEN_CANDIDATE_SOURCE_KEYS_SHA256,
            FROZEN_CANDIDATE_MPHF_SHA256,
            FROZEN_CANDIDATE_FINGERPRINT_SHA256,
        ] {
            validate_sha256(digest, "frozen candidate digest").unwrap();
        }
    }
}
