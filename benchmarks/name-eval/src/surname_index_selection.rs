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
pub(super) const V8_SHA256: &str =
    "55fe9ae0efc7e604e55c997f8c26cd2cfc3e97961514a6a3780cd2f0420ae6c9";
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
const COMPACT_MIN_CORRECT: usize = 5;
const MIB: usize = 1_048_576;
const COMPACT_SIZE_CAPS_MIB: [usize; 5] = [1, 2, 4, 8, 16];
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
const FROZEN_COMPACT_CANDIDATE_MANIFEST_BYTES: usize = 728;
const FROZEN_COMPACT_CANDIDATE_MANIFEST_SHA256: &str =
    "33fbd24462f13b980b0f8a0a8618edcf6209c8032be6875116a472224adf3fb8";
const FROZEN_COMPACT_CANDIDATE_SOURCE_KEYS_SHA256: &str =
    "b5734b8ff0bc6bba300558b30be73c8644db76ab2e823915968422fe028fbdc5";
const FROZEN_COMPACT_CANDIDATE_MPHF_BYTES: usize = 737_024;
const FROZEN_COMPACT_CANDIDATE_MPHF_SHA256: &str =
    "832244b41604149295a1a26f879a633598a144e5a11731bc42597716913dc5c1";
const FROZEN_COMPACT_CANDIDATE_FINGERPRINT_BYTES: usize = 6_837_588;
const FROZEN_COMPACT_CANDIDATE_FINGERPRINT_SHA256: &str =
    "183639758511ba02a8824e4a419b41f7ff6de46530db90524608051824e25ceb";
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
    V8,
}

impl Generation {
    const V1_TO_V7: [Self; 7] = [
        Self::V1,
        Self::V2,
        Self::V3,
        Self::V4,
        Self::V5,
        Self::V6,
        Self::V7,
    ];
    const V1_TO_V8: [Self; 8] = [
        Self::V1,
        Self::V2,
        Self::V3,
        Self::V4,
        Self::V5,
        Self::V6,
        Self::V7,
        Self::V8,
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
            V8_SHA256 => Some(Self::V8),
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
            Self::V8 => "REAL_PROXY_V8",
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

    fn residual_membership_emission(&self, member: bool) -> Option<&str> {
        (self.would_query_surname_index() && member)
            .then_some(self.selected_candidate.as_deref())
            .flatten()
    }

    fn would_query_surname_index(&self) -> bool {
        self.residual_topology && !self.complement_given_observed
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

#[derive(Clone, Debug, Eq, PartialEq)]
struct ActualMembershipStats {
    threshold: u64,
    key_count: usize,
    mphf_bytes: usize,
    fingerprint_bytes: usize,
    manifest_bytes: usize,
    total_bytes: usize,
    mphf_sha256: String,
    fingerprint_sha256: String,
    manifest_sha256: String,
    member_queries: usize,
    member_misses: usize,
    given_negative_queries: usize,
    given_false_accepts: usize,
    generated_negative_queries: usize,
    generated_false_accepts: usize,
    zstd19_mphf_bytes: usize,
    zstd19_fingerprint_bytes: usize,
    zstd19_total_bytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CompactThresholdPoint {
    threshold: u64,
    metrics: Metrics,
    inventory: InventoryRow,
    membership: ActualMembershipStats,
    membership_hits: Vec<bool>,
}

impl CompactThresholdPoint {
    fn retained_correct_denominator(points: &[Self]) -> Result<usize> {
        let denominator = points
            .iter()
            .find(|point| point.threshold == 1)
            .ok_or("compact threshold frontier is missing count 1")?
            .metrics
            .correct;
        if denominator == 0 {
            return Err("count-1 compact threshold has zero correct additions".into());
        }
        Ok(denominator)
    }

    fn retained_at_least_half(&self, denominator: usize) -> bool {
        self.metrics.correct.saturating_mul(2) >= denominator
    }
}

#[derive(Clone, Debug)]
struct CompactSelectionResult {
    rows: Vec<SelectionRow>,
    points: Vec<CompactThresholdPoint>,
    logo: Vec<(Generation, CompactThresholdPoint, Metrics)>,
    selected: CompactThresholdPoint,
    counts_sha256: String,
    inventory_sha256: String,
    probes: Vec<ProbeResult>,
    selected_reproduction_sha256: String,
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
    metadata: FrozenMembershipMetadata,
}

#[derive(Clone, Copy)]
struct FrozenMembershipMetadata {
    surname_count_min: u64,
    key_count: usize,
    manifest_bytes: usize,
    manifest_sha256: &'static str,
    source_keys_sha256: &'static str,
    mphf_bytes: usize,
    mphf_sha256: &'static str,
    fingerprint_bytes: usize,
    fingerprint_sha256: &'static str,
}

impl FrozenSurnameMembership {
    pub(super) fn contains(&self, key: &str) -> bool {
        self.index.contains(key)
    }

    pub(super) const fn surname_count_min(&self) -> u64 {
        self.metadata.surname_count_min
    }

    pub(super) const fn key_count(&self) -> usize {
        self.metadata.key_count
    }

    pub(super) const fn artifact_bytes(&self) -> usize {
        self.metadata.mphf_bytes + self.metadata.fingerprint_bytes + self.metadata.manifest_bytes
    }

    pub(super) const fn manifest_sha256(&self) -> &'static str {
        self.metadata.manifest_sha256
    }

    pub(super) const fn source_keys_sha256(&self) -> &'static str {
        self.metadata.source_keys_sha256
    }

    pub(super) const fn mphf_sha256(&self) -> &'static str {
        self.metadata.mphf_sha256
    }

    pub(super) const fn fingerprint_sha256(&self) -> &'static str {
        self.metadata.fingerprint_sha256
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
    let holdouts = validate_and_order_holdouts(holdouts, &Generation::V1_TO_V7)?;
    let keys = expected_lookup_keys(&holdouts, Some(probes_path))?;
    let bytes = serialize_keys(&keys)?;
    let digest = sha256_hex(&bytes);
    fs::write(output.join("lookup_keys.csv"), &bytes)?;
    fs::write(
        output.join("lookup_manifest.csv"),
        lookup_manifest_csv(
            &holdouts,
            &Generation::V1_TO_V7,
            "all_plain_alphabetic_tokens_from_v1_through_v7_plus_private_probes",
            keys.len(),
            &digest,
        )?,
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
    let holdouts = validate_and_order_holdouts(holdouts, &Generation::V1_TO_V7)?;
    let expected_keys = expected_lookup_keys(&holdouts, Some(probes_path))?;
    let scan = load_targeted_scan(
        surname_counts,
        surname_manifest,
        &expected_keys,
        &holdouts,
        &Generation::V1_TO_V7,
    )?;
    let inventory = load_inventory(inventory_path, &scan.inventory_sha256)?;
    let rows = build_rows(corpus, &holdouts, &Generation::V1_TO_V7, &scan.counts)?;
    let points = threshold_points(&rows, &inventory);
    let selected = select_point(&points).ok_or("surname threshold grid is empty")?;
    let logo = logo_points(&rows, &inventory, &Generation::V1_TO_V7, select_point)?;
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

pub(crate) fn prepare_compact_surname_index_selection(
    output: &Path,
    holdouts: Vec<FrozenHoldout>,
    probes_path: &Path,
) -> Result<String> {
    let holdouts = validate_and_order_holdouts(holdouts, &Generation::V1_TO_V8)?;
    let keys = expected_lookup_keys(&holdouts, Some(probes_path))?;
    let bytes = serialize_keys(&keys)?;
    let digest = sha256_hex(&bytes);
    fs::write(output.join("lookup_keys.csv"), &bytes)?;
    fs::write(
        output.join("lookup_manifest.csv"),
        lookup_manifest_csv(
            &holdouts,
            &Generation::V1_TO_V8,
            "all_plain_alphabetic_tokens_from_v1_through_v8_plus_private_probes",
            keys.len(),
            &digest,
        )?,
    )?;
    let mut report = String::new();
    writeln!(
        report,
        "# Private spent V1-V8 compact surname lookup preparation\n"
    )
    .unwrap();
    writeln!(report, "All eight holdouts were checksum-verified before exporting a lexical superset of {} distinct keys. The lookup-key SHA-256 is `{digest}`. This unredacted material must remain under ignored `_wip/`.", keys.len()).unwrap();
    Ok(report)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_compact_surname_index_selection(
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
    let holdouts = validate_and_order_holdouts(holdouts, &Generation::V1_TO_V8)?;
    let expected_keys = expected_lookup_keys(&holdouts, Some(probes_path))?;
    let scan = load_targeted_scan(
        surname_counts,
        surname_manifest,
        &expected_keys,
        &holdouts,
        &Generation::V1_TO_V8,
    )?;
    let inventory = load_inventory(inventory_path, &scan.inventory_sha256)?;
    let rows = build_rows(corpus, &holdouts, &Generation::V1_TO_V8, &scan.counts)?;
    let points =
        build_compact_threshold_points(output, &rows, &inventory, key_directory, name_totals)?;
    let selected = select_compact_point(&points)?;
    if selected.membership.given_false_accepts != 0
        || selected.membership.generated_false_accepts != 0
    {
        return Err("selected compact candidate has an observed fingerprint false accept".into());
    }
    let logo = compact_logo_points(&rows, &points)?;
    let selected_reproduction_sha256 =
        reproduce_selected_compact_candidate(output, &selected, key_directory, name_totals)?;
    let selected_keys = load_selected_keys(key_directory, &inventory, selected.threshold)?.0;
    let probes = evaluate_probes(
        corpus,
        probes_path,
        &scan.counts,
        &selected_keys,
        selected.threshold,
    )?;
    validate_safety_probe(&probes)?;
    retain_selected_compact_candidate(output, selected.threshold)?;
    let result = CompactSelectionResult {
        rows,
        points,
        logo,
        selected,
        counts_sha256: scan.counts_sha256,
        inventory_sha256: scan.inventory_sha256,
        probes,
        selected_reproduction_sha256,
    };
    let outputs = build_compact_outputs(&result)?;
    let repeated = build_compact_outputs(&result)?;
    if outputs != repeated {
        return Err("compact surname-index selection serialization is not deterministic".into());
    }
    for (name, bytes) in &outputs {
        fs::write(output.join(name), bytes)?;
    }
    Ok(String::from_utf8(
        outputs
            .get("selection_report.md")
            .ok_or("compact selection report missing")?
            .clone(),
    )?)
}

fn validate_and_order_holdouts(
    holdouts: Vec<FrozenHoldout>,
    required: &[Generation],
) -> Result<Vec<FrozenHoldout>> {
    let mut by_generation = BTreeMap::new();
    for holdout in holdouts {
        let generation = Generation::from_digest(&holdout.manifest.holdout_sha256)
            .ok_or("unrecognized spent holdout digest")?;
        if by_generation.insert(generation, holdout).is_some() {
            return Err("duplicate spent holdout generation".into());
        }
    }
    if by_generation.keys().copied().ne(required.iter().copied()) {
        let last = required
            .last()
            .ok_or("surname-index selection requires at least one generation")?;
        return Err(format!(
            "surname-index selection requires exactly REAL_PROXY_V1 through {}",
            last.as_str()
        )
        .into());
    }
    Ok(required
        .iter()
        .copied()
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
    generations: &[Generation],
    selection: &str,
    key_count: usize,
    key_sha256: &str,
) -> Result<Vec<u8>> {
    let mut writer = canonical_writer();
    writer.write_record(["key", "value"])?;
    writer.write_record(["selection", selection])?;
    writer.write_record(["target_keys", &key_count.to_string()])?;
    writer.write_record(["lookup_keys_sha256", key_sha256])?;
    for (generation, holdout) in generations.iter().zip(holdouts) {
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
    generations: &[Generation],
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
    for (generation, holdout) in generations.iter().zip(holdouts) {
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
    generations: &[Generation],
    surname_counts: &BTreeMap<String, u64>,
) -> Result<Vec<SelectionRow>> {
    let mut rows = Vec::new();
    for (generation, holdout) in generations.iter().copied().zip(holdouts) {
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
    generations: &[Generation],
    select: fn(&[ThresholdPoint]) -> Option<ThresholdPoint>,
) -> Result<Vec<(Generation, ThresholdPoint, Metrics)>> {
    let mut output = Vec::new();
    for held_out in generations.iter().copied() {
        let training = SURNAME_THRESHOLDS
            .into_iter()
            .zip(inventory)
            .map(|(threshold, inventory)| ThresholdPoint {
                threshold,
                metrics: evaluate_threshold(rows, threshold, |row| row.generation != held_out),
                artifact_bytes: estimated_mphf_membership_bytes(inventory.key_count),
            })
            .collect::<Vec<_>>();
        let selected = select(&training).ok_or("LOGO threshold grid has no selectable point")?;
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

fn evaluate_membership_hits(
    rows: &[SelectionRow],
    membership_hits: &[bool],
    include: impl Fn(&SelectionRow) -> bool,
) -> Result<Metrics> {
    if rows.len() != membership_hits.len() {
        return Err("compact membership decisions do not align with selection rows".into());
    }
    let mut metrics = Metrics::default();
    for (row, member) in rows.iter().zip(membership_hits) {
        if !include(row) {
            continue;
        }
        let emission = row.residual_membership_emission(*member);
        metrics.observe(row.expected_greeting.as_deref(), emission);
    }
    Ok(metrics)
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

fn build_compact_threshold_points(
    output: &Path,
    rows: &[SelectionRow],
    inventory: &[InventoryRow],
    key_directory: &Path,
    name_totals: &Path,
) -> Result<Vec<CompactThresholdPoint>> {
    let candidates = output.join("candidates");
    fs::create_dir(&candidates)?;
    let mut points = Vec::with_capacity(SURNAME_THRESHOLDS.len());
    for inventory in inventory {
        let threshold = inventory.threshold;
        let (keys, key_sha256) =
            load_selected_keys(key_directory, std::slice::from_ref(inventory), threshold)?;
        let raw_metrics = evaluate_threshold(rows, threshold, |_| true);
        let (membership, index) = build_actual_membership_candidate(
            &candidates.join(format!("count-{threshold}")),
            ThresholdPoint {
                threshold,
                metrics: raw_metrics,
                artifact_bytes: 0,
            },
            &keys,
            &key_sha256,
            name_totals,
        )?;
        let membership_hits = rows
            .iter()
            .map(|row| {
                row.complement
                    .as_deref()
                    .is_some_and(|key| index.contains(key))
            })
            .collect::<Vec<_>>();
        let metrics = evaluate_membership_hits(rows, &membership_hits, |_| true)?;
        points.push(CompactThresholdPoint {
            threshold,
            metrics,
            inventory: inventory.clone(),
            membership,
            membership_hits,
        });
    }
    Ok(points)
}

fn select_compact_point(points: &[CompactThresholdPoint]) -> Result<CompactThresholdPoint> {
    let denominator = CompactThresholdPoint::retained_correct_denominator(points)?;
    let meaningful = points
        .iter()
        .filter(|point| point.metrics.correct >= COMPACT_MIN_CORRECT)
        .collect::<Vec<_>>();
    let minimum_wrong = meaningful
        .iter()
        .map(|point| point.metrics.wrong)
        .min()
        .ok_or("compact threshold grid has no meaningfully useful point")?;
    let minimum_null = meaningful
        .iter()
        .filter(|point| point.metrics.wrong == minimum_wrong)
        .map(|point| point.metrics.null_false_emissions)
        .min()
        .ok_or("compact threshold grid has no safety-optimal point")?;
    meaningful
        .into_iter()
        .filter(|point| {
            point.metrics.wrong == minimum_wrong
                && point.metrics.null_false_emissions == minimum_null
                && point.retained_at_least_half(denominator)
        })
        .min_by(|left, right| compare_compact_sizes(left, right))
        .cloned()
        .ok_or_else(|| {
            "no safety-optimal compact threshold retains at least half of count-1 benefit".into()
        })
}

fn compare_compact_sizes(left: &CompactThresholdPoint, right: &CompactThresholdPoint) -> Ordering {
    left.membership
        .total_bytes
        .cmp(&right.membership.total_bytes)
        .then_with(|| right.metrics.correct.cmp(&left.metrics.correct))
        .then_with(|| right.threshold.cmp(&left.threshold))
}

fn compact_logo_points(
    rows: &[SelectionRow],
    points: &[CompactThresholdPoint],
) -> Result<Vec<(Generation, CompactThresholdPoint, Metrics)>> {
    let mut output = Vec::new();
    for held_out in Generation::V1_TO_V8 {
        let mut training = Vec::with_capacity(points.len());
        for point in points {
            training.push(CompactThresholdPoint {
                threshold: point.threshold,
                metrics: evaluate_membership_hits(rows, &point.membership_hits, |row| {
                    row.generation != held_out
                })?,
                inventory: point.inventory.clone(),
                membership: point.membership.clone(),
                membership_hits: point.membership_hits.clone(),
            });
        }
        let selected = select_compact_point(&training)?;
        let held_out_metrics = evaluate_membership_hits(rows, &selected.membership_hits, |row| {
            row.generation == held_out
        })?;
        output.push((held_out, selected, held_out_metrics));
    }
    Ok(output)
}

fn point_is_pareto(points: &[CompactThresholdPoint], candidate: &CompactThresholdPoint) -> bool {
    !points.iter().any(|other| {
        other.threshold != candidate.threshold
            && other.metrics.wrong <= candidate.metrics.wrong
            && other.metrics.null_false_emissions <= candidate.metrics.null_false_emissions
            && other.metrics.correct >= candidate.metrics.correct
            && other.membership.total_bytes <= candidate.membership.total_bytes
            && (other.metrics.wrong < candidate.metrics.wrong
                || other.metrics.null_false_emissions < candidate.metrics.null_false_emissions
                || other.metrics.correct > candidate.metrics.correct
                || other.membership.total_bytes < candidate.membership.total_bytes)
    })
}

fn best_point_under_cap(
    points: &[CompactThresholdPoint],
    cap_bytes: usize,
) -> Option<&CompactThresholdPoint> {
    points
        .iter()
        .filter(|point| point.membership.total_bytes <= cap_bytes)
        .min_by(|left, right| {
            left.metrics
                .wrong
                .cmp(&right.metrics.wrong)
                .then_with(|| {
                    left.metrics
                        .null_false_emissions
                        .cmp(&right.metrics.null_false_emissions)
                })
                .then_with(|| right.metrics.correct.cmp(&left.metrics.correct))
                .then_with(|| compare_compact_sizes(left, right))
        })
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

fn build_actual_membership_candidate(
    directory: &Path,
    point: ThresholdPoint,
    keys: &[String],
    key_sha256: &str,
    name_totals: &Path,
) -> Result<(ActualMembershipStats, MembershipIndex)> {
    fs::create_dir(directory)?;
    let routing = keys
        .iter()
        .map(|key| xxh3_64_with_seed(key.as_bytes(), ROUTING_SEED))
        .collect::<Vec<_>>();
    let unique_routing = routing.iter().copied().collect::<HashSet<_>>();
    if unique_routing.len() != routing.len() {
        return Err("compact surname keys have a 64-bit routing-hash collision".into());
    }
    let first = build_membership_index(keys, &routing)?;
    let second = build_membership_index(keys, &routing)?;
    if first.1 != second.1 || first.2 != second.2 {
        return Err("compact surname membership is not byte-deterministic".into());
    }
    let (_, mphf_bytes, fingerprint_bytes) = first;
    fs::write(directory.join("names.mphf"), &mphf_bytes)?;
    fs::write(directory.join("fingerprints.u32"), &fingerprint_bytes)?;
    let index = load_membership_candidate(directory, keys.len())?;
    let member_misses = keys.iter().filter(|key| !index.contains(key)).count();
    if member_misses != 0 {
        return Err("round-tripped compact membership candidate missed a member".into());
    }
    let negative_stats = test_core_negative_membership(name_totals, keys, &index)?;
    let manifest = if point.threshold == 1 {
        let legacy_stats = (
            negative_stats.0,
            negative_stats.1,
            negative_stats.2,
            negative_stats.3,
            1_932,
            0,
        );
        candidate_manifest(
            point,
            keys.len(),
            key_sha256,
            &mphf_bytes,
            &fingerprint_bytes,
            &legacy_stats,
            bloom_bytes(keys.len(), BLOOM_REFERENCE_FPR),
            bloom_bytes(keys.len(), BLOOM_FINGERPRINT_FPR),
        )?
    } else {
        compact_candidate_manifest(
            point.threshold,
            keys.len(),
            key_sha256,
            &mphf_bytes,
            &fingerprint_bytes,
            negative_stats,
        )?
    };
    fs::write(directory.join("manifest.csv"), &manifest)?;
    let stats = ActualMembershipStats {
        threshold: point.threshold,
        key_count: keys.len(),
        mphf_bytes: mphf_bytes.len(),
        fingerprint_bytes: fingerprint_bytes.len(),
        manifest_bytes: manifest.len(),
        total_bytes: mphf_bytes.len() + fingerprint_bytes.len() + manifest.len(),
        mphf_sha256: sha256_hex(&mphf_bytes),
        fingerprint_sha256: sha256_hex(&fingerprint_bytes),
        manifest_sha256: sha256_hex(&manifest),
        member_queries: keys.len(),
        member_misses,
        given_negative_queries: negative_stats.0,
        given_false_accepts: negative_stats.1,
        generated_negative_queries: negative_stats.2,
        generated_false_accepts: negative_stats.3,
        zstd19_mphf_bytes: compress_for_shipping(directory, "names.mphf")?,
        zstd19_fingerprint_bytes: compress_for_shipping(directory, "fingerprints.u32")?,
        zstd19_total_bytes: 0,
    };
    let stats = ActualMembershipStats {
        zstd19_total_bytes: stats.zstd19_mphf_bytes
            + stats.zstd19_fingerprint_bytes
            + stats.manifest_bytes,
        ..stats
    };
    authenticate_actual_candidate(directory, &stats)?;
    if point.threshold == 1 {
        validate_rebuilt_count_one_candidate(&stats, key_sha256)?;
    }
    Ok((stats, index))
}

fn test_core_negative_membership(
    name_totals: &Path,
    selected_keys: &[String],
    index: &MembershipIndex,
) -> Result<(usize, usize, usize, usize)> {
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
    for record in reader.records() {
        let record = record?;
        let key = record.get(0).ok_or("missing retained given-name key")?;
        if selected_keys
            .binary_search_by(|candidate| candidate.as_str().cmp(key))
            .is_ok()
        {
            return Err("compact surname-only set duplicates a retained given-name key".into());
        }
        given_queries += 1;
        given_false_accepts += usize::from(index.contains(key));
    }
    if given_queries != GIVEN_KEYS {
        return Err("retained given-name negative-query count changed".into());
    }
    let generated_false_accepts = (0..GENERATED_NEGATIVES)
        .filter(|value| index.contains(&format!("definitely-not-a-surname-{value:06}")))
        .count();
    Ok((
        given_queries,
        given_false_accepts,
        GENERATED_NEGATIVES,
        generated_false_accepts,
    ))
}

fn compact_candidate_manifest(
    threshold: u64,
    key_count: usize,
    key_sha256: &str,
    mphf_bytes: &[u8],
    fingerprint_bytes: &[u8],
    negative_stats: (usize, usize, usize, usize),
) -> Result<Vec<u8>> {
    let mut writer = canonical_writer();
    writer.write_record(["key", "value"])?;
    for (key, value) in [
        (
            "format",
            "compact-surname-membership-candidate-v1".to_string(),
        ),
        ("surname_count_min", threshold.to_string()),
        ("key_count", key_count.to_string()),
        ("source_keys_sha256", key_sha256.to_string()),
        ("routing_seed", format!("0x{ROUTING_SEED:016x}")),
        ("fingerprint_seed", format!("0x{FINGERPRINT_SEED:016x}")),
        ("mphf_gamma", format!("{MPHF_GAMMA:.1}")),
        ("fingerprint_bits", "32".to_string()),
        ("stores_surname_count", "false".to_string()),
        ("names_mphf_bytes", mphf_bytes.len().to_string()),
        ("names_mphf_sha256", sha256_hex(mphf_bytes)),
        ("fingerprints_bytes", fingerprint_bytes.len().to_string()),
        ("fingerprints_sha256", sha256_hex(fingerprint_bytes)),
        ("member_queries", key_count.to_string()),
        ("member_misses", "0".to_string()),
        ("given_negative_queries", negative_stats.0.to_string()),
        ("given_false_accepts", negative_stats.1.to_string()),
        ("generated_negative_queries", negative_stats.2.to_string()),
        ("generated_false_accepts", negative_stats.3.to_string()),
        (
            "nominal_unknown_false_accept_probability",
            "2^-32".to_string(),
        ),
    ] {
        writer.write_record([key, &value])?;
    }
    Ok(writer.into_inner()?)
}

fn compress_for_shipping(directory: &Path, filename: &str) -> Result<usize> {
    let source = directory.join(filename);
    let compressed = directory.join(format!("{filename}.zst"));
    let status = Command::new("zstd")
        .args(["-19", "--long", "-q", "-f"])
        .arg(&source)
        .arg("-o")
        .arg(&compressed)
        .status()?;
    if !status.success() {
        return Err(format!("zstd shipping measurement failed with {status}").into());
    }
    Ok(usize::try_from(fs::metadata(compressed)?.len())?)
}

fn authenticate_actual_candidate(directory: &Path, stats: &ActualMembershipStats) -> Result<()> {
    for (filename, bytes, digest) in [
        ("names.mphf", stats.mphf_bytes, stats.mphf_sha256.as_str()),
        (
            "fingerprints.u32",
            stats.fingerprint_bytes,
            stats.fingerprint_sha256.as_str(),
        ),
        (
            "manifest.csv",
            stats.manifest_bytes,
            stats.manifest_sha256.as_str(),
        ),
    ] {
        let contents = fs::read(directory.join(filename))?;
        if contents.len() != bytes || sha256_hex(&contents) != digest {
            return Err(format!("compact candidate authentication failed for {filename}").into());
        }
    }
    Ok(())
}

fn validate_rebuilt_count_one_candidate(
    stats: &ActualMembershipStats,
    key_sha256: &str,
) -> Result<()> {
    if stats.key_count != PRIOR_KEY_COUNTS[0]
        || key_sha256 != FROZEN_CANDIDATE_SOURCE_KEYS_SHA256
        || stats.mphf_bytes != FROZEN_CANDIDATE_MPHF_BYTES
        || stats.mphf_sha256 != FROZEN_CANDIDATE_MPHF_SHA256
        || stats.fingerprint_bytes != FROZEN_CANDIDATE_FINGERPRINT_BYTES
        || stats.fingerprint_sha256 != FROZEN_CANDIDATE_FINGERPRINT_SHA256
        || stats.manifest_bytes != FROZEN_CANDIDATE_MANIFEST_BYTES
        || stats.manifest_sha256 != FROZEN_CANDIDATE_MANIFEST_SHA256
    {
        return Err("rebuilt count-1 candidate differs from the V8 frozen artifact".into());
    }
    Ok(())
}

fn reproduce_selected_compact_candidate(
    output: &Path,
    selected: &CompactThresholdPoint,
    key_directory: &Path,
    name_totals: &Path,
) -> Result<String> {
    let reproduction = output.join("selected-reproduction");
    let (keys, key_sha256) = load_selected_keys(
        key_directory,
        std::slice::from_ref(&selected.inventory),
        selected.threshold,
    )?;
    let (rebuilt, _) = build_actual_membership_candidate(
        &reproduction,
        ThresholdPoint {
            threshold: selected.threshold,
            metrics: selected.metrics,
            artifact_bytes: selected.membership.total_bytes,
        },
        &keys,
        &key_sha256,
        name_totals,
    )?;
    if rebuilt != selected.membership {
        return Err("selected compact candidate did not reproduce byte-for-byte".into());
    }
    let mut receipt = String::new();
    writeln!(receipt, "threshold={}", selected.threshold).unwrap();
    writeln!(receipt, "manifest_sha256={}", rebuilt.manifest_sha256).unwrap();
    writeln!(receipt, "mphf_sha256={}", rebuilt.mphf_sha256).unwrap();
    writeln!(receipt, "fingerprint_sha256={}", rebuilt.fingerprint_sha256).unwrap();
    writeln!(receipt, "byte_identical=true").unwrap();
    let digest = sha256_hex(receipt.as_bytes());
    fs::write(output.join("selected_reproduction.txt"), receipt)?;
    fs::remove_dir_all(&reproduction)?;
    Ok(digest)
}

fn retain_selected_compact_candidate(output: &Path, threshold: u64) -> Result<()> {
    let candidates = output.join("candidates");
    let selected = candidates.join(format!("count-{threshold}"));
    for filename in ["names.mphf.zst", "fingerprints.u32.zst"] {
        fs::remove_file(selected.join(filename))?;
    }
    fs::rename(&selected, output.join("selected-candidate"))?;
    fs::remove_dir_all(candidates)?;
    Ok(())
}

pub(super) fn load_frozen_membership_candidate(
    directory: &Path,
) -> Result<FrozenSurnameMembership> {
    let metadata = FrozenMembershipMetadata {
        surname_count_min: 1,
        key_count: PRIOR_KEY_COUNTS[0],
        manifest_bytes: FROZEN_CANDIDATE_MANIFEST_BYTES,
        manifest_sha256: FROZEN_CANDIDATE_MANIFEST_SHA256,
        source_keys_sha256: FROZEN_CANDIDATE_SOURCE_KEYS_SHA256,
        mphf_bytes: FROZEN_CANDIDATE_MPHF_BYTES,
        mphf_sha256: FROZEN_CANDIDATE_MPHF_SHA256,
        fingerprint_bytes: FROZEN_CANDIDATE_FINGERPRINT_BYTES,
        fingerprint_sha256: FROZEN_CANDIDATE_FINGERPRINT_SHA256,
    };
    load_authenticated_membership_candidate(
        directory,
        metadata,
        &[
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
            (
                "fingerprints_sha256",
                FROZEN_CANDIDATE_FINGERPRINT_SHA256,
            ),
            ("given_negative_queries", "1803175"),
            ("given_false_accepts", "0"),
            ("generated_negative_queries", "100000"),
            ("generated_false_accepts", "0"),
            ("bloom_0_001_bytes", "63651457"),
            ("bloom_0_001_false_accepts", "1932"),
            ("bloom_2^-32_bytes", "204383975"),
            ("bloom_2^-32_false_accepts", "0"),
        ],
    )
}

pub(super) fn load_frozen_compact_membership_candidate(
    directory: &Path,
) -> Result<FrozenSurnameMembership> {
    let metadata = FrozenMembershipMetadata {
        surname_count_min: 10,
        key_count: PRIOR_KEY_COUNTS[3],
        manifest_bytes: FROZEN_COMPACT_CANDIDATE_MANIFEST_BYTES,
        manifest_sha256: FROZEN_COMPACT_CANDIDATE_MANIFEST_SHA256,
        source_keys_sha256: FROZEN_COMPACT_CANDIDATE_SOURCE_KEYS_SHA256,
        mphf_bytes: FROZEN_COMPACT_CANDIDATE_MPHF_BYTES,
        mphf_sha256: FROZEN_COMPACT_CANDIDATE_MPHF_SHA256,
        fingerprint_bytes: FROZEN_COMPACT_CANDIDATE_FINGERPRINT_BYTES,
        fingerprint_sha256: FROZEN_COMPACT_CANDIDATE_FINGERPRINT_SHA256,
    };
    load_authenticated_membership_candidate(
        directory,
        metadata,
        &[
            ("format", "compact-surname-membership-candidate-v1"),
            ("surname_count_min", "10"),
            ("key_count", "1709397"),
            (
                "source_keys_sha256",
                FROZEN_COMPACT_CANDIDATE_SOURCE_KEYS_SHA256,
            ),
            ("routing_seed", "0x6e616d652d726f75"),
            ("fingerprint_seed", "0x6e616d652d667033"),
            ("mphf_gamma", "1.7"),
            ("fingerprint_bits", "32"),
            ("stores_surname_count", "false"),
            ("names_mphf_bytes", "737024"),
            (
                "names_mphf_sha256",
                FROZEN_COMPACT_CANDIDATE_MPHF_SHA256,
            ),
            ("fingerprints_bytes", "6837588"),
            (
                "fingerprints_sha256",
                FROZEN_COMPACT_CANDIDATE_FINGERPRINT_SHA256,
            ),
            ("member_queries", "1709397"),
            ("member_misses", "0"),
            ("given_negative_queries", "1803175"),
            ("given_false_accepts", "0"),
            ("generated_negative_queries", "100000"),
            ("generated_false_accepts", "0"),
            ("nominal_unknown_false_accept_probability", "2^-32"),
        ],
    )
}

fn load_authenticated_membership_candidate(
    directory: &Path,
    metadata: FrozenMembershipMetadata,
    expected_manifest: &[(&str, &str)],
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
    if manifest_bytes.len() != metadata.manifest_bytes
        || sha256_hex(&manifest_bytes) != metadata.manifest_sha256
    {
        return Err("frozen surname candidate manifest failed authentication".into());
    }
    let manifest = load_key_value_manifest(&manifest_path)?;
    if manifest.len() != expected_manifest.len() {
        return Err("frozen surname candidate manifest has unexpected fields".into());
    }
    for &(key, expected) in expected_manifest {
        validate_manifest_value(&manifest, key, expected)?;
    }

    let mphf_bytes = fs::read(directory.join("names.mphf"))?;
    if mphf_bytes.len() != metadata.mphf_bytes
        || sha256_hex(&mphf_bytes) != metadata.mphf_sha256
    {
        return Err("frozen surname candidate MPHF failed authentication".into());
    }
    let fingerprint_bytes = fs::read(directory.join("fingerprints.u32"))?;
    if fingerprint_bytes.len() != metadata.fingerprint_bytes
        || sha256_hex(&fingerprint_bytes) != metadata.fingerprint_sha256
    {
        return Err("frozen surname candidate fingerprints failed authentication".into());
    }
    let index = decode_membership_candidate(
        &mphf_bytes,
        &fingerprint_bytes,
        metadata.key_count,
    )?;
    Ok(FrozenSurnameMembership { index, metadata })
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

fn build_compact_outputs(
    result: &CompactSelectionResult,
) -> Result<BTreeMap<&'static str, Vec<u8>>> {
    let mut outputs = BTreeMap::new();
    outputs.insert(
        "thresholds_pooled.csv",
        compact_thresholds_pooled_csv(result)?,
    );
    outputs.insert(
        "thresholds_by_generation.csv",
        compact_thresholds_by_generation_csv(result)?,
    );
    outputs.insert("threshold_logo.csv", compact_threshold_logo_csv(result)?);
    outputs.insert("pareto_frontier.csv", compact_pareto_csv(result)?);
    outputs.insert("size_budgets.csv", compact_size_budgets_csv(result)?);
    outputs.insert("query_incidence.csv", compact_query_incidence_csv(result)?);
    outputs.insert(
        "membership_verification.csv",
        compact_membership_verification_csv(result)?,
    );
    outputs.insert(
        "error_diagnostics.csv",
        compact_error_diagnostics_csv(result)?,
    );
    outputs.insert(
        "qualitative_probes.csv",
        compact_qualitative_probes_csv(result)?,
    );
    outputs.insert(
        "selected_candidate.csv",
        compact_selected_candidate_csv(result)?,
    );
    outputs.insert("run_manifest.csv", compact_run_manifest_csv(result)?);
    outputs.insert("selection_report.md", compact_report(result)?.into_bytes());
    Ok(outputs)
}

fn compact_thresholds_pooled_csv(result: &CompactSelectionResult) -> Result<Vec<u8>> {
    let denominator = CompactThresholdPoint::retained_correct_denominator(&result.points)?;
    let mut writer = canonical_writer();
    writer.write_record([
        "surname_count_min",
        "additional_emitted",
        "additional_correct",
        "additional_wrong",
        "additional_null_false_emissions",
        "correct_retained_fraction",
        "qualifying_keys",
        "actual_artifact_bytes",
        "combined_name_data_bytes",
        "zstd19_shipping_bytes",
        "pareto",
        "selected",
    ])?;
    for point in &result.points {
        writer.write_record([
            point.threshold.to_string(),
            point.metrics.emitted.to_string(),
            point.metrics.correct.to_string(),
            point.metrics.wrong.to_string(),
            point.metrics.null_false_emissions.to_string(),
            format_ratio_value(point.metrics.correct, denominator),
            point.inventory.key_count.to_string(),
            point.membership.total_bytes.to_string(),
            (EXISTING_ARTIFACT_BYTES + point.membership.total_bytes).to_string(),
            point.membership.zstd19_total_bytes.to_string(),
            bool_string(point_is_pareto(&result.points, point)).to_string(),
            bool_string(point.threshold == result.selected.threshold).to_string(),
        ])?;
    }
    Ok(writer.into_inner()?)
}

fn compact_thresholds_by_generation_csv(result: &CompactSelectionResult) -> Result<Vec<u8>> {
    let mut writer = canonical_writer();
    writer.write_record([
        "generation",
        "surname_count_min",
        "additional_emitted",
        "additional_correct",
        "additional_wrong",
        "additional_null_false_emissions",
        "correct_retained_fraction",
    ])?;
    for generation in Generation::V1_TO_V8 {
        let denominator = metrics_for_generation(result, generation, 1)?.correct;
        for point in &result.points {
            let metrics = metrics_for_generation(result, generation, point.threshold)?;
            writer.write_record([
                generation.as_str().to_string(),
                point.threshold.to_string(),
                metrics.emitted.to_string(),
                metrics.correct.to_string(),
                metrics.wrong.to_string(),
                metrics.null_false_emissions.to_string(),
                format_ratio_or_empty(metrics.correct, denominator),
            ])?;
        }
    }
    Ok(writer.into_inner()?)
}

fn metrics_for_generation(
    result: &CompactSelectionResult,
    generation: Generation,
    threshold: u64,
) -> Result<Metrics> {
    let point = result
        .points
        .iter()
        .find(|point| point.threshold == threshold)
        .ok_or("generation metrics requested an unknown threshold")?;
    evaluate_membership_hits(&result.rows, &point.membership_hits, |row| {
        row.generation == generation
    })
}

fn compact_threshold_logo_csv(result: &CompactSelectionResult) -> Result<Vec<u8>> {
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

fn compact_pareto_csv(result: &CompactSelectionResult) -> Result<Vec<u8>> {
    let mut writer = canonical_writer();
    writer.write_record([
        "surname_count_min",
        "additional_correct",
        "additional_wrong",
        "additional_null_false_emissions",
        "actual_artifact_bytes",
        "pareto",
    ])?;
    for point in &result.points {
        writer.write_record([
            point.threshold.to_string(),
            point.metrics.correct.to_string(),
            point.metrics.wrong.to_string(),
            point.metrics.null_false_emissions.to_string(),
            point.membership.total_bytes.to_string(),
            bool_string(point_is_pareto(&result.points, point)).to_string(),
        ])?;
    }
    Ok(writer.into_inner()?)
}

fn compact_size_budgets_csv(result: &CompactSelectionResult) -> Result<Vec<u8>> {
    let mut writer = canonical_writer();
    writer.write_record([
        "direct_size_cap_mib",
        "surname_count_min",
        "additional_correct",
        "additional_wrong",
        "additional_null_false_emissions",
        "actual_artifact_bytes",
    ])?;
    for cap in COMPACT_SIZE_CAPS_MIB {
        let point = best_point_under_cap(&result.points, cap * MIB)
            .ok_or("no compact candidate fits a declared size cap")?;
        writer.write_record([
            cap.to_string(),
            point.threshold.to_string(),
            point.metrics.correct.to_string(),
            point.metrics.wrong.to_string(),
            point.metrics.null_false_emissions.to_string(),
            point.membership.total_bytes.to_string(),
        ])?;
    }
    Ok(writer.into_inner()?)
}

fn compact_query_incidence_csv(result: &CompactSelectionResult) -> Result<Vec<u8>> {
    let mut writer = canonical_writer();
    writer.write_record([
        "generation",
        "evaluable_rows",
        "surname_membership_queries",
        "query_fraction",
    ])?;
    for generation in Generation::V1_TO_V8 {
        write_query_incidence(&mut writer, generation.as_str(), &result.rows, |row| {
            row.generation == generation
        })?;
    }
    write_query_incidence(&mut writer, "POOLED_V1_V8", &result.rows, |_| true)?;
    Ok(writer.into_inner()?)
}

fn write_query_incidence(
    writer: &mut csv::Writer<Vec<u8>>,
    label: &str,
    rows: &[SelectionRow],
    include: impl Fn(&SelectionRow) -> bool,
) -> Result<()> {
    let selected = rows.iter().filter(|row| include(row)).collect::<Vec<_>>();
    let queries = selected
        .iter()
        .filter(|row| row.would_query_surname_index())
        .count();
    writer.write_record([
        label.to_string(),
        selected.len().to_string(),
        queries.to_string(),
        format_ratio_or_empty(queries, selected.len()),
    ])?;
    Ok(())
}

fn compact_membership_verification_csv(result: &CompactSelectionResult) -> Result<Vec<u8>> {
    let mut writer = canonical_writer();
    writer.write_record([
        "surname_count_min",
        "member_queries",
        "member_misses",
        "given_negative_queries",
        "given_false_accepts",
        "generated_negative_queries",
        "generated_false_accepts",
        "nominal_unknown_false_accept_probability",
        "mphf_bytes",
        "fingerprint_bytes",
        "manifest_bytes",
        "total_bytes",
        "zstd19_mphf_bytes",
        "zstd19_fingerprint_bytes",
        "zstd19_total_bytes",
        "mphf_sha256",
        "fingerprint_sha256",
        "manifest_sha256",
    ])?;
    for point in &result.points {
        let stats = &point.membership;
        writer.write_record([
            point.threshold.to_string(),
            stats.member_queries.to_string(),
            stats.member_misses.to_string(),
            stats.given_negative_queries.to_string(),
            stats.given_false_accepts.to_string(),
            stats.generated_negative_queries.to_string(),
            stats.generated_false_accepts.to_string(),
            "2^-32".to_string(),
            stats.mphf_bytes.to_string(),
            stats.fingerprint_bytes.to_string(),
            stats.manifest_bytes.to_string(),
            stats.total_bytes.to_string(),
            stats.zstd19_mphf_bytes.to_string(),
            stats.zstd19_fingerprint_bytes.to_string(),
            stats.zstd19_total_bytes.to_string(),
            stats.mphf_sha256.clone(),
            stats.fingerprint_sha256.clone(),
            stats.manifest_sha256.clone(),
        ])?;
    }
    Ok(writer.into_inner()?)
}

fn compact_error_diagnostics_csv(result: &CompactSelectionResult) -> Result<Vec<u8>> {
    let count_one = result
        .points
        .iter()
        .find(|point| point.threshold == 1)
        .ok_or("compact error diagnostic is missing count 1")?;
    let mut writer = canonical_writer();
    writer.write_record([
        "generation",
        "category",
        "raw_surname_count",
        "candidate_quality",
        "role_signal",
        "reliability",
        "selected_position",
        "complement_position",
        "hard_organization_marker",
        "generic_organization_marker",
        "ampersand",
        "candidate_too_short",
    ])?;
    for (row, member) in result.rows.iter().zip(&count_one.membership_hits) {
        let selected = row.residual_membership_emission(*member);
        if selected.is_none()
            || row
                .expected_greeting
                .as_deref()
                .is_some_and(|expected| greeting_matches(Some(expected), selected))
        {
            continue;
        }
        writer.write_record([
            row.generation.as_str(),
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
            "second",
            bool_string(row.hard_organization_marker),
            bool_string(row.generic_organization_marker),
            bool_string(row.ampersand),
            bool_string(row.candidate_too_short),
        ])?;
    }
    Ok(writer.into_inner()?)
}

fn compact_qualitative_probes_csv(result: &CompactSelectionResult) -> Result<Vec<u8>> {
    let mut writer = canonical_writer();
    writer.write_record([
        "probe",
        "selected_candidate",
        "raw_surname_count",
        "complement_in_selected_index",
        "c6_emits",
        "surname_residual_emits",
        "membership_condition_passes",
        "vetoes_pass",
    ])?;
    for probe in &result.probes {
        writer.write_record([
            probe.label.clone(),
            probe.selected_candidate.clone().unwrap_or_default(),
            probe
                .complement_surname_count
                .map_or_else(String::new, |value| value.to_string()),
            bool_string(probe.complement_in_selected_index).to_string(),
            bool_string(probe.first_position_emits).to_string(),
            bool_string(probe.surname_residual_emits).to_string(),
            bool_string(probe.complement_in_selected_index).to_string(),
            bool_string(probe.vetoes_pass).to_string(),
        ])?;
    }
    Ok(writer.into_inner()?)
}

fn compact_selected_candidate_csv(result: &CompactSelectionResult) -> Result<Vec<u8>> {
    let point = &result.selected;
    let mut writer = canonical_writer();
    writer.write_record(["key", "value"])?;
    for (key, value) in [
        ("surname_count_min", point.threshold.to_string()),
        ("additional_correct", point.metrics.correct.to_string()),
        ("additional_wrong", point.metrics.wrong.to_string()),
        (
            "additional_null_false_emissions",
            point.metrics.null_false_emissions.to_string(),
        ),
        ("surname_only_keys", point.inventory.key_count.to_string()),
        ("candidate_bytes", point.membership.total_bytes.to_string()),
        (
            "combined_name_data_bytes",
            (EXISTING_ARTIFACT_BYTES + point.membership.total_bytes).to_string(),
        ),
        ("source_keys_sha256", point.inventory.raw_sha256.clone()),
        (
            "candidate_manifest_sha256",
            point.membership.manifest_sha256.clone(),
        ),
        (
            "candidate_mphf_sha256",
            point.membership.mphf_sha256.clone(),
        ),
        (
            "candidate_fingerprint_sha256",
            point.membership.fingerprint_sha256.clone(),
        ),
        (
            "reproduction_receipt_sha256",
            result.selected_reproduction_sha256.clone(),
        ),
    ] {
        writer.write_record([key, &value])?;
    }
    Ok(writer.into_inner()?)
}

fn compact_run_manifest_csv(result: &CompactSelectionResult) -> Result<Vec<u8>> {
    let mut writer = canonical_writer();
    writer.write_record(["key", "value"])?;
    for (key, value) in [
        ("spent_generations", "REAL_PROXY_V1-V8".to_string()),
        ("surname_counts_sha256", result.counts_sha256.clone()),
        ("surname_inventory_sha256", result.inventory_sha256.clone()),
        (
            "threshold_grid",
            "1/2/5/10/25/50/100/250/500/1000".to_string(),
        ),
        ("selection_minimum_correct", COMPACT_MIN_CORRECT.to_string()),
        ("selection_retention_minimum", "0.5".to_string()),
        (
            "selection_objective",
            "meaningful_then_wrong_then_null_then_half_retention_then_size".to_string(),
        ),
        ("selected_threshold", result.selected.threshold.to_string()),
        ("row_level_output", "forbidden".to_string()),
        ("v9_created", "false".to_string()),
        ("surname_production_integration", "false".to_string()),
    ] {
        writer.write_record([key, &value])?;
    }
    Ok(writer.into_inner()?)
}

fn compact_report(result: &CompactSelectionResult) -> Result<String> {
    let denominator = CompactThresholdPoint::retained_correct_denominator(&result.points)?;
    let selected = &result.selected;
    let pooled_queries = result
        .rows
        .iter()
        .filter(|row| row.would_query_surname_index())
        .count();
    let mut report = String::new();
    writeln!(
        report,
        "# Compact surname-only index selection on spent V1-V8\n"
    )
    .unwrap();
    writeln!(report, "REAL_PROXY_V1 through V8 are spent development evidence. The C6-abstained first-token topology, candidate quality 0.40, reliability 0.00, role signal 0.30, complement-given absence, and every existing veto were frozen. Only the ten declared raw surname-count thresholds changed membership. No V9 data was created or inspected.\n").unwrap();
    writeln!(report, "| Count | Correct | Wrong | NULL FP | Retained | Keys | Direct bytes | zstd-19 shipping bytes | Pareto |\n|---:|---:|---:|---:|---:|---:|---:|---:|---|").unwrap();
    for point in &result.points {
        writeln!(
            report,
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            point.threshold,
            point.metrics.correct,
            point.metrics.wrong,
            point.metrics.null_false_emissions,
            format_percent_ratio(point.metrics.correct, denominator),
            point.inventory.key_count,
            point.membership.total_bytes,
            point.membership.zstd19_total_bytes,
            if point_is_pareto(&result.points, point) {
                "yes"
            } else {
                "no"
            }
        )
        .unwrap();
    }
    writeln!(report, "\nThe fixed selection rule chose `surname_count >= {}`: **+{} correct / {} wrong / {} NULL FP**, retaining {} of the count-1 correct benefit in {} direct bytes. Combined with the unchanged {}-byte given-name artifact, the name-data footprint is {} bytes.\n", selected.threshold, selected.metrics.correct, selected.metrics.wrong, selected.metrics.null_false_emissions, format_percent_ratio(selected.metrics.correct, denominator), selected.membership.total_bytes, EXISTING_ARTIFACT_BYTES, EXISTING_ARTIFACT_BYTES + selected.membership.total_bytes).unwrap();

    writeln!(report, "## Fixed-threshold generation stability\n").unwrap();
    writeln!(report, "| Generation | >=1 | >=2 | >=5 | >=10 | >=25 | >=50 | >=100 | >=250 | >=500 | >=1000 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|").unwrap();
    for generation in Generation::V1_TO_V8 {
        write!(report, "| {} |", generation.as_str()).unwrap();
        for point in &result.points {
            let metrics = metrics_for_generation(result, generation, point.threshold)?;
            write!(
                report,
                " {}/{}/{} |",
                metrics.correct, metrics.wrong, metrics.null_false_emissions
            )
            .unwrap();
        }
        writeln!(report).unwrap();
    }
    writeln!(report, "\nEach cell is `correct/wrong/NULL FP`. V7 and V8 retention relative to their respective count-1 results is reported in `thresholds_by_generation.csv`.\n").unwrap();

    writeln!(report, "## Leave-one-generation-out compact selection\n").unwrap();
    writeln!(report, "| Held out | Selected count | Training correct/wrong/NULL | Held-out correct/wrong/NULL |\n|---|---:|---:|---:|").unwrap();
    for (generation, training, held_out) in &result.logo {
        writeln!(
            report,
            "| {} | {} | {}/{}/{} | {}/{}/{} |",
            generation.as_str(),
            training.threshold,
            training.metrics.correct,
            training.metrics.wrong,
            training.metrics.null_false_emissions,
            held_out.correct,
            held_out.wrong,
            held_out.null_false_emissions
        )
        .unwrap();
    }

    writeln!(report, "\n## Direct-size caps\n").unwrap();
    writeln!(report, "| Cap | Best safety-first count | Correct/wrong/NULL | Direct bytes |\n|---:|---:|---:|---:|").unwrap();
    for cap in COMPACT_SIZE_CAPS_MIB {
        let point = best_point_under_cap(&result.points, cap * MIB)
            .ok_or("no compact candidate fits a declared size cap")?;
        writeln!(
            report,
            "| {cap} MiB | {} | {}/{}/{} | {} |",
            point.threshold,
            point.metrics.correct,
            point.metrics.wrong,
            point.metrics.null_false_emissions,
            point.membership.total_bytes
        )
        .unwrap();
    }

    writeln!(report, "\n## Representation and lookup incidence\n").unwrap();
    writeln!(report, "Every threshold used the same deterministic MPHF routing, independent 32-bit fingerprint, fixed-int bincode encoding, and no count/evidence byte. Every member was queried after disk round-trip. Each candidate also received {} retained-given and {} generated negative queries. Observed false accepts are recorded per threshold in `membership_verification.csv`; the nominal unknown-query risk remains approximately `2^-32` per lookup. zstd-19 sizes are shipping measurements only and did not affect selection.\n", GIVEN_KEYS, GENERATED_NEGATIVES).unwrap();
    writeln!(report, "Only {} of {} evaluable proxy rows ({}) satisfied all frozen pre-membership conditions and would query the surname index. This is spent-proxy incidence, not a production traffic guarantee.\n", pooled_queries, result.rows.len(), format_percent_ratio(pooled_queries, result.rows.len())).unwrap();

    writeln!(report, "The selected candidate contains {} keys: MPHF {} bytes, fingerprints {} bytes, manifest {} bytes, total {} bytes. Its checksums are manifest `{}`, MPHF `{}`, and fingerprints `{}`. Independent reconstruction matched byte-for-byte under reproduction receipt SHA-256 `{}`.\n", selected.inventory.key_count, selected.membership.mphf_bytes, selected.membership.fingerprint_bytes, selected.membership.manifest_bytes, selected.membership.total_bytes, selected.membership.manifest_sha256, selected.membership.mphf_sha256, selected.membership.fingerprint_sha256, result.selected_reproduction_sha256).unwrap();

    writeln!(report, "## Error and qualitative controls\n").unwrap();
    writeln!(report, "Residual errors are serialized only as abstract numeric/veto rows in `error_diagnostics.csv`. No display name, candidate string, complement string, identifier, country, or locale is included. The known high-count V7 semantic error is not converted into a heuristic.\n").unwrap();
    writeln!(report, "| Probe | Selected | Count | Member | C6 emits | Residual emits | Vetoes pass |\n|---|---|---:|---|---|---|---|").unwrap();
    for probe in &result.probes {
        writeln!(
            report,
            "| {} | {} | {} | {} | {} | {} | {} |",
            probe.label,
            probe.selected_candidate.as_deref().unwrap_or(""),
            probe
                .complement_surname_count
                .map_or_else(|| "n/a".to_string(), |value| value.to_string()),
            bool_string(probe.complement_in_selected_index),
            bool_string(probe.first_position_emits),
            bool_string(probe.surname_residual_emits),
            bool_string(probe.vetoes_pass)
        )
        .unwrap();
    }
    writeln!(report, "\nThe two redacted first-position probes already emit through production C6, so complement membership is diagnostic and the additive residual is not reached. The redacted two-candidate probe remains outside the sole-candidate topology, and `Motorcycle Club` remains vetoed.\n").unwrap();
    writeln!(report, "C6 remains production. The selected directory is a frozen V9 validation candidate only; normal inference does not load it. No V9 was created.").unwrap();
    Ok(report)
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
    for generation in Generation::V1_TO_V7 {
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
    for generation in Generation::V1_TO_V7 {
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

fn format_ratio_value(numerator: usize, denominator: usize) -> String {
    format!("{:.6}", numerator as f64 / denominator as f64)
}

fn format_ratio_or_empty(numerator: usize, denominator: usize) -> String {
    if denominator == 0 {
        String::new()
    } else {
        format_ratio_value(numerator, denominator)
    }
}

fn format_percent_ratio(numerator: usize, denominator: usize) -> String {
    if denominator == 0 {
        return "n/a".to_string();
    }
    format!("{:.2}%", numerator as f64 * 100.0 / denominator as f64)
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

    fn compact_point(
        threshold: u64,
        correct: usize,
        wrong: usize,
        nulls: usize,
        bytes: usize,
    ) -> CompactThresholdPoint {
        CompactThresholdPoint {
            threshold,
            metrics: Metrics {
                emitted: correct + wrong,
                correct,
                wrong,
                null_false_emissions: nulls,
            },
            inventory: InventoryRow {
                threshold,
                key_count: 1,
                raw_utf8_bytes: 1,
                zstd19_bytes: 1,
                raw_sha256: "0".repeat(64),
                zstd_sha256: "0".repeat(64),
            },
            membership: ActualMembershipStats {
                threshold,
                key_count: 1,
                mphf_bytes: 0,
                fingerprint_bytes: 0,
                manifest_bytes: bytes,
                total_bytes: bytes,
                mphf_sha256: "0".repeat(64),
                fingerprint_sha256: "0".repeat(64),
                manifest_sha256: "0".repeat(64),
                member_queries: 1,
                member_misses: 0,
                given_negative_queries: 1,
                given_false_accepts: 0,
                generated_negative_queries: 1,
                generated_false_accepts: 0,
                zstd19_mphf_bytes: 0,
                zstd19_fingerprint_bytes: 0,
                zstd19_total_bytes: bytes,
            },
            membership_hits: Vec::new(),
        }
    }

    fn selection_row() -> SelectionRow {
        SelectionRow {
            generation: Generation::V8,
            expected_greeting: Some("ExpectedSecret".to_string()),
            selected_candidate: Some("SelectedSecret".to_string()),
            first_position_emission: false,
            residual_topology: true,
            candidate_quality: 0.4,
            reliability: 0.0,
            role_signal: 0.3,
            complement: Some("ComplementSecret".to_string()),
            complement_given_observed: false,
            complement_surname_count: Some(10),
            hard_organization_marker: false,
            generic_organization_marker: false,
            ampersand: false,
            candidate_too_short: false,
        }
    }

    fn compact_result_with_row(row: SelectionRow, member: bool) -> CompactSelectionResult {
        let mut point = compact_point(1, 0, 1, 0, 1);
        point.membership_hits = vec![member];
        CompactSelectionResult {
            rows: vec![row],
            points: vec![point.clone()],
            logo: Vec::new(),
            selected: point,
            counts_sha256: "0".repeat(64),
            inventory_sha256: "0".repeat(64),
            probes: Vec::new(),
            selected_reproduction_sha256: "0".repeat(64),
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
    fn compact_selection_requires_half_retention_then_chooses_smallest_safe_point() {
        let points = [
            compact_point(1, 100, 1, 1, 100),
            compact_point(2, 70, 1, 1, 80),
            compact_point(5, 50, 1, 1, 40),
            compact_point(10, 49, 1, 1, 20),
        ];
        assert_eq!(select_compact_point(&points).unwrap().threshold, 5);
    }

    #[test]
    fn compact_selection_prioritizes_semantic_errors_and_rejects_no_useful_point() {
        let unsafe_points = [
            compact_point(1, 100, 1, 1, 100),
            compact_point(2, 60, 1, 1, 50),
            compact_point(5, 55, 0, 0, 40),
        ];
        assert_eq!(select_compact_point(&unsafe_points).unwrap().threshold, 5);

        let insufficient = [
            compact_point(1, 4, 0, 0, 100),
            compact_point(2, 3, 0, 0, 50),
        ];
        assert!(select_compact_point(&insufficient).is_err());
        assert!(
            CompactThresholdPoint::retained_correct_denominator(&[compact_point(1, 0, 0, 0, 100)])
                .is_err()
        );
    }

    #[test]
    fn compact_pareto_and_budget_selection_use_semantics_benefit_and_actual_size() {
        let points = [
            compact_point(1, 10, 1, 1, 10 * MIB),
            compact_point(2, 10, 1, 1, 8 * MIB),
            compact_point(5, 8, 1, 1, 2 * MIB),
            compact_point(10, 7, 2, 1, MIB),
        ];
        assert!(!point_is_pareto(&points, &points[0]));
        assert!(point_is_pareto(&points, &points[1]));
        assert_eq!(best_point_under_cap(&points, 2 * MIB).unwrap().threshold, 5);
    }

    #[test]
    fn compact_logo_applies_the_same_selection_rule_to_each_training_fold() {
        let rows = Generation::V1_TO_V8
            .into_iter()
            .map(|generation| {
                let mut row = selection_row();
                row.generation = generation;
                row.expected_greeting = row.selected_candidate.clone();
                row
            })
            .collect::<Vec<_>>();
        let mut count_one = compact_point(1, 8, 0, 0, 100);
        count_one.membership_hits = vec![true; rows.len()];
        let mut count_two = compact_point(2, 8, 0, 0, 50);
        count_two.membership_hits = vec![true; rows.len()];

        let logo = compact_logo_points(&rows, &[count_one, count_two]).unwrap();
        assert_eq!(logo.len(), Generation::V1_TO_V8.len());
        assert!(logo.iter().all(|(_, selected, held_out)| {
            selected.threshold == 2
                && selected.metrics.correct == 7
                && held_out.correct == 1
                && held_out.wrong == 0
        }));
    }

    #[test]
    fn historical_and_compact_generation_sets_are_exact() {
        assert_eq!(Generation::V1_TO_V7.len(), 7);
        assert_eq!(Generation::V1_TO_V8.len(), 8);
        assert_eq!(Generation::V1_TO_V8[..7], Generation::V1_TO_V7);
        assert_eq!(Generation::V1_TO_V8[7], Generation::V8);
        assert_eq!(Generation::from_digest(V8_SHA256), Some(Generation::V8));
    }

    #[test]
    fn compact_query_requires_frozen_topology_and_given_absence() {
        let row = selection_row();
        assert!(row.would_query_surname_index());

        let mut given_observed = row.clone();
        given_observed.complement_given_observed = true;
        assert!(!given_observed.would_query_surname_index());

        let mut outside_topology = row;
        outside_topology.residual_topology = false;
        assert!(!outside_topology.would_query_surname_index());
    }

    #[test]
    fn compact_error_output_is_aggregate_and_redacted() {
        let output = String::from_utf8(
            compact_error_diagnostics_csv(&compact_result_with_row(selection_row(), true)).unwrap(),
        )
        .unwrap();
        for secret in ["ExpectedSecret", "SelectedSecret", "ComplementSecret"] {
            assert!(!output.contains(secret));
        }
        assert!(output.contains("REAL_PROXY_V8,wrong_greeting,10,0.400000,0.300000,0.000000"));
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
        let (_, repeated_mphf, repeated_fingerprints) =
            build_membership_index(&keys, &routing).unwrap();
        assert_eq!(mphf, repeated_mphf);
        assert_eq!(fingerprints, repeated_fingerprints);
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
        assert_eq!(PRIOR_KEY_COUNTS[3], 1_709_397);
        assert_eq!(
            FROZEN_COMPACT_CANDIDATE_MPHF_BYTES
                + FROZEN_COMPACT_CANDIDATE_FINGERPRINT_BYTES
                + FROZEN_COMPACT_CANDIDATE_MANIFEST_BYTES,
            7_575_340
        );
        for digest in [
            FROZEN_COMPACT_CANDIDATE_MANIFEST_SHA256,
            FROZEN_COMPACT_CANDIDATE_SOURCE_KEYS_SHA256,
            FROZEN_COMPACT_CANDIDATE_MPHF_SHA256,
            FROZEN_COMPACT_CANDIDATE_FINGERPRINT_SHA256,
        ] {
            validate_sha256(digest, "frozen compact candidate digest").unwrap();
        }
    }
}
