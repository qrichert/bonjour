use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::path::Path;

use bonjour::benchmark::{
    ALGORITHM_C3, EvidenceSource, candidate_diagnostics, candidate_is_eligible, open_artifact,
};
use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3::xxh3_64_with_seed;

use crate::output::{Result, file_sha256, publish_directory};
use crate::text::{
    contains_organization_component, model_normalize, morphology_family, script_class,
    valid_candidate_form,
};

const TOTALS_SHA256: &str = "e43e8661261b2762d3d4f2581ebb803af94abb7505409873f46041be1470ff62";
const CLEAN_SHA256: &str = "57a82801894facf883769403271e68094bcedb563d31c81f853192dc05e66b47";
const EXPECTED_KEYS: usize = 1_803_175;
const EXPECTED_GIVEN_TOTAL: u64 = 444_154_759;
const EXPECTED_SURNAME_OVERLAP_TOTAL: u64 = 364_386_816;
const SURNAME_TOTAL: u64 = 489_631_377;
const ROLE_SMOOTHING: f64 = 0.5;
const MIN_ROLE_COUNT: u64 = 100;
const MIN_ROLE_LLR: f64 = 2.0;
const SPLIT_SEED: u64 = 0x6e6e_6d6f_7270_6831;
const SPLIT_BUCKETS: u64 = 1_000;
const TRAIN_CUTOFF: u64 = 800;
const VALIDATION_CUTOFF: u64 = 900;
const MAX_UNTRUNCATED_BYTES: usize = 94;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
enum MorphSplit {
    Train,
    Validation,
    MorphTest,
}

impl MorphSplit {
    const ALL: [Self; 3] = [Self::Train, Self::Validation, Self::MorphTest];

    fn from_family(family: &str) -> Self {
        match xxh3_64_with_seed(family.as_bytes(), SPLIT_SEED) % SPLIT_BUCKETS {
            bucket if bucket < TRAIN_CUTOFF => Self::Train,
            bucket if bucket < VALIDATION_CUTOFF => Self::Validation,
            _ => Self::MorphTest,
        }
    }

    fn file_name(self) -> &'static str {
        match self {
            Self::Train => "train.csv",
            Self::Validation => "validation.csv",
            Self::MorphTest => "morph_test.csv",
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Validation => "validation",
            Self::MorphTest => "morph_test",
        }
    }
}

#[derive(Default)]
struct Aggregate {
    given_count: u64,
    surname_count: u64,
    representative: String,
    representative_given_count: u64,
    source_rows: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct PopulationLabels {
    high_confidence_given: bool,
    strong_surname: bool,
    organization_non_name: bool,
    garbage_malformed: bool,
}

#[derive(Clone, Debug, Serialize)]
struct PreparedRow {
    normalized: String,
    family: String,
    split: &'static str,
    label: u8,
    populations: String,
    given_count: u64,
    surname_count: u64,
    role_llr: f64,
    role_signal: f64,
    primary_country: String,
    production_candidate_quality: Option<f64>,
    production_role_llr: Option<f64>,
    production_role_signal: Option<f64>,
    production_reliability: Option<f64>,
    script: &'static str,
    utf8_bytes: usize,
    truncated: bool,
}

#[derive(Debug, Deserialize)]
struct NonNameRow {
    token: String,
    kind: String,
    source: String,
}

#[derive(Debug, Deserialize)]
struct ProbeRow {
    category: String,
    value: String,
}

#[derive(Default, Serialize)]
struct PreparationStats {
    totals_rows: usize,
    normalized_keys: usize,
    source_given_total: u64,
    source_surname_overlap_total: u64,
    unlabeled_keys: usize,
    exact_label_conflicts: usize,
    quarantined_keys: usize,
    output_keys: usize,
    country_rows: usize,
    country_observations: u64,
}

#[derive(Serialize)]
struct DatasetManifest {
    format_version: u32,
    split_seed: String,
    split_buckets: u64,
    train_cutoff: u64,
    validation_cutoff: u64,
    role_smoothing: f64,
    minimum_role_count: u64,
    minimum_role_llr: f64,
    given_total: u64,
    surname_overlap_total: u64,
    surname_total: u64,
    surname_only_status: &'static str,
    dictionary_non_name_status: &'static str,
    source_sha256: BTreeMap<&'static str, String>,
    output_sha256: BTreeMap<String, String>,
    split_counts: BTreeMap<String, usize>,
    population_counts: BTreeMap<String, usize>,
    script_counts: BTreeMap<String, usize>,
    stats: PreparationStats,
}

pub(crate) fn prepare(
    artifact_path: &Path,
    totals_path: &Path,
    clean_path: &Path,
    fixtures_path: &Path,
    output_path: &Path,
) -> Result<()> {
    let totals_sha256 = require_sha256(totals_path, TOTALS_SHA256, "name totals")?;
    let clean_sha256 = require_sha256(clean_path, CLEAN_SHA256, "clean-v1")?;
    let non_names_path = fixtures_path.join("non_name_tokens.csv");
    let probes_path = fixtures_path.join("qualitative_probes.csv");
    let non_names_sha256 = file_sha256(&non_names_path)?;
    let probes_sha256 = file_sha256(&probes_path)?;

    let organization_tokens = load_organization_tokens(&non_names_path)?;
    let quarantined_families = load_quarantined_families(&probes_path)?;
    let artifact = open_artifact(artifact_path)?;
    let (mut aggregates, mut stats) = load_totals(totals_path)?;
    inject_organization_tokens(&mut aggregates, &organization_tokens);
    let mut rows = label_rows(
        &artifact,
        aggregates,
        &organization_tokens,
        &quarantined_families,
        &mut stats,
    )?;
    let selected_keys = rows
        .iter()
        .map(|row| row.normalized.clone())
        .collect::<HashSet<_>>();
    let countries = load_primary_countries(clean_path, &selected_keys, &mut stats)?;
    for row in &mut rows {
        row.primary_country = countries.get(&row.normalized).cloned().unwrap_or_default();
    }
    validate_disjointness(&rows)?;
    stats.output_keys = rows.len();

    if file_sha256(totals_path)? != totals_sha256 || file_sha256(clean_path)? != clean_sha256 {
        return Err("a source CSV changed during preparation".into());
    }

    publish_directory(output_path, |temporary| {
        write_dataset(temporary, &rows)?;
        write_manifest(
            temporary,
            &rows,
            stats,
            [
                ("name_totals", totals_sha256.clone()),
                ("clean_v1", clean_sha256.clone()),
                ("non_name_tokens", non_names_sha256.clone()),
                ("qualitative_probes", probes_sha256.clone()),
            ],
        )
    })?;
    eprintln!("Prepared morphology data: {}", output_path.display());
    Ok(())
}

fn require_sha256(path: &Path, expected: &str, label: &str) -> Result<String> {
    let actual = file_sha256(path)?;
    if actual != expected {
        return Err(format!("{label} checksum mismatch: expected {expected}, got {actual}").into());
    }
    Ok(actual)
}

fn load_organization_tokens(path: &Path) -> Result<BTreeSet<String>> {
    let mut reader = csv::Reader::from_path(path)?;
    require_header(reader.headers()?, &["token", "kind", "source"], path)?;
    let mut tokens = BTreeSet::new();
    for result in reader.deserialize::<NonNameRow>() {
        let row = result?;
        if !matches!(row.kind.as_str(), "legal" | "generic")
            || row.source != "production_c3_vocabulary"
        {
            return Err(format!("invalid non-name metadata for {:?}", row.token).into());
        }
        let normalized = model_normalize(&row.token);
        if normalized.is_empty() || !tokens.insert(normalized) {
            return Err(format!("duplicate or empty non-name token: {:?}", row.token).into());
        }
    }
    if tokens.len() != 31 {
        return Err(format!(
            "expected 31 frozen organization tokens, got {}",
            tokens.len()
        )
        .into());
    }
    Ok(tokens)
}

fn load_quarantined_families(path: &Path) -> Result<BTreeSet<String>> {
    let mut reader = csv::Reader::from_path(path)?;
    require_header(reader.headers()?, &["category", "value"], path)?;
    let mut values = BTreeSet::new();
    let mut families = BTreeSet::new();
    for result in reader.deserialize::<ProbeRow>() {
        let row = result?;
        if row.category.is_empty() || row.value.is_empty() {
            return Err("qualitative probe fields must be non-empty".into());
        }
        let normalized = model_normalize(&row.value);
        if !values.insert(normalized) {
            return Err(format!("duplicate qualitative probe: {:?}", row.value).into());
        }
        families.insert(morphology_family(&row.value));
    }
    if values.len() != 27 {
        return Err(format!("expected 27 qualitative probes, got {}", values.len()).into());
    }
    Ok(families)
}

fn load_totals(path: &Path) -> Result<(HashMap<String, Aggregate>, PreparationStats)> {
    let mut reader = csv::Reader::from_path(path)?;
    require_header(
        reader.headers()?,
        &["name", "given_count", "as_surname_count"],
        path,
    )?;
    let mut aggregates = HashMap::<String, Aggregate>::with_capacity(EXPECTED_KEYS);
    let mut stats = PreparationStats::default();
    let mut previous = None::<Vec<u8>>;
    for result in reader.records() {
        let record = result?;
        let name = field(&record, 0, "name")?;
        if previous
            .as_ref()
            .is_some_and(|previous| previous.as_slice() >= name.as_bytes())
        {
            return Err("name totals are not strictly bytewise ordered".into());
        }
        previous = Some(name.as_bytes().to_vec());
        let given_count = parse_u64(field(&record, 1, "given_count")?, "given_count")?;
        let surname_count = parse_u64(field(&record, 2, "as_surname_count")?, "as_surname_count")?;
        stats.totals_rows += 1;
        stats.source_given_total = stats
            .source_given_total
            .checked_add(given_count)
            .ok_or("given total overflow")?;
        stats.source_surname_overlap_total = stats
            .source_surname_overlap_total
            .checked_add(surname_count)
            .ok_or("surname total overflow")?;
        let normalized = model_normalize(name);
        let aggregate = aggregates.entry(normalized).or_default();
        aggregate.given_count = aggregate
            .given_count
            .checked_add(given_count)
            .ok_or("normalized given count overflow")?;
        aggregate.surname_count = aggregate
            .surname_count
            .checked_add(surname_count)
            .ok_or("normalized surname count overflow")?;
        aggregate.source_rows += 1;
        if representative_order(
            name,
            given_count,
            &aggregate.representative,
            aggregate.representative_given_count,
        ) == Ordering::Greater
        {
            aggregate.representative = name.to_string();
            aggregate.representative_given_count = given_count;
        }
    }
    stats.normalized_keys = aggregates.len();
    if stats.totals_rows != EXPECTED_KEYS
        || stats.source_given_total != EXPECTED_GIVEN_TOTAL
        || stats.source_surname_overlap_total != EXPECTED_SURNAME_OVERLAP_TOTAL
    {
        return Err(format!(
            "name totals invariants changed: rows={}, given={}, surname={}",
            stats.totals_rows, stats.source_given_total, stats.source_surname_overlap_total
        )
        .into());
    }
    Ok((aggregates, stats))
}

fn representative_order(
    candidate: &str,
    candidate_count: u64,
    current: &str,
    current_count: u64,
) -> Ordering {
    candidate_count
        .cmp(&current_count)
        .then_with(|| current.as_bytes().cmp(candidate.as_bytes()))
}

fn inject_organization_tokens(
    aggregates: &mut HashMap<String, Aggregate>,
    organization_tokens: &BTreeSet<String>,
) {
    for token in organization_tokens {
        aggregates
            .entry(token.clone())
            .or_insert_with(|| Aggregate {
                representative: token.clone(),
                ..Aggregate::default()
            });
    }
}

fn label_rows(
    corpus: &impl EvidenceSource,
    aggregates: HashMap<String, Aggregate>,
    organization_tokens: &BTreeSet<String>,
    quarantined_families: &BTreeSet<String>,
    stats: &mut PreparationStats,
) -> Result<Vec<PreparedRow>> {
    let mut rows = Vec::new();
    for (normalized, aggregate) in aggregates {
        let family = morphology_family(&normalized);
        let role_llr = role_llr(aggregate.given_count, aggregate.surname_count);
        let valid = valid_candidate_form(&normalized);
        let labels = population_labels(
            valid,
            contains_organization_component(&normalized, organization_tokens),
            organization_tokens.contains(&normalized),
            candidate_is_eligible(&normalized),
            aggregate.given_count,
            aggregate.surname_count,
            role_llr,
        );
        let negative =
            labels.strong_surname || labels.organization_non_name || labels.garbage_malformed;
        if labels.high_confidence_given && negative {
            stats.exact_label_conflicts += 1;
            continue;
        }
        if !labels.high_confidence_given && !negative {
            stats.unlabeled_keys += 1;
            continue;
        }
        if quarantined_families.contains(&family) {
            stats.quarantined_keys += 1;
            continue;
        }
        let mut populations = Vec::new();
        if labels.high_confidence_given {
            populations.push("high_confidence_given");
        }
        if labels.strong_surname {
            populations.push("strong_surname");
        }
        if labels.organization_non_name {
            populations.push("organization_non_name");
        }
        if labels.garbage_malformed {
            populations.push("garbage_malformed");
        }
        let split = MorphSplit::from_family(&family);
        let production = production_signals(corpus, &aggregate.representative, &normalized);
        rows.push(PreparedRow {
            script: script_class(&normalized),
            utf8_bytes: normalized.len(),
            truncated: normalized.len() > MAX_UNTRUNCATED_BYTES,
            normalized,
            family,
            split: split.as_str(),
            label: u8::from(labels.high_confidence_given),
            populations: populations.join(";"),
            given_count: aggregate.given_count,
            surname_count: aggregate.surname_count,
            role_llr,
            role_signal: role_signal(role_llr),
            primary_country: String::new(),
            production_candidate_quality: production.map(|value| value.0),
            production_role_llr: production.map(|value| value.1),
            production_role_signal: production.map(|value| value.2),
            production_reliability: production.map(|value| value.3),
        });
    }
    rows.sort_by(|left, right| left.normalized.as_bytes().cmp(right.normalized.as_bytes()));
    Ok(rows)
}

fn population_labels(
    valid_candidate: bool,
    contains_organization: bool,
    exact_organization_token: bool,
    lexical_candidate: bool,
    given_count: u64,
    surname_count: u64,
    role_llr: f64,
) -> PopulationLabels {
    PopulationLabels {
        high_confidence_given: valid_candidate
            && given_count >= MIN_ROLE_COUNT
            && role_llr >= MIN_ROLE_LLR
            && !contains_organization,
        strong_surname: valid_candidate
            && surname_count >= MIN_ROLE_COUNT
            && role_llr <= -MIN_ROLE_LLR,
        organization_non_name: exact_organization_token,
        garbage_malformed: !lexical_candidate,
    }
}

fn production_signals(
    corpus: &impl EvidenceSource,
    representative: &str,
    normalized: &str,
) -> Option<(f64, f64, f64, f64)> {
    candidate_diagnostics(corpus, ALGORITHM_C3, representative, None, None)
        .into_iter()
        .find(|candidate| model_normalize(&candidate.display) == normalized)
        .map(|candidate| {
            (
                candidate.score,
                candidate.role_llr,
                candidate.role_signal,
                candidate.reliability,
            )
        })
}

fn load_primary_countries(
    path: &Path,
    selected: &HashSet<String>,
    stats: &mut PreparationStats,
) -> Result<HashMap<String, String>> {
    let mut reader = csv::Reader::from_path(path)?;
    require_header(
        reader.headers()?,
        &["name", "country", "gender", "count"],
        path,
    )?;
    let mut counts = HashMap::<String, HashMap<String, u64>>::new();
    for result in reader.records() {
        let record = result?;
        stats.country_rows += 1;
        let name = model_normalize(field(&record, 0, "name")?);
        let country = field(&record, 1, "country")?;
        let count = parse_u64(field(&record, 3, "count")?, "count")?;
        stats.country_observations = stats
            .country_observations
            .checked_add(count)
            .ok_or("country observation overflow")?;
        if selected.contains(&name) {
            let country_counts = counts.entry(name).or_default();
            let total = country_counts.entry(country.to_string()).or_default();
            *total = total.checked_add(count).ok_or("country count overflow")?;
        }
    }
    if stats.country_observations != EXPECTED_GIVEN_TOTAL {
        return Err(format!(
            "clean-v1 observation total changed: {}",
            stats.country_observations
        )
        .into());
    }
    Ok(counts
        .into_iter()
        .filter_map(|(name, countries)| {
            countries
                .into_iter()
                .max_by(|left, right| {
                    left.1
                        .cmp(&right.1)
                        .then_with(|| right.0.as_bytes().cmp(left.0.as_bytes()))
                })
                .map(|(country, _)| (name, country))
        })
        .collect())
}

fn validate_disjointness(rows: &[PreparedRow]) -> Result<()> {
    let mut exact = HashMap::<&str, &str>::new();
    let mut families = HashMap::<&str, &str>::new();
    for row in rows {
        if exact
            .insert(&row.normalized, row.split)
            .is_some_and(|prior| prior != row.split)
        {
            return Err(format!("exact key leaked across splits: {:?}", row.normalized).into());
        }
        if families
            .insert(&row.family, row.split)
            .is_some_and(|prior| prior != row.split)
        {
            return Err(format!("morphology family leaked across splits: {:?}", row.family).into());
        }
    }
    Ok(())
}

fn write_dataset(output: &Path, rows: &[PreparedRow]) -> Result<()> {
    for split in MorphSplit::ALL {
        let mut writer = csv::Writer::from_path(output.join(split.file_name()))?;
        for row in rows.iter().filter(|row| row.split == split.as_str()) {
            writer.serialize(row)?;
        }
        writer.flush()?;
    }
    Ok(())
}

fn write_manifest(
    output: &Path,
    rows: &[PreparedRow],
    stats: PreparationStats,
    source_hashes: [(&'static str, String); 4],
) -> Result<()> {
    let mut split_counts = BTreeMap::new();
    let mut population_counts = BTreeMap::from([
        ("surname_only".to_string(), 0),
        ("dictionary_non_name".to_string(), 0),
    ]);
    let mut script_counts = BTreeMap::new();
    for row in rows {
        *split_counts.entry(row.split.to_string()).or_insert(0) += 1;
        for population in row.populations.split(';') {
            *population_counts.entry(population.to_string()).or_insert(0) += 1;
        }
        *script_counts.entry(row.script.to_string()).or_insert(0) += 1;
    }
    let output_sha256 = MorphSplit::ALL
        .into_iter()
        .map(|split| {
            let name = split.file_name().to_string();
            Ok((name.clone(), file_sha256(&output.join(name))?))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    let manifest = DatasetManifest {
        format_version: 1,
        split_seed: format!("0x{SPLIT_SEED:016x}"),
        split_buckets: SPLIT_BUCKETS,
        train_cutoff: TRAIN_CUTOFF,
        validation_cutoff: VALIDATION_CUTOFF,
        role_smoothing: ROLE_SMOOTHING,
        minimum_role_count: MIN_ROLE_COUNT,
        minimum_role_llr: MIN_ROLE_LLR,
        given_total: EXPECTED_GIVEN_TOTAL,
        surname_overlap_total: EXPECTED_SURNAME_OVERLAP_TOTAL,
        surname_total: SURNAME_TOTAL,
        surname_only_status: "unavailable: raw upstream surname-only source is not local",
        dictionary_non_name_status: "unavailable: no existing clean dictionary source",
        source_sha256: source_hashes.into_iter().collect(),
        output_sha256,
        split_counts,
        population_counts,
        script_counts,
        stats,
    };
    let mut bytes = serde_json::to_vec_pretty(&manifest)?;
    bytes.push(b'\n');
    fs::write(output.join("dataset_manifest.json"), bytes)?;
    Ok(())
}

fn role_llr(given_count: u64, surname_count: u64) -> f64 {
    ((given_count as f64 + ROLE_SMOOTHING) / EXPECTED_GIVEN_TOTAL as f64).ln()
        - ((surname_count as f64 + ROLE_SMOOTHING) / SURNAME_TOTAL as f64).ln()
}

fn role_signal(role_llr: f64) -> f64 {
    1.0 / (1.0 + (-((role_llr - 1.0) / 1.4)).exp())
}

fn require_header(actual: &csv::StringRecord, expected: &[&str], path: &Path) -> Result<()> {
    if actual.iter().eq(expected.iter().copied()) {
        Ok(())
    } else {
        Err(format!("unexpected header in {}: {actual:?}", path.display()).into())
    }
}

fn field<'a>(record: &'a csv::StringRecord, index: usize, label: &str) -> Result<&'a str> {
    record
        .get(index)
        .ok_or_else(|| format!("missing {label}").into())
}

fn parse_u64(value: &str, label: &str) -> Result<u64> {
    value
        .parse::<u64>()
        .map_err(|error| format!("invalid {label} {value:?}: {error}").into())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use bonjour::benchmark::{Evidence, diagnose_role_inference};

    use super::*;

    struct AlwaysEvidence;

    impl EvidenceSource for AlwaysEvidence {
        fn lookup(&self, _name: &str, _country_hint: Option<[u8; 2]>) -> Option<Evidence> {
            Some(Evidence {
                global_count: 50_000,
                country_count: 0,
                effective_count: 50_000,
                female_count: 0,
                male_count: 50_000,
                surname_count: 100,
                given_total: EXPECTED_GIVEN_TOTAL,
                surname_total: SURNAME_TOTAL,
            })
        }
    }

    #[test]
    fn label_thresholds_are_inclusive() {
        let given = population_labels(true, false, false, true, 100, 0, 2.0);
        assert!(given.high_confidence_given);
        assert_eq!(
            population_labels(true, false, false, true, 99, 0, 2.0),
            PopulationLabels::default()
        );
        assert_eq!(
            population_labels(
                true,
                false,
                false,
                true,
                100,
                0,
                f64::from_bits(2.0_f64.to_bits() - 1),
            ),
            PopulationLabels::default()
        );

        let surname = population_labels(true, false, false, true, 0, 100, -2.0);
        assert!(surname.strong_surname);
        assert_eq!(
            population_labels(true, false, false, true, 0, 99, -2.0),
            PopulationLabels::default()
        );

        let organization = population_labels(true, true, true, true, 100, 0, 2.0);
        assert!(!organization.high_confidence_given);
        assert!(organization.organization_non_name);

        let malformed = population_labels(false, false, false, false, 0, 0, 0.0);
        assert!(malformed.garbage_malformed);
    }

    #[test]
    fn split_is_family_deterministic() {
        assert_eq!(
            MorphSplit::from_family(&morphology_family("Élodie")),
            MorphSplit::from_family(&morphology_family("elodie"))
        );
    }

    #[test]
    fn representative_prefers_count_then_byte_order() {
        assert_eq!(representative_order("B", 2, "A", 1), Ordering::Greater);
        assert_eq!(representative_order("A", 2, "B", 2), Ordering::Greater);
    }

    #[test]
    fn organization_fixture_matches_frozen_c3_flags() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/non_name_tokens.csv");
        let tokens = load_organization_tokens(&path).unwrap();
        let mut reader = csv::Reader::from_path(path).unwrap();
        for result in reader.deserialize::<NonNameRow>() {
            let row = result.unwrap();
            let input = format!("Quentin {}", row.token);
            let diagnostic =
                diagnose_role_inference(&AlwaysEvidence, ALGORITHM_C3, &input, None, None);
            match row.kind.as_str() {
                "legal" => {
                    assert!(diagnostic.hard_organization_abstention, "{input:?}");
                }
                "generic" => {
                    assert!(diagnostic.generic_organization_marker, "{input:?}");
                }
                _ => panic!("unexpected fixture kind"),
            }
            assert!(contains_organization_component(&input, &tokens));
        }
    }
}
