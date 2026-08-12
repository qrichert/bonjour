use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap, HashMap};
use std::error::Error;
use std::fmt::Write as FmtWrite;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use boomphf::Mphf;
use csv::{ByteRecord, Reader, ReaderBuilder, Writer};
use unicode_general_category::get_general_category;
use xxhash_rust::xxh3::xxh3_64_with_seed;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const THRESHOLDS: [u64; 8] = [1, 2, 5, 10, 20, 50, 100, 500];
const BAND_LABELS: [&str; 9] = [
    "1", "2", "3-4", "5-9", "10-19", "20-49", "50-99", "100-499", "500+",
];
const SAMPLE_SIZE: usize = 500;
const FEATURE_SAMPLE_SIZE: usize = 100;
const LONGEST_SIZE: usize = 500;
const PROGRESS_ROWS: u64 = 5_000_000;
const MPHF_GAMMA: f64 = 1.7;
const ROUTING_SEED: u64 = 0x6e61_6d65_2d72_6f75;
const FINGERPRINT_SEED: u64 = 0x6e61_6d65_2d66_7033;
const SAMPLE_SEED: u64 = 0x6e61_6d65_2d73_616d;

struct FirstPass {
    name_ids: HashMap<Box<[u8]>, u32>,
    rows_per_name: Vec<u32>,
    observations_per_name: Vec<u64>,
    country_codes: Vec<u16>,
    row_count: u32,
    total_observations: u128,
    max_row_count: u32,
    row_bands: [BandStats; BAND_LABELS.len()],
    elapsed: Duration,
}

struct Rows {
    offsets_by_id: Vec<u32>,
    countries: Vec<u8>,
    genders: Vec<u8>,
    counts: Vec<u32>,
}

#[derive(Clone, Copy, Default)]
struct BandStats {
    keys_or_rows: u64,
    metadata_rows: u64,
    observations: u128,
}

#[derive(Clone, Copy, Default)]
struct PolicyStats {
    keys: u64,
    metadata_rows: u64,
    observations: u128,
}

#[derive(Clone, Copy, Default)]
struct FeatureStats {
    names: u64,
    metadata_rows: u64,
    observations: u128,
}

#[derive(Default)]
struct QuantizationStats {
    absolute_error_sum: u128,
    original_sum: u128,
    decoded_sum: u128,
    relative_error_sum: f64,
    max_absolute_error: u64,
    max_relative_error: f64,
    relative_error_histogram: Vec<u64>,
    rows: u64,
}

struct ArtifactStats {
    threshold: u64,
    keys: usize,
    metadata_rows: u64,
    observations: u128,
    bytes: u64,
    mphf_bytes: u64,
    fingerprint_bytes: u64,
    metadata_bytes: u64,
    build_elapsed: Duration,
    quantization: QuantizationStats,
}

struct Sampler {
    limit: usize,
    seed: u64,
    entries: BinaryHeap<(u64, u32)>,
}

impl Sampler {
    fn new(limit: usize, seed: u64) -> Self {
        Self {
            limit,
            seed,
            entries: BinaryHeap::with_capacity(limit + 1),
        }
    }

    fn observe(&mut self, id: u32, name: &[u8]) {
        let score = xxh3_64_with_seed(name, self.seed);
        if self.entries.len() < self.limit {
            self.entries.push((score, id));
        } else if self
            .entries
            .peek()
            .is_some_and(|&(largest, _)| score < largest)
        {
            self.entries.pop();
            self.entries.push((score, id));
        }
    }

    fn sorted_ids(&self) -> Vec<u32> {
        let mut entries: Vec<_> = self.entries.iter().copied().collect();
        entries.sort_unstable();
        entries.into_iter().map(|(_, id)| id).collect()
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let input = PathBuf::from(
        args.next()
            .ok_or("usage: name-corpus-audit <normalized.csv> <new-output-directory>")?,
    );
    let output = PathBuf::from(
        args.next()
            .ok_or("usage: name-corpus-audit <normalized.csv> <new-output-directory>")?,
    );
    if args.next().is_some() {
        return Err("usage: name-corpus-audit <normalized.csv> <new-output-directory>".into());
    }
    if !input.is_file() {
        return Err(format!("input is not a file: {}", input.display()).into());
    }
    if output.exists() {
        return Err(format!("refusing to overwrite: {}", output.display()).into());
    }

    let output_parent = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(output_parent)?;
    let output_name = output
        .file_name()
        .ok_or_else(|| format!("output has no final component: {}", output.display()))?;
    let temporary_output = output_parent.join(format!(
        ".{}.tmp-{}",
        output_name.to_string_lossy(),
        std::process::id()
    ));
    if temporary_output.exists() {
        return Err(format!(
            "refusing to overwrite temporary output: {}",
            temporary_output.display()
        )
        .into());
    }

    let started = Instant::now();
    let result = audit_and_build(&input, &temporary_output);
    match result {
        Ok(report) => {
            fs::write(temporary_output.join("report.md"), &report)?;
            fs::rename(&temporary_output, &output)?;
            println!("{report}");
            println!("Output: {}", output.display());
            println!("Total elapsed: {:.1?}", started.elapsed());
            Ok(())
        }
        Err(error) => {
            if let Err(cleanup_error) = fs::remove_dir_all(&temporary_output)
                && cleanup_error.kind() != io::ErrorKind::NotFound
            {
                eprintln!(
                    "warning: failed to remove {}: {cleanup_error}",
                    temporary_output.display()
                );
            }
            Err(error)
        }
    }
}

#[allow(clippy::too_many_lines)]
fn audit_and_build(input: &Path, output: &Path) -> Result<String> {
    eprintln!("Pass 1/2: aggregating exact names and row distributions");
    let first = first_pass(input)?;
    eprintln!(
        "Pass 1 complete: {} rows, {} names, {} observations, {:.1?}",
        first.row_count,
        first.name_ids.len(),
        first.total_observations,
        first.elapsed
    );

    let mut names_by_id: Vec<&[u8]> = vec![&[]; first.name_ids.len()];
    for (name, &id) in &first.name_ids {
        names_by_id[id as usize] = name;
    }

    fs::create_dir(output)?;
    let audit_directory = output.join("audit");
    let artifacts_directory = output.join("artifacts");
    fs::create_dir(&audit_directory)?;
    fs::create_dir(&artifacts_directory)?;

    eprintln!("Auditing name-frequency bands, tail samples, and string features");
    let audit_started = Instant::now();
    let name_bands = audit_names(&audit_directory, &first, &names_by_id)?;
    write_frequency_bands(&audit_directory, &name_bands, &first)?;
    write_row_frequency_bands(&audit_directory, &first)?;
    let audit_elapsed = audit_started.elapsed();

    eprintln!("Computing independent routing hashes and membership fingerprints");
    let routing_hashes: Vec<u64> = names_by_id
        .iter()
        .map(|name| xxh3_64_with_seed(name, ROUTING_SEED))
        .collect();
    check_routing_collisions(&routing_hashes, &names_by_id)?;
    let fingerprints: Vec<u32> = names_by_id.iter().map(|name| fingerprint(name)).collect();
    let (duplicate_fingerprint_values, duplicate_fingerprint_pairs) =
        fingerprint_collision_stats(&fingerprints);
    eprintln!(
        "32-bit fingerprint duplicates: {duplicate_fingerprint_values} values, \
         {duplicate_fingerprint_pairs} colliding pairs (allowed)"
    );

    eprintln!("Pass 2/2: packing source metadata rows in memory");
    let pass_two_started = Instant::now();
    let rows = second_pass(input, &first)?;
    let pass_two_elapsed = pass_two_started.elapsed();
    eprintln!("Pass 2 complete: {pass_two_elapsed:.1?}");

    eprintln!("Computing global/country-row threshold policy matrix");
    let policy_started = Instant::now();
    let policy = compute_policy_matrix(&first, &rows)?;
    write_policy_matrix(&audit_directory, &policy, first.total_observations)?;
    let policy_elapsed = policy_started.elapsed();

    let mut artifact_stats = Vec::with_capacity(THRESHOLDS.len());
    for threshold in THRESHOLDS {
        eprintln!("Building C32 artifact for global count >= {threshold}");
        artifact_stats.push(build_artifact(
            &artifacts_directory,
            threshold,
            &first,
            &names_by_id,
            &routing_hashes,
            &fingerprints,
            &rows,
        )?);
    }
    write_artifact_stats(&audit_directory, &artifact_stats, first.total_observations)?;

    Ok(format_report(
        input,
        &first,
        &name_bands,
        &artifact_stats,
        duplicate_fingerprint_values,
        duplicate_fingerprint_pairs,
        audit_elapsed,
        pass_two_elapsed,
        policy_elapsed,
    ))
}

fn open_csv(path: &Path) -> Result<Reader<File>> {
    let mut reader = ReaderBuilder::new().from_path(path)?;
    let headers = reader.byte_headers()?;
    let expected = ByteRecord::from(vec!["name", "country", "gender", "count"]);
    if headers != &expected {
        return Err(format!("unexpected CSV header: {headers:?}").into());
    }
    Ok(reader)
}

fn first_pass(path: &Path) -> Result<FirstPass> {
    let started = Instant::now();
    let mut reader = open_csv(path)?;
    let mut name_ids = HashMap::<Box<[u8]>, u32>::with_capacity(31_000_000);
    let mut rows_per_name = Vec::<u32>::with_capacity(31_000_000);
    let mut observations_per_name = Vec::<u64>::with_capacity(31_000_000);
    let mut country_codes = BTreeSet::<u16>::new();
    let mut row_count = 0_u32;
    let mut total_observations = 0_u128;
    let mut max_row_count = 0_u32;
    let mut row_bands = [BandStats::default(); BAND_LABELS.len()];

    for result in reader.byte_records() {
        let record = result?;
        let (name, country, _gender, count) = parse_record(&record, u64::from(row_count) + 2)?;
        if count == 0 {
            return Err(format!("line {}: zero count", u64::from(row_count) + 2).into());
        }
        let id = if let Some(&id) = name_ids.get(name) {
            id
        } else {
            let id = u32::try_from(name_ids.len())?;
            name_ids.insert(name.to_vec().into_boxed_slice(), id);
            rows_per_name.push(0);
            observations_per_name.push(0);
            id
        };
        rows_per_name[id as usize] = rows_per_name[id as usize]
            .checked_add(1)
            .ok_or("too many rows for one name")?;
        observations_per_name[id as usize] = observations_per_name[id as usize]
            .checked_add(u64::from(count))
            .ok_or("per-name observation count overflow")?;
        country_codes.insert(country);
        total_observations += u128::from(count);
        max_row_count = max_row_count.max(count);
        let band = frequency_band(u64::from(count));
        row_bands[band].keys_or_rows += 1;
        row_bands[band].observations += u128::from(count);
        row_count = row_count.checked_add(1).ok_or("more than u32::MAX rows")?;
        if u64::from(row_count).is_multiple_of(PROGRESS_ROWS) {
            eprintln!("  pass 1: {row_count} rows");
        }
    }

    Ok(FirstPass {
        name_ids,
        rows_per_name,
        observations_per_name,
        country_codes: country_codes.into_iter().collect(),
        row_count,
        total_observations,
        max_row_count,
        row_bands,
        elapsed: started.elapsed(),
    })
}

fn second_pass(path: &Path, first: &FirstPass) -> Result<Rows> {
    let mut offsets_by_id = Vec::with_capacity(first.rows_per_name.len() + 1);
    offsets_by_id.push(0_u32);
    for &count in &first.rows_per_name {
        let next = offsets_by_id
            .last()
            .copied()
            .ok_or("missing initial row offset")?
            .checked_add(count)
            .ok_or("row offset overflow")?;
        offsets_by_id.push(next);
    }
    if offsets_by_id.last().copied() != Some(first.row_count) {
        return Err("row offsets do not cover the source rows".into());
    }

    let row_count = usize::try_from(first.row_count)?;
    let mut countries = vec![0_u8; row_count];
    let mut genders = vec![0_u8; row_count];
    let mut counts = vec![0_u32; row_count];
    let mut positions = offsets_by_id[..first.rows_per_name.len()].to_vec();
    let mut country_ids = vec![u8::MAX; 65_536].into_boxed_slice();
    for (id, &code) in first.country_codes.iter().enumerate() {
        country_ids[usize::from(code)] = u8::try_from(id)?;
    }

    let mut reader = open_csv(path)?;
    let mut seen_rows = 0_u32;
    for result in reader.byte_records() {
        let record = result?;
        let (name, country, gender, count) = parse_record(&record, u64::from(seen_rows) + 2)?;
        let id = *first
            .name_ids
            .get(name)
            .ok_or("name disappeared between CSV passes")?;
        let position = positions[id as usize];
        if position >= offsets_by_id[id as usize + 1] {
            return Err(format!("too many second-pass rows for name ID {id}").into());
        }
        let position = usize::try_from(position)?;
        let country_id = country_ids[usize::from(country)];
        if country_id == u8::MAX {
            return Err("country disappeared between CSV passes".into());
        }
        countries[position] = country_id;
        genders[position] = gender;
        counts[position] = count;
        positions[id as usize] += 1;
        seen_rows += 1;
        if u64::from(seen_rows).is_multiple_of(PROGRESS_ROWS) {
            eprintln!("  pass 2: {seen_rows} rows");
        }
    }
    if seen_rows != first.row_count {
        return Err("CSV row count changed between passes".into());
    }

    Ok(Rows {
        offsets_by_id,
        countries,
        genders,
        counts,
    })
}

fn parse_record(record: &ByteRecord, line: u64) -> Result<(&[u8], u16, u8, u32)> {
    if record.len() != 4 {
        return Err(format!("line {line}: expected 4 fields, got {}", record.len()).into());
    }
    let name = record.get(0).ok_or("missing name")?;
    if name.is_empty() {
        return Err(format!("line {line}: empty name").into());
    }
    std::str::from_utf8(name)
        .map_err(|error| format!("line {line}: name is not UTF-8: {error}"))?;
    let country = record.get(1).ok_or("missing country")?;
    if country.len() != 2 || !country.iter().all(u8::is_ascii_uppercase) {
        return Err(format!("line {line}: invalid country code {country:?}").into());
    }
    let country = u16::from_be_bytes([country[0], country[1]]);
    let gender = match record.get(2).ok_or("missing gender")? {
        b"" => 0,
        b"F" => 1,
        b"M" => 2,
        value => return Err(format!("line {line}: invalid gender {value:?}").into()),
    };
    let count_bytes = record.get(3).ok_or("missing count")?;
    let count = std::str::from_utf8(count_bytes)
        .map_err(|error| format!("line {line}: count is not UTF-8: {error}"))?
        .parse::<u32>()
        .map_err(|error| format!("line {line}: invalid count: {error}"))?;
    Ok((name, country, gender, count))
}

#[allow(clippy::too_many_lines)]
fn audit_names(
    directory: &Path,
    first: &FirstPass,
    names: &[&[u8]],
) -> Result<[BandStats; BAND_LABELS.len()]> {
    let mut bands = [BandStats::default(); BAND_LABELS.len()];
    let mut band_samplers: Vec<Sampler> = (0..BAND_LABELS.len())
        .map(|band| Sampler::new(SAMPLE_SIZE, SAMPLE_SEED ^ u64::try_from(band).unwrap()))
        .collect();
    let feature_names = [
        "non_ascii",
        "whitespace",
        "number",
        "punctuation",
        "symbol",
        "mark",
        "control_or_format",
        "private_or_unassigned",
    ];
    let mut feature_stats = [FeatureStats::default(); 8];
    let mut feature_samplers: Vec<Sampler> = (0..feature_names.len())
        .map(|feature| {
            Sampler::new(
                FEATURE_SAMPLE_SIZE,
                SAMPLE_SEED ^ 0x8000 ^ u64::try_from(feature).unwrap(),
            )
        })
        .collect();
    let mut longest_bytes = BinaryHeap::<Reverse<(usize, u64, u32)>>::new();
    let mut longest_chars = BinaryHeap::<Reverse<(usize, u64, u32)>>::new();

    for (id, name) in names.iter().enumerate() {
        let id = u32::try_from(id)?;
        let observations = first.observations_per_name[id as usize];
        let metadata_rows = first.rows_per_name[id as usize];
        let band = frequency_band(observations);
        bands[band].keys_or_rows += 1;
        bands[band].metadata_rows += u64::from(metadata_rows);
        bands[band].observations += u128::from(observations);
        band_samplers[band].observe(id, name);

        let text = std::str::from_utf8(name)?;
        for (index, present) in classify(text).into_iter().enumerate() {
            if present {
                feature_stats[index].names += 1;
                feature_stats[index].metadata_rows += u64::from(metadata_rows);
                feature_stats[index].observations += u128::from(observations);
                feature_samplers[index].observe(id, name);
            }
        }
        let score = xxh3_64_with_seed(name, SAMPLE_SEED ^ 0x4c4f_4e47);
        observe_longest(&mut longest_bytes, name.len(), score, id);
        observe_longest(&mut longest_chars, text.chars().count(), score, id);
    }

    write_tail_samples(directory, &band_samplers, first, names)?;
    write_feature_summary(directory, &feature_names, &feature_stats, first)?;
    write_feature_samples(directory, &feature_names, &feature_samplers, first, names)?;
    write_longest_names(directory, &longest_bytes, &longest_chars, first, names)?;
    Ok(bands)
}

fn frequency_band(count: u64) -> usize {
    match count {
        0 | 1 => 0,
        2 => 1,
        3..=4 => 2,
        5..=9 => 3,
        10..=19 => 4,
        20..=49 => 5,
        50..=99 => 6,
        100..=499 => 7,
        _ => 8,
    }
}

fn classify(text: &str) -> [bool; 8] {
    let mut features = [false; 8];
    features[0] = !text.is_ascii();
    for character in text.chars() {
        let abbreviation = get_general_category(character).abbreviation();
        features[1] |= character.is_whitespace() || abbreviation.starts_with('Z');
        features[2] |= abbreviation.starts_with('N');
        features[3] |= abbreviation.starts_with('P');
        features[4] |= abbreviation.starts_with('S');
        features[5] |= abbreviation.starts_with('M');
        features[6] |= matches!(abbreviation, "Cc" | "Cf" | "Cs");
        features[7] |= matches!(abbreviation, "Co" | "Cn");
    }
    features
}

fn observe_longest(
    heap: &mut BinaryHeap<Reverse<(usize, u64, u32)>>,
    length: usize,
    score: u64,
    id: u32,
) {
    let entry = Reverse((length, score, id));
    if heap.len() < LONGEST_SIZE {
        heap.push(entry);
    } else if heap.peek().is_some_and(|smallest| entry.0 > smallest.0) {
        heap.pop();
        heap.push(entry);
    }
}

fn write_frequency_bands(
    directory: &Path,
    bands: &[BandStats; BAND_LABELS.len()],
    first: &FirstPass,
) -> Result<()> {
    let mut writer = Writer::from_path(directory.join("global_frequency_bands.csv"))?;
    writer.write_record([
        "frequency_band",
        "distinct_names",
        "metadata_rows",
        "observations",
        "name_percentage",
        "observation_percentage",
    ])?;
    for (label, stats) in BAND_LABELS.iter().zip(bands) {
        writer.write_record([
            *label,
            &stats.keys_or_rows.to_string(),
            &stats.metadata_rows.to_string(),
            &stats.observations.to_string(),
            &format_percentage(u128::from(stats.keys_or_rows), first.name_ids.len() as u128),
            &format_percentage(stats.observations, first.total_observations),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn write_row_frequency_bands(directory: &Path, first: &FirstPass) -> Result<()> {
    let mut writer = Writer::from_path(directory.join("country_gender_row_frequency_bands.csv"))?;
    writer.write_record([
        "frequency_band",
        "metadata_rows",
        "observations",
        "row_percentage",
        "observation_percentage",
    ])?;
    for (label, stats) in BAND_LABELS.iter().zip(&first.row_bands) {
        writer.write_record([
            *label,
            &stats.keys_or_rows.to_string(),
            &stats.observations.to_string(),
            &format_percentage(u128::from(stats.keys_or_rows), u128::from(first.row_count)),
            &format_percentage(stats.observations, first.total_observations),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn write_tail_samples(
    directory: &Path,
    samplers: &[Sampler],
    first: &FirstPass,
    names: &[&[u8]],
) -> Result<()> {
    let mut writer = Writer::from_path(directory.join("frequency_samples.csv"))?;
    writer.write_record([
        "frequency_band",
        "global_count",
        "metadata_rows",
        "byte_length",
        "character_length",
        "name",
    ])?;
    for (label, sampler) in BAND_LABELS.iter().zip(samplers) {
        for id in sampler.sorted_ids() {
            let name = std::str::from_utf8(names[id as usize])?;
            writer.write_record([
                *label,
                &first.observations_per_name[id as usize].to_string(),
                &first.rows_per_name[id as usize].to_string(),
                &name.len().to_string(),
                &name.chars().count().to_string(),
                name,
            ])?;
        }
    }
    writer.flush()?;
    Ok(())
}

fn write_feature_summary(
    directory: &Path,
    names: &[&str],
    stats: &[FeatureStats],
    first: &FirstPass,
) -> Result<()> {
    let mut writer = Writer::from_path(directory.join("feature_summary.csv"))?;
    writer.write_record([
        "feature",
        "distinct_names",
        "metadata_rows",
        "observations",
        "name_percentage",
        "observation_percentage",
    ])?;
    for (name, stats) in names.iter().zip(stats) {
        writer.write_record([
            *name,
            &stats.names.to_string(),
            &stats.metadata_rows.to_string(),
            &stats.observations.to_string(),
            &format_percentage(u128::from(stats.names), first.name_ids.len() as u128),
            &format_percentage(stats.observations, first.total_observations),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn write_feature_samples(
    directory: &Path,
    feature_names: &[&str],
    samplers: &[Sampler],
    first: &FirstPass,
    names: &[&[u8]],
) -> Result<()> {
    let mut writer = Writer::from_path(directory.join("feature_samples.csv"))?;
    writer.write_record(["feature", "global_count", "metadata_rows", "name"])?;
    for (feature, sampler) in feature_names.iter().zip(samplers) {
        for id in sampler.sorted_ids() {
            writer.write_record([
                *feature,
                &first.observations_per_name[id as usize].to_string(),
                &first.rows_per_name[id as usize].to_string(),
                std::str::from_utf8(names[id as usize])?,
            ])?;
        }
    }
    writer.flush()?;
    Ok(())
}

fn write_longest_names(
    directory: &Path,
    longest_bytes: &BinaryHeap<Reverse<(usize, u64, u32)>>,
    longest_chars: &BinaryHeap<Reverse<(usize, u64, u32)>>,
    first: &FirstPass,
    names: &[&[u8]],
) -> Result<()> {
    let mut writer = Writer::from_path(directory.join("longest_names.csv"))?;
    writer.write_record([
        "ranking",
        "byte_length",
        "character_length",
        "global_count",
        "metadata_rows",
        "name",
    ])?;
    for (ranking, heap) in [("bytes", longest_bytes), ("characters", longest_chars)] {
        let mut entries: Vec<_> = heap.iter().map(|entry| entry.0).collect();
        entries.sort_unstable_by(|left, right| right.cmp(left));
        for (_, _, id) in entries {
            let name = std::str::from_utf8(names[id as usize])?;
            writer.write_record([
                ranking,
                &name.len().to_string(),
                &name.chars().count().to_string(),
                &first.observations_per_name[id as usize].to_string(),
                &first.rows_per_name[id as usize].to_string(),
                name,
            ])?;
        }
    }
    writer.flush()?;
    Ok(())
}

fn check_routing_collisions(hashes: &[u64], names: &[&[u8]]) -> Result<()> {
    let mut sorted = hashes.to_vec();
    sorted.sort_unstable();
    if let Some(collision) = sorted.windows(2).find(|pair| pair[0] == pair[1]) {
        let hash = collision[0];
        let colliding_names: Vec<_> = hashes
            .iter()
            .enumerate()
            .filter(|(_, candidate)| **candidate == hash)
            .map(|(id, _)| String::from_utf8_lossy(names[id]).into_owned())
            .collect();
        return Err(format!(
            "64-bit routing collision for hash {hash}: {colliding_names:?}; change ROUTING_SEED"
        )
        .into());
    }
    Ok(())
}

fn fingerprint_collision_stats(fingerprints: &[u32]) -> (u64, u64) {
    let mut sorted = fingerprints.to_vec();
    sorted.sort_unstable();
    let mut duplicate_values = 0_u64;
    let mut pairs = 0_u64;
    let mut start = 0_usize;
    while start < sorted.len() {
        let mut end = start + 1;
        while end < sorted.len() && sorted[end] == sorted[start] {
            end += 1;
        }
        let count = u64::try_from(end - start).expect("fingerprint group length fits u64");
        if count > 1 {
            duplicate_values += 1;
            pairs += count * (count - 1) / 2;
        }
        start = end;
    }
    (duplicate_values, pairs)
}

fn compute_policy_matrix(
    first: &FirstPass,
    rows: &Rows,
) -> Result<[[PolicyStats; THRESHOLDS.len()]; THRESHOLDS.len()]> {
    let mut matrix = [[PolicyStats::default(); THRESHOLDS.len()]; THRESHOLDS.len()];
    for id in 0..first.name_ids.len() {
        let total = first.observations_per_name[id];
        let mut retained_rows = [0_u64; THRESHOLDS.len()];
        let mut retained_observations = [0_u128; THRESHOLDS.len()];
        let range =
            usize::try_from(rows.offsets_by_id[id])?..usize::try_from(rows.offsets_by_id[id + 1])?;
        for position in range {
            let count = u64::from(rows.counts[position]);
            for (index, &threshold) in THRESHOLDS.iter().enumerate() {
                if count < threshold {
                    break;
                }
                retained_rows[index] += 1;
                retained_observations[index] += u128::from(count);
            }
        }
        for (global_index, &global_threshold) in THRESHOLDS.iter().enumerate() {
            if total < global_threshold {
                break;
            }
            for row_index in 0..THRESHOLDS.len() {
                if retained_rows[row_index] != 0 {
                    matrix[global_index][row_index].keys += 1;
                    matrix[global_index][row_index].metadata_rows += retained_rows[row_index];
                    matrix[global_index][row_index].observations +=
                        retained_observations[row_index];
                }
            }
        }
        if (id as u64 + 1).is_multiple_of(PROGRESS_ROWS) {
            eprintln!("  policy matrix: {} names", id + 1);
        }
    }
    Ok(matrix)
}

fn write_policy_matrix(
    directory: &Path,
    matrix: &[[PolicyStats; THRESHOLDS.len()]; THRESHOLDS.len()],
    total_observations: u128,
) -> Result<()> {
    let mut writer = Writer::from_path(directory.join("threshold_policy_matrix.csv"))?;
    writer.write_record([
        "minimum_global_count",
        "minimum_country_gender_row_count",
        "distinct_names",
        "metadata_rows",
        "observations_retained",
        "observation_percentage",
    ])?;
    for (global_index, &global_threshold) in THRESHOLDS.iter().enumerate() {
        for (row_index, &row_threshold) in THRESHOLDS.iter().enumerate() {
            let stats = matrix[global_index][row_index];
            writer.write_record([
                &global_threshold.to_string(),
                &row_threshold.to_string(),
                &stats.keys.to_string(),
                &stats.metadata_rows.to_string(),
                &stats.observations.to_string(),
                &format_percentage(stats.observations, total_observations),
            ])?;
        }
    }
    writer.flush()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_artifact(
    artifacts_directory: &Path,
    threshold: u64,
    first: &FirstPass,
    names: &[&[u8]],
    routing_hashes: &[u64],
    fingerprints: &[u32],
    rows: &Rows,
) -> Result<ArtifactStats> {
    let started = Instant::now();
    let selected_ids: Vec<u32> = first
        .observations_per_name
        .iter()
        .enumerate()
        .filter(|(_, count)| **count >= threshold)
        .map(|(id, _)| u32::try_from(id))
        .collect::<std::result::Result<_, _>>()?;
    let selected_hashes: Vec<u64> = selected_ids
        .iter()
        .map(|&id| routing_hashes[id as usize])
        .collect();
    let mphf = Mphf::new_parallel(MPHF_GAMMA, &selected_hashes, None);
    let mut ids_by_slot = vec![u32::MAX; selected_ids.len()];
    let mut fingerprints_by_slot = vec![0_u32; selected_ids.len()];
    for (&id, &routing_hash) in selected_ids.iter().zip(&selected_hashes) {
        let slot = usize::try_from(mphf.hash(&routing_hash))?;
        if ids_by_slot[slot] != u32::MAX {
            return Err(format!("MPHF assigned more than one key to slot {slot}").into());
        }
        ids_by_slot[slot] = id;
        fingerprints_by_slot[slot] = fingerprints[id as usize];
    }

    let directory = artifacts_directory.join(format!("min-global-{threshold:03}"));
    fs::create_dir(&directory)?;
    write_country_dictionary(&directory, &first.country_codes)?;
    let mut mphf_writer = BufWriter::new(File::create(directory.join("names.mphf"))?);
    bincode::serialize_into(&mut mphf_writer, &mphf)?;
    mphf_writer.flush()?;
    let mut fingerprint_writer = BufWriter::new(File::create(directory.join("fingerprints.u32"))?);
    for fingerprint in &fingerprints_by_slot {
        fingerprint_writer.write_all(&fingerprint.to_le_bytes())?;
    }
    fingerprint_writer.flush()?;

    let (metadata_rows, observations, quantization) =
        write_quantized_metadata(&directory, &ids_by_slot, rows, first.max_row_count)?;
    validate_artifact(
        &directory,
        &mphf,
        &selected_ids,
        names,
        routing_hashes,
        fingerprints,
        &fingerprints_by_slot,
        metadata_rows,
    )?;

    let mphf_bytes = fs::metadata(directory.join("names.mphf"))?.len();
    let fingerprint_bytes = fs::metadata(directory.join("fingerprints.u32"))?.len();
    let bytes = directory_size(&directory)?;
    Ok(ArtifactStats {
        threshold,
        keys: selected_ids.len(),
        metadata_rows,
        observations,
        bytes,
        mphf_bytes,
        fingerprint_bytes,
        metadata_bytes: bytes - mphf_bytes - fingerprint_bytes,
        build_elapsed: started.elapsed(),
        quantization,
    })
}

fn write_country_dictionary(directory: &Path, country_codes: &[u16]) -> Result<()> {
    let mut writer = BufWriter::new(File::create(directory.join("countries.dict"))?);
    for &code in country_codes {
        writer.write_all(&code.to_be_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

fn write_quantized_metadata(
    directory: &Path,
    order: &[u32],
    rows: &Rows,
    max_count: u32,
) -> Result<(u64, u128, QuantizationStats)> {
    let mut row_offsets = Vec::with_capacity(order.len() + 1);
    row_offsets.push(0_u32);
    let mut country_writer = BufWriter::new(File::create(directory.join("country_ids.u8"))?);
    let gender_file = File::create(directory.join("genders.2bit"))?;
    let mut gender_writer = TwoBitWriter::new(BufWriter::new(gender_file));
    let mut count_writer = BufWriter::new(File::create(directory.join("counts.q8"))?);
    let mut stats = QuantizationStats {
        relative_error_histogram: vec![0; 10_001],
        ..QuantizationStats::default()
    };
    let mut observations = 0_u128;

    for &id in order {
        let id = usize::try_from(id)?;
        let range =
            usize::try_from(rows.offsets_by_id[id])?..usize::try_from(rows.offsets_by_id[id + 1])?;
        for position in range.clone() {
            country_writer.write_all(&[rows.countries[position]])?;
            gender_writer.write(rows.genders[position])?;
            let original = rows.counts[position];
            let quantized = quantize_count(original, max_count);
            let decoded = dequantize_count(quantized, max_count);
            count_writer.write_all(&[quantized])?;
            stats.observe(original, decoded);
            observations += u128::from(original);
        }
        row_offsets.push(
            row_offsets
                .last()
                .copied()
                .ok_or("missing row offset")?
                .checked_add(u32::try_from(range.len())?)
                .ok_or("row offset overflow")?,
        );
    }
    country_writer.flush()?;
    gender_writer.finish()?;
    count_writer.flush()?;
    write_u32_file(&directory.join("row_offsets.u32"), &row_offsets)?;
    fs::write(
        directory.join("quantization_max_count.u32"),
        max_count.to_le_bytes(),
    )?;
    Ok((stats.rows, observations, stats))
}

struct TwoBitWriter<W: Write> {
    inner: W,
    byte: u8,
    occupied_slots: u8,
}

impl<W: Write> TwoBitWriter<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            byte: 0,
            occupied_slots: 0,
        }
    }

    fn write(&mut self, value: u8) -> Result<()> {
        if value > 3 {
            return Err(format!("2-bit value out of range: {value}").into());
        }
        self.byte |= value << (self.occupied_slots * 2);
        self.occupied_slots += 1;
        if self.occupied_slots == 4 {
            self.inner.write_all(&[self.byte])?;
            self.byte = 0;
            self.occupied_slots = 0;
        }
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        if self.occupied_slots != 0 {
            self.inner.write_all(&[self.byte])?;
        }
        self.inner.flush()?;
        Ok(())
    }
}

fn write_u32_file(path: &Path, values: &[u32]) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    for &value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn quantize_count(count: u32, max_count: u32) -> u8 {
    if count == 0 || max_count <= 1 {
        return 0;
    }
    let position = f64::from(count).ln() / f64::from(max_count).ln();
    (1.0 + position * 254.0).round().clamp(1.0, 255.0) as u8
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn dequantize_count(value: u8, max_count: u32) -> u32 {
    if value == 0 || max_count <= 1 {
        return u32::from(value != 0);
    }
    let position = (f64::from(value) - 1.0) / 254.0;
    f64::from(max_count)
        .ln()
        .mul_add(position, 0.0)
        .exp()
        .round()
        .clamp(1.0, f64::from(max_count)) as u32
}

impl QuantizationStats {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn observe(&mut self, original: u32, decoded: u32) {
        let absolute = original.abs_diff(decoded);
        let relative = f64::from(absolute) / f64::from(original);
        self.absolute_error_sum += u128::from(absolute);
        self.original_sum += u128::from(original);
        self.decoded_sum += u128::from(decoded);
        self.relative_error_sum += relative;
        self.max_absolute_error = self.max_absolute_error.max(u64::from(absolute));
        self.max_relative_error = self.max_relative_error.max(relative);
        let histogram_index = (relative * 10_000.0).round().clamp(0.0, 10_000.0) as usize;
        self.relative_error_histogram[histogram_index] += 1;
        self.rows += 1;
    }

    fn percentile_relative_error(&self, percentile: u64) -> f64 {
        let target = self.rows.saturating_mul(percentile).div_ceil(100);
        let mut cumulative = 0_u64;
        for (index, &count) in self.relative_error_histogram.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                return f64::from(u32::try_from(index).expect("histogram index fits u32"))
                    / 10_000.0;
            }
        }
        1.0
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_artifact(
    directory: &Path,
    mphf: &Mphf<u64>,
    selected_ids: &[u32],
    names: &[&[u8]],
    routing_hashes: &[u64],
    fingerprints: &[u32],
    fingerprints_by_slot: &[u32],
    metadata_rows: u64,
) -> Result<()> {
    let reloaded: Mphf<u64> =
        bincode::deserialize_from(BufReader::new(File::open(directory.join("names.mphf"))?))?;
    for &id in selected_ids {
        let slot = usize::try_from(reloaded.hash(&routing_hashes[id as usize]))?;
        if fingerprints_by_slot[slot] != fingerprints[id as usize] {
            return Err(format!(
                "known-name membership mismatch for {:?}",
                String::from_utf8_lossy(names[id as usize])
            )
            .into());
        }
        if mphf.hash(&routing_hashes[id as usize]) != reloaded.hash(&routing_hashes[id as usize]) {
            return Err("serialized MPHF changed a known-key slot".into());
        }
    }
    for unknown in [
        b"definitely-not-a-known-name".as_slice(),
        b"supercalifragilisticexpialidocious".as_slice(),
        b"__bonjour_unknown_probe__".as_slice(),
    ] {
        let routing = xxh3_64_with_seed(unknown, ROUTING_SEED);
        let fingerprint = fingerprint(unknown);
        let slot = usize::try_from(reloaded.hash(&routing))?;
        if fingerprints_by_slot[slot] == fingerprint {
            return Err("unknown-name fingerprint unexpectedly passed membership check".into());
        }
    }
    expect_file_size(
        &directory.join("fingerprints.u32"),
        u64::try_from(selected_ids.len())? * 4,
    )?;
    expect_file_size(
        &directory.join("row_offsets.u32"),
        u64::try_from(selected_ids.len() + 1)? * 4,
    )?;
    expect_file_size(&directory.join("country_ids.u8"), metadata_rows)?;
    expect_file_size(&directory.join("counts.q8"), metadata_rows)?;
    expect_file_size(&directory.join("genders.2bit"), metadata_rows.div_ceil(4))?;
    Ok(())
}

fn expect_file_size(path: &Path, expected: u64) -> Result<()> {
    let actual = fs::metadata(path)?.len();
    if actual != expected {
        return Err(format!("{} has size {actual}, expected {expected}", path.display()).into());
    }
    Ok(())
}

fn directory_size(path: &Path) -> Result<u64> {
    let mut total = 0_u64;
    for entry in fs::read_dir(path)? {
        let metadata = entry?.metadata()?;
        if metadata.is_file() {
            total = total.checked_add(metadata.len()).ok_or("size overflow")?;
        }
    }
    Ok(total)
}

fn write_artifact_stats(
    directory: &Path,
    artifacts: &[ArtifactStats],
    total_observations: u128,
) -> Result<()> {
    let mut writer = Writer::from_path(directory.join("c32_artifact_sizes.csv"))?;
    writer.write_record([
        "minimum_global_count",
        "distinct_names",
        "metadata_rows",
        "observations_retained",
        "observation_percentage",
        "total_bytes",
        "total_mib",
        "mphf_bytes",
        "fingerprint_bytes",
        "metadata_bytes",
        "q8_p99_relative_error_percentage",
        "build_seconds",
    ])?;
    for artifact in artifacts {
        writer.write_record([
            &artifact.threshold.to_string(),
            &artifact.keys.to_string(),
            &artifact.metadata_rows.to_string(),
            &artifact.observations.to_string(),
            &format_percentage(artifact.observations, total_observations),
            &artifact.bytes.to_string(),
            &format!("{:.2}", mib(artifact.bytes)),
            &artifact.mphf_bytes.to_string(),
            &artifact.fingerprint_bytes.to_string(),
            &artifact.metadata_bytes.to_string(),
            &format!(
                "{:.4}",
                artifact.quantization.percentile_relative_error(99) * 100.0
            ),
            &format!("{:.3}", artifact.build_elapsed.as_secs_f64()),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

#[allow(clippy::too_many_arguments, clippy::cast_precision_loss)]
fn format_report(
    input: &Path,
    first: &FirstPass,
    name_bands: &[BandStats; BAND_LABELS.len()],
    artifacts: &[ArtifactStats],
    duplicate_fingerprint_values: u64,
    duplicate_fingerprint_pairs: u64,
    audit_elapsed: Duration,
    pass_two_elapsed: Duration,
    policy_elapsed: Duration,
) -> String {
    let mut report = format!(
        "# Name corpus audit and C32 benchmark\n\n\
         Input: `{}` (opened read-only)\n\n\
         - Source metadata rows: {}\n\
         - Distinct exact names: {}\n\
         - Total observations: {}\n\
         - Countries: {}\n\
         - Maximum country/gender-row count: {}\n\n\
         ## Global frequency bands\n\n\
         | Frequency | Names | Name share | Observations | Observation share |\n\
         |---:|---:|---:|---:|---:|\n",
        input.display(),
        first.row_count,
        first.name_ids.len(),
        first.total_observations,
        first.country_codes.len(),
        first.max_row_count,
    );
    for (label, stats) in BAND_LABELS.iter().zip(name_bands) {
        writeln!(
            report,
            "| {label} | {} | {}% | {} | {}% |",
            stats.keys_or_rows,
            format_percentage(u128::from(stats.keys_or_rows), first.name_ids.len() as u128),
            stats.observations,
            format_percentage(stats.observations, first.total_observations),
        )
        .expect("writing to a String cannot fail");
    }
    report.push_str(
        "\n## C + independent 32-bit fingerprint + q8\n\n\
         | Minimum global count | Names | Metadata rows | Observation share | Direct size |\n\
         |---:|---:|---:|---:|---:|\n",
    );
    for artifact in artifacts {
        writeln!(
            report,
            "| {} | {} | {} | {}% | {:.2} MiB |",
            artifact.threshold,
            artifact.keys,
            artifact.metadata_rows,
            format_percentage(artifact.observations, first.total_observations),
            mib(artifact.bytes),
        )
        .expect("writing to a String cannot fail");
    }
    write!(
        report,
        "\nThe MPHF uses a seeded 64-bit routing hash. Its seed produced no collision \
         across the full corpus. Membership uses a separately seeded 32-bit \
         fingerprint. Duplicate stored fingerprints are allowed: \
         {duplicate_fingerprint_values} fingerprint values account for \
         {duplicate_fingerprint_pairs} colliding stored-key pairs.\n\n\
         These are structural coverage measurements, not classifier precision or \
         recall. No held-out labeled evaluation corpus exists in this repository.\n\n\
         ## Timings\n\n\
         - CSV aggregation pass: {:.1?}\n\
         - Name audit and samples: {audit_elapsed:.1?}\n\
         - CSV metadata pass: {pass_two_elapsed:.1?}\n\
         - Threshold policy matrix: {policy_elapsed:.1?}\n",
        first.elapsed,
    )
    .expect("writing to a String cannot fail");
    report
}

#[allow(clippy::cast_possible_truncation)]
fn fingerprint(name: &[u8]) -> u32 {
    xxh3_64_with_seed(name, FINGERPRINT_SEED) as u32
}

#[allow(clippy::cast_precision_loss)]
fn format_percentage(numerator: u128, denominator: u128) -> String {
    if denominator == 0 {
        return "0.000000".to_string();
    }
    format!("{:.6}", numerator as f64 / denominator as f64 * 100.0)
}

#[allow(clippy::cast_precision_loss)]
fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}
