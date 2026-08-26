use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime};

use boomphf::Mphf;
use csv::{ByteRecord, ReaderBuilder, Writer};
use sha2::{Digest, Sha256};
use xxhash_rust::xxh3::xxh3_64_with_seed;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const ROUTING_SEED: u64 = 0x6e61_6d65_2d72_6f75;
const FINGERPRINT_SEED: u64 = 0x6e61_6d65_2d66_7033;
const PROGRESS_ROWS: u64 = 10_000_000;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct GivenPair {
    name_id: u32,
    country: u16,
    counts: [u64; 3],
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct SurnamePair {
    name_id: u32,
    country: u16,
    count: u64,
}

struct CleanData {
    names: Vec<Box<[u8]>>,
    name_ids: HashMap<Box<[u8]>, u32>,
    global_given: Vec<u64>,
    given_pairs: Vec<GivenPair>,
    source_rows: u64,
    total_given: u128,
}

#[derive(Clone, Copy, Debug, Default)]
struct CountryStats {
    clean_given: u128,
    raw_rows: u64,
    nonempty_surnames: u64,
    matched_surnames: u64,
    matched_keys: u64,
}

struct SurnameData {
    pairs: Vec<SurnamePair>,
    global_counts: Vec<u64>,
    countries: BTreeMap<u16, CountryStats>,
    raw_files: usize,
    raw_rows: u64,
    nonempty_surnames: u64,
    matched_surnames: u64,
}

#[derive(Clone)]
struct FileSnapshot {
    path: PathBuf,
    bytes: u64,
    modified: Option<SystemTime>,
}

#[derive(Clone)]
struct Checksum {
    bytes: u64,
    sha256: String,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct V2Stats {
    rows: u64,
    both_rows: u64,
    given_only_rows: u64,
    surname_only_country_rows: u64,
    given_sum: u128,
    surname_sum: u128,
}

#[derive(Clone, Debug, Default)]
struct QuantizationStats {
    rows: u64,
    original_sum: u128,
    decoded_sum: u128,
    relative_error_sum: f64,
    max_relative_error: f64,
    histogram: Vec<u64>,
}

#[derive(Clone, Copy)]
struct Sizes {
    clean_v2_csv: u64,
    clean_v2_gzip: u64,
    clean_v2_zstd: u64,
    baseline_direct: u64,
    global_direct: u64,
    global_archive: u64,
    country_direct: u64,
    country_archive: u64,
}

struct ArtifactStats {
    global_quantization: QuantizationStats,
    country_quantization: QuantizationStats,
    pairs_min_1: usize,
    pairs_min_2: usize,
    pairs_min_5: usize,
    countries: usize,
    sizes: Sizes,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let mut arguments = std::env::args_os().skip(1);
    let clean_v1 = PathBuf::from(arguments.next().ok_or_else(usage)?);
    let raw_directory = PathBuf::from(arguments.next().ok_or_else(usage)?);
    let baseline_artifact = PathBuf::from(arguments.next().ok_or_else(usage)?);
    let output = PathBuf::from(arguments.next().ok_or_else(usage)?);
    if arguments.next().is_some() {
        return Err(usage().into());
    }
    if !clean_v1.is_file() {
        return Err(format!("clean-v1 is not a file: {}", clean_v1.display()).into());
    }
    if !raw_directory.is_dir() {
        return Err(format!("raw source is not a directory: {}", raw_directory.display()).into());
    }
    if !baseline_artifact.is_dir() {
        return Err(format!(
            "baseline artifact is not a directory: {}",
            baseline_artifact.display()
        )
        .into());
    }
    if output.exists() {
        return Err(format!("refusing to overwrite: {}", output.display()).into());
    }

    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(
        ".{}.tmp-{}",
        output
            .file_name()
            .ok_or("output has no final component")?
            .to_string_lossy(),
        std::process::id()
    ));
    if temporary.exists() {
        return Err(format!("refusing to overwrite: {}", temporary.display()).into());
    }

    let started = Instant::now();
    let result = generate(&clean_v1, &raw_directory, &baseline_artifact, &temporary);
    match result {
        Ok(report) => {
            fs::write(temporary.join("report.md"), report)?;
            fs::rename(&temporary, &output)?;
            eprintln!("Output: {}", output.display());
            eprintln!("Total elapsed: {:.1?}", started.elapsed());
            Ok(())
        }
        Err(error) => {
            let _ = fs::remove_dir_all(&temporary);
            Err(error)
        }
    }
}

fn usage() -> String {
    "usage: name-surname-v2 <clean-v1.csv> <raw-country-directory> <baseline-c32-directory> <new-output-directory>".to_string()
}

fn generate(
    clean_v1: &Path,
    raw_directory: &Path,
    baseline_artifact: &Path,
    output: &Path,
) -> Result<String> {
    fs::create_dir(output)?;
    eprintln!("Hashing clean-v1 and fixed baseline artifact");
    let clean_checksum_before = checksum(clean_v1)?;
    let artifact_checksums_before = directory_checksums(baseline_artifact)?;
    let raw_snapshot_before = snapshot_csv_directory(raw_directory)?;

    eprintln!("Loading clean-v1 keys and exact given-name counts");
    let clean = load_clean_v1(clean_v1)?;
    eprintln!(
        "Loaded {} keys, {} rows, {} observations",
        clean.names.len(),
        clean.source_rows,
        clean.total_given
    );

    eprintln!("Scanning raw surname columns");
    let mut surname = scan_raw_surnames(raw_directory, &clean)?;
    validate_observation_denominators(clean.total_given, surname.nonempty_surnames)?;
    add_clean_country_totals(&mut surname.countries, &clean.given_pairs);
    eprintln!(
        "Scanned {} files and {} rows; {} surname observations matched retained keys",
        surname.raw_files, surname.raw_rows, surname.matched_surnames
    );

    eprintln!("Writing exact clean-v2 evidence tables");
    let v2_stats = write_clean_v2(output, &clean, &surname)?;
    write_name_totals(output, &clean, &surname)?;
    write_country_totals(output, &surname)?;
    validate_clean_v2(&output.join("clean-v2.csv"), &clean, &surname, v2_stats)?;

    eprintln!("Building surname metadata variants over the fixed C32 key slots");
    let artifact_stats = build_artifact_variants(output, baseline_artifact, &clean, &surname)?;

    eprintln!("Compressing exact CSV");
    let clean_v2_path = output.join("clean-v2.csv");
    let clean_v2_gzip = output.join("clean-v2.csv.gz");
    let clean_v2_zstd = output.join("clean-v2.csv.zst");
    gzip(&clean_v2_path, &clean_v2_gzip)?;
    zstd_file(&clean_v2_path, &clean_v2_zstd)?;
    validate_compressed(&clean_v2_gzip, &clean_v2_zstd)?;

    let sizes = Sizes {
        clean_v2_csv: fs::metadata(&clean_v2_path)?.len(),
        clean_v2_gzip: fs::metadata(&clean_v2_gzip)?.len(),
        clean_v2_zstd: fs::metadata(&clean_v2_zstd)?.len(),
        ..artifact_stats.sizes
    };
    let artifact_stats = ArtifactStats {
        sizes,
        ..artifact_stats
    };

    eprintln!("Rechecking all read-only inputs");
    let clean_checksum_after = checksum(clean_v1)?;
    let artifact_checksums_after = directory_checksums(baseline_artifact)?;
    let raw_snapshot_after = snapshot_csv_directory(raw_directory)?;
    if clean_checksum_before.bytes != clean_checksum_after.bytes
        || clean_checksum_before.sha256 != clean_checksum_after.sha256
    {
        return Err("clean-v1 changed during generation".into());
    }
    if artifact_checksums_before != artifact_checksums_after {
        return Err("baseline C32 artifact changed during generation".into());
    }
    if !same_snapshot(&raw_snapshot_before, &raw_snapshot_after) {
        return Err(
            "raw source file size, timestamp, or membership changed during generation".into(),
        );
    }

    Ok(format_report(
        clean_v1,
        raw_directory,
        &clean_checksum_before,
        &clean,
        &surname,
        v2_stats,
        &artifact_stats,
    ))
}

fn load_clean_v1(path: &Path) -> Result<CleanData> {
    let mut reader = csv::Reader::from_path(path)?;
    let expected = ByteRecord::from(vec!["name", "country", "gender", "count"]);
    if reader.byte_headers()? != &expected {
        return Err("unexpected clean-v1 header".into());
    }

    let mut names = Vec::<Box<[u8]>>::with_capacity(1_900_000);
    let mut name_ids = HashMap::<Box<[u8]>, u32>::with_capacity(1_900_000);
    let mut global_given = Vec::<u64>::with_capacity(1_900_000);
    let mut given_pairs = Vec::<GivenPair>::with_capacity(7_000_000);
    let mut previous_name = Vec::<u8>::new();
    let mut previous_tuple = None::<(u16, u8)>;
    let mut current_id = None::<u32>;
    let mut source_rows = 0_u64;
    let mut total_given = 0_u128;

    for result in reader.byte_records() {
        let record = result?;
        source_rows = source_rows.checked_add(1).ok_or("clean-v1 row overflow")?;
        if record.len() != 4 {
            return Err(
                format!("clean-v1 row {} does not have four fields", source_rows + 1).into(),
            );
        }
        let name = record.get(0).ok_or("missing clean name")?;
        if name.is_empty() {
            return Err(format!("clean-v1 row {} has an empty name", source_rows + 1).into());
        }
        let country = parse_country(record.get(1).ok_or("missing clean country")?)?;
        let gender = parse_gender(record.get(2).ok_or("missing clean gender")?)?;
        let count = parse_u64(record.get(3).ok_or("missing clean count")?)?;
        if count < 2 {
            return Err(format!("clean-v1 row {} has count below 2", source_rows + 1).into());
        }

        if previous_name.as_slice() != name {
            if !previous_name.is_empty() && previous_name.as_slice() >= name {
                return Err(
                    "clean-v1 names are not strictly byte-lexicographically ordered".into(),
                );
            }
            let id = u32::try_from(names.len())?;
            let boxed = name.to_vec().into_boxed_slice();
            if name_ids.insert(boxed.clone(), id).is_some() {
                return Err("duplicate clean-v1 name ID".into());
            }
            names.push(boxed);
            global_given.push(0);
            previous_name.clear();
            previous_name.extend_from_slice(name);
            previous_tuple = None;
            current_id = Some(id);
        }

        let id = current_id.ok_or("missing current clean-v1 name ID")?;
        let tuple = (country, gender);
        if previous_tuple.is_some_and(|previous| previous >= tuple) {
            return Err(
                format!("clean-v1 tuples are not strictly ordered for name ID {id}").into(),
            );
        }
        previous_tuple = Some(tuple);
        if given_pairs
            .last()
            .is_none_or(|row| row.name_id != id || row.country != country)
        {
            given_pairs.push(GivenPair {
                name_id: id,
                country,
                counts: [0; 3],
            });
        }
        let pair = given_pairs.last_mut().ok_or("missing given pair")?;
        pair.counts[usize::from(gender)] = count;
        global_given[id as usize] = global_given[id as usize]
            .checked_add(count)
            .ok_or("given-name global count overflow")?;
        total_given = total_given
            .checked_add(u128::from(count))
            .ok_or("given-name denominator overflow")?;
    }

    if names.is_empty() {
        return Err("clean-v1 has no keys".into());
    }
    if global_given.iter().any(|&count| count < 5) {
        return Err("clean-v1 contains a name below the global minimum 5".into());
    }
    Ok(CleanData {
        names,
        name_ids,
        global_given,
        given_pairs,
        source_rows,
        total_given,
    })
}

fn scan_raw_surnames(directory: &Path, clean: &CleanData) -> Result<SurnameData> {
    let mut paths = fs::read_dir(directory)?
        .map(|entry| Ok(entry?.path()))
        .collect::<Result<Vec<_>>>()?;
    paths.retain(|path| path.extension().is_some_and(|extension| extension == "csv"));
    paths.sort();
    if paths.is_empty() {
        return Err("raw directory contains no CSV files".into());
    }

    let mut pairs = Vec::<SurnamePair>::new();
    let mut global_counts = vec![0_u64; clean.names.len()];
    let mut countries = BTreeMap::<u16, CountryStats>::new();
    let mut raw_rows = 0_u64;
    let mut nonempty_surnames = 0_u64;
    let mut matched_surnames = 0_u64;

    for (file_index, path) in paths.iter().enumerate() {
        let filename = path
            .file_stem()
            .and_then(|value| value.to_str())
            .ok_or_else(|| format!("invalid raw filename: {}", path.display()))?;
        let file_country = parse_country(filename.as_bytes())?;
        let started = Instant::now();
        let mut reader = ReaderBuilder::new().has_headers(false).from_path(path)?;
        let mut local = HashMap::<u32, u64>::new();
        let mut file_rows = 0_u64;
        let mut file_nonempty = 0_u64;
        let mut file_matched = 0_u64;

        for result in reader.byte_records() {
            let record = result?;
            file_rows = file_rows.checked_add(1).ok_or("raw row count overflow")?;
            if record.len() != 4 {
                return Err(format!(
                    "{} row {} has {} fields, expected 4",
                    path.display(),
                    file_rows,
                    record.len()
                )
                .into());
            }
            let surname = record.get(1).ok_or("missing raw surname")?;
            parse_gender(record.get(2).ok_or("missing raw gender")?)?;
            let row_country = parse_country(record.get(3).ok_or("missing raw country")?)?;
            if row_country != file_country {
                return Err(format!(
                    "{} row {} country does not match its filename",
                    path.display(),
                    file_rows
                )
                .into());
            }
            if surname.is_empty() {
                continue;
            }
            file_nonempty = file_nonempty
                .checked_add(1)
                .ok_or("surname denominator overflow")?;
            if let Some(&id) = clean.name_ids.get(surname) {
                let local_count = local.entry(id).or_insert(0);
                *local_count = local_count
                    .checked_add(1)
                    .ok_or("country surname count overflow")?;
                global_counts[id as usize] = global_counts[id as usize]
                    .checked_add(1)
                    .ok_or("global surname count overflow")?;
                file_matched = file_matched
                    .checked_add(1)
                    .ok_or("matched surname count overflow")?;
            }
        }

        let mut local = local.into_iter().collect::<Vec<_>>();
        local.sort_unstable_by_key(|&(id, _)| id);
        pairs.extend(local.iter().map(|&(name_id, count)| SurnamePair {
            name_id,
            country: file_country,
            count,
        }));
        countries.insert(
            file_country,
            CountryStats {
                raw_rows: file_rows,
                nonempty_surnames: file_nonempty,
                matched_surnames: file_matched,
                matched_keys: u64::try_from(local.len())?,
                ..CountryStats::default()
            },
        );
        raw_rows = raw_rows
            .checked_add(file_rows)
            .ok_or("raw row count overflow")?;
        nonempty_surnames = nonempty_surnames
            .checked_add(file_nonempty)
            .ok_or("surname denominator overflow")?;
        matched_surnames = matched_surnames
            .checked_add(file_matched)
            .ok_or("matched surname count overflow")?;
        eprintln!(
            "  [{}/{}] {}: {} rows, {} matched observations, {} matched keys, {:.1?}",
            file_index + 1,
            paths.len(),
            path.file_name().unwrap_or_default().to_string_lossy(),
            file_rows,
            file_matched,
            local.len(),
            started.elapsed()
        );
        if raw_rows / PROGRESS_ROWS != (raw_rows - file_rows) / PROGRESS_ROWS {
            eprintln!("  cumulative raw rows: {raw_rows}");
        }
    }

    pairs.sort_unstable_by_key(|row| (row.name_id, row.country));
    if pairs
        .windows(2)
        .any(|rows| (rows[0].name_id, rows[0].country) >= (rows[1].name_id, rows[1].country))
    {
        return Err("duplicate or unordered surname pairs after raw scan".into());
    }
    if global_counts
        .iter()
        .map(|&count| u128::from(count))
        .sum::<u128>()
        != u128::from(matched_surnames)
    {
        return Err("global surname counts do not sum to matched observations".into());
    }
    Ok(SurnameData {
        pairs,
        global_counts,
        countries,
        raw_files: paths.len(),
        raw_rows,
        nonempty_surnames,
        matched_surnames,
    })
}

fn validate_observation_denominators(given: u128, surname: u64) -> Result<()> {
    if given == 0 || surname == 0 {
        return Err("given and surname denominators must be nonzero".into());
    }
    u64::try_from(given).map_err(|_| "given-name denominator exceeds u64")?;
    Ok(())
}

fn add_clean_country_totals(
    countries: &mut BTreeMap<u16, CountryStats>,
    given_pairs: &[GivenPair],
) {
    for row in given_pairs {
        countries.entry(row.country).or_default().clean_given += row
            .counts
            .iter()
            .map(|&count| u128::from(count))
            .sum::<u128>();
    }
}

fn write_clean_v2(output: &Path, clean: &CleanData, surname: &SurnameData) -> Result<V2Stats> {
    let mut writer = Writer::from_path(output.join("clean-v2.csv"))?;
    writer.write_record([
        "name",
        "country",
        "given_unknown_count",
        "given_female_count",
        "given_male_count",
        "as_surname_count",
    ])?;
    let mut given_position = 0_usize;
    let mut surname_position = 0_usize;
    let mut stats = V2Stats::default();
    while given_position < clean.given_pairs.len() || surname_position < surname.pairs.len() {
        let given_key = clean
            .given_pairs
            .get(given_position)
            .map(|row| (row.name_id, row.country));
        let surname_key = surname
            .pairs
            .get(surname_position)
            .map(|row| (row.name_id, row.country));
        let key = match (given_key, surname_key) {
            (Some(left), Some(right)) => left.min(right),
            (Some(left), None) => left,
            (None, Some(right)) => right,
            (None, None) => break,
        };
        let given = if given_key == Some(key) {
            let row = clean.given_pairs[given_position];
            given_position += 1;
            row.counts
        } else {
            [0; 3]
        };
        let surname_count = if surname_key == Some(key) {
            let count = surname.pairs[surname_position].count;
            surname_position += 1;
            count
        } else {
            0
        };
        write_v2_record(
            &mut writer,
            &clean.names[key.0 as usize],
            key.1,
            given,
            surname_count,
        )?;
        let has_given = given.iter().any(|&count| count != 0);
        let has_surname = surname_count != 0;
        stats.rows += 1;
        stats.given_sum += given.iter().map(|&count| u128::from(count)).sum::<u128>();
        stats.surname_sum += u128::from(surname_count);
        match (has_given, has_surname) {
            (true, true) => stats.both_rows += 1,
            (true, false) => stats.given_only_rows += 1,
            (false, true) => stats.surname_only_country_rows += 1,
            (false, false) => return Err("clean-v2 merge produced an empty row".into()),
        }
    }
    writer.flush()?;
    Ok(stats)
}

fn write_v2_record(
    writer: &mut Writer<File>,
    name: &[u8],
    country: u16,
    given: [u64; 3],
    surname_count: u64,
) -> Result<()> {
    let mut record = ByteRecord::new();
    record.push_field(name);
    record.push_field(&country.to_be_bytes());
    let values = [given[0], given[1], given[2], surname_count];
    let strings = values.map(|value| value.to_string());
    for value in &strings {
        record.push_field(value.as_bytes());
    }
    writer.write_byte_record(&record)?;
    Ok(())
}

fn write_name_totals(output: &Path, clean: &CleanData, surname: &SurnameData) -> Result<()> {
    let mut writer = Writer::from_path(output.join("name-totals.csv"))?;
    writer.write_record(["name", "given_count", "as_surname_count"])?;
    for id in 0..clean.names.len() {
        let mut record = ByteRecord::new();
        record.push_field(&clean.names[id]);
        let given = clean.global_given[id].to_string();
        let surname_count = surname.global_counts[id].to_string();
        record.push_field(given.as_bytes());
        record.push_field(surname_count.as_bytes());
        writer.write_byte_record(&record)?;
    }
    writer.flush()?;
    Ok(())
}

fn write_country_totals(output: &Path, surname: &SurnameData) -> Result<()> {
    let mut writer = Writer::from_path(output.join("country-totals.csv"))?;
    writer.write_record([
        "country",
        "clean_given_observations",
        "raw_rows",
        "nonempty_surname_observations",
        "matched_surname_observations",
        "matched_clean_v1_keys",
    ])?;
    for (&country, stats) in &surname.countries {
        writer.write_record([
            country_text(country)?,
            stats.clean_given.to_string(),
            stats.raw_rows.to_string(),
            stats.nonempty_surnames.to_string(),
            stats.matched_surnames.to_string(),
            stats.matched_keys.to_string(),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn validate_clean_v2(
    path: &Path,
    clean: &CleanData,
    surname: &SurnameData,
    expected: V2Stats,
) -> Result<()> {
    let mut reader = csv::Reader::from_path(path)?;
    let expected_header = ByteRecord::from(vec![
        "name",
        "country",
        "given_unknown_count",
        "given_female_count",
        "given_male_count",
        "as_surname_count",
    ]);
    if reader.byte_headers()? != &expected_header {
        return Err("unexpected clean-v2 header".into());
    }
    let mut previous = None::<(u32, u16)>;
    let mut seen_names = vec![false; clean.names.len()];
    let mut actual = V2Stats::default();
    for result in reader.byte_records() {
        let record = result?;
        if record.len() != 6 {
            return Err("clean-v2 validation found a non-six-field row".into());
        }
        let name = record.get(0).ok_or("missing clean-v2 name")?;
        let id = *clean
            .name_ids
            .get(name)
            .ok_or("clean-v2 contains a surname-only key")?;
        let country = parse_country(record.get(1).ok_or("missing clean-v2 country")?)?;
        let key = (id, country);
        if previous.is_some_and(|value| value >= key) {
            return Err("clean-v2 rows are not strictly ordered".into());
        }
        previous = Some(key);
        seen_names[id as usize] = true;
        let given = [
            parse_u64(record.get(2).ok_or("missing unknown count")?)?,
            parse_u64(record.get(3).ok_or("missing female count")?)?,
            parse_u64(record.get(4).ok_or("missing male count")?)?,
        ];
        let surname_count = parse_u64(record.get(5).ok_or("missing surname count")?)?;
        let has_given = given.iter().any(|&count| count != 0);
        let has_surname = surname_count != 0;
        if !has_given && !has_surname {
            return Err("clean-v2 contains an empty evidence row".into());
        }
        actual.rows += 1;
        actual.given_sum += given.iter().map(|&count| u128::from(count)).sum::<u128>();
        actual.surname_sum += u128::from(surname_count);
        match (has_given, has_surname) {
            (true, true) => actual.both_rows += 1,
            (true, false) => actual.given_only_rows += 1,
            (false, true) => actual.surname_only_country_rows += 1,
            (false, false) => unreachable!(),
        }
    }
    if actual != expected {
        return Err("clean-v2 independent validation totals differ".into());
    }
    if seen_names.contains(&false) {
        return Err("clean-v2 does not retain every clean-v1 key".into());
    }
    if actual.given_sum != clean.total_given {
        return Err("clean-v2 given observations do not equal clean-v1".into());
    }
    if actual.surname_sum != u128::from(surname.matched_surnames) {
        return Err("clean-v2 surname observations do not equal scan totals".into());
    }
    Ok(())
}

fn build_artifact_variants(
    output: &Path,
    baseline: &Path,
    clean: &CleanData,
    surname: &SurnameData,
) -> Result<ArtifactStats> {
    let mphf: Mphf<u64> =
        bincode::deserialize_from(BufReader::new(File::open(baseline.join("names.mphf"))?))?;
    let fingerprints = read_u32_file(&baseline.join("fingerprints.u32"))?;
    if fingerprints.len() != clean.names.len() {
        return Err(format!(
            "baseline has {} keys but clean-v1 has {}",
            fingerprints.len(),
            clean.names.len()
        )
        .into());
    }
    let mut slots = vec![usize::MAX; clean.names.len()];
    let mut occupied = vec![false; clean.names.len()];
    for (id, name) in clean.names.iter().enumerate() {
        let routing = xxh3_64_with_seed(name, ROUTING_SEED);
        let slot = usize::try_from(
            mphf.try_hash(&routing)
                .ok_or("baseline MPHF rejected a clean-v1 key")?,
        )?;
        if slot >= fingerprints.len() || occupied[slot] {
            return Err("baseline MPHF produced an invalid or duplicate slot".into());
        }
        let fingerprint = xxh3_64_with_seed(name, FINGERPRINT_SEED) as u32;
        if fingerprints[slot] != fingerprint {
            return Err(format!("baseline fingerprint mismatch for clean key ID {id}").into());
        }
        occupied[slot] = true;
        slots[id] = slot;
    }
    if occupied.contains(&false) {
        return Err("baseline MPHF left an unused slot".into());
    }

    let global_directory = output.join("c32-q8-surname-global");
    let country_directory = output.join("c32-q8-surname-country");
    copy_flat_directory(baseline, &global_directory)?;
    copy_flat_directory(baseline, &country_directory)?;
    let max_global = surname.global_counts.iter().copied().max().unwrap_or(0);
    let max_global_u32 = u32::try_from(max_global)?;
    let mut global_quantization = QuantizationStats::new();
    let mut global_by_slot = vec![0_u8; clean.names.len()];
    for (id, &count) in surname.global_counts.iter().enumerate() {
        let quantized = quantize_count(count, max_global);
        global_by_slot[slots[id]] = quantized;
        global_quantization.observe(count, dequantize_count(quantized, max_global));
    }
    for directory in [&global_directory, &country_directory] {
        fs::write(directory.join("surname_counts.q8"), &global_by_slot)?;
        fs::write(
            directory.join("surname_quantization_max_count.u32"),
            max_global_u32.to_le_bytes(),
        )?;
        fs::write(
            directory.join("surname_total_observations.u64"),
            surname.nonempty_surnames.to_le_bytes(),
        )?;
        fs::write(
            directory.join("clean_given_total_observations.u64"),
            u64::try_from(clean.total_given)?.to_le_bytes(),
        )?;
    }

    let country_codes = surname.countries.keys().copied().collect::<Vec<_>>();
    if country_codes.len() > 256 {
        return Err("more than 256 surname countries".into());
    }
    let country_ids = country_codes
        .iter()
        .enumerate()
        .map(|(id, &country)| Ok((country, u8::try_from(id)?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let mut rows = surname
        .pairs
        .iter()
        .map(|row| (slots[row.name_id as usize], row.country, row.count))
        .collect::<Vec<_>>();
    rows.sort_unstable_by_key(|&(slot, country, _)| (slot, country));
    let max_country = rows.iter().map(|row| row.2).max().unwrap_or(0);
    let max_country_u32 = u32::try_from(max_country)?;
    let mut offsets = Vec::<u32>::with_capacity(clean.names.len() + 1);
    let mut packed_countries = Vec::<u8>::with_capacity(rows.len());
    let mut packed_counts = Vec::<u8>::with_capacity(rows.len());
    let mut country_quantization = QuantizationStats::new();
    let mut position = 0_usize;
    offsets.push(0);
    for slot in 0..clean.names.len() {
        while position < rows.len() && rows[position].0 == slot {
            let (_, country, count) = rows[position];
            packed_countries.push(country_ids[&country]);
            let quantized = quantize_count(count, max_country);
            packed_counts.push(quantized);
            country_quantization.observe(count, dequantize_count(quantized, max_country));
            position += 1;
        }
        offsets.push(u32::try_from(position)?);
    }
    if position != rows.len() {
        return Err("surname country rows were not fully packed".into());
    }
    write_u32_file(&country_directory.join("surname_row_offsets.u32"), &offsets)?;
    fs::write(
        country_directory.join("surname_country_ids.u8"),
        &packed_countries,
    )?;
    fs::write(
        country_directory.join("surname_country_counts.q8"),
        &packed_counts,
    )?;
    fs::write(
        country_directory.join("surname_country_quantization_max_count.u32"),
        max_country_u32.to_le_bytes(),
    )?;
    let mut dictionary = Vec::with_capacity(country_codes.len() * 2);
    let mut surname_totals = Vec::with_capacity(country_codes.len() * 8);
    let mut given_totals = Vec::with_capacity(country_codes.len() * 8);
    for country in &country_codes {
        dictionary.extend_from_slice(&country.to_be_bytes());
        surname_totals
            .extend_from_slice(&surname.countries[country].nonempty_surnames.to_le_bytes());
        given_totals.extend_from_slice(
            &u64::try_from(surname.countries[country].clean_given)?.to_le_bytes(),
        );
    }
    fs::write(country_directory.join("surname_countries.dict"), dictionary)?;
    fs::write(
        country_directory.join("surname_country_total_observations.u64"),
        surname_totals,
    )?;
    fs::write(
        country_directory.join("clean_given_country_total_observations.u64"),
        given_totals,
    )?;

    validate_sidecars(
        &global_directory,
        &country_directory,
        clean.names.len(),
        rows.len(),
        country_codes.len(),
    )?;
    let baseline_direct = directory_size(baseline)?;
    let global_direct = directory_size(&global_directory)?;
    let country_direct = directory_size(&country_directory)?;
    let global_archive = output.join("c32-q8-surname-global.tar.zst");
    let country_archive = output.join("c32-q8-surname-country.tar.zst");
    tar_zstd(&global_directory, &global_archive)?;
    tar_zstd(&country_directory, &country_archive)?;
    zstd_test(&global_archive)?;
    zstd_test(&country_archive)?;

    Ok(ArtifactStats {
        global_quantization,
        country_quantization,
        pairs_min_1: rows.len(),
        pairs_min_2: rows.iter().filter(|row| row.2 >= 2).count(),
        pairs_min_5: rows.iter().filter(|row| row.2 >= 5).count(),
        countries: country_codes.len(),
        sizes: Sizes {
            clean_v2_csv: 0,
            clean_v2_gzip: 0,
            clean_v2_zstd: 0,
            baseline_direct,
            global_direct,
            global_archive: fs::metadata(global_archive)?.len(),
            country_direct,
            country_archive: fs::metadata(country_archive)?.len(),
        },
    })
}

fn validate_sidecars(
    global: &Path,
    country: &Path,
    names: usize,
    rows: usize,
    countries: usize,
) -> Result<()> {
    expect_size(&global.join("surname_counts.q8"), u64::try_from(names)?)?;
    expect_size(&global.join("surname_quantization_max_count.u32"), 4)?;
    expect_size(&global.join("surname_total_observations.u64"), 8)?;
    expect_size(&global.join("clean_given_total_observations.u64"), 8)?;
    expect_size(&country.join("surname_counts.q8"), u64::try_from(names)?)?;
    expect_size(
        &country.join("surname_row_offsets.u32"),
        u64::try_from((names + 1) * 4)?,
    )?;
    expect_size(
        &country.join("surname_country_ids.u8"),
        u64::try_from(rows)?,
    )?;
    expect_size(
        &country.join("surname_country_counts.q8"),
        u64::try_from(rows)?,
    )?;
    expect_size(
        &country.join("surname_countries.dict"),
        u64::try_from(countries * 2)?,
    )?;
    expect_size(
        &country.join("surname_country_total_observations.u64"),
        u64::try_from(countries * 8)?,
    )?;
    expect_size(
        &country.join("clean_given_country_total_observations.u64"),
        u64::try_from(countries * 8)?,
    )?;
    Ok(())
}

fn format_report(
    clean_path: &Path,
    raw_directory: &Path,
    checksum: &Checksum,
    clean: &CleanData,
    surname: &SurnameData,
    v2: V2Stats,
    artifact: &ArtifactStats,
) -> String {
    let overlap_keys = surname
        .global_counts
        .iter()
        .filter(|&&count| count != 0)
        .count();
    let overlap_key_percentage = overlap_keys as f64 / clean.names.len() as f64 * 100.0;
    let overlap_observation_percentage =
        surname.matched_surnames as f64 / surname.nonempty_surnames as f64 * 100.0;
    let global_delta = artifact.sizes.global_direct - artifact.sizes.baseline_direct;
    let country_delta = artifact.sizes.country_direct - artifact.sizes.baseline_direct;
    let min_2_direct = country_direct_estimate(
        artifact.sizes.global_direct,
        clean.names.len(),
        artifact.pairs_min_2,
        artifact.countries,
    );
    let min_5_direct = country_direct_estimate(
        artifact.sizes.global_direct,
        clean.names.len(),
        artifact.pairs_min_5,
        artifact.countries,
    );
    let examples = role_examples(clean, surname);
    format!(
        "# clean-v2 surname-evidence report\n\n\
         `clean-v2` retains exactly the clean-v1 first-name key set and adds exact evidence that those same strings occur in the raw surname column. Surname-only strings never become index keys. Matching is exact UTF-8 byte equality; no new normalization or sanitation policy is introduced.\n\n\
         ## Inputs and integrity\n\n\
         - clean-v1: `{}`\n\
         - clean-v1 bytes: {}\n\
         - clean-v1 SHA-256 before/after: `{}` (unchanged)\n\
         - raw directory: `{}`\n\
         - raw CSV files: {}\n\
         - raw file membership, byte sizes, and modification times: unchanged before/after\n\
         - fixed baseline C32 files: byte-for-byte unchanged before/after\n\n\
         ## Evidence totals\n\n\
         - retained MPHF keys: {}\n\
         - clean-v1 metadata rows: {}\n\
         - clean-v1 given observations: {}\n\
         - raw person rows scanned: {}\n\
         - non-empty surname observations (the surname denominator): {}\n\
         - surname observations matching retained keys: {} ({:.6}% of non-empty surname observations)\n\
         - retained keys observed as surnames: {} ({:.6}% of retained keys)\n\
         - matched `(name,country)` surname pairs: {}\n\
         - matched pairs with count >=2: {}\n\
         - matched pairs with count >=5: {}\n\n\
         `sum(as_surname_count)` is {}: the overlap numerator, not the surname denominator. The likelihood denominator is all {} non-empty surname observations, including surname-only strings that are deliberately absent from the index.\n\n\
         ## Exact clean-v2 CSV\n\n\
         Schema: `name,country,given_unknown_count,given_female_count,given_male_count,as_surname_count`.\n\n\
         - rows: {}\n\
         - rows with both given and surname evidence: {}\n\
         - given-only rows: {}\n\
         - surname-only country rows for retained keys: {}\n\
         - given observation sum: {} (matches clean-v1)\n\
         - surname observation sum: {} (matches the overlap scan)\n\
         - uncompressed: {} bytes ({})\n\
         - gzip -9: {} bytes ({})\n\
         - zstd -19: {} bytes ({})\n\n\
         ## C32 + q8 metadata benchmark\n\n\
         Both variants copy the fixed 1,803,175-key C32 artifact and preserve its MPHF slots and fingerprints. No surname keys are added.\n\n\
         | Variant | Direct bytes | Direct size | Added versus baseline | zstd-19 archive |\n\
         |---|---:|---:|---:|---:|\n\
         | clean-v1 baseline | {} | {} | — | 20,554,776 bytes (previous measurement) |\n\
         | global surname q8 | {} | {} | {} | {} bytes ({}) |\n\
         | global + sparse country surname q8 (`row >=1`) | {} | {} | {} | {} bytes ({}) |\n\n\
         Sparse-country direct-size estimates if row thresholds are applied only during binary generation:\n\n\
         | Surname country minimum | Sparse rows | Estimated direct bytes | Estimated size |\n\
         |---:|---:|---:|---:|\n\
         | 1 | {} | {} | {} |\n\
         | 2 | {} | {} | {} |\n\
         | 5 | {} | {} | {} |\n\n\
         Global surname q8 error: mean row {:.4}%, p99 {:.4}%, maximum {:.4}%, signed aggregate {:+.4}%.\n\
         Country surname q8 error: mean row {:.4}%, p99 {:.4}%, maximum {:.4}%, signed aggregate {:+.4}%.\n\n\
         ## Illustrative global role evidence\n\n\
         The log likelihood ratio shown is `ln(given_count / clean_given_total) - ln(surname_count / all_nonempty_surname_total)` when both counts are non-zero. It is descriptive evidence, not a calibrated classifier score.\n\n\
         {}\n\
         ## Validation\n\n\
         - all raw rows had exactly four CSV fields and a country matching their filename\n\
         - all clean-v2 names were exhaustively checked against the retained clean-v1 key set\n\
         - clean-v2 ordering and `(name,country)` uniqueness were exhaustively checked\n\
         - every clean-v1 key remains represented; no surname-only key was added\n\
         - exact given and surname sums were independently reread from clean-v2\n\
         - all known keys reproduced their original C32 MPHF slot and fingerprint\n\
         - binary array lengths and compressed-stream integrity were checked\n\
         - no classifier behavior or application runtime file was changed\n\n\
         ## Interpretation boundary\n\n\
         Exact matching intentionally preserves clean-v1's current case/spelling fragmentation. Runtime variant lookup and any future role-score smoothing/calibration remain classifier-design work; this corpus pass does not tune either.\n",
        clean_path.display(),
        checksum.bytes,
        checksum.sha256,
        raw_directory.display(),
        surname.raw_files,
        clean.names.len(),
        clean.source_rows,
        clean.total_given,
        surname.raw_rows,
        surname.nonempty_surnames,
        surname.matched_surnames,
        overlap_observation_percentage,
        overlap_keys,
        overlap_key_percentage,
        artifact.pairs_min_1,
        artifact.pairs_min_2,
        artifact.pairs_min_5,
        surname.matched_surnames,
        surname.nonempty_surnames,
        v2.rows,
        v2.both_rows,
        v2.given_only_rows,
        v2.surname_only_country_rows,
        v2.given_sum,
        v2.surname_sum,
        artifact.sizes.clean_v2_csv,
        human_bytes(artifact.sizes.clean_v2_csv),
        artifact.sizes.clean_v2_gzip,
        human_bytes(artifact.sizes.clean_v2_gzip),
        artifact.sizes.clean_v2_zstd,
        human_bytes(artifact.sizes.clean_v2_zstd),
        artifact.sizes.baseline_direct,
        human_bytes(artifact.sizes.baseline_direct),
        artifact.sizes.global_direct,
        human_bytes(artifact.sizes.global_direct),
        human_bytes(global_delta),
        artifact.sizes.global_archive,
        human_bytes(artifact.sizes.global_archive),
        artifact.sizes.country_direct,
        human_bytes(artifact.sizes.country_direct),
        human_bytes(country_delta),
        artifact.sizes.country_archive,
        human_bytes(artifact.sizes.country_archive),
        artifact.pairs_min_1,
        artifact.sizes.country_direct,
        human_bytes(artifact.sizes.country_direct),
        artifact.pairs_min_2,
        min_2_direct,
        human_bytes(min_2_direct),
        artifact.pairs_min_5,
        min_5_direct,
        human_bytes(min_5_direct),
        artifact.global_quantization.mean_relative() * 100.0,
        artifact.global_quantization.percentile_relative(99) * 100.0,
        artifact.global_quantization.max_relative_error * 100.0,
        artifact.global_quantization.signed_aggregate() * 100.0,
        artifact.country_quantization.mean_relative() * 100.0,
        artifact.country_quantization.percentile_relative(99) * 100.0,
        artifact.country_quantization.max_relative_error * 100.0,
        artifact.country_quantization.signed_aggregate() * 100.0,
        examples,
    )
}

fn role_examples(clean: &CleanData, surname: &SurnameData) -> String {
    let mut output = String::from(
        "| Name | Given count | As-surname count | Global log likelihood ratio |\n|---|---:|---:|---:|\n",
    );
    for name in ["Jean", "Martin", "Quentin", "Élodie", "Elodie"] {
        let Some(&id) = clean.name_ids.get(name.as_bytes()) else {
            continue;
        };
        let given = clean.global_given[id as usize];
        let surname_count = surname.global_counts[id as usize];
        let ratio = if given == 0 || surname_count == 0 {
            "n/a".to_string()
        } else {
            let value = (given as f64 / clean.total_given as f64).ln()
                - (surname_count as f64 / surname.nonempty_surnames as f64).ln();
            format!("{value:+.4}")
        };
        output.push_str(&format!(
            "| {name} | {given} | {surname_count} | {ratio} |\n"
        ));
    }
    output
}

fn country_direct_estimate(global_direct: u64, names: usize, rows: usize, countries: usize) -> u64 {
    global_direct
        + u64::try_from((names + 1) * 4).unwrap()
        + u64::try_from(rows * 2).unwrap()
        + 4
        + u64::try_from(countries * (2 + 8 + 8)).unwrap()
}

fn parse_country(value: &[u8]) -> Result<u16> {
    if value.len() != 2 || !value.iter().all(u8::is_ascii_uppercase) {
        return Err(format!("invalid country code: {value:?}").into());
    }
    Ok(u16::from_be_bytes([value[0], value[1]]))
}

fn country_text(country: u16) -> Result<String> {
    Ok(String::from_utf8(country.to_be_bytes().to_vec())?)
}

fn parse_gender(value: &[u8]) -> Result<u8> {
    match value {
        b"" => Ok(0),
        b"F" => Ok(1),
        b"M" => Ok(2),
        _ => Err(format!("invalid gender: {value:?}").into()),
    }
}

fn parse_u64(value: &[u8]) -> Result<u64> {
    Ok(std::str::from_utf8(value)?.parse::<u64>()?)
}

fn checksum(path: &Path) -> Result<Checksum> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut digest = Sha256::new();
    let mut bytes = 0_u64;
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
        bytes += u64::try_from(read)?;
    }
    Ok(Checksum {
        bytes,
        sha256: format!("{:x}", digest.finalize()),
    })
}

fn directory_checksums(path: &Path) -> Result<BTreeMap<String, (u64, String)>> {
    let mut output = BTreeMap::new();
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            let checksum = checksum(&entry.path())?;
            output.insert(
                entry.file_name().to_string_lossy().into_owned(),
                (checksum.bytes, checksum.sha256),
            );
        }
    }
    Ok(output)
}

fn snapshot_csv_directory(path: &Path) -> Result<Vec<FileSnapshot>> {
    let mut output = Vec::new();
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        if entry
            .path()
            .extension()
            .is_some_and(|extension| extension == "csv")
        {
            let metadata = entry.metadata()?;
            output.push(FileSnapshot {
                path: entry.path(),
                bytes: metadata.len(),
                modified: metadata.modified().ok(),
            });
        }
    }
    output.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(output)
}

fn same_snapshot(left: &[FileSnapshot], right: &[FileSnapshot]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            left.path == right.path && left.bytes == right.bytes && left.modified == right.modified
        })
}

fn copy_flat_directory(source: &Path, destination: &Path) -> Result<()> {
    fs::create_dir(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            fs::copy(entry.path(), destination.join(entry.file_name()))?;
        }
    }
    Ok(())
}

fn read_u32_file(path: &Path) -> Result<Vec<u32>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(format!("{} is not a u32 stream", path.display()).into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn write_u32_file(path: &Path, values: &[u32]) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    for value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

fn quantize_count(count: u64, max_count: u64) -> u8 {
    if count == 0 || max_count <= 1 {
        return u8::from(count != 0);
    }
    let position = (count as f64).ln() / (max_count as f64).ln();
    (1.0 + position * 254.0).round().clamp(1.0, 255.0) as u8
}

fn dequantize_count(value: u8, max_count: u64) -> u64 {
    if value == 0 || max_count <= 1 {
        return u64::from(value != 0);
    }
    let position = (f64::from(value) - 1.0) / 254.0;
    ((max_count as f64).ln() * position)
        .exp()
        .round()
        .clamp(1.0, max_count as f64) as u64
}

impl QuantizationStats {
    fn new() -> Self {
        Self {
            histogram: vec![0; 10_001],
            ..Self::default()
        }
    }

    fn observe(&mut self, original: u64, decoded: u64) {
        if original == 0 {
            debug_assert_eq!(decoded, 0);
            return;
        }
        let absolute = original.abs_diff(decoded);
        let relative = absolute as f64 / original as f64;
        self.rows += 1;
        self.original_sum += u128::from(original);
        self.decoded_sum += u128::from(decoded);
        self.relative_error_sum += relative;
        self.max_relative_error = self.max_relative_error.max(relative);
        let bucket = (relative * 10_000.0).round().clamp(0.0, 10_000.0) as usize;
        self.histogram[bucket] += 1;
    }

    fn mean_relative(&self) -> f64 {
        if self.rows == 0 {
            0.0
        } else {
            self.relative_error_sum / self.rows as f64
        }
    }

    fn percentile_relative(&self, percentile: u64) -> f64 {
        if self.rows == 0 {
            return 0.0;
        }
        let target = self.rows.saturating_mul(percentile).div_ceil(100);
        let mut cumulative = 0_u64;
        for (index, &count) in self.histogram.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                return index as f64 / 10_000.0;
            }
        }
        1.0
    }

    fn signed_aggregate(&self) -> f64 {
        if self.original_sum == 0 {
            0.0
        } else {
            (self.decoded_sum as f64 - self.original_sum as f64) / self.original_sum as f64
        }
    }
}

fn expect_size(path: &Path, expected: u64) -> Result<()> {
    let actual = fs::metadata(path)?.len();
    if actual != expected {
        return Err(format!("{} has {actual} bytes, expected {expected}", path.display()).into());
    }
    Ok(())
}

fn directory_size(path: &Path) -> Result<u64> {
    let mut total = 0_u64;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            total = total
                .checked_add(entry.metadata()?.len())
                .ok_or("size overflow")?;
        }
    }
    Ok(total)
}

fn gzip(source: &Path, destination: &Path) -> Result<()> {
    let output = File::create(destination)?;
    let status = Command::new("gzip")
        .args(["-9", "-n", "-c"])
        .arg(source)
        .stdout(Stdio::from(output))
        .status()?;
    if !status.success() {
        return Err(format!("gzip failed with {status}").into());
    }
    Ok(())
}

fn zstd_file(source: &Path, destination: &Path) -> Result<()> {
    let status = Command::new("zstd")
        .args(["-19", "-T0", "-q", "-f", "-o"])
        .arg(destination)
        .arg(source)
        .status()?;
    if !status.success() {
        return Err(format!("zstd failed with {status}").into());
    }
    Ok(())
}

fn validate_compressed(gzip_path: &Path, zstd_path: &Path) -> Result<()> {
    let gzip_status = Command::new("gzip").arg("-t").arg(gzip_path).status()?;
    if !gzip_status.success() {
        return Err("gzip integrity check failed".into());
    }
    zstd_test(zstd_path)
}

fn tar_zstd(directory: &Path, destination: &Path) -> Result<()> {
    let parent = directory
        .parent()
        .ok_or("artifact directory has no parent")?;
    let name = directory
        .file_name()
        .ok_or("artifact directory has no name")?;
    let mut tar = Command::new("tar")
        .args(["-cf", "-", "-C"])
        .arg(parent)
        .arg(name)
        .stdout(Stdio::piped())
        .spawn()?;
    let tar_output = tar.stdout.take().ok_or("tar did not expose stdout")?;
    let status = Command::new("zstd")
        .args(["-19", "-T0", "-q", "-f", "-o"])
        .arg(destination)
        .stdin(Stdio::from(tar_output))
        .status()?;
    let tar_status = tar.wait()?;
    if !tar_status.success() || !status.success() {
        return Err(format!("artifact compression failed: tar={tar_status}, zstd={status}").into());
    }
    Ok(())
}

fn zstd_test(path: &Path) -> Result<()> {
    let status = Command::new("zstd").args(["-t", "-q"]).arg(path).status()?;
    if !status.success() {
        return Err(format!("zstd integrity check failed for {}", path.display()).into());
    }
    Ok(())
}

fn human_bytes(bytes: u64) -> String {
    format!("{:.2} MiB", bytes as f64 / 1024.0 / 1024.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn q8_round_trip_is_exact_at_endpoints() {
        let max = 1_000_000;
        assert_eq!(quantize_count(0, max), 0);
        assert_eq!(dequantize_count(0, max), 0);
        assert_eq!(dequantize_count(quantize_count(1, max), max), 1);
        assert_eq!(dequantize_count(quantize_count(max, max), max), max);
    }

    #[test]
    fn country_size_estimate_has_expected_components() {
        assert_eq!(
            country_direct_estimate(100, 4, 7, 2),
            100 + 20 + 14 + 4 + 36
        );
    }

    #[test]
    fn observation_denominators_are_nonzero_and_fit_the_format() {
        assert!(validate_observation_denominators(1, 1).is_ok());
        assert!(validate_observation_denominators(0, 1).is_err());
        assert!(validate_observation_denominators(1, 0).is_err());
        assert!(validate_observation_denominators(u128::from(u64::MAX) + 1, 1).is_err());
    }

    #[test]
    fn empty_clean_corpus_is_rejected() -> Result<()> {
        let temporary = tempdir()?;
        let clean_path = temporary.path().join("clean-v1.csv");
        fs::write(&clean_path, "name,country,gender,count\n")?;
        assert!(load_clean_v1(&clean_path).is_err());
        Ok(())
    }

    #[test]
    fn raw_snapshot_comparison_is_strict() {
        let now = SystemTime::now();
        let left = vec![FileSnapshot {
            path: PathBuf::from("AA.csv"),
            bytes: 10,
            modified: Some(now),
        }];
        assert!(same_snapshot(&left, &left));
        let mut right = left.clone();
        right[0].bytes += 1;
        assert!(!same_snapshot(&left, &right));
    }

    #[test]
    fn surname_scan_retains_only_clean_keys_and_preserves_sums() -> Result<()> {
        let temporary = tempdir()?;
        let clean_path = temporary.path().join("clean-v1.csv");
        let raw = temporary.path().join("raw");
        let output = temporary.path().join("output");
        fs::create_dir(&raw)?;
        fs::create_dir(&output)?;
        fs::write(
            &clean_path,
            "name,country,gender,count\nJean,AA,M,5\nMartin,AA,M,4\nMartin,BB,M,3\n",
        )?;
        fs::write(
            raw.join("AA.csv"),
            "Jean,Martin,M,AA\nX,Martin,F,AA\nX,Jean,F,AA\nX,Dupont,M,AA\n",
        )?;
        fs::write(raw.join("BB.csv"), "X,Martin,M,BB\nX,,M,BB\n")?;

        let clean = load_clean_v1(&clean_path)?;
        let mut surname = scan_raw_surnames(&raw, &clean)?;
        add_clean_country_totals(&mut surname.countries, &clean.given_pairs);
        assert_eq!(surname.raw_rows, 6);
        assert_eq!(surname.nonempty_surnames, 5);
        assert_eq!(surname.matched_surnames, 4);
        assert_eq!(surname.pairs.len(), 3);
        assert_eq!(surname.global_counts, [1, 3]);

        let stats = write_clean_v2(&output, &clean, &surname)?;
        validate_clean_v2(&output.join("clean-v2.csv"), &clean, &surname, stats)?;
        assert_eq!(stats.given_sum, 12);
        assert_eq!(stats.surname_sum, 4);
        assert_eq!(stats.rows, 3);
        let csv = fs::read_to_string(output.join("clean-v2.csv"))?;
        assert!(!csv.contains("Dupont"));
        Ok(())
    }
}
