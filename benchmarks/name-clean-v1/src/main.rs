use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::error::Error;
use std::fmt::Write as FmtWrite;
use std::fs::{self, File};
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use csv::{ByteRecord, Reader, ReaderBuilder, Writer};
use sha1::Sha1;
use sha2::{Digest, Sha256};
use unicode_casefold::UnicodeCaseFold;
use unicode_general_category::{GeneralCategory, get_general_category};
use unicode_normalization::UnicodeNormalization;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const GLOBAL_MINIMUM: u64 = 5;
const ROW_MINIMUM: u64 = 2;
const SAMPLE_SIZE: usize = 100;
const PROGRESS_ROWS: u64 = 5_000_000;
const SAMPLE_SEED: u64 = 0x636c_6561_6e2d_7631;

const ORGANIZATION_MARKERS: [&str; 10] = [
    "gmbh",
    "llc",
    "ltd",
    "limited",
    "inc",
    "incorporated",
    "sarl",
    "eurl",
    "sasu",
    "plc",
];

const DIGIT_RULE: u8 = 1 << 0;
const CONTROL_RULE: u8 = 1 << 1;
const URL_EMAIL_RULE: u8 = 1 << 2;
const ORGANIZATION_RULE: u8 = 1 << 3;

#[derive(Clone, Copy, Default)]
struct RemovalStats {
    names: u64,
    rows: u64,
    observations: u128,
}

#[derive(Clone, Copy, Default)]
struct PackedRow {
    count: u64,
    country: u16,
    gender: u8,
}

#[derive(Clone)]
struct Checksum {
    bytes: u64,
    sha1: String,
    sha256: String,
}

struct FirstPass {
    name_ids: HashMap<Box<[u8]>, u32>,
    rows_per_name: Vec<u32>,
    observations_per_name: Vec<u64>,
    rule_masks: Vec<u8>,
    marker_masks: Vec<u16>,
    row_count: u32,
    total_observations: u128,
    digit_sample: Reservoir,
    organization_sample: Reservoir,
    elapsed: Duration,
}

struct SanitationAudit {
    independent: [RemovalStats; 4],
    primary: [RemovalStats; 4],
    markers: [RemovalStats; ORGANIZATION_MARKERS.len()],
    initial_global: RemovalStats,
    candidates: Vec<bool>,
}

struct CompactionAudit {
    duplicate_rows_collapsed: u64,
    row_threshold_rows: u64,
    row_threshold_observations: u128,
    post_row_global: RemovalStats,
    final_names: u64,
    final_rows: u64,
    final_observations: u128,
    max_final_count: u32,
}

#[derive(Default)]
struct QuantizationStats {
    rows: u64,
    original_sum: u128,
    decoded_sum: u128,
    absolute_error_sum: u128,
    relative_error_sum: f64,
    maximum_absolute_error: u64,
    maximum_relative_error: f64,
    histogram: Vec<u64>,
}

struct CompressionSizes {
    csv: u64,
    gzip: u64,
    zstd: u64,
}

struct Reservoir {
    limit: usize,
    seen: u64,
    ids: Vec<u32>,
    rng: SplitMix64,
}

impl Reservoir {
    fn new(seed: u64) -> Self {
        Self {
            limit: SAMPLE_SIZE,
            seen: 0,
            ids: Vec::with_capacity(SAMPLE_SIZE),
            rng: SplitMix64::new(seed),
        }
    }

    fn observe(&mut self, id: u32) {
        self.seen += 1;
        if self.ids.len() < self.limit {
            self.ids.push(id);
            return;
        }
        let replacement = self.rng.below(self.seen);
        if replacement < self.limit as u64 {
            self.ids[usize::try_from(replacement).expect("sample index fits usize")] = id;
        }
    }
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn below(&mut self, upper: u64) -> u64 {
        ((u128::from(self.next()) * u128::from(upper)) >> 64) as u64
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
            .ok_or("usage: name-clean-v1 <normalized.csv> <new-output-directory>")?,
    );
    let output = PathBuf::from(
        args.next()
            .ok_or("usage: name-clean-v1 <normalized.csv> <new-output-directory>")?,
    );
    if args.next().is_some() {
        return Err("usage: name-clean-v1 <normalized.csv> <new-output-directory>".into());
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
    let result = clean(&input, &temporary_output);
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
fn clean(input: &Path, output: &Path) -> Result<String> {
    eprintln!("Hashing original input before processing");
    let checksum_before = checksum(input)?;

    eprintln!("Pass 1/2: indexing names, sanitation rules, and global counts");
    let mut first = first_pass(input)?;
    let sanitation = audit_sanitation(&first);
    eprintln!(
        "Pass 1 complete: {} rows, {} names, {} observations, {:.1?}",
        first.row_count,
        first.name_ids.len(),
        first.total_observations,
        first.elapsed
    );

    eprintln!("Pass 2/2: collecting candidate metadata rows");
    let pass_two_started = Instant::now();
    let (mut rows, source_offsets) = second_pass(input, &first, &sanitation.candidates)?;
    let pass_two_elapsed = pass_two_started.elapsed();

    eprintln!("Aggregating tuples and applying row/global thresholds");
    let compact_started = Instant::now();
    let (clean_offsets, final_names, compaction) = compact_rows(
        &mut rows,
        &source_offsets,
        &sanitation.candidates,
        &mut first.rows_per_name,
        &mut first.observations_per_name,
    )?;
    let compact_elapsed = compact_started.elapsed();

    fs::create_dir(output)?;
    let samples_directory = output.join("samples");
    fs::create_dir(&samples_directory)?;
    write_rejected_samples(&samples_directory, &first)?;
    write_survivor_samples(&samples_directory, &final_names, &first, &clean_offsets)?;

    eprintln!("Writing clean-v1.csv");
    let csv_path = output.join("clean-v1.csv");
    write_clean_csv(
        &csv_path,
        &first.name_ids,
        final_names.len(),
        &rows,
        &clean_offsets,
    )?;
    let validation = validate_clean_csv(&csv_path)?;
    if validation.names != compaction.final_names
        || validation.rows != compaction.final_rows
        || validation.observations != compaction.final_observations
    {
        return Err("independent clean CSV validation totals do not match generation".into());
    }

    eprintln!("Compressing clean CSV with gzip -9 and zstd -19");
    let compression = compress_csv(&csv_path)?;
    let quantization = quantify_q8_error(&rows, compaction.max_final_count);

    eprintln!("Hashing original input after processing");
    let checksum_after = checksum(input)?;
    if checksum_before.sha1 != checksum_after.sha1
        || checksum_before.sha256 != checksum_after.sha256
        || checksum_before.bytes != checksum_after.bytes
    {
        return Err("original input checksum or size changed during processing".into());
    }

    Ok(format_report(
        input,
        &checksum_before,
        &checksum_after,
        &first,
        &sanitation,
        &compaction,
        &compression,
        &quantization,
        pass_two_elapsed,
        compact_elapsed,
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
    let mut rule_masks = Vec::<u8>::with_capacity(31_000_000);
    let mut marker_masks = Vec::<u16>::with_capacity(31_000_000);
    let mut row_count = 0_u32;
    let mut total_observations = 0_u128;
    let mut digit_sample = Reservoir::new(SAMPLE_SEED ^ 0x6469_6769_7473);
    let mut organization_sample = Reservoir::new(SAMPLE_SEED ^ 0x6f72_6761_6e69_7a65);

    for result in reader.byte_records() {
        let record = result?;
        let (name, _country, _gender, count) = parse_record(&record, u64::from(row_count) + 2)?;
        if count == 0 {
            return Err(format!("line {}: zero count", u64::from(row_count) + 2).into());
        }
        let id = if let Some(&id) = name_ids.get(name) {
            id
        } else {
            let id = u32::try_from(name_ids.len())?;
            let text = std::str::from_utf8(name)?;
            let (rule_mask, marker_mask) = classify_name(text);
            name_ids.insert(name.to_vec().into_boxed_slice(), id);
            rows_per_name.push(0);
            observations_per_name.push(0);
            rule_masks.push(rule_mask);
            marker_masks.push(marker_mask);
            if rule_mask & DIGIT_RULE != 0 {
                digit_sample.observe(id);
            }
            if rule_mask & ORGANIZATION_RULE != 0 {
                organization_sample.observe(id);
            }
            id
        };
        rows_per_name[id as usize] = rows_per_name[id as usize]
            .checked_add(1)
            .ok_or("too many rows for one name")?;
        observations_per_name[id as usize] = observations_per_name[id as usize]
            .checked_add(count)
            .ok_or("per-name observation count overflow")?;
        total_observations += u128::from(count);
        row_count = row_count.checked_add(1).ok_or("more than u32::MAX rows")?;
        if u64::from(row_count).is_multiple_of(PROGRESS_ROWS) {
            eprintln!("  pass 1: {row_count} rows");
        }
    }

    Ok(FirstPass {
        name_ids,
        rows_per_name,
        observations_per_name,
        rule_masks,
        marker_masks,
        row_count,
        total_observations,
        digit_sample,
        organization_sample,
        elapsed: started.elapsed(),
    })
}

fn audit_sanitation(first: &FirstPass) -> SanitationAudit {
    let mut independent = [RemovalStats::default(); 4];
    let mut primary = [RemovalStats::default(); 4];
    let mut markers = [RemovalStats::default(); ORGANIZATION_MARKERS.len()];
    let mut initial_global = RemovalStats::default();
    let mut candidates = vec![false; first.name_ids.len()];

    for (id, candidate) in candidates.iter_mut().enumerate() {
        let rows = u64::from(first.rows_per_name[id]);
        let observations = u128::from(first.observations_per_name[id]);
        let rules = first.rule_masks[id];
        for (index, bit) in [DIGIT_RULE, CONTROL_RULE, URL_EMAIL_RULE, ORGANIZATION_RULE]
            .into_iter()
            .enumerate()
        {
            if rules & bit != 0 {
                add_removal(&mut independent[index], rows, observations);
            }
        }
        for (index, stats) in markers.iter_mut().enumerate() {
            if first.marker_masks[id] & (1 << index) != 0 {
                add_removal(stats, rows, observations);
            }
        }

        if let Some(index) = primary_rule_index(rules) {
            add_removal(&mut primary[index], rows, observations);
        } else if first.observations_per_name[id] < GLOBAL_MINIMUM {
            add_removal(&mut initial_global, rows, observations);
        } else {
            *candidate = true;
        }
    }

    SanitationAudit {
        independent,
        primary,
        markers,
        initial_global,
        candidates,
    }
}

fn add_removal(stats: &mut RemovalStats, rows: u64, observations: u128) {
    stats.names += 1;
    stats.rows += rows;
    stats.observations += observations;
}

fn primary_rule_index(rules: u8) -> Option<usize> {
    [DIGIT_RULE, CONTROL_RULE, URL_EMAIL_RULE, ORGANIZATION_RULE]
        .into_iter()
        .position(|bit| rules & bit != 0)
}

fn second_pass(
    path: &Path,
    first: &FirstPass,
    candidates: &[bool],
) -> Result<(Vec<PackedRow>, Vec<u32>)> {
    let mut offsets = Vec::with_capacity(first.name_ids.len() + 1);
    offsets.push(0_u32);
    for (id, &source_rows) in first.rows_per_name.iter().enumerate() {
        let retained = if candidates[id] { source_rows } else { 0 };
        offsets.push(
            offsets
                .last()
                .copied()
                .ok_or("missing source offset")?
                .checked_add(retained)
                .ok_or("candidate row offset overflow")?,
        );
    }
    let mut rows = vec![PackedRow::default(); usize::try_from(*offsets.last().unwrap())?];
    let mut positions = offsets[..first.name_ids.len()].to_vec();
    let mut reader = open_csv(path)?;
    let mut seen_rows = 0_u32;
    for result in reader.byte_records() {
        let record = result?;
        let (name, country, gender, count) = parse_record(&record, u64::from(seen_rows) + 2)?;
        let id = *first
            .name_ids
            .get(name)
            .ok_or("name disappeared between CSV passes")?;
        if candidates[id as usize] {
            let position = usize::try_from(positions[id as usize])?;
            rows[position] = PackedRow {
                count,
                country,
                gender,
            };
            positions[id as usize] += 1;
        }
        seen_rows += 1;
        if u64::from(seen_rows).is_multiple_of(PROGRESS_ROWS) {
            eprintln!("  pass 2: {seen_rows} rows");
        }
    }
    if seen_rows != first.row_count {
        return Err("CSV row count changed between passes".into());
    }
    for (id, &position) in positions.iter().enumerate() {
        if candidates[id] && position != offsets[id + 1] {
            return Err(format!("second pass did not fill candidate name ID {id}").into());
        }
    }
    Ok((rows, offsets))
}

#[allow(clippy::too_many_lines)]
fn compact_rows(
    rows: &mut Vec<PackedRow>,
    source_offsets: &[u32],
    candidates: &[bool],
    rows_per_name: &mut [u32],
    observations_per_name: &mut [u64],
) -> Result<(Vec<u32>, Vec<u32>, CompactionAudit)> {
    let mut clean_offsets = Vec::with_capacity(candidates.len() + 1);
    clean_offsets.push(0_u32);
    let mut final_ids = Vec::new();
    let mut write_position = 0_usize;
    let mut duplicate_rows_collapsed = 0_u64;
    let mut row_threshold_rows = 0_u64;
    let mut row_threshold_observations = 0_u128;
    let mut post_row_global = RemovalStats::default();
    let mut final_observations = 0_u128;
    let mut max_final_count = 0_u32;

    for id in 0..candidates.len() {
        if !candidates[id] {
            clean_offsets.push(u32::try_from(write_position)?);
            continue;
        }
        let start = usize::try_from(source_offsets[id])?;
        let end = usize::try_from(source_offsets[id + 1])?;
        rows[start..end].sort_unstable_by_key(|row| (row.country, row.gender));
        let name_write_start = write_position;
        let mut aggregated_rows = 0_u64;
        let mut read_position = start;
        let mut surviving_observations = 0_u64;
        let mut name_max_count = 0_u32;
        while read_position < end {
            let country = rows[read_position].country;
            let gender = rows[read_position].gender;
            let mut count = 0_u64;
            while read_position < end
                && rows[read_position].country == country
                && rows[read_position].gender == gender
            {
                count = count
                    .checked_add(rows[read_position].count)
                    .ok_or("aggregated tuple count overflow")?;
                read_position += 1;
            }
            aggregated_rows += 1;
            if count < ROW_MINIMUM {
                row_threshold_rows += 1;
                row_threshold_observations += u128::from(count);
            } else {
                let count_u32 = u32::try_from(count)
                    .map_err(|_| format!("aggregated count exceeds u32: {count}"))?;
                rows[write_position] = PackedRow {
                    count,
                    country,
                    gender,
                };
                write_position += 1;
                surviving_observations = surviving_observations
                    .checked_add(count)
                    .ok_or("surviving per-name count overflow")?;
                name_max_count = name_max_count.max(count_u32);
            }
        }
        duplicate_rows_collapsed += u64::try_from(end - start)? - aggregated_rows;
        let surviving_rows = write_position - name_write_start;
        if surviving_observations < GLOBAL_MINIMUM {
            if surviving_rows != 0 {
                add_removal(
                    &mut post_row_global,
                    u64::try_from(surviving_rows)?,
                    u128::from(surviving_observations),
                );
            } else {
                post_row_global.names += 1;
            }
            write_position = name_write_start;
            rows_per_name[id] = 0;
            observations_per_name[id] = 0;
        } else {
            final_ids.push(u32::try_from(id)?);
            rows_per_name[id] = u32::try_from(surviving_rows)?;
            observations_per_name[id] = surviving_observations;
            final_observations += u128::from(surviving_observations);
            max_final_count = max_final_count.max(name_max_count);
        }
        clean_offsets.push(u32::try_from(write_position)?);
        if (id as u64 + 1).is_multiple_of(PROGRESS_ROWS) {
            eprintln!("  compaction: {} names", id + 1);
        }
    }
    rows.truncate(write_position);
    let final_name_count = u64::try_from(final_ids.len())?;
    Ok((
        clean_offsets,
        final_ids,
        CompactionAudit {
            duplicate_rows_collapsed,
            row_threshold_rows,
            row_threshold_observations,
            post_row_global,
            final_names: final_name_count,
            final_rows: u64::try_from(write_position)?,
            final_observations,
            max_final_count,
        },
    ))
}

fn write_clean_csv(
    path: &Path,
    name_ids: &HashMap<Box<[u8]>, u32>,
    expected_names: usize,
    rows: &[PackedRow],
    clean_offsets: &[u32],
) -> Result<()> {
    let mut names: Vec<(&[u8], u32)> = name_ids
        .iter()
        .filter(|(_, id)| clean_offsets[**id as usize] < clean_offsets[**id as usize + 1])
        .map(|(name, &id)| (name.as_ref(), id))
        .collect();
    if names.len() != expected_names {
        return Err("clean CSV name lookup length mismatch".into());
    }
    names.sort_unstable_by(|left, right| left.0.cmp(right.0));
    let mut writer = Writer::from_path(path)?;
    writer.write_record(["name", "country", "gender", "count"])?;
    for (name, id) in names {
        let start = usize::try_from(clean_offsets[id as usize])?;
        let end = usize::try_from(clean_offsets[id as usize + 1])?;
        for row in &rows[start..end] {
            let country_bytes = row.country.to_be_bytes();
            let country = std::str::from_utf8(&country_bytes)?;
            let gender = match row.gender {
                0 => "",
                1 => "F",
                2 => "M",
                value => return Err(format!("invalid packed gender {value}").into()),
            };
            writer.write_record([
                std::str::from_utf8(name)?,
                country,
                gender,
                &row.count.to_string(),
            ])?;
        }
    }
    writer.flush()?;
    Ok(())
}

struct ValidationTotals {
    names: u64,
    rows: u64,
    observations: u128,
}

fn validate_clean_csv(path: &Path) -> Result<ValidationTotals> {
    let mut reader = open_csv(path)?;
    let mut previous_name = Vec::<u8>::new();
    let mut previous_tuple: Option<(u16, u8)> = None;
    let mut current_global = 0_u64;
    let mut names = 0_u64;
    let mut rows = 0_u64;
    let mut observations = 0_u128;
    for result in reader.byte_records() {
        let record = result?;
        let (name, country, gender, count) = parse_record(&record, rows + 2)?;
        if count < ROW_MINIMUM {
            return Err(format!("clean row {} has count {count}", rows + 2).into());
        }
        let text = std::str::from_utf8(name)?;
        if classify_name(text).0 != 0 {
            return Err(format!("clean output retains rejected name: {text:?}").into());
        }
        if previous_name.as_slice() != name {
            if !previous_name.is_empty() && current_global < GLOBAL_MINIMUM {
                return Err(format!(
                    "clean name {:?} has global count {current_global}",
                    String::from_utf8_lossy(&previous_name)
                )
                .into());
            }
            if !previous_name.is_empty() && previous_name.as_slice() >= name {
                return Err("clean names are not strictly lexicographically ordered".into());
            }
            previous_name.clear();
            previous_name.extend_from_slice(name);
            previous_tuple = None;
            current_global = 0;
            names += 1;
        }
        let tuple = (country, gender);
        if previous_tuple.is_some_and(|previous| previous >= tuple) {
            return Err(format!("duplicate or unsorted tuple for name {text:?}").into());
        }
        previous_tuple = Some(tuple);
        current_global += count;
        observations += u128::from(count);
        rows += 1;
    }
    if !previous_name.is_empty() && current_global < GLOBAL_MINIMUM {
        return Err(format!(
            "clean name {:?} has global count {current_global}",
            String::from_utf8_lossy(&previous_name)
        )
        .into());
    }
    Ok(ValidationTotals {
        names,
        rows,
        observations,
    })
}

fn write_rejected_samples(directory: &Path, first: &FirstPass) -> Result<()> {
    let mut wanted = HashSet::new();
    wanted.extend(first.digit_sample.ids.iter().copied());
    wanted.extend(first.organization_sample.ids.iter().copied());
    let mut names = HashMap::<u32, &[u8]>::new();
    for (name, &id) in &first.name_ids {
        if wanted.contains(&id) {
            names.insert(id, name);
        }
    }
    write_rejected_sample(
        &directory.join("rejected_digits.csv"),
        &first.digit_sample.ids,
        first,
        &names,
        false,
    )?;
    write_rejected_sample(
        &directory.join("rejected_organization_markers.csv"),
        &first.organization_sample.ids,
        first,
        &names,
        true,
    )?;
    Ok(())
}

fn write_rejected_sample(
    path: &Path,
    ids: &[u32],
    first: &FirstPass,
    names: &HashMap<u32, &[u8]>,
    include_markers: bool,
) -> Result<()> {
    let mut writer = Writer::from_path(path)?;
    writer.write_record(["global_count", "source_rows", "matched_markers", "name"])?;
    for &id in ids {
        let markers = if include_markers {
            marker_names(first.marker_masks[id as usize])
        } else {
            String::new()
        };
        writer.write_record([
            &first.observations_per_name[id as usize].to_string(),
            &first.rows_per_name[id as usize].to_string(),
            &markers,
            std::str::from_utf8(names[&id])?,
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn write_survivor_samples(
    directory: &Path,
    final_ids: &[u32],
    first: &FirstPass,
    clean_offsets: &[u32],
) -> Result<()> {
    let mut final_names: Vec<(u32, &[u8])> = first
        .name_ids
        .iter()
        .filter(|(_, id)| clean_offsets[**id as usize] < clean_offsets[**id as usize + 1])
        .map(|(name, &id)| (id, name.as_ref()))
        .collect();
    if final_names.len() != final_ids.len() {
        return Err("final-name lookup length mismatch".into());
    }
    final_names.sort_unstable_by_key(|&(id, _)| id);
    let mut low = Reservoir::new(SAMPLE_SEED ^ 0x676c_6f62_616c_3539);
    let mut multi = Reservoir::new(SAMPLE_SEED ^ 0x6d75_6c74_692d_746f);
    let mut single = Reservoir::new(SAMPLE_SEED ^ 0x7369_6e67_6c65_2d74);
    let mut longest = BinaryHeap::<Reverse<(usize, usize, u32)>>::new();
    for &(id, name) in &final_names {
        let global = first.observations_per_name[id as usize];
        if (5..=9).contains(&global) {
            low.observe(id);
        }
        let text = std::str::from_utf8(name)?;
        if text.split_whitespace().count() > 1 {
            multi.observe(id);
        } else {
            single.observe(id);
        }
        let entry = Reverse((text.chars().count(), name.len(), id));
        if longest.len() < SAMPLE_SIZE {
            longest.push(entry);
        } else if longest.peek().is_some_and(|smallest| entry.0 > smallest.0) {
            longest.pop();
            longest.push(entry);
        }
    }
    let names_by_id: HashMap<u32, &[u8]> = final_names.into_iter().collect();
    write_survivor_sample(
        &directory.join("surviving_global_5_9.csv"),
        &low.ids,
        first,
        &names_by_id,
    )?;
    write_survivor_sample(
        &directory.join("surviving_multi_token.csv"),
        &multi.ids,
        first,
        &names_by_id,
    )?;
    write_survivor_sample(
        &directory.join("surviving_single_token.csv"),
        &single.ids,
        first,
        &names_by_id,
    )?;
    let mut longest_entries: Vec<_> = longest.iter().map(|entry| entry.0).collect();
    longest_entries.sort_unstable_by(|left, right| right.cmp(left));
    let mut writer = Writer::from_path(directory.join("longest_surviving.csv"))?;
    writer.write_record([
        "character_length",
        "byte_length",
        "global_count",
        "metadata_rows",
        "name",
    ])?;
    for (characters, bytes, id) in longest_entries {
        writer.write_record([
            &characters.to_string(),
            &bytes.to_string(),
            &first.observations_per_name[id as usize].to_string(),
            &first.rows_per_name[id as usize].to_string(),
            std::str::from_utf8(names_by_id[&id])?,
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn write_survivor_sample(
    path: &Path,
    ids: &[u32],
    first: &FirstPass,
    names: &HashMap<u32, &[u8]>,
) -> Result<()> {
    let mut writer = Writer::from_path(path)?;
    writer.write_record(["global_count", "metadata_rows", "name"])?;
    for &id in ids {
        writer.write_record([
            &first.observations_per_name[id as usize].to_string(),
            &first.rows_per_name[id as usize].to_string(),
            std::str::from_utf8(names[&id])?,
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn marker_names(mask: u16) -> String {
    ORGANIZATION_MARKERS
        .iter()
        .enumerate()
        .filter(|(index, _)| mask & (1 << index) != 0)
        .map(|(_, marker)| *marker)
        .collect::<Vec<_>>()
        .join("|")
}

fn classify_name(text: &str) -> (u8, u16) {
    let mut rules = 0_u8;
    if text.as_bytes().iter().any(u8::is_ascii_digit) {
        rules |= DIGIT_RULE;
    }
    if text
        .chars()
        .any(|character| get_general_category(character) == GeneralCategory::Control)
    {
        rules |= CONTROL_RULE;
    }
    let folded: String = text.case_fold().collect();
    if folded.contains('@')
        || folded.contains("://")
        || contains_boundary_marker(&folded, "http:")
        || contains_boundary_marker(&folded, "https:")
        || contains_boundary_marker(&folded, "www.")
    {
        rules |= URL_EMAIL_RULE;
    }
    let matching_text: String = folded
        .nfd()
        .filter(|character| {
            !matches!(
                get_general_category(*character),
                GeneralCategory::NonspacingMark
                    | GeneralCategory::SpacingMark
                    | GeneralCategory::EnclosingMark
            )
        })
        .collect();
    let mut marker_mask = 0_u16;
    for token in matching_text.split(|character: char| !character.is_alphanumeric()) {
        if let Some(index) = ORGANIZATION_MARKERS
            .iter()
            .position(|marker| *marker == token)
        {
            marker_mask |= 1 << index;
        }
    }
    if marker_mask != 0 {
        rules |= ORGANIZATION_RULE;
    }
    (rules, marker_mask)
}

fn contains_boundary_marker(text: &str, marker: &str) -> bool {
    text.match_indices(marker).any(|(start, _)| {
        start == 0
            || text[..start]
                .chars()
                .next_back()
                .is_some_and(|character| !character.is_alphanumeric())
    })
}

fn parse_record(record: &ByteRecord, line: u64) -> Result<(&[u8], u16, u8, u64)> {
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
        .parse::<u64>()
        .map_err(|error| format!("line {line}: invalid count: {error}"))?;
    Ok((name, country, gender, count))
}

fn checksum(path: &Path) -> Result<Checksum> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut sha1 = Sha1::new();
    let mut sha256 = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    let mut bytes = 0_u64;
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        sha1.update(&buffer[..read]);
        sha256.update(&buffer[..read]);
        bytes += u64::try_from(read)?;
    }
    Ok(Checksum {
        bytes,
        sha1: hexadecimal(sha1.finalize().as_ref()),
        sha256: hexadecimal(sha256.finalize().as_ref()),
    })
}

fn hexadecimal(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(char::from(DIGITS[usize::from(byte >> 4)]));
        output.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    output
}

fn compress_csv(csv_path: &Path) -> Result<CompressionSizes> {
    let gzip_path = csv_path.with_extension("csv.gz");
    let zstd_path = csv_path.with_extension("csv.zst");
    let gzip_output = File::create(&gzip_path)?;
    let gzip_status = Command::new("gzip")
        .args(["-9", "-n", "-c"])
        .arg(csv_path)
        .stdout(Stdio::from(gzip_output))
        .status()?;
    if !gzip_status.success() {
        return Err(format!("gzip failed with status {gzip_status}").into());
    }
    let zstd_status = Command::new("zstd")
        .args(["-19", "-T0", "-q", "-f", "-o"])
        .arg(&zstd_path)
        .arg(csv_path)
        .status()?;
    if !zstd_status.success() {
        return Err(format!("zstd failed with status {zstd_status}").into());
    }
    if !Command::new("gzip")
        .arg("-t")
        .arg(&gzip_path)
        .status()?
        .success()
    {
        return Err("gzip integrity validation failed".into());
    }
    if !Command::new("zstd")
        .args(["-t", "-q"])
        .arg(&zstd_path)
        .status()?
        .success()
    {
        return Err("zstd integrity validation failed".into());
    }
    Ok(CompressionSizes {
        csv: fs::metadata(csv_path)?.len(),
        gzip: fs::metadata(gzip_path)?.len(),
        zstd: fs::metadata(zstd_path)?.len(),
    })
}

fn quantify_q8_error(rows: &[PackedRow], max_count: u32) -> QuantizationStats {
    let mut stats = QuantizationStats {
        histogram: vec![0; 10_001],
        ..QuantizationStats::default()
    };
    for row in rows {
        let original = u32::try_from(row.count).expect("clean count validated as u32");
        let quantized = quantize_count(original, max_count);
        let decoded = dequantize_count(quantized, max_count);
        stats.observe(original, decoded);
    }
    stats
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
        self.rows += 1;
        self.original_sum += u128::from(original);
        self.decoded_sum += u128::from(decoded);
        self.absolute_error_sum += u128::from(absolute);
        self.relative_error_sum += relative;
        self.maximum_absolute_error = self.maximum_absolute_error.max(u64::from(absolute));
        self.maximum_relative_error = self.maximum_relative_error.max(relative);
        let index = (relative * 10_000.0).round().clamp(0.0, 10_000.0) as usize;
        self.histogram[index] += 1;
    }

    fn percentile(&self, percentile: u64) -> f64 {
        let target = self.rows.saturating_mul(percentile).div_ceil(100);
        let mut cumulative = 0_u64;
        for (index, &count) in self.histogram.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                return f64::from(u32::try_from(index).expect("histogram index fits u32"))
                    / 10_000.0;
            }
        }
        1.0
    }
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_precision_loss
)]
fn format_report(
    input: &Path,
    checksum_before: &Checksum,
    checksum_after: &Checksum,
    first: &FirstPass,
    sanitation: &SanitationAudit,
    compaction: &CompactionAudit,
    compression: &CompressionSizes,
    quantization: &QuantizationStats,
    pass_two_elapsed: Duration,
    compact_elapsed: Duration,
) -> String {
    let labels = ["ASCII digits", "control", "URL/email", "organization"];
    let mut primary_rows = String::new();
    let mut independent_rows = String::new();
    for (index, label) in labels.iter().enumerate() {
        let primary = sanitation.primary[index];
        let independent = sanitation.independent[index];
        writeln!(
            primary_rows,
            "| {label} | {} | {} | {} |",
            primary.names, primary.rows, primary.observations
        )
        .expect("writing to a String cannot fail");
        writeln!(
            independent_rows,
            "| {label} | {} | {} | {} |",
            independent.names, independent.rows, independent.observations
        )
        .expect("writing to a String cannot fail");
    }
    let mut marker_rows = String::new();
    for (marker, stats) in ORGANIZATION_MARKERS.iter().zip(&sanitation.markers) {
        writeln!(
            marker_rows,
            "| {marker} | {} | {} | {} |",
            stats.names, stats.rows, stats.observations
        )
        .expect("writing to a String cannot fail");
    }
    let mean_relative = quantization.relative_error_sum / quantization.rows as f64;
    let total_absolute = quantization.absolute_error_sum as f64 / quantization.original_sum as f64;
    let signed_total = (quantization.decoded_sum as f64 - quantization.original_sum as f64)
        / quantization.original_sum as f64;
    format!(
        "# clean-v1 corpus report\n\n\
         Input: `{}`\n\n\
         ## Original-file integrity\n\n\
         | | Before | After |\n\
         |---|---|---|\n\
         | Bytes | {} | {} |\n\
         | SHA-1 | `{}` | `{}` |\n\
         | SHA-256 | `{}` | `{}` |\n\n\
         The original file was opened read-only and both checksums are unchanged.\n\n\
         ## Corpus totals\n\n\
         - Original rows: {}\n\
         - Original distinct names: {}\n\
         - Original observations: {}\n\
         - Initial global `<5`: {} names and {} source rows removed\n\
         - Duplicate candidate rows collapsed during tuple aggregation: {}\n\
         - Aggregated rows removed by row `<2`: {} ({} observations)\n\
         - Post-row global `<5` recheck: {} names and {} surviving rows removed\n\
         - Final rows: {}\n\
         - Final distinct names: {}\n\
         - Final observations: {}\n\
         - Observations retained: {}%\n\n\
         The global threshold is rechecked after row pruning so every final name \
         still has a surviving global count of at least 5.\n\n\
         ## Primary sanitation attribution\n\n\
         Precedence is `digits → controls → URL/email → organization`, making \
         these rows mutually exclusive.\n\n\
         | Rule | Names removed | Source rows removed | Observations removed |\n\
         |---|---:|---:|---:|\n\
         {primary_rows}\n\
         ## Independent sanitation hits\n\n\
         These counts overlap when a key matches multiple rules.\n\n\
         | Rule | Matching names | Matching source rows | Matching observations |\n\
         |---|---:|---:|---:|\n\
         {independent_rows}\n\
         ## Organization-marker hits\n\n\
         | Marker | Matching names | Matching source rows | Matching observations |\n\
         |---|---:|---:|---:|\n\
         {marker_rows}\n\
         The configured marker list is exactly: `{}`. Matching uses full Unicode \
         case folding and Unicode-alphanumeric token boundaries. No generic \
         organization words are hard exclusions.\n\n\
         URL/email sanitation is limited to `@`, `://`, boundary `http:`, \
         boundary `https:`, and boundary `www.`.\n\n\
         ## Output sizes\n\n\
         - `clean-v1.csv`: {} bytes ({:.2} MiB)\n\
         - `clean-v1.csv.gz` (`gzip -9 -n`): {} bytes ({:.2} MiB)\n\
         - `clean-v1.csv.zst` (`zstd -19`): {} bytes ({:.2} MiB)\n\n\
         ## q8 error over clean-v1 rows\n\n\
         - Mean row relative error: {:.4}%\n\
         - p50 row relative error: {:.4}%\n\
         - p95 row relative error: {:.4}%\n\
         - p99 row relative error: {:.4}%\n\
         - Maximum row relative error: {:.4}%\n\
         - Maximum row absolute error: {}\n\
         - Total absolute error / exact observations: {:.4}%\n\
         - Signed decoded-total error: {:+.4}%\n\n\
         Quantization is measured only for the comparison artifact; `clean-v1.csv` \
         retains the original integer counts.\n\n\
         ## Audit samples\n\n\
         Samples contain up to 100 rows per requested category (all available \
         values when fewer than 100 exist) and were selected automatically by \
         reservoir sampling with SplitMix64 seed \
         `0x{SAMPLE_SEED:016x}` (domain-separated per category). Longest names are \
         ranked by Unicode scalar-value count, then UTF-8 byte length. “Multi-token” \
         means more than one whitespace-separated token.\n\n\
         ## Timings\n\n\
         - First CSV pass: {:.1?}\n\
         - Second CSV pass: {pass_two_elapsed:.1?}\n\
         - Aggregation and threshold compaction: {compact_elapsed:.1?}\n\n\
         ## C32 comparison\n\n\
         The C32 artifact benchmark is run separately with the existing benchmark \
         implementation; its results are appended after that exhaustive validation.\n",
        input.display(),
        checksum_before.bytes,
        checksum_after.bytes,
        checksum_before.sha1,
        checksum_after.sha1,
        checksum_before.sha256,
        checksum_after.sha256,
        first.row_count,
        first.name_ids.len(),
        first.total_observations,
        sanitation.initial_global.names,
        sanitation.initial_global.rows,
        compaction.duplicate_rows_collapsed,
        compaction.row_threshold_rows,
        compaction.row_threshold_observations,
        compaction.post_row_global.names,
        compaction.post_row_global.rows,
        compaction.final_rows,
        compaction.final_names,
        compaction.final_observations,
        percentage(compaction.final_observations, first.total_observations),
        ORGANIZATION_MARKERS.join(", "),
        compression.csv,
        mib(compression.csv),
        compression.gzip,
        mib(compression.gzip),
        compression.zstd,
        mib(compression.zstd),
        mean_relative * 100.0,
        quantization.percentile(50) * 100.0,
        quantization.percentile(95) * 100.0,
        quantization.percentile(99) * 100.0,
        quantization.maximum_relative_error * 100.0,
        quantization.maximum_absolute_error,
        total_absolute * 100.0,
        signed_total * 100.0,
        first.elapsed,
    )
}

#[allow(clippy::cast_precision_loss)]
fn percentage(numerator: u128, denominator: u128) -> String {
    format!("{:.6}", numerator as f64 / denominator as f64 * 100.0)
}

#[allow(clippy::cast_precision_loss)]
fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_allowed_name_structures_and_generic_words() {
        for name in [
            "Anne Marie",
            "Jean Pierre",
            "Jean-Pierre",
            "O'Connor",
            "İbrahim",
            "Élodie",
            "Martin Club",
            "The Restaurant & Shop",
            "Association Studio Group",
            "Gmbheimer",
        ] {
            assert_eq!(classify_name(name).0, 0, "name: {name:?}");
        }
    }

    #[test]
    fn rejects_organization_markers_at_token_boundaries() {
        for name in [
            "Mersim Montage GmbH",
            "Mhb-Gmbh",
            "ACME Ltd",
            "Foo LLC",
            "Example,Incorporated",
            "Example PLC",
        ] {
            assert_ne!(
                classify_name(name).0 & ORGANIZATION_RULE,
                0,
                "name: {name:?}"
            );
        }
    }

    #[test]
    fn rejects_other_high_confidence_rules() {
        assert_ne!(classify_name("Martin92").0 & DIGIT_RULE, 0);
        assert_ne!(classify_name("Jean\u{7}Pierre").0 & CONTROL_RULE, 0);
        for name in [
            "name@example.com",
            "https://example.com",
            "Visit WWW.Example.com",
        ] {
            assert_ne!(classify_name(name).0 & URL_EMAIL_RULE, 0);
        }
    }

    #[test]
    fn reapplies_global_minimum_after_row_pruning() {
        let mut rows = vec![
            PackedRow {
                count: 2,
                country: 1,
                gender: 0,
            },
            PackedRow {
                count: 1,
                country: 2,
                gender: 0,
            },
            PackedRow {
                count: 1,
                country: 3,
                gender: 0,
            },
            PackedRow {
                count: 1,
                country: 4,
                gender: 0,
            },
        ];
        let mut rows_per_name = [4];
        let mut observations_per_name = [5];
        let (_, final_ids, audit) = compact_rows(
            &mut rows,
            &[0, 4],
            &[true],
            &mut rows_per_name,
            &mut observations_per_name,
        )
        .unwrap();

        assert!(final_ids.is_empty());
        assert!(rows.is_empty());
        assert_eq!(audit.post_row_global.names, 1);
        assert_eq!(audit.post_row_global.observations, 2);
    }

    #[test]
    fn rejects_aggregated_tuple_count_above_u32() {
        let mut rows = vec![PackedRow {
            count: u64::from(u32::MAX) + 1,
            country: 1,
            gender: 0,
        }];
        let mut rows_per_name = [1];
        let mut observations_per_name = [u64::from(u32::MAX) + 1];
        let result = compact_rows(
            &mut rows,
            &[0, 1],
            &[true],
            &mut rows_per_name,
            &mut observations_per_name,
        );
        assert!(result.is_err());
    }
}
