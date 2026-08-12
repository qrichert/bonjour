use std::collections::{BTreeSet, HashMap};
use std::error::Error;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use boomphf::Mphf;
use csv::{ByteRecord, Reader, ReaderBuilder};
use fst::{Map, MapBuilder};
use xxhash_rust::xxh3::xxh3_64;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const PROGRESS_ROWS: u64 = 5_000_000;
const MPHF_GAMMA: f64 = 1.7;

struct FirstPass {
    name_ids: HashMap<Box<[u8]>, u32>,
    rows_per_name: Vec<u32>,
    country_codes: Vec<u16>,
    row_count: u32,
    max_count: u32,
    elapsed: Duration,
}

struct Rows {
    offsets_by_id: Vec<u32>,
    countries: Vec<u8>,
    genders: Vec<u8>,
    counts: Vec<u32>,
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

struct ArtifactSizes {
    fst_lossless: u64,
    mphf_lossless: u64,
    mphf_quantized: u64,
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
            .ok_or("usage: name-indexes-benchmark <normalized.csv> <new-output-directory>")?,
    );
    let output = PathBuf::from(
        args.next()
            .ok_or("usage: name-indexes-benchmark <normalized.csv> <new-output-directory>")?,
    );
    if args.next().is_some() {
        return Err("usage: name-indexes-benchmark <normalized.csv> <new-output-directory>".into());
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

    let total_started = Instant::now();
    let result = build_all(&input, &temporary_output);
    match result {
        Ok((sizes, report)) => {
            fs::write(temporary_output.join("report.txt"), &report)?;
            fs::rename(&temporary_output, &output)?;
            println!("{report}");
            println!("Output: {}", output.display());
            println!("Total elapsed: {:.1?}", total_started.elapsed());
            println!(
                "Artifact bytes: A={}, B={}, C={}",
                sizes.fst_lossless, sizes.mphf_lossless, sizes.mphf_quantized
            );
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
fn build_all(input: &Path, output: &Path) -> Result<(ArtifactSizes, String)> {
    eprintln!("Pass 1/2: indexing names and counting rows");
    let first = first_pass(input)?;
    eprintln!(
        "Pass 1 complete: {} rows, {} names, {} countries, max count {}, {:.1?}",
        first.row_count,
        first.name_ids.len(),
        first.country_codes.len(),
        first.max_count,
        first.elapsed
    );

    let mut names_by_id: Vec<&[u8]> = vec![&[]; first.name_ids.len()];
    for (name, &id) in &first.name_ids {
        names_by_id[id as usize] = name;
    }

    eprintln!("Pass 2/2: packing source rows in memory");
    let pass_two_started = Instant::now();
    let rows = second_pass(input, &first)?;
    let pass_two_elapsed = pass_two_started.elapsed();
    eprintln!("Pass 2 complete: {pass_two_elapsed:.1?}");

    eprintln!("Sorting names for FST construction");
    let sort_started = Instant::now();
    let mut lexicographic_ids: Vec<u32> = (0..u32::try_from(names_by_id.len())?).collect();
    lexicographic_ids.sort_unstable_by(|&left, &right| {
        names_by_id[left as usize].cmp(names_by_id[right as usize])
    });
    let sort_elapsed = sort_started.elapsed();

    eprintln!("Hashing names and checking 64-bit fingerprint collisions");
    let fingerprints_by_id: Vec<u64> = names_by_id.iter().map(|name| xxh3_64(name)).collect();
    check_fingerprint_collisions(&fingerprints_by_id, &names_by_id)?;

    eprintln!("Building MPHF");
    let mphf_started = Instant::now();
    let mphf = Mphf::new_parallel(MPHF_GAMMA, &fingerprints_by_id, None);
    let mphf_elapsed = mphf_started.elapsed();
    let mut ids_by_mphf_slot = vec![u32::MAX; names_by_id.len()];
    let mut fingerprints_by_mphf_slot = vec![0_u64; names_by_id.len()];
    for (id, &fingerprint) in fingerprints_by_id.iter().enumerate() {
        let slot = usize::try_from(mphf.hash(&fingerprint))?;
        if ids_by_mphf_slot[slot] != u32::MAX {
            return Err(format!("MPHF assigned more than one key to slot {slot}").into());
        }
        ids_by_mphf_slot[slot] = u32::try_from(id)?;
        fingerprints_by_mphf_slot[slot] = fingerprint;
    }
    if ids_by_mphf_slot.contains(&u32::MAX) {
        return Err("MPHF left at least one slot unused".into());
    }

    fs::create_dir(output)?;
    let a_directory = output.join("a-fst-lossless");
    let b_directory = output.join("b-mphf-lossless");
    let c_directory = output.join("c-mphf-quantized");
    fs::create_dir(&a_directory)?;
    fs::create_dir(&b_directory)?;
    fs::create_dir(&c_directory)?;

    eprintln!("Writing A: FST + lossless packed metadata");
    let output_started = Instant::now();
    write_country_dictionary(&a_directory, &first.country_codes)?;
    write_fst(&a_directory, &lexicographic_ids, &names_by_id)?;
    write_lossless_metadata(&a_directory, &lexicographic_ids, &rows)?;

    eprintln!("Writing B: MPHF + fingerprints + lossless packed metadata");
    write_country_dictionary(&b_directory, &first.country_codes)?;
    write_mphf_index(&b_directory, &mphf, &fingerprints_by_mphf_slot)?;
    write_lossless_metadata(&b_directory, &ids_by_mphf_slot, &rows)?;

    eprintln!("Writing C: MPHF + fingerprints + quantized packed metadata");
    write_country_dictionary(&c_directory, &first.country_codes)?;
    write_mphf_index(&c_directory, &mphf, &fingerprints_by_mphf_slot)?;
    let quantization =
        write_quantized_metadata(&c_directory, &ids_by_mphf_slot, &rows, first.max_count)?;
    let output_elapsed = output_started.elapsed();

    eprintln!("Validating serialized indexes");
    validate_fst(&a_directory, &lexicographic_ids, &names_by_id)?;
    validate_mphf(&b_directory, &first.name_ids, &fingerprints_by_mphf_slot)?;
    validate_metadata_sizes(
        &a_directory,
        first.name_ids.len(),
        usize::try_from(first.row_count)?,
        false,
    )?;
    validate_metadata_sizes(
        &b_directory,
        first.name_ids.len(),
        usize::try_from(first.row_count)?,
        false,
    )?;
    validate_metadata_sizes(
        &c_directory,
        first.name_ids.len(),
        usize::try_from(first.row_count)?,
        true,
    )?;

    let sizes = ArtifactSizes {
        fst_lossless: directory_size(&a_directory)?,
        mphf_lossless: directory_size(&b_directory)?,
        mphf_quantized: directory_size(&c_directory)?,
    };
    let report = format_report(
        input,
        &first,
        pass_two_elapsed,
        sort_elapsed,
        mphf_elapsed,
        output_elapsed,
        &sizes,
        &quantization,
    );
    Ok((sizes, report))
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
    let mut name_ids = HashMap::<Box<[u8]>, u32>::with_capacity(1_000_000);
    let mut rows_per_name = Vec::<u32>::with_capacity(1_000_000);
    let mut country_codes = BTreeSet::<u16>::new();
    let mut row_count = 0_u32;
    let mut max_count = 0_u32;

    for result in reader.byte_records() {
        let record = result?;
        let (name, country, _gender, count) = parse_record(&record, u64::from(row_count) + 2)?;
        let id = if let Some(&id) = name_ids.get(name) {
            id
        } else {
            let id = u32::try_from(name_ids.len())?;
            name_ids.insert(name.to_vec().into_boxed_slice(), id);
            rows_per_name.push(0);
            id
        };
        rows_per_name[id as usize] = rows_per_name[id as usize]
            .checked_add(1)
            .ok_or("too many rows for one name")?;
        country_codes.insert(country);
        max_count = max_count.max(count);
        row_count = row_count.checked_add(1).ok_or("more than u32::MAX rows")?;
        if u64::from(row_count).is_multiple_of(PROGRESS_ROWS) {
            eprintln!("  pass 1: {row_count} rows");
        }
    }

    Ok(FirstPass {
        name_ids,
        rows_per_name,
        country_codes: country_codes.into_iter().collect(),
        row_count,
        max_count,
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
        let end = offsets_by_id[id as usize + 1];
        if position >= end {
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
    for (id, &position) in positions.iter().enumerate() {
        if position != offsets_by_id[id + 1] {
            return Err(format!("second pass did not fill name ID {id}").into());
        }
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

fn check_fingerprint_collisions(fingerprints: &[u64], names: &[&[u8]]) -> Result<()> {
    let mut sorted: Vec<(u64, u32)> = fingerprints
        .iter()
        .copied()
        .enumerate()
        .map(|(id, fingerprint)| Ok((fingerprint, u32::try_from(id)?)))
        .collect::<Result<_>>()?;
    sorted.sort_unstable();
    for pair in sorted.windows(2) {
        if pair[0].0 == pair[1].0 {
            let left = String::from_utf8_lossy(names[pair[0].1 as usize]);
            let right = String::from_utf8_lossy(names[pair[1].1 as usize]);
            return Err(
                format!("64-bit fingerprint collision between {left:?} and {right:?}").into(),
            );
        }
    }
    Ok(())
}

fn write_country_dictionary(directory: &Path, country_codes: &[u16]) -> Result<()> {
    let mut writer = BufWriter::new(File::create(directory.join("countries.dict"))?);
    for &code in country_codes {
        writer.write_all(&code.to_be_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

fn write_fst(directory: &Path, order: &[u32], names: &[&[u8]]) -> Result<()> {
    let writer = BufWriter::new(File::create(directory.join("names.fst"))?);
    let mut builder = MapBuilder::new(writer)?;
    for (slot, &id) in order.iter().enumerate() {
        builder.insert(names[id as usize], u64::try_from(slot)?)?;
    }
    builder.finish()?;
    Ok(())
}

fn write_mphf_index(
    directory: &Path,
    mphf: &Mphf<u64>,
    fingerprints_by_slot: &[u64],
) -> Result<()> {
    let mut mphf_writer = BufWriter::new(File::create(directory.join("names.mphf"))?);
    bincode::serialize_into(&mut mphf_writer, mphf)?;
    mphf_writer.flush()?;

    let mut fingerprint_writer = BufWriter::new(File::create(directory.join("fingerprints.u64"))?);
    for &fingerprint in fingerprints_by_slot {
        fingerprint_writer.write_all(&fingerprint.to_le_bytes())?;
    }
    fingerprint_writer.flush()?;
    Ok(())
}

fn write_lossless_metadata(directory: &Path, order: &[u32], rows: &Rows) -> Result<()> {
    let mut row_offsets = Vec::with_capacity(order.len() + 1);
    let mut count_offsets = Vec::with_capacity(order.len() + 1);
    row_offsets.push(0_u32);
    count_offsets.push(0_u32);

    let mut country_writer = BufWriter::new(File::create(directory.join("country_ids.u8"))?);
    let gender_file = File::create(directory.join("genders.2bit"))?;
    let mut gender_writer = TwoBitWriter::new(BufWriter::new(gender_file));
    let mut count_writer = BufWriter::new(File::create(directory.join("counts.varint"))?);
    let mut count_bytes = 0_u32;

    for &id in order {
        let range = row_range(rows, id)?;
        for position in range.clone() {
            country_writer.write_all(&[rows.countries[position]])?;
            gender_writer.write(rows.genders[position])?;
            count_bytes = count_bytes
                .checked_add(write_varint(
                    &mut count_writer,
                    u64::from(rows.counts[position]),
                )?)
                .ok_or("count stream exceeds u32::MAX bytes")?;
        }
        row_offsets.push(
            row_offsets
                .last()
                .copied()
                .ok_or("missing row offset")?
                .checked_add(u32::try_from(range.len())?)
                .ok_or("row offset overflow")?,
        );
        count_offsets.push(count_bytes);
    }
    country_writer.flush()?;
    gender_writer.finish()?;
    count_writer.flush()?;
    write_u32_file(&directory.join("row_offsets.u32"), &row_offsets)?;
    write_u32_file(&directory.join("count_offsets.u32"), &count_offsets)?;
    Ok(())
}

fn write_quantized_metadata(
    directory: &Path,
    order: &[u32],
    rows: &Rows,
    max_count: u32,
) -> Result<QuantizationStats> {
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

    for &id in order {
        let range = row_range(rows, id)?;
        for position in range.clone() {
            country_writer.write_all(&[rows.countries[position]])?;
            gender_writer.write(rows.genders[position])?;
            let original = rows.counts[position];
            let quantized = quantize_count(original, max_count);
            let decoded = dequantize_count(quantized, max_count);
            count_writer.write_all(&[quantized])?;
            stats.observe(original, decoded);
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
    Ok(stats)
}

fn row_range(rows: &Rows, id: u32) -> Result<std::ops::Range<usize>> {
    let id = usize::try_from(id)?;
    Ok(usize::try_from(rows.offsets_by_id[id])?..usize::try_from(rows.offsets_by_id[id + 1])?)
}

fn write_u32_file(path: &Path, values: &[u32]) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    for &value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

fn write_varint(writer: &mut impl Write, mut value: u64) -> Result<u32> {
    let mut buffer = [0_u8; 10];
    let mut length = 0_usize;
    loop {
        let mut byte = u8::try_from(value & 0x7f)?;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        buffer[length] = byte;
        length += 1;
        if value == 0 {
            break;
        }
    }
    writer.write_all(&buffer[..length])?;
    Ok(u32::try_from(length)?)
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

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn quantize_count(count: u32, max_count: u32) -> u8 {
    if count == 0 || max_count <= 1 {
        return 0;
    }
    let position = f64::from(count).ln() / f64::from(max_count).ln();
    let bucket = 1.0 + position * 254.0;
    bucket.round().clamp(1.0, 255.0) as u8
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
        let relative = if original == 0 {
            f64::from(decoded != 0)
        } else {
            f64::from(absolute) / f64::from(original)
        };
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

fn validate_fst(directory: &Path, order: &[u32], names: &[&[u8]]) -> Result<()> {
    let map = Map::new(fs::read(directory.join("names.fst"))?)?;
    if map.len() != order.len() {
        return Err("serialized FST has the wrong key count".into());
    }
    for (slot, &id) in order.iter().enumerate() {
        if map.get(names[id as usize]) != Some(u64::try_from(slot)?) {
            return Err(format!("FST lookup mismatch at slot {slot}").into());
        }
    }
    Ok(())
}

fn validate_mphf(
    directory: &Path,
    known_names: &HashMap<Box<[u8]>, u32>,
    expected_fingerprints: &[u64],
) -> Result<()> {
    let reader = BufReader::new(File::open(directory.join("names.mphf"))?);
    let mphf: Mphf<u64> = bincode::deserialize_from(reader)?;
    for (slot, &fingerprint) in expected_fingerprints.iter().enumerate() {
        if usize::try_from(mphf.hash(&fingerprint))? != slot {
            return Err(format!("serialized MPHF lookup mismatch at slot {slot}").into());
        }
    }
    for unknown in [
        b"definitely-not-a-known-name".as_slice(),
        b"supercalifragilisticexpialidocious".as_slice(),
        b"__bonjour_unknown_probe__".as_slice(),
    ] {
        if known_names.contains_key(unknown) {
            continue;
        }
        let fingerprint = xxh3_64(unknown);
        let slot = usize::try_from(mphf.hash(&fingerprint))?;
        if expected_fingerprints[slot] == fingerprint {
            return Err("unknown-name fingerprint unexpectedly passed membership check".into());
        }
    }
    Ok(())
}

fn validate_metadata_sizes(
    directory: &Path,
    names: usize,
    rows: usize,
    quantized: bool,
) -> Result<()> {
    let expected_row_offsets = u64::try_from((names + 1) * 4)?;
    expect_file_size(&directory.join("row_offsets.u32"), expected_row_offsets)?;
    expect_file_size(&directory.join("country_ids.u8"), u64::try_from(rows)?)?;
    expect_file_size(
        &directory.join("genders.2bit"),
        u64::try_from(rows.div_ceil(4))?,
    )?;
    if quantized {
        expect_file_size(&directory.join("counts.q8"), u64::try_from(rows)?)?;
    } else {
        expect_file_size(&directory.join("count_offsets.u32"), expected_row_offsets)?;
    }
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
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.is_file() {
            total = total.checked_add(metadata.len()).ok_or("size overflow")?;
        }
    }
    Ok(total)
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::cast_precision_loss)]
fn format_report(
    input: &Path,
    first: &FirstPass,
    pass_two_elapsed: Duration,
    sort_elapsed: Duration,
    mphf_elapsed: Duration,
    output_elapsed: Duration,
    sizes: &ArtifactSizes,
    quantization: &QuantizationStats,
) -> String {
    let mean_relative = quantization.relative_error_sum / quantization.rows as f64;
    let total_relative = if quantization.original_sum == 0 {
        0.0
    } else {
        quantization.absolute_error_sum as f64 / quantization.original_sum as f64
    };
    let signed_total = if quantization.original_sum == 0 {
        0.0
    } else {
        (quantization.decoded_sum as f64 - quantization.original_sum as f64)
            / quantization.original_sum as f64
    };
    format!(
        "Name index representation benchmark\n\
         ===================================\n\
         Input: {}\n\
         Rows: {}\n\
         Distinct exact names: {}\n\
         Countries: {}\n\
         Maximum exact count: {}\n\
         \n\
         A. FST + lossless packed metadata: {} bytes ({})\n\
         B. MPHF + 64-bit fingerprint + lossless packed metadata: {} bytes ({})\n\
         C. MPHF + 64-bit fingerprint + q8 logarithmic counts: {} bytes ({})\n\
         \n\
         C versus B: {:+.2}%\n\
         \n\
         Quantization error (decoded count versus exact count):\n\
           mean row relative error: {:.4}%\n\
           p50 row relative error: {:.4}%\n\
           p95 row relative error: {:.4}%\n\
           p99 row relative error: {:.4}%\n\
           maximum row relative error: {:.4}%\n\
           maximum row absolute error: {}\n\
           total absolute error / total exact counts: {:.4}%\n\
           signed decoded-total error: {:+.4}%\n\
         \n\
         Timings:\n\
           CSV pass 1: {:.1?}\n\
           CSV pass 2: {:.1?}\n\
           lexicographic sort: {:.1?}\n\
           MPHF build: {:.1?}\n\
           artifact writes: {:.1?}",
        input.display(),
        first.row_count,
        first.name_ids.len(),
        first.country_codes.len(),
        first.max_count,
        sizes.fst_lossless,
        human_bytes(sizes.fst_lossless),
        sizes.mphf_lossless,
        human_bytes(sizes.mphf_lossless),
        sizes.mphf_quantized,
        human_bytes(sizes.mphf_quantized),
        (sizes.mphf_quantized as f64 / sizes.mphf_lossless as f64 - 1.0) * 100.0,
        mean_relative * 100.0,
        quantization.percentile_relative_error(50) * 100.0,
        quantization.percentile_relative_error(95) * 100.0,
        quantization.percentile_relative_error(99) * 100.0,
        quantization.max_relative_error * 100.0,
        quantization.max_absolute_error,
        total_relative * 100.0,
        signed_total * 100.0,
        first.elapsed,
        pass_two_elapsed,
        sort_elapsed,
        mphf_elapsed,
        output_elapsed,
    )
}

#[allow(clippy::cast_precision_loss)]
fn human_bytes(bytes: u64) -> String {
    const MEBIBYTE: f64 = 1024.0 * 1024.0;
    format!("{:.2} MiB", bytes as f64 / MEBIBYTE)
}
