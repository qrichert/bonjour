use std::error::Error;
use std::fs::{self, File};
use std::io::BufReader;
use std::path::Path;

use boomphf::Mphf;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use xxhash_rust::xxh3::xxh3_64_with_seed;

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

const ROUTING_SEED: u64 = 0x6e61_6d65_2d72_6f75;
const FINGERPRINT_SEED: u64 = 0x6e61_6d65_2d66_7033;
const CLEAN_V1_KEYS: usize = 1_803_175;
const CLEAN_V1_ROWS: usize = 8_722_920;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GenderHint {
    Female,
    Male,
}

impl GenderHint {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Female => "F",
            Self::Male => "M",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Evidence {
    pub global_count: u64,
    pub country_count: u64,
    pub effective_count: u64,
    pub female_count: u64,
    pub male_count: u64,
    pub surname_count: u64,
    pub given_total: u64,
    pub surname_total: u64,
}

pub trait EvidenceSource {
    fn lookup(&self, name: &str, country_hint: Option<[u8; 2]>) -> Option<Evidence>;
}

pub struct C32Artifact {
    mphf: Mphf<u64>,
    fingerprints: Vec<u32>,
    offsets: Vec<u32>,
    countries: Vec<[u8; 2]>,
    country_ids: Vec<u8>,
    genders: Vec<u8>,
    counts: Vec<u8>,
    max_count: u32,
    surname_counts: Vec<u8>,
    surname_max_count: u32,
    given_total: u64,
    surname_total: u64,
}

#[derive(Deserialize)]
struct ManifestRow {
    file: String,
    bytes: u64,
    sha256: String,
}

impl C32Artifact {
    pub fn open(directory: &Path, manifest: &Path, surname_manifest: &Path) -> Result<Self> {
        validate_manifest(directory, manifest, 8)?;
        validate_manifest(directory, surname_manifest, 4)?;

        let mphf =
            bincode::deserialize_from(BufReader::new(File::open(directory.join("names.mphf"))?))?;
        let fingerprints = read_u32_file(&directory.join("fingerprints.u32"))?;
        let offsets = read_u32_file(&directory.join("row_offsets.u32"))?;
        let country_bytes = fs::read(directory.join("countries.dict"))?;
        let countries = country_bytes
            .chunks_exact(2)
            .map(|bytes| [bytes[0], bytes[1]])
            .collect::<Vec<_>>();
        if country_bytes.len() != countries.len() * 2 {
            return Err("countries.dict has a trailing byte".into());
        }
        let country_ids = fs::read(directory.join("country_ids.u8"))?;
        let genders = fs::read(directory.join("genders.2bit"))?;
        let counts = fs::read(directory.join("counts.q8"))?;
        let max_count_bytes = fs::read(directory.join("quantization_max_count.u32"))?;
        let max_count = u32::from_le_bytes(
            max_count_bytes
                .as_slice()
                .try_into()
                .map_err(|_| "quantization maximum is not one u32")?,
        );
        let surname_counts = fs::read(directory.join("surname_counts.q8"))?;
        let surname_max_count =
            read_one_u32(&directory.join("surname_quantization_max_count.u32"))?;
        let given_total = read_one_u64(&directory.join("clean_given_total_observations.u64"))?;
        let surname_total = read_one_u64(&directory.join("surname_total_observations.u64"))?;

        if fingerprints.len() != CLEAN_V1_KEYS {
            return Err(format!(
                "artifact has {} keys, expected clean-v1's {CLEAN_V1_KEYS}",
                fingerprints.len()
            )
            .into());
        }
        if offsets.len() != fingerprints.len() + 1 {
            return Err("row-offset count does not equal key count plus one".into());
        }
        if offsets.first() != Some(&0)
            || offsets.last().copied().map(|value| value as usize) != Some(CLEAN_V1_ROWS)
            || offsets.windows(2).any(|pair| pair[0] > pair[1])
        {
            return Err("invalid row offsets".into());
        }
        if country_ids.len() != CLEAN_V1_ROWS || counts.len() != CLEAN_V1_ROWS {
            return Err("metadata arrays do not contain the clean-v1 row count".into());
        }
        if surname_counts.len() != CLEAN_V1_KEYS {
            return Err("surname count array does not contain one value per clean-v1 key".into());
        }
        if given_total == 0 || surname_total == 0 {
            return Err("role-evidence denominators must be non-zero".into());
        }
        if genders.len() != CLEAN_V1_ROWS.div_ceil(4) {
            return Err("packed gender array has the wrong length".into());
        }
        if country_ids
            .iter()
            .any(|&id| usize::from(id) >= countries.len())
        {
            return Err("country ID lies outside countries.dict".into());
        }

        Ok(Self {
            mphf,
            fingerprints,
            offsets,
            countries,
            country_ids,
            genders,
            counts,
            max_count,
            surname_counts,
            surname_max_count,
            given_total,
            surname_total,
        })
    }

    pub fn key_count(&self) -> usize {
        self.fingerprints.len()
    }

    pub fn row_count(&self) -> usize {
        self.counts.len()
    }

    pub fn given_total(&self) -> u64 {
        self.given_total
    }

    pub fn surname_total(&self) -> u64 {
        self.surname_total
    }

    fn gender_at(&self, row: usize) -> u8 {
        (self.genders[row / 4] >> ((row % 4) * 2)) & 0b11
    }
}

impl EvidenceSource for C32Artifact {
    fn lookup(&self, name: &str, country_hint: Option<[u8; 2]>) -> Option<Evidence> {
        let bytes = name.as_bytes();
        let routing = xxh3_64_with_seed(bytes, ROUTING_SEED);
        let slot = usize::try_from(self.mphf.try_hash(&routing)?).ok()?;
        let fingerprint = xxh3_64_with_seed(bytes, FINGERPRINT_SEED) as u32;
        if self.fingerprints.get(slot).copied()? != fingerprint {
            return None;
        }

        let start = usize::try_from(*self.offsets.get(slot)?).ok()?;
        let end = usize::try_from(*self.offsets.get(slot + 1)?).ok()?;
        let mut global_count = 0_u64;
        let mut hinted_count = 0_u64;
        let mut global_female = 0_u64;
        let mut global_male = 0_u64;
        let mut hinted_female = 0_u64;
        let mut hinted_male = 0_u64;

        for row in start..end {
            let count = u64::from(dequantize_count(self.counts[row], self.max_count));
            global_count = global_count.saturating_add(count);
            match self.gender_at(row) {
                1 => global_female = global_female.saturating_add(count),
                2 => global_male = global_male.saturating_add(count),
                _ => {}
            }
            if country_hint
                .is_some_and(|hint| self.countries[self.country_ids[row] as usize] == hint)
            {
                hinted_count = hinted_count.saturating_add(count);
                match self.gender_at(row) {
                    1 => hinted_female = hinted_female.saturating_add(count),
                    2 => hinted_male = hinted_male.saturating_add(count),
                    _ => {}
                }
            }
        }

        let use_hint = hinted_count != 0;
        Some(Evidence {
            global_count,
            country_count: hinted_count,
            effective_count: if use_hint { hinted_count } else { global_count },
            female_count: if use_hint {
                hinted_female
            } else {
                global_female
            },
            male_count: if use_hint { hinted_male } else { global_male },
            surname_count: u64::from(dequantize_count(
                self.surname_counts[slot],
                self.surname_max_count,
            )),
            given_total: self.given_total,
            surname_total: self.surname_total,
        })
    }
}

fn validate_manifest(directory: &Path, manifest: &Path, expected_rows: usize) -> Result<()> {
    let mut reader = csv::Reader::from_path(manifest)?;
    let mut rows = 0_usize;
    for row in reader.deserialize::<ManifestRow>() {
        let row = row?;
        if row.file.contains('/') || row.file.contains('\\') {
            return Err(format!("artifact manifest filename is not local: {}", row.file).into());
        }
        let path = directory.join(&row.file);
        let bytes = fs::read(&path)?;
        if bytes.len() as u64 != row.bytes {
            return Err(format!("{} has the wrong byte length", path.display()).into());
        }
        let actual = format!("{:x}", Sha256::digest(&bytes));
        if actual != row.sha256 {
            return Err(format!("{} does not match the fixed baseline", path.display()).into());
        }
        rows += 1;
    }
    if rows != expected_rows {
        return Err(format!("artifact manifest has {rows} rows, expected {expected_rows}").into());
    }
    Ok(())
}

fn read_one_u32(path: &Path) -> Result<u32> {
    let bytes = fs::read(path)?;
    Ok(u32::from_le_bytes(bytes.as_slice().try_into().map_err(
        |_| format!("{} is not one u32", path.display()),
    )?))
}

fn read_one_u64(path: &Path) -> Result<u64> {
    let bytes = fs::read(path)?;
    Ok(u64::from_le_bytes(bytes.as_slice().try_into().map_err(
        |_| format!("{} is not one u64", path.display()),
    )?))
}

fn read_u32_file(path: &Path) -> Result<Vec<u32>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(format!("{} is not a u32 array", path.display()).into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
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
