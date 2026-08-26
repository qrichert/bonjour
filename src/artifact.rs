#![allow(clippy::redundant_pub_crate)]

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
#[cfg(any(feature = "standalone", test))]
use std::path::PathBuf;

use bincode::Options;
use boomphf::Mphf;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use xxhash_rust::xxh3::xxh3_64_with_seed;

use crate::{LoadError, LoadErrorKind};

pub(crate) const ARTIFACT_ID: &str = "bonjour-name-data-v1";
pub(crate) const FORMAT_VERSION: u32 = 1;
pub(crate) const KEY_COUNT: usize = 1_803_175;
pub(crate) const ROW_COUNT: usize = 8_722_920;
pub(crate) const GIVEN_TOTAL: u64 = 444_154_759;
pub(crate) const SURNAME_TOTAL: u64 = 489_631_377;
pub(crate) const PINNED_MANIFEST_BYTES: &[u8] = include_bytes!("../data/name-v1/manifest.json");

const ROUTING_SEED: u64 = 0x6e61_6d65_2d72_6f75;
const FINGERPRINT_SEED: u64 = 0x6e61_6d65_2d66_7033;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum GenderHint {
    Female,
    Male,
}

impl GenderHint {
    /// Parse `f`/`female` or `m`/`male`, ignoring ASCII case and whitespace.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "f" | "female" => Some(Self::Female),
            "m" | "male" => Some(Self::Male),
            _ => None,
        }
    }

    /// Return the compact uppercase representation.
    #[must_use]
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

enum ArtifactBytes {
    Owned(Box<[u8]>),
    #[cfg(feature = "standalone")]
    Embedded(&'static [u8]),
}

impl ArtifactBytes {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Owned(bytes) => bytes,
            #[cfg(feature = "standalone")]
            Self::Embedded(bytes) => bytes,
        }
    }
}

impl std::fmt::Debug for ArtifactBytes {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ArtifactBytes")
            .field("len", &self.as_slice().len())
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "standalone")]
#[derive(Clone, Copy)]
pub(crate) struct EmbeddedFile {
    pub name: &'static str,
    pub bytes: &'static [u8],
}

#[cfg(feature = "standalone")]
#[derive(Clone, Copy)]
pub(crate) struct EmbeddedArtifact {
    pub readme: &'static [u8],
    pub notice: &'static [u8],
    pub files: [EmbeddedFile; 12],
}

pub struct C32Artifact {
    mphf: Mphf<u64>,
    fingerprints: ArtifactBytes,
    offsets: ArtifactBytes,
    countries: ArtifactBytes,
    country_ids: ArtifactBytes,
    genders: ArtifactBytes,
    counts: ArtifactBytes,
    max_count: u32,
    surname_counts: ArtifactBytes,
    surname_max_count: u32,
    given_total: u64,
    surname_total: u64,
}

impl std::fmt::Debug for C32Artifact {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("C32Artifact")
            .field("artifact_id", &ARTIFACT_ID)
            .field("format_version", &FORMAT_VERSION)
            .field("key_count", &self.key_count())
            .field("row_count", &self.row_count())
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Deserialize)]
struct Manifest {
    #[serde(rename = "manifest_schema")]
    schema: u32,
    artifact_id: String,
    format_version: u32,
    key_count: u64,
    row_count: u64,
    given_total_observations: u64,
    surname_total_observations: u64,
    files: Vec<ManifestFile>,
    readme_sha256: String,
    notice_sha256: String,
}

#[derive(Debug, Deserialize)]
struct ManifestFile {
    name: String,
    bytes: u64,
    sha256: String,
}

impl C32Artifact {
    pub(crate) fn from_dir(directory: &Path) -> Result<Self, LoadError> {
        require_directory(directory)?;
        let manifest_path = directory.join("manifest.json");
        let manifest_bytes = read_required_file(&manifest_path)?;
        let manifest = parse_and_authenticate_manifest(&manifest_bytes, &manifest_path)?;

        let readme_path = directory.join("README.md");
        let readme = read_required_file(&readme_path)?;
        validate_digest(&readme, &manifest.readme_sha256, &readme_path)?;
        let notice_path = directory.join("NOTICE");
        let notice = read_required_file(&notice_path)?;
        validate_digest(&notice, &manifest.notice_sha256, &notice_path)?;

        let mut files = BTreeMap::new();
        for expected in &manifest.files {
            let path = directory.join(&expected.name);
            let bytes = read_required_file(&path)?;
            validate_file(&bytes, expected, &path)?;
            files.insert(
                expected.name.clone(),
                ArtifactBytes::Owned(bytes.into_boxed_slice()),
            );
        }
        Self::from_files(files, directory)
    }

    #[cfg(feature = "standalone")]
    pub(crate) fn from_embedded(embedded: &EmbeddedArtifact) -> Result<Self, LoadError> {
        let manifest_path = Path::new("<embedded>/manifest.json");
        let manifest = parse_and_authenticate_manifest(PINNED_MANIFEST_BYTES, manifest_path)?;
        validate_digest(
            embedded.readme,
            &manifest.readme_sha256,
            Path::new("<embedded>/README.md"),
        )?;
        validate_digest(
            embedded.notice,
            &manifest.notice_sha256,
            Path::new("<embedded>/NOTICE"),
        )?;

        let embedded_by_name = embedded
            .files
            .iter()
            .map(|file| (file.name, file.bytes))
            .collect::<BTreeMap<_, _>>();
        let mut files = BTreeMap::new();
        for expected in &manifest.files {
            let path = PathBuf::from(format!("<embedded>/{}", expected.name));
            let Some(bytes) = embedded_by_name.get(expected.name.as_str()).copied() else {
                return Err(LoadError::new(
                    LoadErrorKind::MissingData,
                    Some(path),
                    format!("embedded artifact is missing {}", expected.name),
                ));
            };
            validate_file(bytes, expected, &path)?;
            files.insert(expected.name.clone(), ArtifactBytes::Embedded(bytes));
        }
        Self::from_files(files, Path::new("<embedded>"))
    }

    fn from_files(
        mut files: BTreeMap<String, ArtifactBytes>,
        root: &Path,
    ) -> Result<Self, LoadError> {
        let mphf_bytes = take_file(&mut files, "names.mphf", root)?;
        let mphf = bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .reject_trailing_bytes()
            .deserialize(mphf_bytes.as_slice())
            .map_err(|error| {
                LoadError::with_source(
                    LoadErrorKind::CorruptArtifact,
                    Some(root.join("names.mphf")),
                    "cannot deserialize MPHF state",
                    error,
                )
            })?;
        let fingerprints = take_file(&mut files, "fingerprints.u32", root)?;
        let offsets = take_file(&mut files, "row_offsets.u32", root)?;
        let countries = take_file(&mut files, "countries.dict", root)?;
        let country_ids = take_file(&mut files, "country_ids.u8", root)?;
        let genders = take_file(&mut files, "genders.2bit", root)?;
        let counts = take_file(&mut files, "counts.q8", root)?;
        let max_count_file = take_file(&mut files, "quantization_max_count.u32", root)?;
        let surname_counts = take_file(&mut files, "surname_counts.q8", root)?;
        let surname_max_file = take_file(&mut files, "surname_quantization_max_count.u32", root)?;
        let given_total_file = take_file(&mut files, "clean_given_total_observations.u64", root)?;
        let surname_total_file = take_file(&mut files, "surname_total_observations.u64", root)?;
        if !files.is_empty() {
            return Err(corrupt(
                root,
                "manifest contains unexpected constituent files",
            ));
        }

        let max_count = read_one_u32(
            max_count_file.as_slice(),
            root,
            "quantization_max_count.u32",
        )?;
        let surname_max_count = read_one_u32(
            surname_max_file.as_slice(),
            root,
            "surname_quantization_max_count.u32",
        )?;
        let given_total = read_one_u64(
            given_total_file.as_slice(),
            root,
            "clean_given_total_observations.u64",
        )?;
        let surname_total = read_one_u64(
            surname_total_file.as_slice(),
            root,
            "surname_total_observations.u64",
        )?;

        validate_structure(
            fingerprints.as_slice(),
            offsets.as_slice(),
            countries.as_slice(),
            country_ids.as_slice(),
            genders.as_slice(),
            counts.as_slice(),
            max_count,
            surname_counts.as_slice(),
            surname_max_count,
            given_total,
            surname_total,
            root,
        )?;

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

    #[must_use]
    pub fn key_count(&self) -> usize {
        self.fingerprints.as_slice().len() / 4
    }

    #[must_use]
    pub fn row_count(&self) -> usize {
        self.counts.as_slice().len()
    }

    #[cfg(any(feature = "benchmark-internals", test))]
    #[must_use]
    pub fn given_total(&self) -> u64 {
        self.given_total
    }

    #[cfg(any(feature = "benchmark-internals", test))]
    #[must_use]
    pub fn surname_total(&self) -> u64 {
        self.surname_total
    }

    fn gender_at(&self, row: usize) -> u8 {
        (self.genders.as_slice()[row / 4] >> ((row % 4) * 2)) & 0b11
    }
}

impl EvidenceSource for C32Artifact {
    #[allow(clippy::cast_possible_truncation)]
    fn lookup(&self, name: &str, country_hint: Option<[u8; 2]>) -> Option<Evidence> {
        let routing = xxh3_64_with_seed(name.as_bytes(), ROUTING_SEED);
        let slot = usize::try_from(self.mphf.try_hash(&routing)?).ok()?;
        let fingerprint = xxh3_64_with_seed(name.as_bytes(), FINGERPRINT_SEED) as u32;
        if read_u32_at(self.fingerprints.as_slice(), slot)? != fingerprint {
            return None;
        }

        let start = usize::try_from(read_u32_at(self.offsets.as_slice(), slot)?).ok()?;
        let end = usize::try_from(read_u32_at(self.offsets.as_slice(), slot + 1)?).ok()?;
        let mut global_count = 0_u64;
        let mut hinted_count = 0_u64;
        let mut global_female = 0_u64;
        let mut global_male = 0_u64;
        let mut hinted_female = 0_u64;
        let mut hinted_male = 0_u64;

        for row in start..end {
            let count = u64::from(dequantize_count(
                self.counts.as_slice()[row],
                self.max_count,
            ));
            global_count = global_count.saturating_add(count);
            match self.gender_at(row) {
                1 => global_female = global_female.saturating_add(count),
                2 => global_male = global_male.saturating_add(count),
                _ => {}
            }
            let country_offset = usize::from(self.country_ids.as_slice()[row]) * 2;
            if country_hint.is_some_and(|hint| {
                self.countries.as_slice()[country_offset..country_offset + 2] == hint
            }) {
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
                *self.surname_counts.as_slice().get(slot)?,
                self.surname_max_count,
            )),
            given_total: self.given_total,
            surname_total: self.surname_total,
        })
    }
}

fn require_directory(path: &Path) -> Result<(), LoadError> {
    match fs::metadata(path) {
        Ok(metadata) if metadata.is_dir() => Ok(()),
        Ok(_) => Err(LoadError::new(
            LoadErrorKind::MissingData,
            Some(path.to_path_buf()),
            "artifact path is not a directory",
        )),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Err(LoadError::with_source(
            LoadErrorKind::MissingData,
            Some(path.to_path_buf()),
            "artifact directory does not exist",
            error,
        )),
        Err(error) => Err(LoadError::with_source(
            LoadErrorKind::Io,
            Some(path.to_path_buf()),
            "cannot inspect artifact directory",
            error,
        )),
    }
}

fn read_required_file(path: &Path) -> Result<Vec<u8>, LoadError> {
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        let kind = if error.kind() == std::io::ErrorKind::NotFound {
            LoadErrorKind::MissingData
        } else {
            LoadErrorKind::Io
        };
        LoadError::with_source(
            kind,
            Some(path.to_path_buf()),
            "cannot inspect artifact file",
            error,
        )
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(corrupt(path, "artifact entry is not a direct regular file"));
    }
    fs::read(path).map_err(|error| {
        let kind = if error.kind() == std::io::ErrorKind::NotFound {
            LoadErrorKind::MissingData
        } else {
            LoadErrorKind::Io
        };
        LoadError::with_source(
            kind,
            Some(path.to_path_buf()),
            "cannot read artifact file",
            error,
        )
    })
}

fn parse_and_authenticate_manifest(bytes: &[u8], path: &Path) -> Result<Manifest, LoadError> {
    let manifest: Manifest = serde_json::from_slice(bytes).map_err(|error| {
        LoadError::with_source(
            LoadErrorKind::CorruptArtifact,
            Some(path.to_path_buf()),
            "artifact manifest is malformed JSON",
            error,
        )
    })?;
    if manifest.schema != 1 || manifest.format_version != FORMAT_VERSION {
        return Err(LoadError::new(
            LoadErrorKind::UnsupportedFormat,
            Some(path.to_path_buf()),
            "artifact schema or format version is unsupported",
        ));
    }
    if bytes != PINNED_MANIFEST_BYTES {
        return Err(LoadError::new(
            LoadErrorKind::ManifestMismatch,
            Some(path.to_path_buf()),
            "artifact manifest does not match bonjour-name-data-v1",
        ));
    }
    validate_manifest_contract(&manifest, path)?;
    Ok(manifest)
}

fn validate_manifest_contract(manifest: &Manifest, path: &Path) -> Result<(), LoadError> {
    if manifest.artifact_id != ARTIFACT_ID
        || manifest.key_count != KEY_COUNT as u64
        || manifest.row_count != ROW_COUNT as u64
        || manifest.given_total_observations != GIVEN_TOTAL
        || manifest.surname_total_observations != SURNAME_TOTAL
        || manifest.files.len() != 12
    {
        return Err(corrupt(
            path,
            "artifact manifest constants are inconsistent",
        ));
    }
    let mut previous = None;
    for file in &manifest.files {
        if file.name.contains('/') || file.name.contains('\\') || file.name.is_empty() {
            return Err(corrupt(
                path,
                "artifact manifest contains an invalid filename",
            ));
        }
        if previous.is_some_and(|name: &str| name >= file.name.as_str()) {
            return Err(corrupt(
                path,
                "artifact manifest filenames are not strictly bytewise sorted",
            ));
        }
        previous = Some(file.name.as_str());
    }
    Ok(())
}

fn validate_file(bytes: &[u8], expected: &ManifestFile, path: &Path) -> Result<(), LoadError> {
    if u64::try_from(bytes.len()).ok() != Some(expected.bytes) {
        return Err(corrupt(
            path,
            "artifact constituent has the wrong byte length",
        ));
    }
    validate_digest(bytes, &expected.sha256, path)
}

fn validate_digest(bytes: &[u8], expected: &str, path: &Path) -> Result<(), LoadError> {
    let actual = format!("{:x}", Sha256::digest(bytes));
    if actual != expected {
        return Err(corrupt(
            path,
            "artifact file checksum does not match the pinned manifest",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_structure(
    fingerprints: &[u8],
    offsets: &[u8],
    countries: &[u8],
    country_ids: &[u8],
    genders: &[u8],
    counts: &[u8],
    max_count: u32,
    surname_counts: &[u8],
    surname_max_count: u32,
    given_total: u64,
    surname_total: u64,
    root: &Path,
) -> Result<(), LoadError> {
    if fingerprints.len() != KEY_COUNT * 4
        || offsets.len() != (KEY_COUNT + 1) * 4
        || country_ids.len() != ROW_COUNT
        || counts.len() != ROW_COUNT
        || genders.len() != ROW_COUNT.div_ceil(4)
        || surname_counts.len() != KEY_COUNT
    {
        return Err(corrupt(root, "artifact arrays have inconsistent lengths"));
    }
    if countries.is_empty() || !countries.len().is_multiple_of(2) || countries.len() / 2 > 256 {
        return Err(corrupt(root, "country dictionary has an invalid length"));
    }
    if max_count == 0 || surname_max_count == 0 {
        return Err(corrupt(root, "quantization maxima must be nonzero"));
    }
    if given_total != GIVEN_TOTAL || surname_total != SURNAME_TOTAL {
        return Err(corrupt(root, "role-evidence denominators are inconsistent"));
    }
    if read_u32_at(offsets, 0) != Some(0)
        || read_u32_at(offsets, KEY_COUNT) != u32::try_from(ROW_COUNT).ok()
    {
        return Err(corrupt(root, "row-offset endpoints are invalid"));
    }
    let mut previous = 0;
    for index in 1..=KEY_COUNT {
        let current = read_u32_at(offsets, index)
            .expect("the exact row-offset byte length was validated above");
        if current < previous {
            return Err(corrupt(root, "row offsets are not monotonic"));
        }
        previous = current;
    }
    let country_count = countries.len() / 2;
    if country_ids
        .iter()
        .any(|id| usize::from(*id) >= country_count)
    {
        return Err(corrupt(root, "country ID lies outside the dictionary"));
    }
    Ok(())
}

fn take_file(
    files: &mut BTreeMap<String, ArtifactBytes>,
    name: &str,
    root: &Path,
) -> Result<ArtifactBytes, LoadError> {
    files.remove(name).ok_or_else(|| {
        LoadError::new(
            LoadErrorKind::MissingData,
            Some(root.join(name)),
            format!("artifact is missing {name}"),
        )
    })
}

fn read_one_u32(bytes: &[u8], root: &Path, name: &str) -> Result<u32, LoadError> {
    let array: [u8; 4] = bytes
        .try_into()
        .map_err(|_| corrupt(&root.join(name), "artifact value is not one u32"))?;
    Ok(u32::from_le_bytes(array))
}

fn read_one_u64(bytes: &[u8], root: &Path, name: &str) -> Result<u64, LoadError> {
    let array: [u8; 8] = bytes
        .try_into()
        .map_err(|_| corrupt(&root.join(name), "artifact value is not one u64"))?;
    Ok(u64::from_le_bytes(array))
}

fn read_u32_at(bytes: &[u8], index: usize) -> Option<u32> {
    let start = index.checked_mul(4)?;
    let chunk = bytes.get(start..start + 4)?;
    Some(u32::from_le_bytes(chunk.try_into().ok()?))
}

fn corrupt(path: &Path, message: impl Into<String>) -> LoadError {
    LoadError::new(
        LoadErrorKind::CorruptArtifact,
        Some(path.to_path_buf()),
        message,
    )
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

#[cfg(test)]
mod tests {
    use super::*;

    fn production_directory() -> Option<PathBuf> {
        std::env::var_os("BONJOUR_TEST_DATA_DIR")
            .map(PathBuf::from)
            .or_else(|| {
                let root = Path::new(env!("CARGO_MANIFEST_DIR"));
                let directory = root.join("data/name-v1/files");
                directory.join("counts.q8").is_file().then_some(directory)
            })
    }

    fn production_files(directory: &Path) -> BTreeMap<String, ArtifactBytes> {
        let manifest: Manifest = serde_json::from_slice(PINNED_MANIFEST_BYTES).unwrap();
        manifest
            .files
            .into_iter()
            .map(|file| {
                let bytes = fs::read(directory.join(&file.name)).unwrap();
                (file.name, ArtifactBytes::Owned(bytes.into_boxed_slice()))
            })
            .collect()
    }

    #[test]
    fn pinned_manifest_is_well_formed_and_fixed() {
        let manifest = parse_and_authenticate_manifest(
            PINNED_MANIFEST_BYTES,
            Path::new("data/name-v1/manifest.json"),
        )
        .unwrap();
        assert_eq!(manifest.artifact_id, ARTIFACT_ID);
        assert_eq!(manifest.files.len(), 12);
    }

    #[test]
    fn dequantization_handles_degenerate_maximum() {
        assert_eq!(dequantize_count(0, 1), 0);
        assert_eq!(dequantize_count(1, 1), 1);
    }

    #[test]
    fn public_benchmark_values_and_debug_output_are_stable() {
        assert_eq!(GenderHint::Female.as_str(), "F");
        assert_eq!(GenderHint::Male.as_str(), "M");
        assert_eq!(GenderHint::parse(" f "), Some(GenderHint::Female));
        assert_eq!(GenderHint::parse("FEMALE"), Some(GenderHint::Female));
        assert_eq!(GenderHint::parse("m"), Some(GenderHint::Male));
        assert_eq!(GenderHint::parse("Male"), Some(GenderHint::Male));
        assert_eq!(GenderHint::parse("unknown"), None);
        assert!(format!("{:?}", ArtifactBytes::Owned(Box::from([]))).contains("len: 0"));

        let Some(directory) = production_directory() else {
            return;
        };
        let artifact = C32Artifact::from_files(production_files(&directory), &directory).unwrap();
        let debug = format!("{artifact:?}");
        assert!(debug.contains(ARTIFACT_ID));
        assert!(debug.contains(&KEY_COUNT.to_string()));
        assert_eq!(artifact.key_count(), KEY_COUNT);
        assert_eq!(artifact.row_count(), ROW_COUNT);
        assert_eq!(artifact.given_total(), GIVEN_TOTAL);
        assert_eq!(artifact.surname_total(), SURNAME_TOTAL);
    }

    #[cfg(feature = "standalone")]
    #[test]
    fn embedded_loader_rejects_missing_constituents() {
        const MISSING: EmbeddedFile = EmbeddedFile {
            name: "not-a-constituent",
            bytes: b"",
        };
        let embedded = EmbeddedArtifact {
            readme: include_bytes!("../data/name-v1/README.md"),
            notice: include_bytes!("../data/name-v1/NOTICE"),
            files: [MISSING; 12],
        };
        let error = C32Artifact::from_embedded(&embedded).unwrap_err();
        assert_eq!(error.kind(), LoadErrorKind::MissingData);
    }

    #[test]
    fn private_loader_rejects_missing_invalid_and_extra_files() {
        let root = Path::new("test-artifact");
        let error = C32Artifact::from_files(BTreeMap::new(), root).unwrap_err();
        assert_eq!(error.kind(), LoadErrorKind::MissingData);

        let invalid_mphf = BTreeMap::from([(
            "names.mphf".to_string(),
            ArtifactBytes::Owned(Box::from([])),
        )]);
        let error = C32Artifact::from_files(invalid_mphf, root).unwrap_err();
        assert_eq!(error.kind(), LoadErrorKind::CorruptArtifact);

        let Some(directory) = production_directory() else {
            return;
        };
        let mut files = production_files(&directory);
        files.insert(
            "unexpected.bin".to_string(),
            ArtifactBytes::Owned(Box::from([])),
        );
        let error = C32Artifact::from_files(files, &directory).unwrap_err();
        assert_eq!(error.kind(), LoadErrorKind::CorruptArtifact);
    }

    #[test]
    fn manifest_contract_rejects_inconsistent_constants_names_and_order() {
        let path = Path::new("manifest.json");

        let mut inconsistent: Manifest = serde_json::from_slice(PINNED_MANIFEST_BYTES).unwrap();
        inconsistent.key_count = 0;
        assert_eq!(
            validate_manifest_contract(&inconsistent, path)
                .unwrap_err()
                .kind(),
            LoadErrorKind::CorruptArtifact
        );

        let mut invalid_name: Manifest = serde_json::from_slice(PINNED_MANIFEST_BYTES).unwrap();
        invalid_name.files[0].name = "../escape".to_string();
        assert_eq!(
            validate_manifest_contract(&invalid_name, path)
                .unwrap_err()
                .kind(),
            LoadErrorKind::CorruptArtifact
        );

        let mut unordered: Manifest = serde_json::from_slice(PINNED_MANIFEST_BYTES).unwrap();
        unordered.files.swap(0, 1);
        assert_eq!(
            validate_manifest_contract(&unordered, path)
                .unwrap_err()
                .kind(),
            LoadErrorKind::CorruptArtifact
        );
    }

    #[test]
    fn constituent_length_is_validated_before_checksum() {
        let manifest: Manifest = serde_json::from_slice(PINNED_MANIFEST_BYTES).unwrap();
        let error = validate_file(b"", &manifest.files[0], Path::new("constituent")).unwrap_err();
        assert_eq!(error.kind(), LoadErrorKind::CorruptArtifact);
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn structural_validator_rejects_every_mutable_format_invariant() {
        let Some(directory) = production_directory() else {
            return;
        };
        let artifact = C32Artifact::from_files(production_files(&directory), &directory).unwrap();
        let root = Path::new("test-artifact");
        let validate = |fingerprints: &[u8],
                        offsets: &[u8],
                        countries: &[u8],
                        country_ids: &[u8],
                        max_count: u32,
                        given_total: u64| {
            validate_structure(
                fingerprints,
                offsets,
                countries,
                country_ids,
                artifact.genders.as_slice(),
                artifact.counts.as_slice(),
                max_count,
                artifact.surname_counts.as_slice(),
                artifact.surname_max_count,
                given_total,
                artifact.surname_total,
                root,
            )
        };

        assert!(
            validate(
                &artifact.fingerprints.as_slice()[1..],
                artifact.offsets.as_slice(),
                artifact.countries.as_slice(),
                artifact.country_ids.as_slice(),
                artifact.max_count,
                artifact.given_total,
            )
            .is_err()
        );
        assert!(
            validate(
                artifact.fingerprints.as_slice(),
                artifact.offsets.as_slice(),
                b"",
                artifact.country_ids.as_slice(),
                artifact.max_count,
                artifact.given_total,
            )
            .is_err()
        );
        assert!(
            validate(
                artifact.fingerprints.as_slice(),
                artifact.offsets.as_slice(),
                artifact.countries.as_slice(),
                artifact.country_ids.as_slice(),
                0,
                artifact.given_total,
            )
            .is_err()
        );
        assert!(
            validate(
                artifact.fingerprints.as_slice(),
                artifact.offsets.as_slice(),
                artifact.countries.as_slice(),
                artifact.country_ids.as_slice(),
                artifact.max_count,
                0,
            )
            .is_err()
        );

        let mut invalid_endpoints = artifact.offsets.as_slice().to_vec();
        invalid_endpoints[..4].copy_from_slice(&1_u32.to_le_bytes());
        assert!(
            validate(
                artifact.fingerprints.as_slice(),
                &invalid_endpoints,
                artifact.countries.as_slice(),
                artifact.country_ids.as_slice(),
                artifact.max_count,
                artifact.given_total,
            )
            .is_err()
        );

        let mut nonmonotonic = artifact.offsets.as_slice().to_vec();
        nonmonotonic[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(
            validate(
                artifact.fingerprints.as_slice(),
                &nonmonotonic,
                artifact.countries.as_slice(),
                artifact.country_ids.as_slice(),
                artifact.max_count,
                artifact.given_total,
            )
            .is_err()
        );

        let country_count = artifact.countries.as_slice().len() / 2;
        if let Ok(invalid_id) = u8::try_from(country_count) {
            let mut invalid_country_ids = artifact.country_ids.as_slice().to_vec();
            invalid_country_ids[0] = invalid_id;
            assert!(
                validate(
                    artifact.fingerprints.as_slice(),
                    artifact.offsets.as_slice(),
                    artifact.countries.as_slice(),
                    &invalid_country_ids,
                    artifact.max_count,
                    artifact.given_total,
                )
                .is_err()
            );
        }
    }
}
