use std::fs;
use std::path::Path;

use bonjour::{Classifier, LoadErrorKind};
use tempfile::tempdir;

const MANIFEST: &[u8] = include_bytes!("../data/name-v1/manifest.json");
const README: &[u8] = include_bytes!("../data/name-v1/README.md");
const NOTICE: &[u8] = include_bytes!("../data/name-v1/NOTICE");

#[test]
fn missing_root_and_missing_manifest_are_missing_data() {
    let temporary = tempdir().unwrap();
    let absent = temporary.path().join("absent");
    assert_kind(&absent, LoadErrorKind::MissingData);

    let empty = temporary.path().join("empty");
    fs::create_dir(&empty).unwrap();
    assert_kind(&empty, LoadErrorKind::MissingData);
}

#[test]
fn non_directory_root_is_missing_data() {
    let temporary = tempdir().unwrap();
    let file = temporary.path().join("artifact");
    fs::write(&file, b"not a directory").unwrap();
    assert_kind(&file, LoadErrorKind::MissingData);
}

#[test]
fn uninspectable_root_is_io() {
    let temporary = tempdir().unwrap();
    let path = temporary.path().join("x".repeat(300));
    assert_kind(&path, LoadErrorKind::Io);
}

#[test]
fn malformed_manifest_is_corrupt() {
    let temporary = tempdir().unwrap();
    fs::write(temporary.path().join("manifest.json"), b"{").unwrap();
    assert_kind(temporary.path(), LoadErrorKind::CorruptArtifact);
}

#[test]
fn unsupported_schema_is_reported_before_manifest_authentication() {
    let temporary = tempdir().unwrap();
    fs::write(
        temporary.path().join("manifest.json"),
        manifest_with_versions(2, 1),
    )
    .unwrap();
    assert_kind(temporary.path(), LoadErrorKind::UnsupportedFormat);
}

#[test]
fn supported_but_unpinned_manifest_is_a_mismatch() {
    let temporary = tempdir().unwrap();
    fs::write(
        temporary.path().join("manifest.json"),
        manifest_with_versions(1, 1),
    )
    .unwrap();
    assert_kind(temporary.path(), LoadErrorKind::ManifestMismatch);
}

#[test]
fn missing_constituent_is_missing_data() {
    let temporary = tempdir().unwrap();
    write_trust_files(temporary.path());
    assert_kind(temporary.path(), LoadErrorKind::MissingData);
}

#[test]
fn present_constituent_with_bad_checksum_is_corrupt() {
    let temporary = tempdir().unwrap();
    write_trust_files(temporary.path());
    fs::write(
        temporary.path().join("clean_given_total_observations.u64"),
        [0; 8],
    )
    .unwrap();
    assert_kind(temporary.path(), LoadErrorKind::CorruptArtifact);
}

#[cfg(unix)]
#[test]
fn symlinked_manifest_is_corrupt() {
    use std::os::unix::fs::symlink;

    let temporary = tempdir().unwrap();
    let source = temporary.path().join("source-manifest.json");
    fs::write(&source, MANIFEST).unwrap();
    symlink(&source, temporary.path().join("manifest.json")).unwrap();
    assert_kind(temporary.path(), LoadErrorKind::CorruptArtifact);
}

#[cfg(unix)]
#[test]
fn permission_failure_is_io() {
    use std::os::unix::fs::PermissionsExt;

    let temporary = tempdir().unwrap();
    let manifest = temporary.path().join("manifest.json");
    fs::write(&manifest, MANIFEST).unwrap();
    fs::set_permissions(&manifest, fs::Permissions::from_mode(0o000)).unwrap();
    let result = Classifier::from_dir(temporary.path());
    fs::set_permissions(&manifest, fs::Permissions::from_mode(0o600)).unwrap();
    let error = result.unwrap_err();
    assert_eq!(error.kind(), LoadErrorKind::Io);
}

fn assert_kind(path: &Path, expected: LoadErrorKind) {
    let error = Classifier::from_dir(path).unwrap_err();
    assert_eq!(error.kind(), expected, "{error}");
    assert!(error.path().is_some());
}

fn write_trust_files(path: &Path) {
    fs::write(path.join("manifest.json"), MANIFEST).unwrap();
    fs::write(path.join("README.md"), README).unwrap();
    fs::write(path.join("NOTICE"), NOTICE).unwrap();
}

fn manifest_with_versions(schema: u32, format: u32) -> String {
    format!(
        "{{\n  \"manifest_schema\": {schema},\n  \"artifact_id\": \"bonjour-name-data-v1\",\n  \"format_version\": {format},\n  \"key_count\": 1803175,\n  \"row_count\": 8722920,\n  \"given_total_observations\": 444154759,\n  \"surname_total_observations\": 489631377,\n  \"files\": [],\n  \"readme_sha256\": \"x\",\n  \"notice_sha256\": \"x\"\n}}\n"
    )
}
