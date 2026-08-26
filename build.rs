use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

const DATA_DIRECTORY: &str = "data/name-v1";
const REPOSITORY_FILES_DIRECTORY: &str = "files";
const GENERATED_FILE: &str = "embedded_data.rs";

#[derive(Deserialize)]
struct Manifest {
    files: Vec<ManifestFile>,
    readme_sha256: String,
    notice_sha256: String,
}

#[derive(Deserialize)]
struct ManifestFile {
    name: String,
    bytes: u64,
    sha256: String,
}

fn main() {
    println!("cargo::rustc-check-cfg=cfg(bonjour_embedded_data)");
    println!("cargo::rerun-if-env-changed=BONJOUR_DATA_DIR");

    let crate_root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"));
    let repository_data = crate_root.join(DATA_DIRECTORY);
    let repository_files = repository_data.join(REPOSITORY_FILES_DIRECTORY);
    let pinned_manifest_path = repository_data.join("manifest.json");
    let pinned_manifest_bytes = fs::read(&pinned_manifest_path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", pinned_manifest_path.display()));
    let manifest: Manifest = serde_json::from_slice(&pinned_manifest_bytes)
        .unwrap_or_else(|error| panic!("invalid pinned name-data manifest: {error}"));

    emit_repository_watches(&repository_data, &repository_files, &manifest);
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR")).join(GENERATED_FILE);
    if env::var_os("CARGO_FEATURE_STANDALONE").is_none() {
        write_unavailable(&output);
        return;
    }

    let selected = if let Some(directory) = env::var_os("BONJOUR_DATA_DIR") {
        let directory = PathBuf::from(directory);
        let directory = fs::canonicalize(&directory).unwrap_or_else(|error| {
            panic!(
                "cannot resolve BONJOUR_DATA_DIR {}: {error}",
                directory.display()
            )
        });
        emit_external_watches(&directory, &manifest);
        validate_directory(&directory, &directory, &pinned_manifest_bytes, &manifest)
            .unwrap_or_else(|error| panic!("invalid BONJOUR_DATA_DIR: {error}"));
        (directory.clone(), directory)
    } else {
        let present = manifest
            .files
            .iter()
            .filter(|file| repository_files.join(&file.name).exists())
            .count();
        if present == manifest.files.len() {
            validate_directory(
                &repository_data,
                &repository_files,
                &pinned_manifest_bytes,
                &manifest,
            )
            .unwrap_or_else(|error| panic!("invalid repository name data: {error}"));
            (repository_data, repository_files)
        } else if present == 0 {
            panic!(
                "the standalone feature requires bonjour-name-data-v1; set BONJOUR_DATA_DIR to the extracted artifact, place all twelve constituents in data/name-v1/files, or download a self-contained binary from https://github.com/qrichert/bonjour/releases"
            );
        } else {
            panic!(
                "repository name data is incomplete: found {present} of {} constituents",
                manifest.files.len()
            );
        }
    };

    println!("cargo::rustc-cfg=bonjour_embedded_data");
    write_embedded(&output, &selected.0, &selected.1, &manifest);
}

fn emit_repository_watches(documents: &Path, files: &Path, manifest: &Manifest) {
    emit_watches(documents, files, manifest);
}

fn emit_external_watches(directory: &Path, manifest: &Manifest) {
    emit_watches(directory, directory, manifest);
}

fn emit_watches(documents: &Path, files: &Path, manifest: &Manifest) {
    for name in ["manifest.json", "README.md", "NOTICE"] {
        println!("cargo::rerun-if-changed={}", documents.join(name).display());
    }
    for file in &manifest.files {
        println!(
            "cargo::rerun-if-changed={}",
            files.join(&file.name).display()
        );
    }
}

fn validate_directory(
    documents: &Path,
    files: &Path,
    pinned_manifest_bytes: &[u8],
    manifest: &Manifest,
) -> Result<(), String> {
    validate_direct_directory(documents)?;
    validate_direct_directory(files)?;
    validate_regular_file(
        &documents.join("manifest.json"),
        Some(pinned_manifest_bytes),
        None,
        None,
    )?;
    validate_regular_file(
        &documents.join("README.md"),
        None,
        None,
        Some(&manifest.readme_sha256),
    )?;
    validate_regular_file(
        &documents.join("NOTICE"),
        None,
        None,
        Some(&manifest.notice_sha256),
    )?;
    for file in &manifest.files {
        validate_regular_file(
            &files.join(&file.name),
            None,
            Some(file.bytes),
            Some(&file.sha256),
        )?;
    }
    Ok(())
}

fn validate_direct_directory(directory: &Path) -> Result<(), String> {
    let metadata = fs::symlink_metadata(directory)
        .map_err(|error| format!("cannot inspect {}: {error}", directory.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(format!("{} is not a direct directory", directory.display()));
    }
    Ok(())
}

fn validate_regular_file(
    path: &Path,
    exact: Option<&[u8]>,
    expected_length: Option<u64>,
    expected_digest: Option<&str>,
) -> Result<(), String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot inspect {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!("{} is not a direct regular file", path.display()));
    }
    let bytes =
        fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    if exact.is_some_and(|expected| bytes != expected) {
        return Err(format!(
            "{} does not match the pinned manifest",
            path.display()
        ));
    }
    if expected_length.is_some_and(|expected| bytes.len() as u64 != expected) {
        return Err(format!("{} has the wrong byte length", path.display()));
    }
    if expected_digest.is_some_and(|expected| format!("{:x}", Sha256::digest(&bytes)) != expected) {
        return Err(format!("{} has the wrong checksum", path.display()));
    }
    Ok(())
}

fn write_embedded(output: &Path, documents: &Path, files: &Path, manifest: &Manifest) {
    let mut source = String::from(
        "const EMBEDDED_ARTIFACT: Option<EmbeddedArtifact> = Some(EmbeddedArtifact {\n",
    );
    writeln!(
        source,
        "    readme: include_bytes!({}),",
        rust_string(&documents.join("README.md"))
    )
    .expect("writing to a String cannot fail");
    writeln!(
        source,
        "    notice: include_bytes!({}),",
        rust_string(&documents.join("NOTICE"))
    )
    .expect("writing to a String cannot fail");
    source.push_str("    files: [\n");
    for file in &manifest.files {
        writeln!(
            source,
            "        crate::artifact::EmbeddedFile {{ name: {:?}, bytes: include_bytes!({}) }},",
            file.name,
            rust_string(&files.join(&file.name))
        )
        .expect("writing to a String cannot fail");
    }
    source.push_str("    ],\n});\n");
    fs::write(output, source)
        .unwrap_or_else(|error| panic!("cannot write {}: {error}", output.display()));
}

fn write_unavailable(output: &Path) {
    fs::write(
        output,
        "const EMBEDDED_ARTIFACT: Option<EmbeddedArtifact> = None;\n",
    )
    .unwrap_or_else(|error| panic!("cannot write {}: {error}", output.display()));
}

fn rust_string(path: &Path) -> String {
    serde_json::to_string(&path.to_string_lossy())
        .expect("filesystem path must serialize as a Rust-compatible string")
}
