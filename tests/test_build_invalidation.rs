use std::fs;
use std::process::{Command, Output};

use tempfile::tempdir;

const FILES: [&str; 15] = [
    "manifest.json",
    "README.md",
    "NOTICE",
    "clean_given_total_observations.u64",
    "countries.dict",
    "country_ids.u8",
    "counts.q8",
    "fingerprints.u32",
    "genders.2bit",
    "names.mphf",
    "quantization_max_count.u32",
    "row_offsets.u32",
    "surname_counts.q8",
    "surname_quantization_max_count.u32",
    "surname_total_observations.u64",
];

#[test]
fn standalone_build_reacts_to_environment_and_external_file_changes() {
    let Some((source_documents, source_files)) = test_data_directories() else {
        return;
    };
    let temporary = tempdir().unwrap();
    let external = temporary.path().join("name-data");
    let probe = temporary.path().join("probe");
    let target = temporary.path().join("target");
    fs::create_dir(&external).unwrap();
    fs::create_dir(&probe).unwrap();
    for name in FILES {
        let source = if matches!(name, "manifest.json" | "README.md" | "NOTICE") {
            &source_documents
        } else {
            &source_files
        };
        fs::copy(source.join(name), external.join(name)).unwrap();
    }
    write_probe(&probe);

    let repository = run_probe(&probe, &target, None);
    assert_embedded(&repository);

    let embedded = run_probe(&probe, &target, Some(&external));
    assert_embedded(&embedded);

    let repository_again = run_probe(&probe, &target, None);
    assert_embedded(&repository_again);

    let embedded_again = run_probe(&probe, &target, Some(&external));
    assert_embedded(&embedded_again);
    let counts = external.join("counts.q8");
    let mut bytes = fs::read(&counts).unwrap();
    bytes[0] ^= 1;
    fs::write(&counts, bytes).unwrap();
    let rejected = run_probe(&probe, &target, Some(&external));
    assert!(!rejected.status.success());
    assert!(
        stderr(&rejected).contains("wrong checksum"),
        "{}",
        stderr(&rejected)
    );
}

#[allow(
    clippy::literal_string_with_formatting_args,
    clippy::unnecessary_debug_formatting
)]
fn write_probe(directory: &std::path::Path) {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    fs::write(
        directory.join("Cargo.toml"),
        format!(
            "[package]\nname = \"bonjour-build-probe\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\n[dependencies]\nbonjour = {{ path = {root:?}, features = [\"standalone\"] }}\n"
        ),
    )
    .unwrap();
    fs::create_dir(directory.join("src")).unwrap();
    fs::write(
        directory.join("src/main.rs"),
        "fn main() {\n    match bonjour::Classifier::standalone() {\n        Ok(classifier) => println!(\"{}\", classifier.infer(\"Quentin Richert\", None, None).greeting_name.unwrap_or(\"none\")),\n        Err(error) => println!(\"{:?}\", error.kind()),\n    }\n}\n",
    )
    .unwrap();
}

fn run_probe(
    probe: &std::path::Path,
    target: &std::path::Path,
    data: Option<&std::path::Path>,
) -> Output {
    let mut command = Command::new(env!("CARGO"));
    command
        .args(["run", "--quiet", "--manifest-path"])
        .arg(probe.join("Cargo.toml"))
        .env("CARGO_TARGET_DIR", target);
    if let Some(data) = data {
        command.env("BONJOUR_DATA_DIR", data);
    } else {
        command.env_remove("BONJOUR_DATA_DIR");
    }
    command.output().unwrap()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn assert_embedded(output: &Output) {
    assert!(output.status.success(), "{}", stderr(output));
    assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "Quentin");
}

fn test_data_directories() -> Option<(std::path::PathBuf, std::path::PathBuf)> {
    std::env::var_os("BONJOUR_TEST_DATA_DIR")
        .map(std::path::PathBuf::from)
        .map(|directory| (directory.clone(), directory))
        .or_else(|| {
            let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
            let repository_documents = root.join("data/name-v1");
            let repository_files = repository_documents.join("files");
            repository_files
                .join("counts.q8")
                .is_file()
                .then_some((repository_documents, repository_files))
        })
}
