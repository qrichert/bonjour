use std::path::PathBuf;
#[cfg(not(feature = "standalone"))]
use std::process::Command;

use bonjour::{Classifier, GenderHint};

#[cfg(not(feature = "standalone"))]
const BONJOUR: &str = env!("CARGO_BIN_EXE_bonjour");

#[test]
fn production_inference_preserves_source_spans_and_hard_abstention() {
    let Some(classifier) = test_classifier() else {
        return;
    };
    let input = "  Quentin   Richert  ";
    let inference = classifier.infer(input, None, None);
    assert_eq!(inference.greeting_name, Some("Quentin"));
    assert_eq!(inference.greeting(), Some("Quentin"));
    assert!(input.contains(inference.greeting_name.unwrap()));

    let repeated = "QuentinQuentin42";
    let inference = classifier.infer(repeated, None, None);
    assert_eq!(inference.greeting_name, Some(&repeated[.."Quentin".len()]));

    let organization = classifier.infer("Quentin Richert GmbH", None, None);
    assert_eq!(organization.greeting_name, None);
    assert_eq!(organization.confidence.to_bits(), 0.0_f64.to_bits());
    assert_eq!(organization.gender_hint, None);
    assert_eq!(organization.gender_confidence.to_bits(), 0.0_f64.to_bits());
}

#[test]
fn country_hint_parsing_preserves_frozen_syntactic_behavior() {
    let Some(classifier) = test_classifier() else {
        return;
    };
    let locale = classifier.infer("Quentin Richert", Some("invalid"), Some("fr_FR"));
    let explicit = classifier.infer("Quentin Richert", Some(" fr "), None);
    assert_eq!(locale, explicit);

    let syntactic_unknown = classifier.infer("Quentin Richert", Some("ZZ"), Some("fr_FR"));
    let syntactic_unknown_without_locale = classifier.infer("Quentin Richert", Some("ZZ"), None);
    assert_eq!(syntactic_unknown, syntactic_unknown_without_locale);
}

#[test]
fn gender_hint_disambiguates_supported_candidate_gender() {
    let Some(classifier) = test_classifier() else {
        return;
    };
    let unhinted = classifier.infer("Simone", None, None);
    assert_eq!(unhinted.gender_hint, None);

    let male = classifier.infer_with_gender("Simone", None, None, Some(GenderHint::Male));
    assert_eq!(male.greeting_name, Some("Simone"));
    assert_eq!(male.gender_hint, Some(GenderHint::Male));

    let female = classifier.infer_with_gender("Simone", None, None, Some(GenderHint::Female));
    assert_eq!(female.greeting_name, Some("Simone"));
    assert_eq!(female.gender_hint, Some(GenderHint::Female));
}

#[cfg(not(feature = "standalone"))]
#[test]
fn cli_greeting_and_json_contracts_are_exact_with_runtime_data() {
    let Some(directory) = test_data_directory() else {
        return;
    };
    let greeting = Command::new(BONJOUR)
        .arg(format!("--data-dir={}", directory.display()))
        .arg("Quentin Richert")
        .output()
        .unwrap();
    assert!(
        greeting.status.success(),
        "{}",
        String::from_utf8_lossy(&greeting.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&greeting.stdout),
        "Bonjour Quentin !\n"
    );

    let abstention = Command::new(BONJOUR)
        .arg(format!("--data-dir={}", directory.display()))
        .arg("Quentin Richert SAS")
        .output()
        .unwrap();
    assert!(
        abstention.status.success(),
        "{}",
        String::from_utf8_lossy(&abstention.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&abstention.stdout),
        "Bonjour Quentin Richert SAS !\n"
    );

    let stricter = Command::new(BONJOUR)
        .arg(format!("--data-dir={}", directory.display()))
        .args(["--threshold=0.83", "Quentin Richert"])
        .output()
        .unwrap();
    assert!(
        stricter.status.success(),
        "{}",
        String::from_utf8_lossy(&stricter.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&stricter.stdout),
        "Bonjour Quentin Richert !\n"
    );

    let json = Command::new(BONJOUR)
        .arg(format!("--data-dir={}", directory.display()))
        .args(["--json", "Quentin Richert"])
        .output()
        .unwrap();
    assert!(
        json.status.success(),
        "{}",
        String::from_utf8_lossy(&json.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&json.stdout),
        "{\n  \"input\": \"Quentin Richert\",\n  \"greeting_name\": \"Quentin\",\n  \"confidence\": 0.8258187425766436,\n  \"gender_hint\": \"male\",\n  \"gender_confidence\": 0.9170640418908462\n}\n"
    );

    let below_default = Command::new(BONJOUR)
        .arg(format!("--data-dir={}", directory.display()))
        .args(["--json", "Martin Emmanuel"])
        .output()
        .unwrap();
    assert!(
        below_default.status.success(),
        "{}",
        String::from_utf8_lossy(&below_default.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&below_default.stdout),
        "{\n  \"input\": \"Martin Emmanuel\",\n  \"greeting_name\": \"Martin Emmanuel\",\n  \"confidence\": 0.5481962760808583,\n  \"gender_hint\": null,\n  \"gender_confidence\": 0.0\n}\n"
    );

    let gender = Command::new(BONJOUR)
        .arg(format!("--data-dir={}", directory.display()))
        .args(["--json", "--gender=M", "Simone"])
        .output()
        .unwrap();
    assert!(
        gender.status.success(),
        "{}",
        String::from_utf8_lossy(&gender.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&gender.stdout),
        "{\n  \"input\": \"Simone\",\n  \"greeting_name\": \"Simone\",\n  \"confidence\": 0.8100985093918445,\n  \"gender_hint\": \"male\",\n  \"gender_confidence\": 0.714385674755892\n}\n"
    );
}

fn test_classifier() -> Option<Classifier> {
    let directory = test_data_directory()?;
    Some(Classifier::from_dir(directory).unwrap())
}

fn test_data_directory() -> Option<PathBuf> {
    std::env::var_os("BONJOUR_TEST_DATA_DIR").map(PathBuf::from)
}
