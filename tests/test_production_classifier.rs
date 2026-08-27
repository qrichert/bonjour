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
    assert_eq!(organization.decision_score.to_bits(), 0.0_f64.to_bits());
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
        "{\n  \"input\": \"Quentin Richert\",\n  \"selected_candidate\": \"Quentin\",\n  \"decision_score\": 0.8258187425766436,\n  \"decision\": {\n    \"candidate_quality\": 0.9341785978125992,\n    \"winner_margin\": 0.7342375610072307,\n    \"margin_signal\": 1.0,\n    \"role_llr\": 3.1788968086994007,\n    \"role_signal\": 0.8258296772199872,\n    \"reliability\": 0.7386898426132628,\n    \"alphabetic_length\": 7,\n    \"minimum_alphabetic_length\": 3,\n    \"contributions\": {\n      \"candidate_quality\": 0.0,\n      \"winner_margin\": 0.1,\n      \"role\": 0.578080774053991,\n      \"reliability\": 0.14773796852265256\n    },\n    \"pre_veto_score\": 0.8258187425766436,\n    \"post_veto_score\": 0.8258187425766436,\n    \"segmented_candidate\": false,\n    \"segmentation_mechanism\": null,\n    \"segmented_candidate_penalty\": 0.0,\n    \"vetoes\": {\n      \"strong_organization_marker\": false,\n      \"generic_organization_marker\": false,\n      \"ampersand\": false,\n      \"candidate_too_short\": false\n    }\n  },\n  \"candidates\": [\n    {\n      \"candidate\": \"Quentin\",\n      \"ranking_score\": 0.9341785978125992,\n      \"signals\": {\n        \"corpus_score\": 0.9341785978125992\n      }\n    },\n    {\n      \"candidate\": \"Richert\",\n      \"ranking_score\": 0.1999410368053685,\n      \"signals\": {\n        \"corpus_score\": 0.1999410368053685\n      }\n    },\n    {\n      \"candidate\": \"Quentin Richert\",\n      \"ranking_score\": null,\n      \"signals\": {\n        \"corpus_score\": null\n      }\n    }\n  ],\n  \"gender_hint\": \"male\",\n  \"gender_confidence\": 0.9170640418908462\n}\n"
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
        "{\n  \"input\": \"Martin Emmanuel\",\n  \"selected_candidate\": \"Martin Emmanuel\",\n  \"decision_score\": 0.5481962760808583,\n  \"decision\": {\n    \"candidate_quality\": 0.7068577742176809,\n    \"winner_margin\": 0.04391163594672676,\n    \"margin_signal\": 0.08782327189345351,\n    \"role_llr\": 2.4953750309459153,\n    \"role_signal\": 0.7442401833792305,\n    \"reliability\": 0.09222910263025777,\n    \"alphabetic_length\": 14,\n    \"minimum_alphabetic_length\": 3,\n    \"contributions\": {\n      \"candidate_quality\": 0.0,\n      \"winner_margin\": 0.008782327189345351,\n      \"role\": 0.5209681283654614,\n      \"reliability\": 0.018445820526051555\n    },\n    \"pre_veto_score\": 0.5481962760808583,\n    \"post_veto_score\": 0.5481962760808583,\n    \"segmented_candidate\": false,\n    \"segmentation_mechanism\": null,\n    \"segmented_candidate_penalty\": 0.0,\n    \"vetoes\": {\n      \"strong_organization_marker\": false,\n      \"generic_organization_marker\": false,\n      \"ampersand\": false,\n      \"candidate_too_short\": false\n    }\n  },\n  \"candidates\": [\n    {\n      \"candidate\": \"Martin Emmanuel\",\n      \"ranking_score\": 0.7068577742176809,\n      \"signals\": {\n        \"corpus_score\": 0.7068577742176809\n      }\n    },\n    {\n      \"candidate\": \"Emmanuel\",\n      \"ranking_score\": 0.6629461382709542,\n      \"signals\": {\n        \"corpus_score\": 0.6629461382709542\n      }\n    },\n    {\n      \"candidate\": \"Martin\",\n      \"ranking_score\": 0.614387611173123,\n      \"signals\": {\n        \"corpus_score\": 0.614387611173123\n      }\n    }\n  ],\n  \"gender_hint\": null,\n  \"gender_confidence\": 0.0\n}\n"
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
        "{\n  \"input\": \"Simone\",\n  \"selected_candidate\": \"Simone\",\n  \"decision_score\": 0.8100985093918445,\n  \"decision\": {\n    \"candidate_quality\": 0.8365742000974182,\n    \"winner_margin\": 1.0,\n    \"margin_signal\": 1.0,\n    \"role_llr\": 2.685054369563951,\n    \"role_signal\": 0.7691664066737898,\n    \"reliability\": 0.858410123600958,\n    \"alphabetic_length\": 6,\n    \"minimum_alphabetic_length\": 3,\n    \"contributions\": {\n      \"candidate_quality\": 0.0,\n      \"winner_margin\": 0.1,\n      \"role\": 0.5384164846716528,\n      \"reliability\": 0.17168202472019162\n    },\n    \"pre_veto_score\": 0.8100985093918445,\n    \"post_veto_score\": 0.8100985093918445,\n    \"segmented_candidate\": false,\n    \"segmentation_mechanism\": null,\n    \"segmented_candidate_penalty\": 0.0,\n    \"vetoes\": {\n      \"strong_organization_marker\": false,\n      \"generic_organization_marker\": false,\n      \"ampersand\": false,\n      \"candidate_too_short\": false\n    }\n  },\n  \"candidates\": [\n    {\n      \"candidate\": \"Simone\",\n      \"ranking_score\": 0.8365742000974182,\n      \"signals\": {\n        \"corpus_score\": 0.8365742000974182\n      }\n    }\n  ],\n  \"gender_hint\": \"male\",\n  \"gender_confidence\": 0.714385674755892\n}\n"
    );

    let hard_abstention = Command::new(BONJOUR)
        .arg(format!("--data-dir={}", directory.display()))
        .args(["--json", "Quentin Richert SAS"])
        .output()
        .unwrap();
    assert!(
        hard_abstention.status.success(),
        "{}",
        String::from_utf8_lossy(&hard_abstention.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&hard_abstention.stdout),
        "{\n  \"input\": \"Quentin Richert SAS\",\n  \"selected_candidate\": null,\n  \"decision_score\": 0.0,\n  \"decision\": {\n    \"candidate_quality\": null,\n    \"winner_margin\": null,\n    \"margin_signal\": null,\n    \"role_llr\": null,\n    \"role_signal\": null,\n    \"reliability\": null,\n    \"alphabetic_length\": null,\n    \"minimum_alphabetic_length\": 3,\n    \"contributions\": null,\n    \"pre_veto_score\": null,\n    \"post_veto_score\": 0.0,\n    \"segmented_candidate\": null,\n    \"segmentation_mechanism\": null,\n    \"segmented_candidate_penalty\": 0.0,\n    \"vetoes\": {\n      \"strong_organization_marker\": true,\n      \"generic_organization_marker\": false,\n      \"ampersand\": false,\n      \"candidate_too_short\": false\n    }\n  },\n  \"candidates\": [\n    {\n      \"candidate\": \"Quentin\",\n      \"ranking_score\": 0.9237220385528407,\n      \"signals\": {\n        \"corpus_score\": 0.9237220385528407\n      }\n    },\n    {\n      \"candidate\": \"SAS\",\n      \"ranking_score\": 0.33597018983124183,\n      \"signals\": {\n        \"corpus_score\": 0.33597018983124183\n      }\n    },\n    {\n      \"candidate\": \"Richert\",\n      \"ranking_score\": 0.1999410368053685,\n      \"signals\": {\n        \"corpus_score\": 0.1999410368053685\n      }\n    },\n    {\n      \"candidate\": \"Quentin Richert\",\n      \"ranking_score\": null,\n      \"signals\": {\n        \"corpus_score\": null\n      }\n    },\n    {\n      \"candidate\": \"Richert SAS\",\n      \"ranking_score\": null,\n      \"signals\": {\n        \"corpus_score\": null\n      }\n    }\n  ],\n  \"gender_hint\": null,\n  \"gender_confidence\": 0.0\n}\n"
    );
}

fn test_classifier() -> Option<Classifier> {
    let directory = test_data_directory()?;
    Some(Classifier::from_dir(directory).unwrap())
}

fn test_data_directory() -> Option<PathBuf> {
    std::env::var_os("BONJOUR_TEST_DATA_DIR").map(PathBuf::from)
}
