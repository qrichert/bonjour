use std::collections::BTreeSet;

use bonjour::benchmark::{candidate_is_eligible, canonicalize};
use unicode_casefold::UnicodeCaseFold;
use unicode_general_category::{GeneralCategory, get_general_category};
use unicode_normalization::UnicodeNormalization;
use unicode_script::{Script, UnicodeScript};

pub(crate) fn model_normalize(value: &str) -> String {
    canonicalize(value)
        .case_fold()
        .collect::<String>()
        .nfc()
        .collect()
}

pub(crate) fn morphology_family(value: &str) -> String {
    let letters = model_normalize(value)
        .nfd()
        .filter(|character| !is_mark(*character))
        .filter(|character| character.is_alphabetic())
        .collect::<String>()
        .nfc()
        .collect::<String>();
    if letters.is_empty() {
        model_normalize(value)
    } else {
        letters
    }
}

pub(crate) fn valid_candidate_form(value: &str) -> bool {
    candidate_is_eligible(value)
        && value
            .chars()
            .filter(|character| character.is_alphabetic())
            .count()
            >= 3
        && value
            .split(|character: char| character.is_whitespace() || matches!(character, '-' | '\''))
            .all(|component| {
                !component.is_empty()
                    && component.chars().next().is_some_and(char::is_alphabetic)
                    && component
                        .chars()
                        .last()
                        .is_some_and(|character| character.is_alphabetic() || is_mark(character))
            })
}

pub(crate) fn contains_organization_component(
    value: &str,
    organization_tokens: &BTreeSet<String>,
) -> bool {
    model_normalize(value)
        .split(|character: char| character.is_whitespace() || matches!(character, '-' | '\''))
        .any(|component| organization_tokens.contains(component))
}

pub(crate) fn greeting_matches(expected: Option<&str>, actual: Option<&str>) -> bool {
    match (expected, actual) {
        (Some(expected), Some(actual)) => canonicalize(expected) == canonicalize(actual),
        (None, None) => true,
        _ => false,
    }
}

pub(crate) fn script_class(value: &str) -> &'static str {
    let scripts = value
        .chars()
        .filter(|character| character.is_alphabetic())
        .map(|character| character.script())
        .filter(|script| !matches!(script, Script::Common | Script::Inherited | Script::Unknown))
        .map(broad_script)
        .collect::<BTreeSet<_>>();
    match scripts.len() {
        0 => "unknown",
        1 => scripts.first().copied().expect("single script"),
        _ => "mixed",
    }
}

fn broad_script(script: Script) -> &'static str {
    match script {
        Script::Latin => "latin",
        Script::Cyrillic => "cyrillic",
        Script::Greek => "greek",
        Script::Arabic => "arabic",
        Script::Han => "han",
        Script::Common | Script::Inherited | Script::Unknown => "unknown",
        _ => "other",
    }
}

fn is_mark(character: char) -> bool {
    matches!(
        get_general_category(character),
        GeneralCategory::NonspacingMark
            | GeneralCategory::SpacingMark
            | GeneralCategory::EnclosingMark
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_case_and_separator_variants() {
        assert_eq!(model_normalize("  JEAN–LUC  "), "jean-luc");
        assert_eq!(model_normalize("O’CONNOR"), "o'connor");
        assert_eq!(model_normalize("İBRAHİM"), "i\u{307}brahi\u{307}m");
    }

    #[test]
    fn groups_accent_and_punctuation_variants() {
        assert_eq!(morphology_family("Élodie"), morphology_family("elodie"));
        assert_eq!(morphology_family("O'Connor"), morphology_family("OConnor"));
        assert_eq!(morphology_family("Anne."), morphology_family("Anne"));
    }

    #[test]
    fn validates_candidate_components() {
        assert!(valid_candidate_form("Jean-Luc"));
        assert!(valid_candidate_form("Anne Marie"));
        assert!(!valid_candidate_form("-Anne"));
        assert!(!valid_candidate_form("Anne."));
        assert!(!valid_candidate_form("Al"));
    }

    #[test]
    fn classifies_broad_scripts() {
        assert_eq!(script_class("Élodie"), "latin");
        assert_eq!(script_class("Ольга"), "cyrillic");
        assert_eq!(script_class("李"), "han");
    }
}
