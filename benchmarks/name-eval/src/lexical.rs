use unicode_general_category::{GeneralCategory, get_general_category};

pub fn candidate_is_eligible(candidate: &str) -> bool {
    let mut has_alphabetic = false;
    for character in candidate.chars() {
        if character.is_alphabetic() {
            has_alphabetic = true;
            continue;
        }
        if is_mark(character)
            || character.is_whitespace()
            || is_apostrophe_separator(character)
            || is_hyphen_separator(character)
        {
            continue;
        }
        return false;
    }
    has_alphabetic
}

fn is_mark(character: char) -> bool {
    matches!(
        get_general_category(character),
        GeneralCategory::NonspacingMark
            | GeneralCategory::SpacingMark
            | GeneralCategory::EnclosingMark
    )
}

fn is_apostrophe_separator(character: char) -> bool {
    matches!(character, '\'' | '‘' | '’' | '‛' | 'ʻ' | 'ʼ' | '＇')
}

fn is_hyphen_separator(character: char) -> bool {
    matches!(character, '-' | '‐' | '‑' | '‒' | '–' | '—' | '―' | '−')
}

#[cfg(test)]
mod tests {
    use unicode_normalization::UnicodeNormalization;

    use super::*;

    #[test]
    fn accepts_name_lexical_structure() {
        for candidate in [
            "Anne Marie",
            "Jean-Pierre",
            "O'Connor",
            "O’Connor",
            "Élodie",
            "İbrahim",
            &"Élodie".nfd().collect::<String>(),
        ] {
            assert!(candidate_is_eligible(candidate), "{candidate:?}");
        }
    }

    #[test]
    fn rejects_non_name_punctuation_and_symbols() {
        for candidate in [
            "A/J*C", "A[[Ine", "A_Kim", "A`S", "A:B", "A<B>", r"A\B", "A1", "[Anne]", "Anne.",
        ] {
            assert!(!candidate_is_eligible(candidate), "{candidate:?}");
        }
    }
}
