//! Extract a probable first name from an arbitrary display name, with a
//! confidence score.
//!
//! A display name is not necessarily a person: it may be a company or a club
//! (`ACME Corporation`, `Club de Tennis Strasbourg`). So rather than assuming a
//! `[first] [last]` structure, we treat the name as a *bag of tokens*, look each
//! one up in a frequency-weighted first-name table, and penalize organization
//! evidence. Uncertain cases yield low confidence (or `None`), and the caller
//! decides whether to greet by name. Precision over recall: a missed greeting is
//! cheap, greeting a tennis club "Bonjour, Martin" is not.

use std::collections::HashMap;
use std::sync::OnceLock;

use serde::Serialize;

/// Embedded first-name table (`name,weight` CSV). Weights are synthetic
/// placeholders — see the file header.
const FIRST_NAMES_CSV: &str = include_str!("../data/first_names.csv");

/// Lowercased organization markers. Their presence strongly suggests the display
/// name is a company/club rather than a person, so we crush confidence.
/// Multilingual on purpose — obvious non-person names should read as such
/// regardless of country.
const ORG_MARKERS: &[&str] = &[
    "sas",
    "sarl",
    "sa",
    "sasu",
    "eurl",
    "sci",
    "gie",
    "gmbh",
    "ag",
    "ug",
    "ltd",
    "llc",
    "inc",
    "corp",
    "corporation",
    "co",
    "company",
    "plc",
    "bv",
    "nv",
    "oy",
    "ab",
    "srl",
    "spa",
    "pty",
    "association",
    "asso",
    "club",
    "fc",
    "foundation",
    "fondation",
    "group",
    "groupe",
    "holding",
    "holdings",
    "partners",
    "fils",
    "frères",
    "cie",
];

/// Confidence multiplier applied once when any organization evidence is present.
/// Tuned so a strong name (`quentin` ≈ 0.95) drops to ≈ 0.1, matching the
/// README's `Quentin Richert SAS` example.
const ORG_MULTIPLIER: f64 = 0.1;

/// Suggested confidence cut-off for callers deciding whether to use the greeting
/// name. Below this, fall back to the full display name.
pub const DEFAULT_THRESHOLD: f64 = 0.4;

/// The result of [`extract`].
#[derive(Debug, Clone, Serialize)]
pub struct Extraction {
    /// The original display name, unchanged.
    pub input: String,
    /// Best first-name candidate, echoed as it appeared in `input` (so output
    /// reads `Quentin`, not `quentin`). `None` only when no token matched the
    /// first-name table at all.
    pub first_name: Option<String>,
    /// Confidence in `first_name`, in `0.0..=1.0`.
    pub confidence: f64,
    // TODO(gender/country): add `gender` + `country`. Gender depends on country
    // (e.g. "Simone" -> FR=F, IT=M) and is needed for gendered greetings.
}

impl Extraction {
    /// The greeting name, but only when we are confident enough
    /// (`confidence >= threshold`).
    ///
    /// This is the value a consumer would persist as `greeting_name`; below the
    /// threshold it returns `None` so the caller falls back to the display name.
    #[must_use]
    pub fn greeting_name(&self, threshold: f64) -> Option<&str> {
        match &self.first_name {
            Some(name) if self.confidence >= threshold => Some(name),
            _ => None,
        }
    }
}

/// Extract a probable first name and confidence from a display name.
///
/// # Examples
///
/// ```
/// let e = bonjour::extract("Quentin Richert");
/// assert_eq!(e.first_name.as_deref(), Some("Quentin"));
///
/// let e = bonjour::extract("ACME Corporation");
/// assert_eq!(e.first_name, None);
/// assert_eq!(e.confidence, 0.0);
/// ```
#[must_use]
pub fn extract(name: &str) -> Extraction {
    let table = first_names();

    // Organization evidence applies to the whole name, once (not per token).
    let looks_like_org = name.contains('&')
        || name.split_whitespace().any(|token| {
            let normalized = strip_surrounding_punctuation(token).to_lowercase();
            ORG_MARKERS.contains(&normalized.as_str())
        });
    let org_multiplier = if looks_like_org { ORG_MULTIPLIER } else { 1.0 };

    // Score each token by its first-name weight and keep the strongest. A
    // hyphenated token is looked up whole — `jean-pierre` is a single row.
    let best = name
        .split_whitespace()
        .filter_map(|token| table.get(&token.to_lowercase()).map(|&w| (token, w)))
        .max_by(|a, b| a.1.total_cmp(&b.1));

    match best {
        Some((token, weight)) => Extraction {
            input: name.to_string(),
            first_name: Some(token.to_string()),
            confidence: (weight * org_multiplier).clamp(0.0, 1.0),
        },
        None => Extraction {
            input: name.to_string(),
            first_name: None,
            confidence: 0.0,
        },
    }
}

/// Parse the embedded CSV once into a `name -> weight` map.
fn first_names() -> &'static HashMap<String, f64> {
    static TABLE: OnceLock<HashMap<String, f64>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = HashMap::new();
        for line in FIRST_NAMES_CSV.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue; // blank or comment line
            }
            let Some((name, weight)) = line.split_once(',') else {
                continue;
            };
            // The header row's `weight` fails to parse and is skipped here too.
            let Ok(weight) = weight.trim().parse::<f64>() else {
                continue;
            };
            table.insert(name.trim().to_lowercase(), weight);
        }
        table
    })
}

/// Strip surrounding non-alphanumerics (for org-marker matching), keeping inner
/// characters such as the hyphen in `jean-pierre`.
fn strip_surrounding_punctuation(token: &str) -> &str {
    token.trim_matches(|c: char| !c.is_alphanumeric())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_first_last() {
        let e = extract("Quentin Richert");
        assert_eq!(e.first_name.as_deref(), Some("Quentin"));
        assert!(e.confidence > 0.9, "confidence was {}", e.confidence);
    }

    #[test]
    fn order_independent() {
        // The family-name-first ordering must still find the first name.
        assert_eq!(name_of("Richert Quentin").as_deref(), Some("Quentin"));
    }

    #[test]
    fn compound_name_is_atomic() {
        assert_eq!(
            name_of("Jean-Pierre Dupont").as_deref(),
            Some("Jean-Pierre")
        );
    }

    #[test]
    fn frequency_beats_ambiguous_surname() {
        // Both "Jean" and "Martin" are in the table; the more common wins.
        assert_eq!(name_of("Jean Martin").as_deref(), Some("Jean"));
    }

    #[test]
    fn accents_match_and_output_is_preserved() {
        assert_eq!(name_of("Éric Tabarly").as_deref(), Some("Éric"));
    }

    #[test]
    fn org_marker_crushes_confidence() {
        let e = extract("Quentin Richert SAS");
        assert_eq!(e.first_name.as_deref(), Some("Quentin"));
        assert!(e.confidence < 0.2, "confidence was {}", e.confidence);
        assert!(e.greeting_name(DEFAULT_THRESHOLD).is_none());
    }

    #[test]
    fn ampersand_is_org_evidence() {
        let e = extract("Martin & Fils");
        assert!(
            e.confidence < DEFAULT_THRESHOLD,
            "confidence was {}",
            e.confidence
        );
        assert!(e.greeting_name(DEFAULT_THRESHOLD).is_none());
    }

    #[test]
    fn no_candidate_yields_null() {
        for input in [
            "Les Motards d'Alsace",
            "ACME Corporation",
            "Club de Tennis Strasbourg",
        ] {
            let e = extract(input);
            assert_eq!(e.first_name, None, "input: {input}");
            assert!(e.confidence.abs() < f64::EPSILON, "input: {input}");
        }
    }

    #[test]
    fn greeting_name_respects_threshold() {
        let e = extract("Quentin Richert");
        assert_eq!(e.greeting_name(DEFAULT_THRESHOLD), Some("Quentin"));
        assert_eq!(e.greeting_name(0.99), None); // 0.95 < 0.99
    }

    #[test]
    fn empty_input_is_null() {
        let e = extract("   ");
        assert_eq!(e.first_name, None);
        assert!(e.confidence.abs() < f64::EPSILON);
    }

    fn name_of(input: &str) -> Option<String> {
        extract(input).first_name
    }
}
