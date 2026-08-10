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
use unicode_normalization::{UnicodeNormalization, char::is_combining_mark};

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

// TODO: Check out <https://github.com/postgres/postgres/blob/master/contrib/unaccent/unaccent.rules>
// TODO: Not sure it's a good idea to do that, removing accents could conflict with words.
trait Unaccent {
    fn unaccent(&self) -> String;
}

impl<T: AsRef<str>> Unaccent for T {
    fn unaccent(&self) -> String {
        self.as_ref()
            // TODO: Try `nfkd`?
            .nfd()
            .filter(|c| !is_combining_mark(*c))
            .nfc()
            .collect()
    }
}

/// Normalize canonically equivalent spellings (for example, precomposed `é`
/// and `e` plus a combining acute accent) while preserving accents.
fn normalize(s: &str) -> String {
    s.to_lowercase().nfc().collect()
}

/// Binary gender label for a name in a given country.
///
/// Gender is country-dependent (`Simone` is `Female` in France, `Male` in
/// Italy), so it belongs to a `(name, country)` row, never to a name alone.
/// This is an output/normalization type; the `extract` hints are plain strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Gender {
    Female,
    Male,
}

impl Gender {
    /// Parse a gender hint or dataset cell, leniently: `f`/`female` and
    /// `m`/`male` in any case. Anything else is `None`, so an unrecognized hint
    /// is simply treated as absent.
    fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "f" | "female" => Some(Self::Female),
            "m" | "male" => Some(Self::Male),
            _ => None,
        }
    }
}

/// One `(country, gender, weight)` observation for a name. A name maps to a
/// `Vec` of these; several rows are how country-dependent gender and popularity
/// are expressed.
#[derive(Debug, Clone)]
struct NameEntry {
    /// ISO 3166-1 alpha-2 country code, uppercase.
    country: String,
    gender: Gender,
    /// First-name likelihood in `0.0..=1.0` for this country.
    weight: f64,
}

/// Exact name rows plus unambiguous accent-folded aliases.
struct NameTable {
    exact: HashMap<String, Vec<NameEntry>>,
    unaccented: HashMap<String, Option<String>>,
}

impl NameTable {
    /// Look up an exact normalized spelling first, then an unambiguous
    /// accent-folded alias.
    fn get(&self, name: &str) -> Option<&[NameEntry]> {
        let normalized = normalize(name);
        if let Some(entries) = self.exact.get(&normalized) {
            return Some(entries);
        }

        let canonical = self.unaccented.get(&normalized.unaccent())?.as_deref()?;
        self.exact.get(canonical).map(Vec::as_slice)
    }

    /// Parse CSV rows while retaining their canonical accented spelling.
    fn from_csv(csv: &str) -> Self {
        let mut exact: HashMap<String, Vec<NameEntry>> = HashMap::new();
        let mut unaccented: HashMap<String, Option<String>> = HashMap::new();

        for line in csv.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue; // blank or comment line
            }
            // Exactly four fields, or skip the row.
            let mut fields = line.split(',');
            let (Some(name), Some(country), Some(gender), Some(weight), None) = (
                fields.next(),
                fields.next(),
                fields.next(),
                fields.next(),
                fields.next(),
            ) else {
                continue;
            };
            // The header row's non-numeric `weight`/`gender` fail to parse and
            // are skipped here too.
            let Ok(weight) = weight.trim().parse::<f64>() else {
                continue;
            };
            let Some(gender) = Gender::parse(gender) else {
                continue;
            };

            let name = normalize(name.trim());
            unaccented
                .entry(name.unaccent())
                .and_modify(|canonical| {
                    if canonical.as_deref() != Some(name.as_str()) {
                        *canonical = None;
                    }
                })
                .or_insert_with(|| Some(name.clone()));
            exact.entry(name).or_default().push(NameEntry {
                country: country.trim().to_uppercase(),
                gender,
                weight,
            });
        }

        Self { exact, unaccented }
    }
}

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
    /// Resolved gender, reported only when the surviving candidate rows agree
    /// on one (a country/gender hint can force agreement). `None` when the name
    /// is unknown, or when rows disagree and nothing disambiguates — the caller
    /// then greets neutrally rather than guessing.
    pub gender: Option<Gender>,
    /// Country of the resolved row: the hinted one, else the highest-weight one.
    /// `None` only when the name is unknown.
    pub country: Option<String>,
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

/// Extract a probable first name from a display name, with confidence, gender
/// and country.
///
/// `country` and `gender` are optional hints — typically the user's locale and
/// profile gender. They act as *symmetric filters*: a country pins down gender
/// (`Simone` + `IT` → male), a gender pins down country (`Simone` + `M` → `IT`).
/// A hint that matches no row for the name is ignored (a hint never rejects a
/// name we know). With no hint and rows that disagree on gender, `gender` is
/// left `None` and `country` reports the highest-weight row.
///
/// # Examples
///
/// ```
/// use bonjour::Gender;
///
/// let e = bonjour::extract("Quentin Richert", None, None);
/// assert_eq!(e.first_name.as_deref(), Some("Quentin"));
/// assert_eq!(e.gender, Some(Gender::Male));
/// assert_eq!(e.country.as_deref(), Some("FR"));
///
/// // Country resolves the gender of an otherwise-ambiguous name.
/// assert_eq!(bonjour::extract("Simone", Some("IT"), None).gender, Some(Gender::Male));
/// assert_eq!(bonjour::extract("Simone", Some("FR"), None).gender, Some(Gender::Female));
///
/// let e = bonjour::extract("ACME Corporation", None, None);
/// assert_eq!(e.first_name, None);
/// assert_eq!(e.confidence, 0.0);
/// ```
#[must_use]
pub fn extract(input: &str, country: Option<&str>, gender: Option<&str>) -> Extraction {
    let table = first_names();

    // Organization evidence applies to the whole name, once (not per token).
    let looks_like_org = input.contains('&')
        || input.split_whitespace().any(|token| {
            let normalized = strip_surrounding_punctuation(token).to_lowercase();
            ORG_MARKERS.contains(&normalized.as_str())
        });
    let org_multiplier = if looks_like_org { ORG_MULTIPLIER } else { 1.0 };

    // Normalize hints once. Country matches case-insensitively via uppercase; an
    // unparsable gender hint is dropped (treated as absent).
    let country_hint = country.map(|c| c.trim().to_uppercase());
    let gender_hint = gender.and_then(Gender::parse);

    // Resolve each matching token under the hints, then keep the token whose
    // resolved weight is highest. A hyphenated token is looked up whole —
    // `jean-pierre` is a single row.
    let best = input
        .split_whitespace()
        .filter_map(|token| {
            let entries = table.get(token)?;
            Some((
                token,
                resolve(entries, country_hint.as_deref(), gender_hint)?,
            ))
        })
        .max_by(|a, b| a.1.weight.total_cmp(&b.1.weight));

    match best {
        Some((token, resolution)) => Extraction {
            input: input.to_string(),
            first_name: Some(token.to_string()),
            confidence: (resolution.weight * org_multiplier).clamp(0.0, 1.0),
            gender: resolution.gender,
            country: Some(resolution.country),
        },
        None => Extraction {
            input: input.to_string(),
            first_name: None,
            confidence: 0.0,
            gender: None,
            country: None,
        },
    }
}

/// A name's rows collapsed to a single answer under the active hints.
struct Resolution {
    country: String,
    gender: Option<Gender>,
    weight: f64,
}

/// Resolve a name's rows under optional country/gender hints.
///
/// Hints are equality filters: keep rows matching every hint present. If that
/// leaves nothing (the hint doesn't apply to this name), fall back to all rows —
/// a hint must never reject a name we know. From the survivors, the
/// highest-weight row sets `country` and `weight`; `gender` is reported only
/// when every survivor agrees on it.
fn resolve(
    entries: &[NameEntry],
    country: Option<&str>,
    gender: Option<Gender>,
) -> Option<Resolution> {
    let matches = |e: &&NameEntry| {
        country.is_none_or(|c| e.country == c) && gender.is_none_or(|g| e.gender == g)
    };

    let mut candidates: Vec<&NameEntry> = entries.iter().filter(matches).collect();
    if candidates.is_empty() {
        candidates = entries.iter().collect();
    }

    let best = *candidates
        .iter()
        .max_by(|a, b| a.weight.total_cmp(&b.weight))?;

    let first = candidates[0].gender;
    let gender = candidates
        .iter()
        .all(|e| e.gender == first)
        .then_some(first);

    Some(Resolution {
        country: best.country.clone(),
        gender,
        weight: best.weight,
    })
}

/// Parse the embedded CSV once into exact rows and unaccented aliases.
fn first_names() -> &'static NameTable {
    static TABLE: OnceLock<NameTable> = OnceLock::new();
    TABLE.get_or_init(|| NameTable::from_csv(FIRST_NAMES_CSV))
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
        let e = extract("Quentin Richert", None, None);
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
        assert_eq!(name_of("Eric Tabarly").as_deref(), Some("Eric"));
    }

    #[test]
    fn unaccented_fallback_requires_an_unambiguous_name() {
        let table = NameTable::from_csv("rene,GB,M,0.60\nrené,FR,M,0.70");

        assert!((table.get("Rene").unwrap()[0].weight - 0.60).abs() < f64::EPSILON);
        assert!((table.get("René").unwrap()[0].weight - 0.70).abs() < f64::EPSILON);
        assert!(table.get("Rène").is_none());
    }

    #[test]
    fn unambiguous_name_reports_gender_and_country() {
        let e = extract("Quentin Richert", None, None);
        assert_eq!(e.gender, Some(Gender::Male));
        assert_eq!(e.country.as_deref(), Some("FR"));
    }

    #[test]
    fn ambiguous_gender_without_hint_is_none() {
        // "Simone" is FR=F, IT=M — no signal to choose, so no gender is claimed.
        let e = extract("Simone", None, None);
        assert_eq!(e.first_name.as_deref(), Some("Simone"));
        assert_eq!(e.gender, None);
        assert_eq!(e.country.as_deref(), Some("FR")); // best weight: 0.70 > 0.65
    }

    #[test]
    fn country_hint_resolves_gender() {
        let it = extract("Simone", Some("IT"), None);
        assert_eq!(it.gender, Some(Gender::Male));
        assert_eq!(it.country.as_deref(), Some("IT"));
        assert!((it.confidence - 0.65).abs() < 1e-9, "was {}", it.confidence);

        let fr = extract("Simone", Some("fr"), None); // case-insensitive
        assert_eq!(fr.gender, Some(Gender::Female));
        assert_eq!(fr.country.as_deref(), Some("FR"));
    }

    #[test]
    fn gender_hint_resolves_country() {
        // Symmetric: knowing the gender pins the country for a unisex name.
        let e = extract("Simone", None, Some("M"));
        assert_eq!(e.gender, Some(Gender::Male));
        assert_eq!(e.country.as_deref(), Some("IT"));
    }

    #[test]
    fn gender_hint_accepts_long_form_and_ignores_garbage() {
        assert_eq!(
            extract("Simone", None, Some("female")).country.as_deref(),
            Some("FR")
        );
        // Unparsable hint → treated as absent → still ambiguous.
        assert_eq!(extract("Simone", None, Some("bogus")).gender, None);
    }

    #[test]
    fn hint_that_matches_nothing_is_ignored() {
        // No IT row for "Quentin" — a hint must not reject a name we know.
        let e = extract("Quentin", Some("IT"), None);
        assert_eq!(e.first_name.as_deref(), Some("Quentin"));
        assert_eq!(e.gender, Some(Gender::Male));
        assert_eq!(e.country.as_deref(), Some("FR"));
    }

    #[test]
    fn org_marker_crushes_confidence() {
        let e = extract("Quentin Richert SAS", None, None);
        assert_eq!(e.first_name.as_deref(), Some("Quentin"));
        assert!(e.confidence < 0.2, "confidence was {}", e.confidence);
        assert!(e.greeting_name(DEFAULT_THRESHOLD).is_none());
    }

    #[test]
    fn ampersand_is_org_evidence() {
        let e = extract("Martin & Fils", None, None);
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
            let e = extract(input, None, None);
            assert_eq!(e.first_name, None, "input: {input}");
            assert_eq!(e.gender, None, "input: {input}");
            assert_eq!(e.country, None, "input: {input}");
            assert!(e.confidence.abs() < f64::EPSILON, "input: {input}");
        }
    }

    #[test]
    fn greeting_name_respects_threshold() {
        let e = extract("Quentin Richert", None, None);
        assert_eq!(e.greeting_name(DEFAULT_THRESHOLD), Some("Quentin"));
        assert_eq!(e.greeting_name(0.99), None); // 0.95 < 0.99
    }

    #[test]
    fn empty_input_is_null() {
        let e = extract("   ", None, None);
        assert_eq!(e.first_name, None);
        assert!(e.confidence.abs() < f64::EPSILON);
    }

    fn name_of(input: &str) -> Option<String> {
        extract(input, None, None).first_name
    }
}
