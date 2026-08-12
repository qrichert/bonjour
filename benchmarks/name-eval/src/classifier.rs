use std::collections::HashSet;

use unicode_normalization::{UnicodeNormalization, char::is_combining_mark};

use crate::artifact::{Evidence, EvidenceSource, GenderHint};
use crate::lexical::candidate_is_eligible;

const STRONG_ORGANIZATION_MARKERS: &[&str] = &[
    "gmbh",
    "llc",
    "ltd",
    "limited",
    "inc",
    "incorporated",
    "sarl",
    "eurl",
    "sasu",
    "plc",
    "sas",
    "corp",
    "corporation",
];
const C_ADDITIONAL_STRONG_ORGANIZATION_MARKERS: &[&str] = &["bv"];
const GENERIC_ORGANIZATION_MARKERS: &[&str] = &[
    "association",
    "atelier",
    "cafe",
    "club",
    "consulting",
    "fils",
    "foundation",
    "garage",
    "group",
    "groupe",
    "hotel",
    "kebab",
    "market",
    "restaurant",
    "services",
    "shop",
    "studio",
];

#[derive(Clone, Copy, Debug)]
pub struct AlgorithmConfig {
    pub name: &'static str,
    pub kind: AlgorithmKind,
    pub frequency_floor: f64,
    pub frequency_weight: f64,
    pub country_weight: f64,
    pub first_position_bonus: f64,
    pub last_position_bonus: f64,
    pub multi_token_bonus: f64,
    pub single_display_bonus: f64,
    pub competition_penalty: f64,
    pub strong_organization_multiplier: f64,
    pub generic_organization_multiplier: f64,
    pub gender_threshold: f64,
    pub role_score_floor: f64,
    pub role_weight: f64,
    pub role_center: f64,
    pub role_scale: f64,
    pub role_smoothing: f64,
    pub role_reliability_weight: f64,
    pub compound_evidence_weight: f64,
    pub remainder_role_weight: f64,
    pub hard_legal_abstention: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AlgorithmKind {
    Legacy,
    RoleHypothesis,
}

pub const ALGORITHM_A: AlgorithmConfig = AlgorithmConfig {
    name: "A-frequency-v1",
    kind: AlgorithmKind::Legacy,
    frequency_floor: 0.25,
    frequency_weight: 0.70,
    country_weight: 0.03,
    first_position_bonus: 0.0,
    last_position_bonus: 0.0,
    multi_token_bonus: 0.04,
    single_display_bonus: 0.02,
    competition_penalty: 0.04,
    strong_organization_multiplier: 0.05,
    generic_organization_multiplier: 0.40,
    gender_threshold: 0.80,
    role_score_floor: 0.0,
    role_weight: 0.0,
    role_center: 0.0,
    role_scale: 1.0,
    role_smoothing: 0.5,
    role_reliability_weight: 0.0,
    compound_evidence_weight: 0.0,
    remainder_role_weight: 0.0,
    hard_legal_abstention: false,
};

pub const ALGORITHM_B: AlgorithmConfig = AlgorithmConfig {
    name: "B-simple-signals-v1",
    kind: AlgorithmKind::Legacy,
    frequency_floor: 0.25,
    frequency_weight: 0.70,
    country_weight: 0.08,
    first_position_bonus: 0.04,
    last_position_bonus: 0.015,
    multi_token_bonus: 0.08,
    single_display_bonus: 0.05,
    competition_penalty: 0.12,
    strong_organization_multiplier: 0.02,
    generic_organization_multiplier: 0.12,
    gender_threshold: 0.80,
    role_score_floor: 0.0,
    role_weight: 0.0,
    role_center: 0.0,
    role_scale: 1.0,
    role_smoothing: 0.5,
    role_reliability_weight: 0.0,
    compound_evidence_weight: 0.0,
    remainder_role_weight: 0.0,
    hard_legal_abstention: false,
};

pub const ALGORITHM_C: AlgorithmConfig = AlgorithmConfig {
    name: "C-global-role-v1",
    kind: AlgorithmKind::RoleHypothesis,
    frequency_floor: 0.0,
    frequency_weight: 0.0,
    country_weight: 0.08,
    first_position_bonus: 0.0,
    last_position_bonus: 0.0,
    multi_token_bonus: 0.0,
    single_display_bonus: 0.04,
    competition_penalty: 0.08,
    strong_organization_multiplier: 0.0,
    generic_organization_multiplier: 0.12,
    gender_threshold: 0.80,
    role_score_floor: 0.28,
    role_weight: 0.56,
    role_center: 1.0,
    role_scale: 1.4,
    role_smoothing: 0.5,
    role_reliability_weight: 0.10,
    compound_evidence_weight: 0.18,
    remainder_role_weight: 0.12,
    hard_legal_abstention: true,
};

#[derive(Clone, Debug)]
pub struct RawInference {
    pub greeting_candidate: Option<String>,
    pub confidence: f64,
    pub gender_hint: Option<GenderHint>,
    pub gender_confidence: f64,
}

impl RawInference {
    pub fn greeting_at(&self, threshold: f64) -> Option<&str> {
        self.greeting_candidate
            .as_deref()
            .filter(|_| self.confidence >= threshold)
    }

    pub fn gender_at(&self, threshold: f64) -> Option<GenderHint> {
        self.greeting_at(threshold).and(self.gender_hint)
    }
}

#[derive(Clone)]
struct Candidate {
    display: String,
    score: f64,
    evidence: Evidence,
}

#[derive(Clone, Debug)]
struct RoleCandidate {
    display: String,
    start: usize,
    length: usize,
    score: f64,
    role_llr: f64,
    role_signal: f64,
    reliability: f64,
    country_support: f64,
    compound_evidence: f64,
    remainder_evidence: f64,
    evidence: Evidence,
}

#[derive(Clone, Debug)]
pub struct CandidateDiagnostic {
    pub display: String,
    pub start: usize,
    pub length: usize,
    pub global_given_count: u64,
    pub country_given_count: u64,
    pub global_surname_count: u64,
    pub role_llr: f64,
    pub role_signal: f64,
    pub reliability: f64,
    pub country_support: f64,
    pub compound_evidence: f64,
    pub remainder_evidence: f64,
    pub score: f64,
    pub algorithm_a_score: f64,
    pub algorithm_b_score: f64,
}

pub fn infer_prethreshold(
    corpus: &impl EvidenceSource,
    config: AlgorithmConfig,
    display_name: &str,
    country_hint: Option<&str>,
    locale_hint: Option<&str>,
) -> RawInference {
    if config.kind == AlgorithmKind::RoleHypothesis {
        return infer_role_prethreshold(corpus, config, display_name, country_hint, locale_hint);
    }
    infer_legacy_prethreshold(corpus, config, display_name, country_hint, locale_hint)
}

fn infer_legacy_prethreshold(
    corpus: &impl EvidenceSource,
    config: AlgorithmConfig,
    display_name: &str,
    country_hint: Option<&str>,
    locale_hint: Option<&str>,
) -> RawInference {
    let country = resolve_country(country_hint, locale_hint);
    let tokens = tokenize(display_name);
    let mut candidates = Vec::<Candidate>::new();

    for start in 0..tokens.len() {
        for length in 1..=2.min(tokens.len() - start) {
            let display = tokens[start..start + length].join(" ");
            let Some(evidence) = lookup_with_variants(corpus, &display, country) else {
                continue;
            };
            let score = legacy_candidate_score(evidence, start, length, tokens.len(), config);
            candidates.push(Candidate {
                display,
                score,
                evidence,
            });
        }
    }

    candidates.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| {
                right
                    .display
                    .chars()
                    .count()
                    .cmp(&left.display.chars().count())
            })
            .then_with(|| left.display.cmp(&right.display))
    });
    let Some(best) = candidates.first() else {
        return RawInference {
            greeting_candidate: None,
            confidence: 0.0,
            gender_hint: None,
            gender_confidence: 0.0,
        };
    };

    let mut confidence = best.score;
    if let Some(second) = candidates.get(1) {
        let margin = (best.score - second.score).clamp(0.0, 1.0);
        confidence -= config.competition_penalty * (1.0 - margin);
    }
    let organization_multiplier = organization_multiplier(display_name, config);
    confidence = (confidence * organization_multiplier).clamp(0.0, 1.0);

    let gender_total = best.evidence.effective_count;
    let (gender_hint, gender_confidence) = if gender_total == 0 {
        (None, 0.0)
    } else {
        let (gender, count) = if best.evidence.female_count > best.evidence.male_count {
            (GenderHint::Female, best.evidence.female_count)
        } else {
            (GenderHint::Male, best.evidence.male_count)
        };
        let gender_confidence = count as f64 / gender_total as f64;
        (
            (gender_confidence >= config.gender_threshold).then_some(gender),
            gender_confidence,
        )
    };

    RawInference {
        greeting_candidate: Some(best.display.clone()),
        confidence,
        gender_hint,
        gender_confidence,
    }
}

fn infer_role_prethreshold(
    corpus: &impl EvidenceSource,
    config: AlgorithmConfig,
    display_name: &str,
    country_hint: Option<&str>,
    locale_hint: Option<&str>,
) -> RawInference {
    if config.hard_legal_abstention && has_strong_organization_marker(display_name) {
        return empty_inference();
    }
    let mut candidates = role_candidates(corpus, config, display_name, country_hint, locale_hint);
    candidates.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| right.length.cmp(&left.length))
            .then_with(|| left.start.cmp(&right.start))
            .then_with(|| left.display.cmp(&right.display))
    });
    let Some(best) = candidates.first() else {
        return empty_inference();
    };

    let mut confidence = best.score;
    if let Some(second) = candidates.get(1) {
        let margin = (best.score - second.score).clamp(0.0, 1.0);
        confidence -= config.competition_penalty * (1.0 - margin);
    }
    if has_generic_organization_marker(display_name) || display_name.contains('&') {
        confidence *= config.generic_organization_multiplier;
    }
    let (gender_hint, gender_confidence) = gender_inference(best.evidence, config);
    RawInference {
        greeting_candidate: Some(best.display.clone()),
        confidence: confidence.clamp(0.0, 1.0),
        gender_hint,
        gender_confidence,
    }
}

fn role_candidates(
    corpus: &impl EvidenceSource,
    config: AlgorithmConfig,
    display_name: &str,
    country_hint: Option<&str>,
    locale_hint: Option<&str>,
) -> Vec<RoleCandidate> {
    let country = resolve_country(country_hint, locale_hint);
    let tokens = tokenize(display_name);
    let mut candidates = Vec::<RoleCandidate>::new();
    for start in 0..tokens.len() {
        for length in 1..=2.min(tokens.len() - start) {
            let display = tokens[start..start + length].join(" ");
            let Some(evidence) = lookup_with_variants(corpus, &display, country) else {
                continue;
            };
            let role_llr = role_llr(evidence, config.role_smoothing);
            let role_signal = logistic((role_llr - config.role_center) / config.role_scale);
            let reliability = count_reliability(evidence.global_count);
            let country_support = country_support(evidence);
            let mut score = config.role_score_floor
                + config.role_weight * role_signal
                + config.role_reliability_weight * reliability
                + config.country_weight * country_support;
            if tokens.len() == 1 {
                score += config.single_display_bonus;
            }
            candidates.push(RoleCandidate {
                display,
                start,
                length,
                score,
                role_llr,
                role_signal,
                reliability,
                country_support,
                compound_evidence: 0.0,
                remainder_evidence: 0.0,
                evidence,
            });
        }
    }

    for index in 0..candidates.len() {
        if candidates[index].length == 2 {
            let component_max = candidates
                .iter()
                .filter(|candidate| {
                    candidate.length == 1
                        && candidate.start >= candidates[index].start
                        && candidate.start < candidates[index].start + candidates[index].length
                })
                .map(|candidate| candidate.evidence.global_count)
                .max();
            let direct = candidates[index].evidence.global_count;
            let comparison = component_max.map_or(1.0, |component| {
                ((direct as f64 + 1.0) / (direct as f64 + component as f64 + 2.0)).sqrt()
            });
            candidates[index].compound_evidence = comparison * candidates[index].role_signal;
            candidates[index].score +=
                config.compound_evidence_weight * candidates[index].compound_evidence;
        }
    }

    for index in 0..candidates.len() {
        let strongest_disjoint = candidates
            .iter()
            .enumerate()
            .filter(|(other_index, other)| {
                *other_index != index
                    && spans_are_disjoint(
                        candidates[index].start,
                        candidates[index].length,
                        other.start,
                        other.length,
                    )
            })
            .map(|(_, candidate)| candidate.role_llr)
            .max_by(f64::total_cmp);
        if let Some(other_role) = strongest_disjoint {
            let margin = (candidates[index].role_llr - other_role) / config.role_scale;
            candidates[index].remainder_evidence = 2.0 * (logistic(margin) - 0.5);
            candidates[index].score +=
                config.remainder_role_weight * candidates[index].remainder_evidence;
        }
        candidates[index].score = candidates[index].score.clamp(0.0, 1.0);
    }
    candidates
}

pub fn candidate_diagnostics(
    corpus: &impl EvidenceSource,
    config: AlgorithmConfig,
    display_name: &str,
    country_hint: Option<&str>,
    locale_hint: Option<&str>,
) -> Vec<CandidateDiagnostic> {
    role_candidates(corpus, config, display_name, country_hint, locale_hint)
        .into_iter()
        .map(|candidate| CandidateDiagnostic {
            display: candidate.display,
            start: candidate.start,
            length: candidate.length,
            global_given_count: candidate.evidence.global_count,
            country_given_count: candidate.evidence.country_count,
            global_surname_count: candidate.evidence.surname_count,
            role_llr: candidate.role_llr,
            role_signal: candidate.role_signal,
            reliability: candidate.reliability,
            country_support: candidate.country_support,
            compound_evidence: candidate.compound_evidence,
            remainder_evidence: candidate.remainder_evidence,
            score: candidate.score,
            algorithm_a_score: legacy_candidate_score(
                candidate.evidence,
                candidate.start,
                candidate.length,
                tokenize(display_name).len(),
                ALGORITHM_A,
            ),
            algorithm_b_score: legacy_candidate_score(
                candidate.evidence,
                candidate.start,
                candidate.length,
                tokenize(display_name).len(),
                ALGORITHM_B,
            ),
        })
        .collect()
}

fn legacy_candidate_score(
    evidence: Evidence,
    start: usize,
    length: usize,
    token_count: usize,
    config: AlgorithmConfig,
) -> f64 {
    let mut score = frequency_score(evidence.global_count, config);
    if evidence.country_count != 0 && evidence.global_count != 0 {
        score += config.country_weight
            * (evidence.country_count as f64 / evidence.global_count as f64).sqrt();
    }
    if start == 0 {
        score += config.first_position_bonus;
    }
    if start + length == token_count {
        score += config.last_position_bonus;
    }
    if length > 1 {
        score += config.multi_token_bonus;
    }
    if token_count == 1 {
        score += config.single_display_bonus;
    }
    score.clamp(0.0, 1.0)
}

pub fn role_llr(evidence: Evidence, smoothing: f64) -> f64 {
    debug_assert!(smoothing > 0.0);
    ((evidence.global_count as f64 + smoothing) / evidence.given_total as f64).ln()
        - ((evidence.surname_count as f64 + smoothing) / evidence.surname_total as f64).ln()
}

fn count_reliability(count: u64) -> f64 {
    const MINIMUM: f64 = 5.0;
    const SATURATION: f64 = 1_500_000.0;
    (((count as f64).ln() - MINIMUM.ln()) / (SATURATION.ln() - MINIMUM.ln())).clamp(0.0, 1.0)
}

fn country_support(evidence: Evidence) -> f64 {
    if evidence.country_count == 0 || evidence.global_count == 0 {
        0.0
    } else {
        ((evidence.country_count as f64 + 1.0).ln() / (evidence.global_count as f64 + 1.0).ln())
            .clamp(0.0, 1.0)
    }
}

fn logistic(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

fn spans_are_disjoint(
    left_start: usize,
    left_length: usize,
    right_start: usize,
    right_length: usize,
) -> bool {
    left_start + left_length <= right_start || right_start + right_length <= left_start
}

fn empty_inference() -> RawInference {
    RawInference {
        greeting_candidate: None,
        confidence: 0.0,
        gender_hint: None,
        gender_confidence: 0.0,
    }
}

fn gender_inference(evidence: Evidence, config: AlgorithmConfig) -> (Option<GenderHint>, f64) {
    let gender_total = evidence.effective_count;
    if gender_total == 0 {
        return (None, 0.0);
    }
    let (gender, count) = if evidence.female_count > evidence.male_count {
        (GenderHint::Female, evidence.female_count)
    } else {
        (GenderHint::Male, evidence.male_count)
    };
    let confidence = count as f64 / gender_total as f64;
    (
        (confidence >= config.gender_threshold).then_some(gender),
        confidence,
    )
}

fn frequency_score(count: u64, config: AlgorithmConfig) -> f64 {
    const MINIMUM: f64 = 5.0;
    const SATURATION: f64 = 1_500_000.0;
    let normalized = ((count as f64).ln() - MINIMUM.ln()) / (SATURATION.ln() - MINIMUM.ln());
    config.frequency_floor + config.frequency_weight * normalized.clamp(0.0, 1.0)
}

fn lookup_with_variants(
    corpus: &impl EvidenceSource,
    display: &str,
    country: Option<[u8; 2]>,
) -> Option<Evidence> {
    if !candidate_is_eligible(display) {
        return None;
    }
    let canonical = canonicalize(display);
    let title = title_case(&canonical);
    let lowercase = canonical.to_lowercase();
    let mut variants = vec![canonical, title, lowercase];
    let unaccented = variants
        .iter()
        .map(|value| strip_accents(value))
        .collect::<Vec<_>>();
    variants.extend(unaccented);
    let mut seen = HashSet::new();
    variants
        .into_iter()
        .filter(|variant| seen.insert(variant.clone()))
        .find_map(|variant| corpus.lookup(&variant, country))
}

fn organization_multiplier(display_name: &str, config: AlgorithmConfig) -> f64 {
    if display_name.contains('&') {
        return config.strong_organization_multiplier;
    }
    let markers = canonicalize(display_name)
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_lowercase)
        .collect::<Vec<_>>();
    if markers
        .iter()
        .any(|marker| STRONG_ORGANIZATION_MARKERS.contains(&marker.as_str()))
    {
        config.strong_organization_multiplier
    } else if markers
        .iter()
        .any(|marker| GENERIC_ORGANIZATION_MARKERS.contains(&marker.as_str()))
    {
        config.generic_organization_multiplier
    } else {
        1.0
    }
}

fn organization_markers(display_name: &str) -> Vec<String> {
    canonicalize(display_name)
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_lowercase)
        .collect()
}

fn has_strong_organization_marker(display_name: &str) -> bool {
    organization_markers(display_name).iter().any(|marker| {
        STRONG_ORGANIZATION_MARKERS.contains(&marker.as_str())
            || C_ADDITIONAL_STRONG_ORGANIZATION_MARKERS.contains(&marker.as_str())
    })
}

fn has_generic_organization_marker(display_name: &str) -> bool {
    organization_markers(display_name)
        .iter()
        .any(|marker| GENERIC_ORGANIZATION_MARKERS.contains(&marker.as_str()))
}

fn tokenize(display_name: &str) -> Vec<String> {
    canonicalize(display_name)
        .split_whitespace()
        .map(str::to_string)
        .collect()
}

fn canonicalize(value: &str) -> String {
    value
        .nfc()
        .map(|character| match character {
            '‐' | '‑' | '‒' | '–' | '—' | '―' | '−' => '-',
            '‘' | '’' | '‛' | 'ʻ' | 'ʼ' | '＇' => '\'',
            _ => character,
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn title_case(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    let mut at_word_start = true;
    for character in value.chars() {
        if character.is_alphabetic() {
            if at_word_start {
                output.extend(character.to_uppercase());
                at_word_start = false;
            } else {
                output.extend(character.to_lowercase());
            }
        } else {
            output.push(character);
            at_word_start = character.is_whitespace() || matches!(character, '-' | '\'');
        }
    }
    output
}

fn strip_accents(value: &str) -> String {
    value
        .nfd()
        .filter(|character| !is_combining_mark(*character))
        .nfc()
        .collect()
}

fn resolve_country(country_hint: Option<&str>, locale_hint: Option<&str>) -> Option<[u8; 2]> {
    country_hint
        .and_then(parse_country)
        .or_else(|| locale_hint.and_then(country_from_locale))
}

fn parse_country(value: &str) -> Option<[u8; 2]> {
    let bytes = value.trim().as_bytes();
    (bytes.len() == 2 && bytes.iter().all(u8::is_ascii_alphabetic))
        .then(|| [bytes[0].to_ascii_uppercase(), bytes[1].to_ascii_uppercase()])
}

fn country_from_locale(locale: &str) -> Option<[u8; 2]> {
    locale.split(['-', '_']).rev().find_map(parse_country)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use proptest::prelude::*;

    use super::*;

    struct FakeCorpus(HashMap<String, Evidence>);

    impl FakeCorpus {
        fn names() -> Self {
            let evidence = Evidence {
                global_count: 50_000,
                country_count: 40_000,
                effective_count: 40_000,
                female_count: 0,
                male_count: 38_000,
                surname_count: 1_000,
                given_total: 444_154_759,
                surname_total: 489_631_377,
            };
            Self(
                ["Martin", "Élodie", "Jean-Pierre", "O'Connor"]
                    .into_iter()
                    .map(|name| (name.to_string(), evidence))
                    .collect(),
            )
        }
    }

    impl EvidenceSource for FakeCorpus {
        fn lookup(&self, name: &str, _country_hint: Option<[u8; 2]>) -> Option<Evidence> {
            self.0.get(name).copied()
        }
    }

    fn infer(input: &str) -> RawInference {
        infer_prethreshold(&FakeCorpus::names(), ALGORITHM_B, input, Some("FR"), None)
    }

    fn role_evidence(given: u64, surname: u64) -> Evidence {
        Evidence {
            global_count: given,
            country_count: 0,
            effective_count: given,
            female_count: 0,
            male_count: given,
            surname_count: surname,
            given_total: 444_154_759,
            surname_total: 489_631_377,
        }
    }

    proptest! {
        #[test]
        fn whitespace_normalization_preserves_inference(left in 1_usize..8, middle in 1_usize..8, right in 1_usize..8) {
            let input = format!("{}Martin{}Dupont{}", " ".repeat(left), " ".repeat(middle), " ".repeat(right));
            let baseline = infer("Martin Dupont");
            let transformed = infer(&input);
            prop_assert_eq!(transformed.greeting_candidate, baseline.greeting_candidate);
            prop_assert!((transformed.confidence - baseline.confidence).abs() < f64::EPSILON);
        }

        #[test]
        fn strong_organization_evidence_is_monotonic(marker in prop::sample::select(STRONG_ORGANIZATION_MARKERS)) {
            let person = infer("Martin");
            let organization = infer(&format!("Martin {marker}"));
            prop_assert!(organization.confidence <= person.confidence);
        }

        #[test]
        fn algorithm_c_hard_abstains_on_legal_markers(marker in prop::sample::select(STRONG_ORGANIZATION_MARKERS)) {
            let organization = infer_prethreshold(
                &FakeCorpus::names(),
                ALGORITHM_C,
                &format!("Martin {marker}"),
                Some("FR"),
                None,
            );
            prop_assert_eq!(organization.greeting_candidate, None);
            prop_assert!(organization.confidence.abs() < f64::EPSILON);
        }
    }

    #[test]
    fn nfc_and_nfd_are_equivalent() {
        let nfc = infer("Élodie Dupont");
        let nfd = infer(&"Élodie Dupont".nfd().collect::<String>());
        assert_eq!(nfc.greeting_candidate, nfd.greeting_candidate);
        assert!((nfc.confidence - nfd.confidence).abs() < f64::EPSILON);
    }

    #[test]
    fn casing_preserves_candidate_and_confidence() {
        let title = infer("Martin Dupont");
        for transformed in ["martin dupont", "MARTIN DUPONT"] {
            let inference = infer(transformed);
            assert_eq!(
                inference
                    .greeting_candidate
                    .as_deref()
                    .map(str::to_lowercase),
                Some("martin".to_string())
            );
            assert!((inference.confidence - title.confidence).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn equivalent_punctuation_is_canonicalized() {
        let hyphen = infer("Jean-Pierre Dupont");
        let unicode_hyphen = infer("Jean‑Pierre Dupont");
        assert_eq!(hyphen.greeting_candidate, unicode_hyphen.greeting_candidate);
        assert!((hyphen.confidence - unicode_hyphen.confidence).abs() < f64::EPSILON);

        let apostrophe = infer("O'Connor Dupont");
        let curly = infer("O’Connor Dupont");
        assert_eq!(apostrophe.greeting_candidate, curly.greeting_candidate);
        assert!((apostrophe.confidence - curly.confidence).abs() < f64::EPSILON);
    }

    #[test]
    fn lexical_ineligibility_blocks_candidate_lookup() {
        let evidence = Evidence {
            global_count: 1_000_000,
            country_count: 1_000_000,
            effective_count: 1_000_000,
            female_count: 0,
            male_count: 1_000_000,
            surname_count: 0,
            given_total: 444_154_759,
            surname_total: 489_631_377,
        };
        let corpus = FakeCorpus(
            ["A/J*C", "A[[Ine", "A_Kim", "A`S"]
                .into_iter()
                .map(|name| (name.to_string(), evidence))
                .collect(),
        );
        for input in ["A/J*C", "A[[Ine", "A_Kim", "A`S"] {
            let inference = infer_prethreshold(&corpus, ALGORITHM_B, input, None, None);
            assert_eq!(inference.greeting_candidate, None, "{input:?}");
            assert!(inference.confidence.abs() < f64::EPSILON, "{input:?}");
        }
    }

    #[test]
    fn role_evidence_beats_order_and_raw_frequency() {
        let corpus = FakeCorpus(HashMap::from([
            ("Jean".to_string(), role_evidence(58_545, 9_796)),
            ("Martin".to_string(), role_evidence(14_544, 46_409)),
        ]));
        for input in ["Jean Martin", "Martin Jean"] {
            let inference = infer_prethreshold(&corpus, ALGORITHM_C, input, None, None);
            assert_eq!(
                inference.greeting_candidate.as_deref(),
                Some("Jean"),
                "{input}"
            );
        }
    }

    #[test]
    fn direct_compound_evidence_can_beat_components() {
        let corpus = FakeCorpus(HashMap::from([
            ("Anne".to_string(), role_evidence(10_000, 100)),
            ("Marie".to_string(), role_evidence(20_000, 500)),
            ("Anne Marie".to_string(), role_evidence(12_000, 50)),
        ]));
        let inference = infer_prethreshold(&corpus, ALGORITHM_C, "Anne Marie Dupont", None, None);
        assert_eq!(inference.greeting_candidate.as_deref(), Some("Anne Marie"));
    }

    #[test]
    fn algorithm_c_only_legal_marker_does_not_change_legacy_algorithms() {
        let corpus = FakeCorpus::names();
        let legacy = infer_prethreshold(&corpus, ALGORITHM_B, "Martin BV", None, None);
        let role = infer_prethreshold(&corpus, ALGORITHM_C, "Martin BV", None, None);
        assert_eq!(legacy.greeting_candidate.as_deref(), Some("Martin"));
        assert_eq!(role.greeting_candidate, None);
    }
}
