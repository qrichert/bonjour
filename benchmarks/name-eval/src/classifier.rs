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
    pub compositional_role_floor: f64,
    pub compositional_evidence_weight: f64,
    pub hyphen_structure_bonus: f64,
    pub hard_legal_abstention: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AlgorithmKind {
    Legacy,
    RoleHypothesis,
    RoleCompositional,
    RoleHandleSegments,
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
    compositional_role_floor: 0.0,
    compositional_evidence_weight: 0.0,
    hyphen_structure_bonus: 0.0,
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
    compositional_role_floor: 0.0,
    compositional_evidence_weight: 0.0,
    hyphen_structure_bonus: 0.0,
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
    compositional_role_floor: 0.0,
    compositional_evidence_weight: 0.0,
    hyphen_structure_bonus: 0.0,
    hard_legal_abstention: true,
};

pub const ALGORITHM_C1: AlgorithmConfig = AlgorithmConfig {
    name: "C1-compositional-role-v1",
    kind: AlgorithmKind::RoleCompositional,
    compositional_role_floor: 0.75,
    compositional_evidence_weight: 0.20,
    hyphen_structure_bonus: 0.04,
    ..ALGORITHM_C
};

pub const ALGORITHM_C3: AlgorithmConfig = AlgorithmConfig {
    name: "C3-conservative-handle-candidates-v1",
    kind: AlgorithmKind::RoleHandleSegments,
    ..ALGORITHM_C1
};

#[derive(Clone, Debug, PartialEq)]
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
    compositional_evidence: f64,
    remainder_evidence: f64,
    origin: CandidateOrigin,
    segmentation_mechanism: Option<HandleSegmentationMechanism>,
    lookup_query: Option<String>,
    lookup_mode: Option<LookupMode>,
    component_lookup_modes: Option<[LookupMode; 2]>,
    evidence: Evidence,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CandidateOrigin {
    Exact,
    ComposedWhitespace,
    ComposedHyphen,
    HandleSegment,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HandleSegmentationMechanism {
    Digit,
    Underscore,
    Dot,
    LowerUpper,
    Mixed,
}

impl HandleSegmentationMechanism {
    fn as_str(self) -> &'static str {
        match self {
            Self::Digit => "digit",
            Self::Underscore => "underscore",
            Self::Dot => "dot",
            Self::LowerUpper => "lower_to_upper",
            Self::Mixed => "mixed",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct HandleSegment {
    display: String,
    mechanism: HandleSegmentationMechanism,
}

impl CandidateOrigin {
    fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::ComposedWhitespace => "composed_whitespace",
            Self::ComposedHyphen => "composed_hyphen",
            Self::HandleSegment => "handle_segment",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LookupMode {
    Normalized,
    AccentFolded,
}

impl LookupMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Normalized => "normalized",
            Self::AccentFolded => "accent_folded",
        }
    }
}

#[derive(Clone, Debug)]
struct LookupMatch {
    evidence: Evidence,
    query: String,
    mode: LookupMode,
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
    pub compositional_evidence: f64,
    pub remainder_evidence: f64,
    pub origin: &'static str,
    pub segmentation_mechanism: Option<&'static str>,
    pub lookup_query: Option<String>,
    pub lookup_mode: Option<&'static str>,
    pub left_lookup_mode: Option<&'static str>,
    pub right_lookup_mode: Option<&'static str>,
    pub score: f64,
    pub algorithm_a_score: f64,
    pub algorithm_b_score: f64,
}

#[derive(Clone, Debug)]
pub struct ExpectedLookupDiagnostic {
    pub eligible: bool,
    pub matched_query: Option<String>,
    pub lookup_mode: Option<&'static str>,
    pub evidence: Option<Evidence>,
    pub role_llr: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct ExpectedCompositionDiagnostic {
    pub shape: Option<&'static str>,
    pub supported: bool,
    pub left_lookup_mode: Option<&'static str>,
    pub right_lookup_mode: Option<&'static str>,
    pub left_role_llr: Option<f64>,
    pub right_role_llr: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct RoleInferenceDiagnostic {
    pub inference: RawInference,
    pub hard_organization_abstention: bool,
    pub generic_organization_marker: bool,
    pub ampersand_negative_evidence: bool,
    pub candidates: Vec<CandidateDiagnostic>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WinnerFeatures {
    pub greeting_candidate: String,
    pub winner_score: f64,
    pub second_score: Option<f64>,
    pub winner_margin: f64,
    pub no_competitor: bool,
    pub role_llr: f64,
    pub role_signal: f64,
    pub reliability: f64,
    pub global_given_count: u64,
    pub global_surname_count: u64,
    pub candidate_origin: &'static str,
    pub segmentation_mechanism: Option<&'static str>,
    pub candidate_count: usize,
    pub alphabetic_length: usize,
    pub generic_organization_marker: bool,
    pub ampersand_negative_evidence: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct C2EmissionConfig {
    pub quality_weight: f64,
    pub margin_weight: f64,
    pub role_weight: f64,
    pub reliability_weight: f64,
    pub margin_scale: f64,
    pub minimum_candidate_letters: usize,
    pub threshold: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct C31EmissionConfig {
    pub handle_segment_penalty: f64,
}

/// Frozen from REAL_PROXY_V1_DEV plus synthetic VALIDATION. This is a
/// development operating point, not a real-world quality claim.
pub const ALGORITHM_C2: C2EmissionConfig = C2EmissionConfig {
    quality_weight: 0.0,
    margin_weight: 0.1,
    role_weight: 0.7,
    reliability_weight: 0.2,
    margin_scale: 0.5,
    minimum_candidate_letters: 3,
    threshold: 0.789_758_824_057_369_6,
};

/// Frozen from spent REAL_PROXY_V1_DEV, spent REAL_PROXY_V3, and synthetic
/// VALIDATION. This is a development operating point requiring fresh V4
/// evaluation, not a real-world quality claim.
pub const ALGORITHM_C31: C31EmissionConfig = C31EmissionConfig {
    handle_segment_penalty: 0.025,
};

pub fn infer_prethreshold(
    corpus: &impl EvidenceSource,
    config: AlgorithmConfig,
    display_name: &str,
    country_hint: Option<&str>,
    locale_hint: Option<&str>,
) -> RawInference {
    if matches!(
        config.kind,
        AlgorithmKind::RoleHypothesis
            | AlgorithmKind::RoleCompositional
            | AlgorithmKind::RoleHandleSegments
    ) {
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
    sort_role_candidates(&mut candidates);
    role_inference_from_sorted_candidates(config, display_name, &candidates)
}

fn sort_role_candidates(candidates: &mut [RoleCandidate]) {
    candidates.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| right.length.cmp(&left.length))
            .then_with(|| left.start.cmp(&right.start))
            .then_with(|| left.display.cmp(&right.display))
    });
}

fn role_inference_from_sorted_candidates(
    config: AlgorithmConfig,
    display_name: &str,
    candidates: &[RoleCandidate],
) -> RawInference {
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
    let mut candidates = exact_role_candidates(corpus, config, &tokens, country);
    if matches!(
        config.kind,
        AlgorithmKind::RoleCompositional | AlgorithmKind::RoleHandleSegments
    ) {
        add_compositional_candidates(corpus, config, &tokens, country, &mut candidates);
    }
    if config.kind == AlgorithmKind::RoleHandleSegments {
        add_handle_segment_candidates(corpus, config, &tokens, country, &mut candidates);
    }

    add_direct_compound_evidence(config, &mut candidates);
    add_remainder_evidence(config, &mut candidates);
    candidates
}

fn exact_role_candidates(
    corpus: &impl EvidenceSource,
    config: AlgorithmConfig,
    tokens: &[String],
    country: Option<[u8; 2]>,
) -> Vec<RoleCandidate> {
    let mut candidates = Vec::new();
    for start in 0..tokens.len() {
        for length in 1..=2.min(tokens.len() - start) {
            let display = tokens[start..start + length].join(" ");
            let Some(lookup) = lookup_match_with_variants(corpus, &display, country) else {
                continue;
            };
            candidates.push(role_candidate_from_lookup(
                config,
                display,
                start,
                length,
                CandidateOrigin::Exact,
                lookup,
                tokens.len() == 1,
            ));
        }
    }
    candidates
}

fn role_candidate_from_lookup(
    config: AlgorithmConfig,
    display: String,
    start: usize,
    length: usize,
    origin: CandidateOrigin,
    lookup: LookupMatch,
    single_display: bool,
) -> RoleCandidate {
    let evidence = lookup.evidence;
    let role_llr = role_llr(evidence, config.role_smoothing);
    let role_signal = logistic((role_llr - config.role_center) / config.role_scale);
    let reliability = count_reliability(evidence.global_count);
    let country_support = country_support(evidence);
    let mut score = config.role_score_floor
        + config.role_weight * role_signal
        + config.role_reliability_weight * reliability
        + config.country_weight * country_support;
    if single_display {
        score += config.single_display_bonus;
    }
    RoleCandidate {
        display,
        start,
        length,
        score,
        role_llr,
        role_signal,
        reliability,
        country_support,
        compound_evidence: 0.0,
        compositional_evidence: 0.0,
        remainder_evidence: 0.0,
        origin,
        segmentation_mechanism: None,
        lookup_query: Some(lookup.query),
        lookup_mode: Some(lookup.mode),
        component_lookup_modes: None,
        evidence,
    }
}

fn add_handle_segment_candidates(
    corpus: &impl EvidenceSource,
    config: AlgorithmConfig,
    tokens: &[String],
    country: Option<[u8; 2]>,
    candidates: &mut Vec<RoleCandidate>,
) {
    for (start, token) in tokens.iter().enumerate() {
        for segment in conservative_handle_segments(token) {
            if segment
                .display
                .chars()
                .filter(|character| character.is_alphabetic())
                .count()
                < ALGORITHM_C2.minimum_candidate_letters
                || candidates.iter().any(|candidate| {
                    candidate.start == start
                        && candidate.length == 1
                        && canonicalize(&candidate.display).to_lowercase()
                            == canonicalize(&segment.display).to_lowercase()
                })
            {
                continue;
            }
            let Some(lookup) = lookup_match_with_variants(corpus, &segment.display, country) else {
                continue;
            };
            let mut candidate = role_candidate_from_lookup(
                config,
                segment.display,
                start,
                1,
                CandidateOrigin::HandleSegment,
                lookup,
                tokens.len() == 1,
            );
            candidate.segmentation_mechanism = Some(segment.mechanism);
            candidates.push(candidate);
        }
    }
}

fn conservative_handle_segments(token: &str) -> Vec<HandleSegment> {
    if token.chars().any(|character| {
        !(character.is_alphabetic()
            || is_combining_mark(character)
            || character.is_ascii_digit()
            || matches!(character, '\'' | '-' | '_' | '.'))
    }) {
        return Vec::new();
    }

    struct ExplicitPart<'a> {
        display: &'a str,
        left_boundary: Option<HandleSegmentationMechanism>,
        right_boundary: Option<HandleSegmentationMechanism>,
    }

    let mut explicit_parts = Vec::<ExplicitPart<'_>>::new();
    let mut part_start = 0;
    let mut left_boundary = None;
    let mut explicit_boundary = false;
    for (index, character) in token.char_indices() {
        let boundary = if character.is_ascii_digit() {
            Some(HandleSegmentationMechanism::Digit)
        } else if character == '_' {
            Some(HandleSegmentationMechanism::Underscore)
        } else if character == '.' {
            Some(HandleSegmentationMechanism::Dot)
        } else {
            None
        };
        let Some(boundary) = boundary else {
            continue;
        };
        explicit_boundary = true;
        if part_start < index {
            explicit_parts.push(ExplicitPart {
                display: &token[part_start..index],
                left_boundary,
                right_boundary: Some(boundary),
            });
        }
        part_start = index + character.len_utf8();
        left_boundary = Some(boundary);
    }
    if part_start < token.len() {
        explicit_parts.push(ExplicitPart {
            display: &token[part_start..],
            left_boundary,
            right_boundary: None,
        });
    }

    let mut segments = Vec::new();
    let mut camel_boundary = false;
    for part in explicit_parts {
        let mut part_segments = Vec::<&str>::new();
        let mut part_has_camel_boundary = false;
        let mut segment_start = 0;
        let mut previous = None;
        for (index, character) in part.display.char_indices() {
            if previous.is_some_and(char::is_lowercase) && character.is_uppercase() {
                camel_boundary = true;
                part_has_camel_boundary = true;
                if segment_start < index {
                    part_segments.push(&part.display[segment_start..index]);
                }
                segment_start = index;
            }
            previous = Some(character);
        }
        if segment_start < part.display.len() {
            part_segments.push(&part.display[segment_start..]);
        }
        // A trailing all-uppercase fragment is indistinguishable from a
        // credential, club marker, or handle suffix (`PrincessFC`). Treat the
        // entire camel-like part as unsafe rather than extracting its prefix.
        if part_has_camel_boundary
            && part_segments
                .iter()
                .any(|segment| !segment.chars().any(char::is_lowercase))
        {
            continue;
        }
        let last_index = part_segments.len().saturating_sub(1);
        for (index, display) in part_segments.into_iter().enumerate() {
            let left = if index == 0 {
                part.left_boundary
            } else {
                Some(HandleSegmentationMechanism::LowerUpper)
            };
            let right = if index == last_index {
                part.right_boundary
            } else {
                Some(HandleSegmentationMechanism::LowerUpper)
            };
            let mechanism = match (left, right) {
                (Some(left), Some(right)) if left != right => HandleSegmentationMechanism::Mixed,
                (Some(mechanism), _) | (_, Some(mechanism)) => mechanism,
                (None, None) => continue,
            };
            segments.push(HandleSegment {
                display: display.to_string(),
                mechanism,
            });
        }
    }

    if !explicit_boundary && !camel_boundary {
        return Vec::new();
    }
    segments
        .into_iter()
        .filter(|segment| candidate_is_eligible(&segment.display))
        .collect()
}

fn add_direct_compound_evidence(config: AlgorithmConfig, candidates: &mut [RoleCandidate]) {
    for index in 0..candidates.len() {
        if candidates[index].origin == CandidateOrigin::Exact && candidates[index].length == 2 {
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
}

fn add_remainder_evidence(config: AlgorithmConfig, candidates: &mut [RoleCandidate]) {
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
}

fn add_compositional_candidates(
    corpus: &impl EvidenceSource,
    config: AlgorithmConfig,
    tokens: &[String],
    country: Option<[u8; 2]>,
    candidates: &mut Vec<RoleCandidate>,
) {
    // Without direct phrase evidence, a two-token input is inherently ambiguous
    // between a compound greeting name and given + surname. Require a remainder
    // token before synthesizing a whitespace compound.
    if tokens.len() >= 3 {
        for start in 0..tokens.len() - 1 {
            if candidates
                .iter()
                .any(|candidate| candidate.start == start && candidate.length == 2)
            {
                continue;
            }
            let Some(left) = component_evidence(corpus, config, &tokens[start], country) else {
                continue;
            };
            let Some(right) = component_evidence(corpus, config, &tokens[start + 1], country)
            else {
                continue;
            };
            if let Some(candidate) = composed_candidate(
                config,
                tokens[start..=start + 1].join(" "),
                start,
                2,
                CandidateOrigin::ComposedWhitespace,
                left,
                right,
            ) {
                candidates.push(candidate);
            }
        }
    }

    for (start, token) in tokens.iter().enumerate() {
        if candidates.iter().any(|candidate| {
            candidate.start == start
                && candidate.length == 1
                && candidate.display.eq_ignore_ascii_case(token)
        }) {
            continue;
        }
        let mut components = token.split('-');
        let (Some(left_display), Some(right_display), None) =
            (components.next(), components.next(), components.next())
        else {
            continue;
        };
        if left_display.is_empty() || right_display.is_empty() {
            continue;
        }
        let Some(left) = component_evidence(corpus, config, left_display, country) else {
            continue;
        };
        let Some(right) = component_evidence(corpus, config, right_display, country) else {
            continue;
        };
        if let Some(candidate) = composed_candidate(
            config,
            token.clone(),
            start,
            1,
            CandidateOrigin::ComposedHyphen,
            left,
            right,
        ) {
            candidates.push(candidate);
        }
    }
}

#[derive(Clone)]
struct ComponentEvidence {
    evidence: Evidence,
    role_llr: f64,
    role_signal: f64,
    reliability: f64,
    country_support: f64,
    lookup_query: String,
    lookup_mode: LookupMode,
}

fn component_evidence(
    corpus: &impl EvidenceSource,
    config: AlgorithmConfig,
    display: &str,
    country: Option<[u8; 2]>,
) -> Option<ComponentEvidence> {
    let lookup = lookup_match_with_variants(corpus, display, country)?;
    let role_llr = role_llr(lookup.evidence, config.role_smoothing);
    Some(ComponentEvidence {
        evidence: lookup.evidence,
        role_llr,
        role_signal: logistic((role_llr - config.role_center) / config.role_scale),
        reliability: count_reliability(lookup.evidence.global_count),
        country_support: country_support(lookup.evidence),
        lookup_query: lookup.query,
        lookup_mode: lookup.mode,
    })
}

fn composed_candidate(
    config: AlgorithmConfig,
    display: String,
    start: usize,
    length: usize,
    origin: CandidateOrigin,
    left: ComponentEvidence,
    right: ComponentEvidence,
) -> Option<RoleCandidate> {
    if left.role_llr < config.compositional_role_floor
        || right.role_llr < config.compositional_role_floor
    {
        return None;
    }
    let role_llr = left.role_llr.min(right.role_llr);
    let role_signal = left.role_signal.min(right.role_signal);
    let reliability = (left.reliability * right.reliability).sqrt();
    let country_support = (left.country_support * right.country_support).sqrt();
    let compositional_evidence = (left.role_signal * right.role_signal).sqrt();
    let structure_bonus = if origin == CandidateOrigin::ComposedHyphen {
        config.hyphen_structure_bonus
    } else {
        0.0
    };
    let component_lookup_modes = [left.lookup_mode, right.lookup_mode];
    let score = config.role_score_floor
        + config.role_weight * role_signal
        + config.role_reliability_weight * reliability
        + config.country_weight * country_support
        + config.compositional_evidence_weight * compositional_evidence
        + structure_bonus;
    Some(RoleCandidate {
        display,
        start,
        length,
        score,
        role_llr,
        role_signal,
        reliability,
        country_support,
        compound_evidence: 0.0,
        compositional_evidence,
        remainder_evidence: 0.0,
        origin,
        segmentation_mechanism: None,
        lookup_query: Some(format!("{} + {}", left.lookup_query, right.lookup_query)),
        lookup_mode: None,
        component_lookup_modes: Some(component_lookup_modes),
        evidence: combine_component_evidence(left.evidence, right.evidence),
    })
}

fn combine_component_evidence(left: Evidence, right: Evidence) -> Evidence {
    Evidence {
        global_count: left.global_count.min(right.global_count),
        country_count: left.country_count.min(right.country_count),
        effective_count: left.effective_count.saturating_add(right.effective_count),
        female_count: left.female_count.saturating_add(right.female_count),
        male_count: left.male_count.saturating_add(right.male_count),
        surname_count: left.surname_count.min(right.surname_count),
        given_total: left.given_total,
        surname_total: left.surname_total,
    }
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
        .map(|candidate| candidate_diagnostic(candidate, tokenize(display_name).len()))
        .collect()
}

pub fn diagnose_role_inference(
    corpus: &impl EvidenceSource,
    config: AlgorithmConfig,
    display_name: &str,
    country_hint: Option<&str>,
    locale_hint: Option<&str>,
) -> RoleInferenceDiagnostic {
    let hard_organization_abstention =
        config.hard_legal_abstention && has_strong_organization_marker(display_name);
    let generic_organization_marker = has_generic_organization_marker(display_name);
    let ampersand_negative_evidence = display_name.contains('&');
    let mut candidates = role_candidates(corpus, config, display_name, country_hint, locale_hint);
    sort_role_candidates(&mut candidates);
    let inference = if hard_organization_abstention {
        empty_inference()
    } else {
        role_inference_from_sorted_candidates(config, display_name, &candidates)
    };
    let token_count = tokenize(display_name).len();
    RoleInferenceDiagnostic {
        inference,
        hard_organization_abstention,
        generic_organization_marker,
        ampersand_negative_evidence,
        candidates: candidates
            .into_iter()
            .map(|candidate| candidate_diagnostic(candidate, token_count))
            .collect(),
    }
}

pub fn winner_features(diagnostic: &RoleInferenceDiagnostic) -> Option<WinnerFeatures> {
    let selected = diagnostic.inference.greeting_candidate.as_deref()?;
    let winner = diagnostic.candidates.first()?;
    debug_assert_eq!(winner.display, selected);
    let second_score = diagnostic
        .candidates
        .get(1)
        .map(|candidate| candidate.score);
    Some(WinnerFeatures {
        greeting_candidate: selected.to_string(),
        winner_score: winner.score,
        second_score,
        winner_margin: second_score.map_or(1.0, |score| winner.score - score),
        no_competitor: second_score.is_none(),
        role_llr: winner.role_llr,
        role_signal: winner.role_signal,
        reliability: winner.reliability,
        global_given_count: winner.global_given_count,
        global_surname_count: winner.global_surname_count,
        candidate_origin: winner.origin,
        segmentation_mechanism: winner.segmentation_mechanism,
        candidate_count: diagnostic.candidates.len(),
        alphabetic_length: selected
            .chars()
            .filter(|character| character.is_alphabetic())
            .count(),
        generic_organization_marker: diagnostic.generic_organization_marker,
        ampersand_negative_evidence: diagnostic.ampersand_negative_evidence,
    })
}

pub fn c2_decision_score(features: &WinnerFeatures, config: C2EmissionConfig) -> f64 {
    debug_assert!(c2_config_is_valid(config));
    if features.generic_organization_marker
        || features.ampersand_negative_evidence
        || features.alphabetic_length < config.minimum_candidate_letters
    {
        return 0.0;
    }
    let margin_signal = (features.winner_margin / config.margin_scale).clamp(0.0, 1.0);
    (config.quality_weight * features.winner_score
        + config.margin_weight * margin_signal
        + config.role_weight * features.role_signal
        + config.reliability_weight * features.reliability)
        .clamp(0.0, 1.0)
}

pub fn c2_config_is_valid(config: C2EmissionConfig) -> bool {
    let weights = [
        config.quality_weight,
        config.margin_weight,
        config.role_weight,
        config.reliability_weight,
    ];
    weights
        .iter()
        .all(|weight| *weight >= 0.0 && weight.is_finite())
        && (weights.iter().sum::<f64>() - 1.0).abs() < 1e-9
        && config.margin_scale > 0.0
        && config.margin_scale.is_finite()
        && config.minimum_candidate_letters > 0
        && config.threshold > 0.0
        && config.threshold <= 1.0
}

pub fn c2_inference_from_diagnostic(
    diagnostic: &RoleInferenceDiagnostic,
    config: C2EmissionConfig,
) -> RawInference {
    let mut inference = diagnostic.inference.clone();
    inference.confidence = winner_features(diagnostic)
        .as_ref()
        .map_or(0.0, |features| c2_decision_score(features, config));
    inference
}

pub fn c31_inference_from_diagnostic(
    diagnostic: &RoleInferenceDiagnostic,
    c2_config: C2EmissionConfig,
    c31_config: C31EmissionConfig,
) -> RawInference {
    debug_assert!(c31_config.handle_segment_penalty >= 0.0);
    debug_assert!(c31_config.handle_segment_penalty.is_finite());
    let mut inference = c2_inference_from_diagnostic(diagnostic, c2_config);
    if winner_features(diagnostic)
        .as_ref()
        .is_some_and(|features| features.candidate_origin == "handle_segment")
    {
        inference.confidence =
            (inference.confidence - c31_config.handle_segment_penalty).clamp(0.0, 1.0);
    }
    inference
}

pub fn expected_lookup_diagnostic(
    corpus: &impl EvidenceSource,
    config: AlgorithmConfig,
    display: &str,
    country_hint: Option<&str>,
    locale_hint: Option<&str>,
) -> ExpectedLookupDiagnostic {
    let eligible = candidate_is_eligible(display);
    let country = resolve_country(country_hint, locale_hint);
    let lookup = lookup_match_with_variants(corpus, display, country);
    ExpectedLookupDiagnostic {
        eligible,
        matched_query: lookup.as_ref().map(|lookup| lookup.query.clone()),
        lookup_mode: lookup.as_ref().map(|lookup| lookup.mode.as_str()),
        evidence: lookup.as_ref().map(|lookup| lookup.evidence),
        role_llr: lookup
            .as_ref()
            .map(|lookup| role_llr(lookup.evidence, config.role_smoothing)),
    }
}

pub fn expected_composition_diagnostic(
    corpus: &impl EvidenceSource,
    config: AlgorithmConfig,
    display: &str,
    country_hint: Option<&str>,
    locale_hint: Option<&str>,
) -> ExpectedCompositionDiagnostic {
    let canonical = canonicalize(display);
    let tokens = canonical.split_whitespace().collect::<Vec<_>>();
    let components = if let [left, right] = tokens.as_slice() {
        Some((*left, *right, "whitespace"))
    } else if let [token] = tokens.as_slice() {
        let mut parts = token.split('-');
        match (parts.next(), parts.next(), parts.next()) {
            (Some(left), Some(right), None) if !left.is_empty() && !right.is_empty() => {
                Some((left, right, "hyphen"))
            }
            _ => None,
        }
    } else {
        None
    };
    let Some((left_display, right_display, shape)) = components else {
        return ExpectedCompositionDiagnostic {
            shape: None,
            supported: false,
            left_lookup_mode: None,
            right_lookup_mode: None,
            left_role_llr: None,
            right_role_llr: None,
        };
    };
    let country = resolve_country(country_hint, locale_hint);
    let left = component_evidence(corpus, config, left_display, country);
    let right = component_evidence(corpus, config, right_display, country);
    ExpectedCompositionDiagnostic {
        shape: Some(shape),
        supported: left
            .as_ref()
            .is_some_and(|component| component.role_llr >= config.compositional_role_floor)
            && right
                .as_ref()
                .is_some_and(|component| component.role_llr >= config.compositional_role_floor),
        left_lookup_mode: left
            .as_ref()
            .map(|component| component.lookup_mode.as_str()),
        right_lookup_mode: right
            .as_ref()
            .map(|component| component.lookup_mode.as_str()),
        left_role_llr: left.as_ref().map(|component| component.role_llr),
        right_role_llr: right.as_ref().map(|component| component.role_llr),
    }
}

fn candidate_diagnostic(candidate: RoleCandidate, token_count: usize) -> CandidateDiagnostic {
    let [left_lookup_mode, right_lookup_mode] = candidate
        .component_lookup_modes
        .map_or([None, None], |modes| [Some(modes[0]), Some(modes[1])]);
    CandidateDiagnostic {
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
        compositional_evidence: candidate.compositional_evidence,
        remainder_evidence: candidate.remainder_evidence,
        origin: candidate.origin.as_str(),
        segmentation_mechanism: candidate
            .segmentation_mechanism
            .map(HandleSegmentationMechanism::as_str),
        lookup_query: candidate.lookup_query,
        lookup_mode: candidate.lookup_mode.map(LookupMode::as_str),
        left_lookup_mode: left_lookup_mode.map(LookupMode::as_str),
        right_lookup_mode: right_lookup_mode.map(LookupMode::as_str),
        score: candidate.score,
        algorithm_a_score: legacy_candidate_score(
            candidate.evidence,
            candidate.start,
            candidate.length,
            token_count,
            ALGORITHM_A,
        ),
        algorithm_b_score: legacy_candidate_score(
            candidate.evidence,
            candidate.start,
            candidate.length,
            token_count,
            ALGORITHM_B,
        ),
    }
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
    lookup_match_with_variants(corpus, display, country).map(|lookup| lookup.evidence)
}

fn lookup_match_with_variants(
    corpus: &impl EvidenceSource,
    display: &str,
    country: Option<[u8; 2]>,
) -> Option<LookupMatch> {
    if !candidate_is_eligible(display) {
        return None;
    }
    let canonical = canonicalize(display);
    let title = title_case(&canonical);
    let lowercase = canonical.to_lowercase();
    let mut variants = vec![
        (canonical, LookupMode::Normalized),
        (title, LookupMode::Normalized),
        (lowercase, LookupMode::Normalized),
    ];
    let unaccented = variants
        .iter()
        .map(|(value, _)| (strip_accents(value), LookupMode::AccentFolded))
        .collect::<Vec<_>>();
    variants.extend(unaccented);
    let mut seen = HashSet::new();
    variants
        .into_iter()
        .filter(|(variant, _)| seen.insert(variant.clone()))
        .find_map(|(query, mode)| {
            corpus.lookup(&query, country).map(|evidence| LookupMatch {
                evidence,
                query,
                mode,
            })
        })
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
    fn c1_composes_unsupported_whitespace_compound_from_given_like_components() {
        let corpus = FakeCorpus(HashMap::from([
            ("Mary".to_string(), role_evidence(40_000, 200)),
            ("Jane".to_string(), role_evidence(30_000, 150)),
        ]));
        let c0 = infer_prethreshold(&corpus, ALGORITHM_C, "Mary Jane Watson", None, None);
        let c1 = infer_prethreshold(&corpus, ALGORITHM_C1, "Mary Jane Watson", None, None);
        assert_ne!(c0.greeting_candidate.as_deref(), Some("Mary Jane"));
        assert_eq!(c1.greeting_candidate.as_deref(), Some("Mary Jane"));
    }

    #[test]
    fn c1_does_not_invent_whitespace_compound_for_ambiguous_two_token_input() {
        let corpus = FakeCorpus(HashMap::from([
            ("Mary".to_string(), role_evidence(40_000, 200)),
            ("Jane".to_string(), role_evidence(30_000, 150)),
        ]));
        let diagnostics = candidate_diagnostics(&corpus, ALGORITHM_C1, "Mary Jane", None, None);
        assert!(
            diagnostics
                .iter()
                .all(|candidate| candidate.origin != "composed_whitespace")
        );
    }

    #[test]
    fn c1_composes_unsupported_hyphenated_name_from_given_like_components() {
        let corpus = FakeCorpus(HashMap::from([
            ("Jean".to_string(), role_evidence(60_000, 1_000)),
            ("Pierre".to_string(), role_evidence(45_000, 800)),
        ]));
        let c0 = infer_prethreshold(&corpus, ALGORITHM_C, "Jean-Pierre Martin", None, None);
        let c1 = infer_prethreshold(&corpus, ALGORITHM_C1, "Jean-Pierre Martin", None, None);
        assert_eq!(c0.greeting_candidate, None);
        assert_eq!(c1.greeting_candidate.as_deref(), Some("Jean-Pierre"));
    }

    #[test]
    fn c1_does_not_compose_given_and_surname_like_candidates() {
        let corpus = FakeCorpus(HashMap::from([
            ("Jean".to_string(), role_evidence(58_545, 9_796)),
            ("Martin".to_string(), role_evidence(14_544, 46_409)),
        ]));
        for input in ["Jean Martin", "Martin Jean"] {
            let inference = infer_prethreshold(&corpus, ALGORITHM_C1, input, None, None);
            assert_eq!(
                inference.greeting_candidate.as_deref(),
                Some("Jean"),
                "{input}"
            );
            assert!(
                candidate_diagnostics(&corpus, ALGORITHM_C1, input, None, None)
                    .iter()
                    .all(|candidate| candidate.origin != "composed_whitespace")
            );
        }
    }

    #[test]
    fn c3_segments_only_explicit_conservative_handle_boundaries() {
        fn displays(input: &str) -> Vec<String> {
            conservative_handle_segments(input)
                .into_iter()
                .map(|segment| segment.display)
                .collect()
        }

        assert_eq!(displays("Quentin42"), ["Quentin"]);
        assert_eq!(displays("Jean.Dupont_2024"), ["Jean", "Dupont"]);
        assert_eq!(displays("ÉlodieMartin"), ["Élodie", "Martin"]);
        assert_eq!(displays("Fatima-ZahraCarla1"), ["Fatima-Zahra", "Carla"]);

        for input in [
            "quentindupont",
            "quentinnnn",
            "QUENTINDUPONT",
            "PrincessFC",
            "kaggle.com/quentin",
            "quentin@example.com",
            "a:quentin",
        ] {
            assert!(displays(input).is_empty(), "{input:?}");
        }
    }

    #[test]
    fn c3_records_the_boundaries_that_expose_each_segment() {
        let mechanisms = |input| {
            conservative_handle_segments(input)
                .into_iter()
                .map(|segment| (segment.display, segment.mechanism.as_str()))
                .collect::<Vec<_>>()
        };
        assert_eq!(mechanisms("Quentin42"), [("Quentin".to_string(), "digit")]);
        assert_eq!(
            mechanisms("Jean.Dupont"),
            [("Jean".to_string(), "dot"), ("Dupont".to_string(), "dot"),]
        );
        assert_eq!(
            mechanisms("Jean_Dupont"),
            [
                ("Jean".to_string(), "underscore"),
                ("Dupont".to_string(), "underscore"),
            ]
        );
        assert_eq!(
            mechanisms("ÉlodieMartin"),
            [
                ("Élodie".to_string(), "lower_to_upper"),
                ("Martin".to_string(), "lower_to_upper"),
            ]
        );
        assert_eq!(
            mechanisms("Jean.Dupont42"),
            [("Jean".to_string(), "dot"), ("Dupont".to_string(), "mixed"),]
        );
    }

    #[test]
    fn c3_adds_corpus_backed_digit_delimiter_and_camel_case_segments() {
        let corpus = FakeCorpus(HashMap::from([
            ("Quentin".to_string(), role_evidence(100_000, 100)),
            ("Élodie".to_string(), role_evidence(90_000, 100)),
            ("Martin".to_string(), role_evidence(14_544, 46_409)),
        ]));
        for (input, expected) in [
            ("Quentin42", "Quentin"),
            ("Quentin_42", "Quentin"),
            ("Quentin.Martin", "Quentin"),
            ("ÉlodieMartin", "Élodie"),
        ] {
            let diagnostic = diagnose_role_inference(&corpus, ALGORITHM_C3, input, None, None);
            assert!(diagnostic.candidates.iter().any(|candidate| {
                candidate.display == expected
                    && candidate.origin == "handle_segment"
                    && candidate.segmentation_mechanism.is_some()
            }));
            let c3 = c2_inference_from_diagnostic(&diagnostic, ALGORITHM_C2);
            assert_eq!(
                c3.greeting_at(ALGORITHM_C2.threshold),
                Some(expected),
                "{input:?}"
            );
        }
    }

    #[test]
    fn c31_penalizes_only_segmented_winners_without_changing_the_winner() {
        let corpus = FakeCorpus(HashMap::from([
            ("Quentin".to_string(), role_evidence(100_000, 100)),
            ("Martin".to_string(), role_evidence(14_544, 46_409)),
        ]));
        let segmented = diagnose_role_inference(&corpus, ALGORITHM_C3, "Quentin42", None, None);
        let c3 = c2_inference_from_diagnostic(&segmented, ALGORITHM_C2);
        let c31 = c31_inference_from_diagnostic(&segmented, ALGORITHM_C2, ALGORITHM_C31);
        assert_eq!(c31.greeting_candidate, c3.greeting_candidate);
        assert!(
            (c31.confidence - (c3.confidence - ALGORITHM_C31.handle_segment_penalty)).abs()
                < f64::EPSILON
        );

        let native = diagnose_role_inference(&corpus, ALGORITHM_C3, "Quentin Martin", None, None);
        assert_eq!(
            c31_inference_from_diagnostic(&native, ALGORITHM_C2, ALGORITHM_C31),
            c2_inference_from_diagnostic(&native, ALGORITHM_C2)
        );
    }

    #[test]
    fn c3_does_not_scan_arbitrary_substrings_or_unsafe_url_tokens() {
        let corpus = FakeCorpus(HashMap::from([(
            "Quentin".to_string(),
            role_evidence(100_000, 100),
        )]));
        for input in [
            "quentindupont",
            "quentinnnn",
            "QUENTINXYZ",
            "kaggle.com/Quentin",
            "quentin@example.com",
        ] {
            let diagnostic = diagnose_role_inference(&corpus, ALGORITHM_C3, input, None, None);
            assert!(
                diagnostic
                    .candidates
                    .iter()
                    .all(|candidate| candidate.origin != "handle_segment"),
                "{input:?}"
            );
        }
    }

    #[test]
    fn c3_leaves_non_handle_c2_inference_unchanged() {
        let corpus = FakeCorpus(HashMap::from([
            ("Quentin".to_string(), role_evidence(100_000, 100)),
            ("Martin".to_string(), role_evidence(14_544, 46_409)),
        ]));
        for input in ["Quentin Martin", "Martin Quentin", "Quentin GmbH"] {
            let c1 = diagnose_role_inference(&corpus, ALGORITHM_C1, input, None, None);
            let c3 = diagnose_role_inference(&corpus, ALGORITHM_C3, input, None, None);
            assert_eq!(
                c2_inference_from_diagnostic(&c3, ALGORITHM_C2),
                c2_inference_from_diagnostic(&c1, ALGORITHM_C2),
                "{input:?}"
            );
        }
    }

    #[test]
    fn algorithm_c_only_legal_marker_does_not_change_legacy_algorithms() {
        let corpus = FakeCorpus::names();
        let legacy = infer_prethreshold(&corpus, ALGORITHM_B, "Martin BV", None, None);
        let role = infer_prethreshold(&corpus, ALGORITHM_C, "Martin BV", None, None);
        assert_eq!(legacy.greeting_candidate.as_deref(), Some("Martin"));
        assert_eq!(role.greeting_candidate, None);
    }

    #[test]
    fn diagnostics_distinguish_normalized_and_accent_folded_lookups() {
        let corpus = FakeCorpus(HashMap::from([
            ("Martin".to_string(), role_evidence(10_000, 100)),
            ("Elodie".to_string(), role_evidence(9_000, 90)),
        ]));
        let normalized = expected_lookup_diagnostic(&corpus, ALGORITHM_C1, "martin", None, None);
        assert!(normalized.eligible);
        assert_eq!(normalized.lookup_mode, Some("normalized"));
        assert_eq!(normalized.matched_query.as_deref(), Some("Martin"));

        let folded = expected_lookup_diagnostic(&corpus, ALGORITHM_C1, "Élodie", None, None);
        assert!(folded.eligible);
        assert_eq!(folded.lookup_mode, Some("accent_folded"));
        assert_eq!(folded.matched_query.as_deref(), Some("Elodie"));
    }

    #[test]
    fn diagnostic_inference_is_identical_to_production_c1() {
        let corpus = FakeCorpus(HashMap::from([
            ("Mary".to_string(), role_evidence(40_000, 200)),
            ("Jane".to_string(), role_evidence(30_000, 150)),
            ("Martin".to_string(), role_evidence(14_544, 46_409)),
        ]));
        for input in [
            "Mary Jane Watson",
            "Mary-Jane Martin",
            "Mary Consulting",
            "Unknown",
            "Mary GmbH",
        ] {
            let production = infer_prethreshold(&corpus, ALGORITHM_C1, input, None, None);
            let diagnostic = diagnose_role_inference(&corpus, ALGORITHM_C1, input, None, None);
            assert_eq!(diagnostic.inference, production, "{input}");
        }
    }

    #[test]
    fn diagnostic_preserves_composed_lookup_provenance() {
        let corpus = FakeCorpus(HashMap::from([
            ("Mary".to_string(), role_evidence(40_000, 200)),
            ("Jane".to_string(), role_evidence(30_000, 150)),
        ]));
        let diagnostic =
            diagnose_role_inference(&corpus, ALGORITHM_C1, "Mary-Jane Watson", None, None);
        let composed = diagnostic
            .candidates
            .iter()
            .find(|candidate| candidate.origin == "composed_hyphen")
            .unwrap();
        assert_eq!(composed.lookup_mode, None);
        assert_eq!(composed.left_lookup_mode, Some("normalized"));
        assert_eq!(composed.right_lookup_mode, Some("normalized"));
    }

    #[test]
    fn expected_composition_support_is_separate_from_candidate_generation() {
        let corpus = FakeCorpus(HashMap::from([
            ("Mary".to_string(), role_evidence(40_000, 200)),
            ("Jane".to_string(), role_evidence(30_000, 150)),
        ]));
        let composition =
            expected_composition_diagnostic(&corpus, ALGORITHM_C1, "Mary Jane", None, None);
        assert_eq!(composition.shape, Some("whitespace"));
        assert!(composition.supported);
        assert!(
            candidate_diagnostics(&corpus, ALGORITHM_C1, "Mary Jane", None, None)
                .iter()
                .all(|candidate| candidate.display != "Mary Jane")
        );
    }

    #[test]
    fn hard_abstention_diagnostic_keeps_counterfactual_candidates() {
        let corpus = FakeCorpus(HashMap::from([(
            "Martin".to_string(),
            role_evidence(50_000, 1_000),
        )]));
        let diagnostic = diagnose_role_inference(&corpus, ALGORITHM_C1, "Martin GmbH", None, None);
        assert!(diagnostic.hard_organization_abstention);
        assert_eq!(diagnostic.inference, empty_inference());
        assert!(
            diagnostic
                .candidates
                .iter()
                .any(|candidate| candidate.display == "Martin")
        );
    }

    #[test]
    fn winner_features_use_unit_margin_without_a_competitor() {
        let corpus = FakeCorpus(HashMap::from([(
            "Quentin".to_string(),
            role_evidence(100_000, 100),
        )]));
        let diagnostic = diagnose_role_inference(&corpus, ALGORITHM_C1, "Quentin", None, None);
        let features = winner_features(&diagnostic).unwrap();
        assert_eq!(features.greeting_candidate, "Quentin");
        assert!(features.no_competitor);
        assert_eq!(features.second_score, None);
        assert!((features.winner_margin - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn c2_score_is_monotonic_in_positive_weighted_features() {
        let baseline = WinnerFeatures {
            greeting_candidate: "Quentin".to_string(),
            winner_score: 0.4,
            second_score: Some(0.3),
            winner_margin: 0.1,
            no_competitor: false,
            role_llr: 1.0,
            role_signal: 0.4,
            reliability: 0.4,
            global_given_count: 100,
            global_surname_count: 10,
            candidate_origin: "exact",
            segmentation_mechanism: None,
            candidate_count: 2,
            alphabetic_length: 7,
            generic_organization_marker: false,
            ampersand_negative_evidence: false,
        };
        let config = C2EmissionConfig {
            quality_weight: 0.25,
            margin_weight: 0.25,
            role_weight: 0.25,
            reliability_weight: 0.25,
            margin_scale: 0.5,
            minimum_candidate_letters: 3,
            threshold: 0.5,
        };
        let score = c2_decision_score(&baseline, config);
        for improved in [
            WinnerFeatures {
                winner_score: 0.5,
                ..baseline.clone()
            },
            WinnerFeatures {
                winner_margin: 0.2,
                ..baseline.clone()
            },
            WinnerFeatures {
                role_signal: 0.5,
                ..baseline.clone()
            },
            WinnerFeatures {
                reliability: 0.5,
                ..baseline.clone()
            },
        ] {
            assert!(c2_decision_score(&improved, config) > score);
        }
    }

    #[test]
    fn c2_configuration_and_safety_vetoes_are_enforced() {
        assert!(c2_config_is_valid(ALGORITHM_C2));
        assert!(!c2_config_is_valid(C2EmissionConfig {
            reliability_weight: 0.3,
            ..ALGORITHM_C2
        }));
        let baseline = WinnerFeatures {
            greeting_candidate: "Martin".to_string(),
            winner_score: 1.0,
            second_score: None,
            winner_margin: 1.0,
            no_competitor: true,
            role_llr: 5.0,
            role_signal: 1.0,
            reliability: 1.0,
            global_given_count: 1_000_000,
            global_surname_count: 0,
            candidate_origin: "exact",
            segmentation_mechanism: None,
            candidate_count: 1,
            alphabetic_length: 6,
            generic_organization_marker: false,
            ampersand_negative_evidence: false,
        };
        assert!(c2_decision_score(&baseline, ALGORITHM_C2) > 0.0);
        assert_eq!(
            c2_decision_score(
                &WinnerFeatures {
                    generic_organization_marker: true,
                    ..baseline.clone()
                },
                ALGORITHM_C2
            ),
            0.0
        );
        assert_eq!(
            c2_decision_score(
                &WinnerFeatures {
                    ampersand_negative_evidence: true,
                    ..baseline.clone()
                },
                ALGORITHM_C2
            ),
            0.0
        );
        assert_eq!(
            c2_decision_score(
                &WinnerFeatures {
                    greeting_candidate: "MD".to_string(),
                    alphabetic_length: 2,
                    ..baseline
                },
                ALGORITHM_C2
            ),
            0.0
        );
    }

    #[test]
    fn c2_preserves_c1_winner_and_gender_but_blocks_short_fragments() {
        let corpus = FakeCorpus(HashMap::from([
            ("Quentin".to_string(), role_evidence(1_500_000, 0)),
            ("MD".to_string(), role_evidence(1_500_000, 0)),
        ]));
        let c1 = infer_prethreshold(&corpus, ALGORITHM_C1, "Quentin", None, None);
        let diagnostic = diagnose_role_inference(&corpus, ALGORITHM_C1, "Quentin", None, None);
        let c2 = c2_inference_from_diagnostic(&diagnostic, ALGORITHM_C2);
        assert_eq!(c2.greeting_candidate, c1.greeting_candidate);
        assert_eq!(c2.gender_hint, c1.gender_hint);
        assert_eq!(c2.gender_confidence, c1.gender_confidence);
        assert!(c2.greeting_at(ALGORITHM_C2.threshold).is_some());

        let diagnostic = diagnose_role_inference(&corpus, ALGORITHM_C1, "MD", None, None);
        let short = c2_inference_from_diagnostic(&diagnostic, ALGORITHM_C2);
        assert_eq!(short.greeting_candidate.as_deref(), Some("MD"));
        assert_eq!(short.greeting_at(ALGORITHM_C2.threshold), None);
    }
}
