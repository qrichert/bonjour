//! Extract a probable greeting name from an arbitrary display name, with a
//! decision score.
//!
//! A display name is not necessarily a person: it may be a company or a club
//! (`ACME Corporation`, `Club de Tennis Strasbourg`). So rather than assuming a
//! `[first] [last]` structure, we evaluate candidate spans using
//! frequency-weighted first-name evidence, surname-role evidence, and
//! organization evidence. Uncertain cases yield `None`; precision over recall:
//! a missed greeting is cheap, greeting a tennis club "Bonjour, Martin" is not.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};

use serde::Serialize;

mod artifact;
mod classifier;
#[cfg(feature = "standalone")]
mod embedded;
mod lexical;

pub use artifact::GenderHint;

/// Proxy-validated default for selecting a greeting from an inference.
pub const DEFAULT_GREETING_THRESHOLD: f64 = classifier::ALGORITHM_C2.threshold;

/// Invalid greeting threshold supplied by a caller.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InvalidThreshold {
    threshold: f64,
}

impl InvalidThreshold {
    /// Return the rejected threshold.
    #[must_use]
    pub fn threshold(self) -> f64 {
        self.threshold
    }
}

impl fmt::Display for InvalidThreshold {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "greeting threshold must be a finite value in 0.0..=1.0, got {}",
            self.threshold
        )
    }
}

impl Error for InvalidThreshold {}

/// One unthresholded C3.1 inference over a display name.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Inference<'a> {
    /// Best exact source-span candidate, or `None` when no safe candidate exists.
    pub greeting_name: Option<&'a str>,
    /// Pre-threshold C3.1 decision score, not a calibrated probability.
    pub decision_score: f64,
    /// Conservatively gated gender evidence for the candidate.
    pub gender_hint: Option<GenderHint>,
    /// Majority-gender share, or zero when no gender evidence is emitted.
    pub gender_confidence: f64,
}

impl<'a> Inference<'a> {
    /// Select the greeting candidate using the proxy-validated default threshold.
    #[must_use]
    pub fn greeting(&self) -> Option<&'a str> {
        self.greeting_name
            .filter(|_| self.decision_score >= DEFAULT_GREETING_THRESHOLD)
    }

    /// Select the greeting candidate using an explicit threshold.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidThreshold`] unless `threshold` is finite and within
    /// `0.0..=1.0`.
    pub fn greeting_at(&self, threshold: f64) -> Result<Option<&'a str>, InvalidThreshold> {
        validate_threshold(threshold)?;
        Ok(self
            .greeting_name
            .filter(|_| self.decision_score >= threshold))
    }
}

/// One lexically eligible candidate exposed by detailed diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct CandidateScore<'a> {
    /// Exact contiguous candidate span from the original display name.
    pub candidate: &'a str,
    /// Internal candidate-ranking score, or `None` without scorer support.
    pub ranking_score: Option<f64>,
    /// Scores supplied by the currently available evidence layers.
    pub signals: CandidateSignals,
}

/// Extensible scorer signals attached to an enumerated candidate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct CandidateSignals {
    /// Score supplied by the statistical name corpus, when available.
    pub corpus_score: Option<f64>,
}

/// Weighted terms in the frozen C2 emission score.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct DecisionContributions {
    /// Candidate-quality contribution. This is zero in frozen C3.1.
    pub candidate_quality: f64,
    /// Contribution from separation between the top two candidates.
    pub winner_margin: f64,
    /// Contribution from given-name versus surname-role evidence.
    pub role: f64,
    /// Contribution from the amount of supporting evidence.
    pub reliability: f64,
}

/// Existing hard and soft vetoes applied by frozen C3.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct DecisionVetoes {
    /// A strong legal or organization marker forced an immediate abstention.
    pub strong_organization_marker: bool,
    /// A generic organization marker forced the emission score to zero.
    pub generic_organization_marker: bool,
    /// An ampersand forced the emission score to zero.
    pub ampersand: bool,
    /// The selected candidate did not contain the minimum number of letters.
    pub candidate_too_short: bool,
}

/// Diagnostic trace of the frozen C3.1 pre-threshold decision.
///
/// Winner-only values are absent when no candidate was selected, including a
/// hard organization-marker abstention. The separately ranked candidates may
/// still contain counterfactual entries in that case.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct DecisionTrace {
    /// Ranking score of the selected candidate.
    pub candidate_quality: Option<f64>,
    /// Difference between the first- and second-ranked candidates.
    pub winner_margin: Option<f64>,
    /// Winner margin normalized by the frozen margin scale.
    pub margin_signal: Option<f64>,
    /// Log likelihood ratio of given-name to surname evidence.
    pub role_llr: Option<f64>,
    /// Bounded role signal derived from `role_llr`.
    pub role_signal: Option<f64>,
    /// Evidence reliability used by the emission model.
    pub reliability: Option<f64>,
    /// Number of Unicode alphabetic characters in the selected candidate.
    pub alphabetic_length: Option<usize>,
    /// Frozen minimum alphabetic length used by the veto.
    pub minimum_alphabetic_length: usize,
    /// Weighted C2 terms before vetoes, when a winner exists.
    pub contributions: Option<DecisionContributions>,
    /// Clamped weighted sum before soft vetoes, when a winner exists.
    pub pre_veto_score: Option<f64>,
    /// Score after soft vetoes and before the segmentation penalty.
    pub post_veto_score: f64,
    /// Whether the winner came from conservative handle segmentation.
    pub segmented_candidate: Option<bool>,
    /// Handle segmentation boundary used for the winner, when applicable.
    pub segmentation_mechanism: Option<&'static str>,
    /// C3.1 provenance penalty actually applied to the winner.
    pub segmented_candidate_penalty: f64,
    /// Veto states used by the decision.
    pub vetoes: DecisionVetoes,
}

/// Unthresholded inference together with every eligible candidate.
///
/// Candidate output is diagnostic: inclusion or scorer support does not imply
/// that a candidate is safe to use as a greeting.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DetailedInference<'a> {
    /// Selected pre-threshold candidate and final C3.1 decision score.
    pub inference: Inference<'a>,
    /// Existing inputs and arithmetic behind the final decision score.
    pub decision: DecisionTrace,
    /// Ranked candidates followed by unscored candidates in source order.
    pub candidates: Vec<CandidateScore<'a>>,
}

fn validate_threshold(threshold: f64) -> Result<(), InvalidThreshold> {
    if threshold.is_finite() && (0.0..=1.0).contains(&threshold) {
        Ok(())
    } else {
        Err(InvalidThreshold { threshold })
    }
}

/// Stable category for artifact-loading failures.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoadErrorKind {
    MissingData,
    StandaloneDataUnavailable,
    ManifestMismatch,
    UnsupportedFormat,
    CorruptArtifact,
    Io,
}

/// Failure to load or validate the pinned production artifact.
#[derive(Debug)]
pub struct LoadError {
    kind: LoadErrorKind,
    path: Option<PathBuf>,
    message: String,
    source: Option<Box<dyn Error + Send + Sync>>,
}

impl LoadError {
    /// Return the stable failure category.
    #[must_use]
    pub fn kind(&self) -> LoadErrorKind {
        self.kind
    }

    /// Return the file or directory associated with the failure, if any.
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    pub(crate) fn new(
        kind: LoadErrorKind,
        path: Option<PathBuf>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            path,
            message: message.into(),
            source: None,
        }
    }

    pub(crate) fn with_source(
        kind: LoadErrorKind,
        path: Option<PathBuf>,
        message: impl Into<String>,
        source: impl Error + Send + Sync + 'static,
    ) -> Self {
        Self {
            kind,
            path,
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }
}

impl fmt::Display for LoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)?;
        if let Some(path) = &self.path {
            write!(formatter, ": {}", path.display())?;
        }
        Ok(())
    }
}

impl Error for LoadError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source
            .as_deref()
            .map(|source| source as &(dyn Error + 'static))
    }
}

/// Immutable, reusable greeting-name classifier.
pub struct Classifier {
    artifact: artifact::C32Artifact,
}

impl fmt::Debug for Classifier {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Classifier")
            .field("artifact_id", &artifact::ARTIFACT_ID)
            .field("format_version", &artifact::FORMAT_VERSION)
            .field("key_count", &self.artifact.key_count())
            .field("row_count", &self.artifact.row_count())
            .finish_non_exhaustive()
    }
}

impl Classifier {
    /// Load the exact pinned production artifact from `path`.
    ///
    /// # Errors
    ///
    /// Returns [`LoadError`] when the directory is unavailable or any pinned
    /// manifest, constituent, or structural invariant fails validation.
    pub fn from_dir(path: impl AsRef<Path>) -> Result<Self, LoadError> {
        artifact::C32Artifact::from_dir(path.as_ref()).map(|artifact| Self { artifact })
    }

    /// Load the production artifact embedded by the `standalone` feature.
    ///
    /// # Errors
    ///
    /// Returns [`LoadError`] when the feature was built without the pinned
    /// artifact or when embedded data fail structural validation.
    #[cfg(feature = "standalone")]
    pub fn standalone() -> Result<Self, LoadError> {
        embedded::artifact().map(|artifact| Self { artifact })
    }

    /// Infer the best exact greeting candidate from `display_name` using C3.1.
    #[must_use]
    pub fn infer<'a>(
        &self,
        display_name: &'a str,
        country_hint: Option<&str>,
        locale_hint: Option<&str>,
    ) -> Inference<'a> {
        self.infer_with_gender(display_name, country_hint, locale_hint, None)
    }

    /// Infer with an optional caller-supplied gender hint.
    #[must_use]
    pub fn infer_with_gender<'a>(
        &self,
        display_name: &'a str,
        country_hint: Option<&str>,
        locale_hint: Option<&str>,
        gender_hint: Option<GenderHint>,
    ) -> Inference<'a> {
        let (diagnostic, raw) =
            self.diagnose_with_gender(display_name, country_hint, locale_hint, gender_hint);
        inference_from_diagnostic(display_name, &diagnostic, &raw)
    }

    /// Infer and return every ranked C3.1 candidate for diagnostics.
    #[must_use]
    pub fn infer_detailed<'a>(
        &self,
        display_name: &'a str,
        country_hint: Option<&str>,
        locale_hint: Option<&str>,
    ) -> DetailedInference<'a> {
        self.infer_detailed_with_gender(display_name, country_hint, locale_hint, None)
    }

    /// Infer with a gender hint and return every ranked C3.1 candidate.
    #[must_use]
    pub fn infer_detailed_with_gender<'a>(
        &self,
        display_name: &'a str,
        country_hint: Option<&str>,
        locale_hint: Option<&str>,
        gender_hint: Option<GenderHint>,
    ) -> DetailedInference<'a> {
        let (diagnostic, raw) =
            self.diagnose_with_gender(display_name, country_hint, locale_hint, gender_hint);
        let decision = decision_trace(&diagnostic);
        let candidates = candidate_scores(display_name, &diagnostic.candidates);
        let inference = inference_from_diagnostic(display_name, &diagnostic, &raw);
        DetailedInference {
            inference,
            decision,
            candidates,
        }
    }

    fn diagnose_with_gender(
        &self,
        display_name: &str,
        country_hint: Option<&str>,
        locale_hint: Option<&str>,
        gender_hint: Option<GenderHint>,
    ) -> (
        classifier::RoleInferenceDiagnostic,
        classifier::RawInference,
    ) {
        let diagnostic = classifier::diagnose_role_inference(
            &self.artifact,
            classifier::ALGORITHM_C3,
            display_name,
            country_hint,
            locale_hint,
        );
        let mut raw = classifier::c31_inference_from_diagnostic(
            &diagnostic,
            classifier::ALGORITHM_C2,
            classifier::ALGORITHM_C31,
        );
        if let Some(gender_hint) = gender_hint {
            classifier::apply_gender_hint(&diagnostic, &mut raw, gender_hint);
        }
        (diagnostic, raw)
    }
}

fn decision_trace(diagnostic: &classifier::RoleInferenceDiagnostic) -> DecisionTrace {
    let breakdown = classifier::c31_decision_breakdown(
        diagnostic,
        classifier::ALGORITHM_C2,
        classifier::ALGORITHM_C31,
    );
    let winner = breakdown.winner.as_ref();
    DecisionTrace {
        candidate_quality: winner.map(|features| features.winner_score),
        winner_margin: winner.map(|features| features.winner_margin),
        margin_signal: breakdown.margin_signal,
        role_llr: winner.map(|features| features.role_llr),
        role_signal: winner.map(|features| features.role_signal),
        reliability: winner.map(|features| features.reliability),
        alphabetic_length: winner.map(|features| features.alphabetic_length),
        minimum_alphabetic_length: classifier::ALGORITHM_C2.minimum_candidate_letters,
        contributions: breakdown
            .contributions
            .map(|contributions| DecisionContributions {
                candidate_quality: contributions.candidate_quality,
                winner_margin: contributions.winner_margin,
                role: contributions.role,
                reliability: contributions.reliability,
            }),
        pre_veto_score: breakdown.pre_veto_score,
        post_veto_score: breakdown.post_veto_score,
        segmented_candidate: breakdown.segmented_candidate,
        segmentation_mechanism: winner.and_then(|features| features.segmentation_mechanism),
        segmented_candidate_penalty: breakdown.segmented_candidate_penalty,
        vetoes: DecisionVetoes {
            strong_organization_marker: breakdown.hard_organization_marker,
            generic_organization_marker: breakdown.generic_organization_marker,
            ampersand: breakdown.ampersand,
            candidate_too_short: breakdown.candidate_too_short,
        },
    }
}

fn inference_from_diagnostic<'a>(
    display_name: &'a str,
    diagnostic: &classifier::RoleInferenceDiagnostic,
    raw: &classifier::RawInference,
) -> Inference<'a> {
    let greeting_name = raw
        .greeting_candidate
        .is_some()
        .then(|| source_greeting_span(display_name, diagnostic.candidates.first()))
        .flatten();
    debug_assert_eq!(raw.greeting_candidate.is_some(), greeting_name.is_some());
    let gender_emitted = greeting_name.is_some()
        && raw.confidence >= DEFAULT_GREETING_THRESHOLD
        && raw.gender_hint.is_some();

    Inference {
        greeting_name,
        decision_score: raw.confidence,
        gender_hint: gender_emitted.then_some(raw.gender_hint).flatten(),
        gender_confidence: if gender_emitted {
            raw.gender_confidence
        } else {
            0.0
        },
    }
}

fn candidate_scores<'a>(
    display_name: &'a str,
    candidates: &[classifier::CandidateDiagnostic],
) -> Vec<CandidateScore<'a>> {
    let mut supported_bounds = HashSet::with_capacity(candidates.len());
    let mut scores = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let (Some(byte_start), Some(byte_end), Some(candidate_span)) = (
            candidate.byte_start,
            candidate.byte_end,
            source_candidate_span(display_name, candidate),
        ) else {
            continue;
        };
        supported_bounds.insert((byte_start, byte_end));
        scores.push(CandidateScore {
            candidate: candidate_span,
            ranking_score: Some(candidate.score),
            signals: CandidateSignals {
                corpus_score: Some(candidate.score),
            },
        });
    }
    debug_assert_eq!(scores.len(), candidates.len());
    scores.extend(
        classifier::enumerate_candidate_spans(display_name)
            .into_iter()
            .filter(|span| !supported_bounds.contains(&(span.byte_start, span.byte_end)))
            .filter_map(|span| {
                display_name
                    .get(span.byte_start..span.byte_end)
                    .map(|candidate| CandidateScore {
                        candidate,
                        ranking_score: None,
                        signals: CandidateSignals { corpus_score: None },
                    })
            }),
    );
    scores
}

fn source_greeting_span<'a>(
    display_name: &'a str,
    winner: Option<&classifier::CandidateDiagnostic>,
) -> Option<&'a str> {
    source_candidate_span(display_name, winner?)
}

fn source_candidate_span<'a>(
    display_name: &'a str,
    candidate: &classifier::CandidateDiagnostic,
) -> Option<&'a str> {
    display_name.get(candidate.byte_start?..candidate.byte_end?)
}

/// Frozen evaluator internals. This is not part of the supported 0.1 API.
#[cfg(feature = "benchmark-internals")]
#[doc(hidden)]
pub mod benchmark {
    use std::path::Path;

    pub use crate::GenderHint;
    pub use crate::artifact::{C32Artifact, Evidence, EvidenceSource};
    pub use crate::classifier::*;
    pub use crate::lexical::candidate_is_eligible;

    pub fn open_artifact(path: &Path) -> Result<C32Artifact, crate::LoadError> {
        C32Artifact::from_dir(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn production_directory() -> Option<PathBuf> {
        std::env::var_os("BONJOUR_TEST_DATA_DIR").map(PathBuf::from)
    }

    #[test]
    fn load_error_kind_is_stable() {
        let error = Classifier::from_dir("definitely-missing-bonjour-name-data").unwrap_err();
        assert_eq!(error.kind(), LoadErrorKind::MissingData);
        assert!(error.path().is_some());
    }

    fn assert_classifier_traits<T: fmt::Debug + Send + Sync>() {}

    fn assert_value_traits<T: fmt::Debug + Clone + Copy + PartialEq + serde::Serialize>() {}

    fn assert_detailed_traits<T: fmt::Debug + Clone + PartialEq + serde::Serialize>() {}

    fn assert_enum_traits<T: fmt::Debug + Clone + Copy + Eq + std::hash::Hash>() {}

    fn assert_error_traits<T: fmt::Debug + Error>() {}

    #[test]
    fn public_types_implement_the_locked_traits() {
        assert_classifier_traits::<Classifier>();
        assert_value_traits::<GenderHint>();
        assert_value_traits::<Inference<'static>>();
        assert_value_traits::<CandidateScore<'static>>();
        assert_value_traits::<CandidateSignals>();
        assert_value_traits::<DecisionContributions>();
        assert_value_traits::<DecisionTrace>();
        assert_value_traits::<DecisionVetoes>();
        assert_detailed_traits::<DetailedInference<'static>>();
        assert_enum_traits::<GenderHint>();
        assert_enum_traits::<LoadErrorKind>();
        assert_error_traits::<InvalidThreshold>();
        assert_error_traits::<LoadError>();
    }

    #[test]
    fn greeting_threshold_is_configurable_and_validated() {
        let inference = Inference {
            greeting_name: Some("Quentin"),
            decision_score: DEFAULT_GREETING_THRESHOLD,
            gender_hint: Some(GenderHint::Male),
            gender_confidence: 0.9,
        };
        assert_eq!(inference.greeting(), Some("Quentin"));
        assert_eq!(inference.greeting_at(0.0).unwrap(), Some("Quentin"));
        assert_eq!(inference.greeting_at(1.0).unwrap(), None);

        for threshold in [f64::NAN, f64::NEG_INFINITY, -0.1, 1.1, f64::INFINITY] {
            let error = inference.greeting_at(threshold).unwrap_err();
            assert_eq!(error.threshold().to_bits(), threshold.to_bits());
            assert!(error.to_string().contains("finite value in 0.0..=1.0"));
        }
    }

    #[test]
    fn load_error_display_and_source_contracts_are_stable() {
        let plain = LoadError::new(LoadErrorKind::MissingData, None, "missing data");
        assert_eq!(plain.to_string(), "missing data");
        assert!(plain.source().is_none());

        let sourced = LoadError::with_source(
            LoadErrorKind::Io,
            Some(PathBuf::from("artifact.bin")),
            "cannot read artifact",
            std::io::Error::other("disk failure"),
        );
        assert_eq!(sourced.to_string(), "cannot read artifact: artifact.bin");
        assert!(sourced.source().is_some());
        assert!(format!("{sourced:?}").contains("disk failure"));
    }

    #[test]
    fn classifier_debug_and_hidden_benchmark_loader_use_pinned_data() {
        let Some(directory) = production_directory() else {
            return;
        };
        let classifier = Classifier::from_dir(&directory).unwrap();
        let debug = format!("{classifier:?}");
        assert!(debug.contains("bonjour-name-data-v1"));
        assert!(debug.contains("1803175"));

        #[cfg(feature = "benchmark-internals")]
        {
            let artifact = benchmark::open_artifact(&directory).unwrap();
            assert_eq!(artifact.key_count(), artifact::KEY_COUNT);
        }
    }

    #[test]
    fn missing_winner_has_no_source_span() {
        assert_eq!(source_greeting_span("Quentin", None), None);
    }

    #[cfg(all(feature = "standalone", bonjour_embedded_data))]
    #[test]
    fn detailed_inference_matches_lightweight_and_preserves_unicode_spans() {
        let classifier = Classifier::standalone().unwrap();
        let input = "E\u{301}lodie Martin";
        let lightweight = classifier.infer(input, Some("FR"), None);
        let detailed = classifier.infer_detailed(input, Some("FR"), None);

        assert_eq!(detailed.inference, lightweight);
        assert_eq!(detailed.inference.greeting(), lightweight.greeting());
        assert_eq!(
            detailed.inference.greeting_at(0.5).unwrap(),
            lightweight.greeting_at(0.5).unwrap()
        );
        assert!(!detailed.candidates.is_empty());
        assert!(
            detailed
                .candidates
                .iter()
                .all(|candidate| input.contains(candidate.candidate))
        );
        let contributions = detailed.decision.contributions.unwrap();
        let reconstructed = contributions.candidate_quality
            + contributions.winner_margin
            + contributions.role
            + contributions.reliability;
        assert_eq!(
            reconstructed.clamp(0.0, 1.0).to_bits(),
            detailed.decision.pre_veto_score.unwrap().to_bits()
        );
        assert_eq!(
            detailed.decision.post_veto_score.to_bits(),
            detailed.inference.decision_score.to_bits()
        );
    }

    #[cfg(all(feature = "standalone", bonjour_embedded_data))]
    #[test]
    fn detailed_hard_abstention_retains_only_counterfactual_candidates() {
        let classifier = Classifier::standalone().unwrap();
        let detailed = classifier.infer_detailed("Quentin Richert GmbH", None, None);

        assert_eq!(detailed.inference.greeting_name, None);
        assert_eq!(
            detailed.inference.decision_score.to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(detailed.decision.candidate_quality, None);
        assert_eq!(detailed.decision.contributions, None);
        assert!(detailed.decision.vetoes.strong_organization_marker);
        assert!(!detailed.candidates.is_empty());
    }

    #[cfg(all(feature = "standalone", bonjour_embedded_data))]
    #[test]
    fn detailed_inference_exposes_soft_veto_and_segmentation_penalty() {
        let classifier = Classifier::standalone().unwrap();
        let organization = classifier.infer_detailed("Baris Kebab", None, None);
        assert!(organization.decision.vetoes.generic_organization_marker);
        assert!(organization.decision.pre_veto_score.unwrap() > 0.0);
        assert_eq!(
            organization.decision.post_veto_score.to_bits(),
            0.0_f64.to_bits()
        );

        let segmented = classifier.infer_detailed("QuentinQuentin42", None, None);
        assert_eq!(segmented.decision.segmented_candidate, Some(true));
        assert_eq!(
            segmented.decision.segmentation_mechanism,
            Some("lower_to_upper")
        );
        assert_eq!(
            segmented.decision.segmented_candidate_penalty.to_bits(),
            classifier::ALGORITHM_C31.handle_segment_penalty.to_bits()
        );
        assert_eq!(
            (segmented.decision.post_veto_score - segmented.decision.segmented_candidate_penalty)
                .clamp(0.0, 1.0)
                .to_bits(),
            segmented.inference.decision_score.to_bits()
        );
    }

    #[cfg(all(feature = "standalone", bonjour_embedded_data))]
    #[test]
    fn detailed_inference_keeps_unknown_eligible_spans_as_unscored_candidates() {
        let classifier = Classifier::standalone().unwrap();
        let detailed = classifier.infer_detailed("Olivier Sieffert", None, None);

        assert_eq!(detailed.inference.greeting_name, Some("Olivier"));
        assert_eq!(detailed.candidates[0].candidate, "Olivier");
        assert!(detailed.candidates[0].ranking_score.is_some());
        assert_eq!(
            detailed.candidates[0].ranking_score,
            detailed.candidates[0].signals.corpus_score
        );
        assert!(detailed.candidates.iter().skip(1).all(|candidate| {
            candidate.ranking_score.is_none() && candidate.signals.corpus_score.is_none()
        }));
        assert!(detailed.candidates.iter().any(|candidate| {
            candidate.candidate == "Olivier Sieffert" && candidate.ranking_score.is_none()
        }));
        assert!(detailed.candidates.iter().any(|candidate| {
            candidate.candidate == "Sieffert" && candidate.ranking_score.is_none()
        }));
    }

    #[cfg(all(feature = "standalone", not(bonjour_embedded_data)))]
    #[test]
    fn unavailable_standalone_returns_typed_error() {
        let error = Classifier::standalone().unwrap_err();
        assert_eq!(error.kind(), LoadErrorKind::StandaloneDataUnavailable);
    }

    #[cfg(all(feature = "standalone", bonjour_embedded_data))]
    #[test]
    fn embedded_production_artifact_performs_known_lookup() {
        let classifier = Classifier::standalone().unwrap();
        let inference = classifier.infer("Quentin Richert", Some("FR"), None);
        assert_eq!(inference.greeting_name, Some("Quentin"));
    }
}
