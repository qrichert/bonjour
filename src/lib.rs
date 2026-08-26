//! Extract a probable greeting name from an arbitrary display name, with a
//! confidence score.
//!
//! A display name is not necessarily a person: it may be a company or a club
//! (`ACME Corporation`, `Club de Tennis Strasbourg`). So rather than assuming a
//! `[first] [last]` structure, we evaluate candidate spans using
//! frequency-weighted first-name evidence, surname-role evidence, and
//! organization evidence. Uncertain cases yield `None`; precision over recall:
//! a missed greeting is cheap, greeting a tennis club "Bonjour, Martin" is not.

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
    pub confidence: f64,
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
            .filter(|_| self.confidence >= DEFAULT_GREETING_THRESHOLD)
    }

    /// Select the greeting candidate using an explicit threshold.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidThreshold`] unless `threshold` is finite and within
    /// `0.0..=1.0`.
    pub fn greeting_at(&self, threshold: f64) -> Result<Option<&'a str>, InvalidThreshold> {
        validate_threshold(threshold)?;
        Ok(self.greeting_name.filter(|_| self.confidence >= threshold))
    }
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
            confidence: raw.confidence,
            gender_hint: gender_emitted.then_some(raw.gender_hint).flatten(),
            gender_confidence: if gender_emitted {
                raw.gender_confidence
            } else {
                0.0
            },
        }
    }
}

fn source_greeting_span<'a>(
    display_name: &'a str,
    winner: Option<&classifier::CandidateDiagnostic>,
) -> Option<&'a str> {
    let winner = winner?;
    display_name.get(winner.byte_start?..winner.byte_end?)
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

    fn assert_enum_traits<T: fmt::Debug + Clone + Copy + Eq + std::hash::Hash>() {}

    fn assert_error_traits<T: fmt::Debug + Error>() {}

    #[test]
    fn public_types_implement_the_locked_traits() {
        assert_classifier_traits::<Classifier>();
        assert_value_traits::<GenderHint>();
        assert_value_traits::<Inference<'static>>();
        assert_enum_traits::<GenderHint>();
        assert_enum_traits::<LoadErrorKind>();
        assert_error_traits::<InvalidThreshold>();
        assert_error_traits::<LoadError>();
    }

    #[test]
    fn greeting_threshold_is_configurable_and_validated() {
        let inference = Inference {
            greeting_name: Some("Quentin"),
            confidence: DEFAULT_GREETING_THRESHOLD,
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
