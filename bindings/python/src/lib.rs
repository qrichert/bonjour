use std::sync::OnceLock;

use bonjour::{CandidateScore, Classifier, DecisionTrace, GenderHint};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use serde::Serialize;

static CLASSIFIER: OnceLock<Result<Classifier, String>> = OnceLock::new();

#[derive(Serialize)]
struct SummaryOutput<'a> {
    best_candidate: Option<&'a str>,
    greeting_name: Option<&'a str>,
    decision_score: f64,
    emission_source: bonjour::EmissionSource,
    gender_hint: Option<GenderHint>,
    gender_confidence: f64,
}

#[derive(Serialize)]
struct DetailedOutput<'a> {
    input: &'a str,
    best_candidate: Option<&'a str>,
    greeting_name: Option<&'a str>,
    decision_score: f64,
    decision: DecisionTrace,
    candidates: Vec<CandidateScore<'a>>,
    gender_hint: Option<GenderHint>,
    gender_confidence: f64,
}

#[pyfunction]
#[pyo3(signature = (display_name, country_hint=None, locale_hint=None, gender_hint=None))]
fn infer_json(
    display_name: &str,
    country_hint: Option<&str>,
    locale_hint: Option<&str>,
    gender_hint: Option<&str>,
) -> PyResult<String> {
    let gender_hint = parse_gender_hint(gender_hint)?;
    let inference =
        classifier()?.infer_with_gender(display_name, country_hint, locale_hint, gender_hint);
    let output = SummaryOutput {
        best_candidate: inference.greeting_name,
        greeting_name: inference.greeting(),
        decision_score: inference.decision_score,
        emission_source: inference.emission_source,
        gender_hint: inference.gender_hint,
        gender_confidence: inference.gender_confidence,
    };
    serialize(&output)
}

#[pyfunction]
#[pyo3(signature = (display_name, country_hint=None, locale_hint=None, gender_hint=None))]
fn infer_detailed_json(
    display_name: &str,
    country_hint: Option<&str>,
    locale_hint: Option<&str>,
    gender_hint: Option<&str>,
) -> PyResult<String> {
    let gender_hint = parse_gender_hint(gender_hint)?;
    let detailed = classifier()?.infer_detailed_with_gender(
        display_name,
        country_hint,
        locale_hint,
        gender_hint,
    );
    let output = DetailedOutput {
        input: display_name,
        best_candidate: detailed.inference.greeting_name,
        greeting_name: detailed.inference.greeting(),
        decision_score: detailed.inference.decision_score,
        decision: detailed.decision,
        candidates: detailed.candidates,
        gender_hint: detailed.inference.gender_hint,
        gender_confidence: detailed.inference.gender_confidence,
    };
    serialize(&output)
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(infer_json, module)?)?;
    module.add_function(wrap_pyfunction!(infer_detailed_json, module)?)?;
    Ok(())
}

fn serialize(value: &impl Serialize) -> PyResult<String> {
    serde_json::to_string(value)
        .map_err(|error| PyRuntimeError::new_err(format!("cannot serialize inference: {error}")))
}

fn classifier() -> PyResult<&'static Classifier> {
    match CLASSIFIER.get_or_init(|| Classifier::standalone().map_err(|error| error.to_string())) {
        Ok(classifier) => Ok(classifier),
        Err(message) => Err(PyRuntimeError::new_err(format!(
            "cannot load embedded bonjour name data: {message}"
        ))),
    }
}

fn parse_gender_hint(value: Option<&str>) -> PyResult<Option<GenderHint>> {
    value
        .map(|value| {
            GenderHint::parse(value)
                .ok_or_else(|| PyValueError::new_err("gender_hint must be F, female, M, or male"))
        })
        .transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_is_initialized_once() {
        let first = classifier().unwrap();
        let second = classifier().unwrap();
        assert!(std::ptr::eq(first, second));
    }

    #[test]
    fn gender_parser_matches_the_public_rust_parser() {
        assert_eq!(
            parse_gender_hint(Some(" female ")).unwrap(),
            Some(GenderHint::Female)
        );
        assert_eq!(
            parse_gender_hint(Some("M")).unwrap(),
            Some(GenderHint::Male)
        );
        assert!(parse_gender_hint(Some("unknown")).is_err());
        assert_eq!(parse_gender_hint(None).unwrap(), None);
    }
}
