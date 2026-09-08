use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::path::{Path, PathBuf};

use bonjour::benchmark::{
    ALGORITHM_C2, ALGORITHM_C3, ALGORITHM_C31, c31_decision_breakdown, diagnose_role_inference,
    open_artifact,
};
use name_eval::holdout::{FrozenHoldout, load_frozen};
use serde::Serialize;

use crate::output::Result;
use crate::text::{greeting_matches, model_normalize};

const V1_SHA256: &str = "de95213f27fc1849032ee6788c8f16d7d515c1a991ae8b2e8414b7b155814c4e";
const V3_SHA256: &str = "d70e4d4b2ed7e49bed09dc1e8d2ba60ade8a752e3b86c772e964bd64883ee6fe";
const V4_SHA256: &str = "d95c589bec836faaeecaeda85b146989d2936914bff0209934f289ccb9446c7f";

#[derive(Clone, Debug)]
pub(crate) struct ProxyInput {
    pub(crate) sealed: PathBuf,
    pub(crate) manifest: PathBuf,
}

#[derive(Serialize)]
struct ProxyRow {
    population: &'static str,
    ordinal: usize,
    normalized_candidate: String,
    outcome: &'static str,
    c31_emits: bool,
    role_llr: Option<f64>,
    role_signal: Option<f64>,
    reliability: Option<f64>,
    winner_margin: Option<f64>,
    candidate_quality: Option<f64>,
    country_hint: String,
}

pub(crate) fn stream_proxy(artifact_path: &Path, inputs: Vec<ProxyInput>) -> Result<()> {
    let artifact = open_artifact(artifact_path)?;
    let holdouts = validate_and_order(inputs)?;
    let stdout = io::stdout();
    let mut writer = csv::Writer::from_writer(stdout.lock());
    for (population, holdout) in holdouts {
        for (ordinal, case) in holdout.cases.iter().enumerate() {
            if !case.is_evaluable() {
                continue;
            }
            let diagnostic = diagnose_role_inference(
                &artifact,
                ALGORITHM_C3,
                &case.display_name,
                nonempty(&case.country_hint),
                nonempty(&case.locale_hint),
            );
            let decision = c31_decision_breakdown(&diagnostic, ALGORITHM_C2, ALGORITHM_C31);
            let row = proxy_row(
                population,
                ordinal,
                case.expected_greeting(),
                &case.country_hint,
                &decision,
            );
            writer.serialize(row)?;
        }
    }
    writer.flush()?;
    Ok(())
}

fn validate_and_order(inputs: Vec<ProxyInput>) -> Result<Vec<(&'static str, FrozenHoldout)>> {
    if inputs.len() != 3 {
        return Err("proxy mode requires exactly three sealed/manifest pairs".into());
    }
    let expected = BTreeSet::from([V1_SHA256, V3_SHA256, V4_SHA256]);
    let mut holdouts = BTreeMap::new();
    for input in inputs {
        let holdout = load_frozen(&input.sealed, &input.manifest)?;
        let digest = holdout.manifest.holdout_sha256.clone();
        let Some(population) = population_from_digest(&digest) else {
            return Err(
                format!("proxy digest is not acknowledged for this experiment: {digest}").into(),
            );
        };
        if holdouts.insert(digest, (population, holdout)).is_some() {
            return Err("duplicate proxy generation".into());
        }
    }
    let actual = holdouts.keys().map(String::as_str).collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(format!("proxy mode requires exactly V1/V3/V4; got {actual:?}").into());
    }
    Ok([V1_SHA256, V3_SHA256, V4_SHA256]
        .into_iter()
        .map(|digest| holdouts.remove(digest).expect("validated holdout"))
        .collect())
}

fn proxy_row(
    population: &'static str,
    ordinal: usize,
    expected: Option<&str>,
    country_hint: &str,
    decision: &bonjour::benchmark::C31DecisionBreakdown,
) -> ProxyRow {
    let Some(winner) = decision.winner.as_ref() else {
        return ProxyRow {
            population,
            ordinal,
            normalized_candidate: String::new(),
            outcome: "no_winner",
            c31_emits: false,
            role_llr: None,
            role_signal: None,
            reliability: None,
            winner_margin: None,
            candidate_quality: None,
            country_hint: normalized_country(country_hint),
        };
    };
    let selected = Some(winner.greeting_candidate.as_str());
    let c31_emits = emits_at_c31_threshold(decision.final_score);
    let outcome = classify_outcome(expected, selected, c31_emits);
    ProxyRow {
        population,
        ordinal,
        normalized_candidate: model_normalize(&winner.greeting_candidate),
        outcome,
        c31_emits,
        role_llr: Some(winner.role_llr),
        role_signal: Some(winner.role_signal),
        reliability: Some(winner.reliability),
        winner_margin: Some(winner.winner_margin),
        candidate_quality: Some(winner.winner_score),
        country_hint: normalized_country(country_hint),
    }
}

fn emits_at_c31_threshold(score: f64) -> bool {
    score >= ALGORITHM_C2.threshold
}

fn classify_outcome(
    expected: Option<&str>,
    selected: Option<&str>,
    c31_emits: bool,
) -> &'static str {
    match expected {
        Some(_) if greeting_matches(expected, selected) && c31_emits => "correct_winner_c31_emits",
        Some(_) if greeting_matches(expected, selected) => "correct_winner_c31_abstains",
        Some(_) => "wrong_winner",
        None if selected.is_some() => "expected_null_winner",
        None => "no_winner",
    }
}

fn population_from_digest(digest: &str) -> Option<&'static str> {
    match digest {
        V1_SHA256 => Some("REAL_PROXY_V1_DEV"),
        V3_SHA256 => Some("REAL_PROXY_V3"),
        V4_SHA256 => Some("REAL_PROXY_V4"),
        _ => None,
    }
}

fn normalized_country(value: &str) -> String {
    let value = value.trim();
    if value.len() == 2 && value.bytes().all(|byte| byte.is_ascii_alphabetic()) {
        value.to_ascii_uppercase()
    } else {
        String::new()
    }
}

fn nonempty(value: &str) -> Option<&str> {
    (!value.is_empty()).then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_only_acknowledged_proxy_digests() {
        assert_eq!(population_from_digest(V1_SHA256), Some("REAL_PROXY_V1_DEV"));
        assert_eq!(population_from_digest(V3_SHA256), Some("REAL_PROXY_V3"));
        assert_eq!(population_from_digest(V4_SHA256), Some("REAL_PROXY_V4"));
        assert_eq!(
            population_from_digest(
                "69070614fee68401b896d6c5bfb4c22c55cca9744237f66213a9dd04291db6c7f"
            ),
            None
        );
    }

    #[test]
    fn normalizes_only_iso_like_country_hints() {
        assert_eq!(normalized_country("fr"), "FR");
        assert_eq!(normalized_country(" FRA "), "");
        assert_eq!(normalized_country(""), "");
    }

    #[test]
    fn classifies_every_proxy_outcome() {
        assert_eq!(
            classify_outcome(Some("Élodie"), Some("Élodie"), true),
            "correct_winner_c31_emits"
        );
        assert_eq!(
            classify_outcome(Some("O’Connor"), Some("O'Connor"), false),
            "correct_winner_c31_abstains"
        );
        assert_eq!(
            classify_outcome(Some("Élodie"), Some("Martin"), false),
            "wrong_winner"
        );
        assert_eq!(
            classify_outcome(None, Some("Martin"), false),
            "expected_null_winner"
        );
        assert_eq!(classify_outcome(None, None, false), "no_winner");
    }

    #[test]
    fn emission_threshold_is_inclusive() {
        assert!(emits_at_c31_threshold(ALGORITHM_C2.threshold));
        assert!(!emits_at_c31_threshold(f64::from_bits(
            ALGORITHM_C2.threshold.to_bits() - 1
        )));
    }
}
