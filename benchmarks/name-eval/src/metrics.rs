use crate::classifier::RawInference;
use crate::dataset::Case;

#[derive(Clone, Copy, Debug, Default)]
pub struct Metrics {
    pub total: usize,
    pub person_cases: usize,
    pub organization_cases: usize,
    pub emitted: usize,
    pub correct_greetings: usize,
    pub wrong_greetings: usize,
    pub abstentions: usize,
    pub organization_false_positives: usize,
    pub person_false_negatives: usize,
    pub gender_labeled: usize,
    pub gender_emitted: usize,
    pub gender_correct: usize,
}

impl Metrics {
    pub fn evaluate(cases: &[&Case], predictions: &[&RawInference], threshold: f64) -> Self {
        assert_eq!(cases.len(), predictions.len());
        let mut metrics = Self::default();
        for (&case, &prediction) in cases.iter().zip(predictions) {
            metrics.observe(case, prediction, threshold);
        }
        metrics
    }

    fn observe(&mut self, case: &Case, prediction: &RawInference, threshold: f64) {
        self.total += 1;
        let greeting = prediction.greeting_at(threshold);
        let correct = greeting_matches(case.expected_greeting.as_deref(), greeting);
        if case.is_person() {
            self.person_cases += 1;
            if greeting.is_none() {
                self.person_false_negatives += 1;
            }
        } else {
            self.organization_cases += 1;
            if greeting.is_some() {
                self.organization_false_positives += 1;
            }
        }
        if greeting.is_some() {
            self.emitted += 1;
            if correct {
                self.correct_greetings += 1;
            } else {
                self.wrong_greetings += 1;
            }
        } else {
            self.abstentions += 1;
        }

        if let Some(expected_gender) = case.expected_gender {
            self.gender_labeled += 1;
            if let Some(gender) = prediction.gender_at(threshold) {
                self.gender_emitted += 1;
                if correct && gender == expected_gender {
                    self.gender_correct += 1;
                }
            }
        }
    }

    pub fn greeting_precision(self) -> Option<f64> {
        ratio(self.correct_greetings, self.emitted)
    }

    pub fn greeting_recall(self) -> Option<f64> {
        ratio(self.correct_greetings, self.person_cases)
    }

    pub fn abstention_rate(self) -> Option<f64> {
        ratio(self.abstentions, self.total)
    }

    pub fn organization_false_positive_rate(self) -> Option<f64> {
        ratio(self.organization_false_positives, self.organization_cases)
    }

    pub fn person_false_negative_rate(self) -> Option<f64> {
        ratio(self.person_false_negatives, self.person_cases)
    }

    pub fn gender_precision(self) -> Option<f64> {
        ratio(self.gender_correct, self.gender_emitted)
    }

    pub fn gender_coverage(self) -> Option<f64> {
        ratio(self.gender_emitted, self.gender_labeled)
    }

    pub fn gender_abstention_rate(self) -> Option<f64> {
        self.gender_coverage().map(|coverage| 1.0 - coverage)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CaseOutcome {
    Correct,
    Wrong,
    Abstained,
}

pub fn outcome(case: &Case, prediction: &RawInference, threshold: f64) -> CaseOutcome {
    let greeting = prediction.greeting_at(threshold);
    if greeting_matches(case.expected_greeting.as_deref(), greeting) {
        CaseOutcome::Correct
    } else if greeting.is_some() {
        CaseOutcome::Wrong
    } else {
        CaseOutcome::Abstained
    }
}

pub fn greeting_matches(expected: Option<&str>, actual: Option<&str>) -> bool {
    match (expected, actual) {
        (Some(expected), Some(actual)) => {
            normalize_greeting(expected) == normalize_greeting(actual)
        }
        (None, None) => true,
        _ => false,
    }
}

fn normalize_greeting(value: &str) -> String {
    use unicode_normalization::UnicodeNormalization;

    value
        .nfc()
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator != 0).then(|| numerator as f64 / denominator as f64)
}

#[cfg(test)]
mod tests {
    use crate::artifact::GenderHint;
    use crate::classifier::RawInference;
    use crate::dataset::Split;

    use super::*;

    fn case(expected: Option<&str>) -> Case {
        Case {
            id: "case".to_string(),
            split: Split::Test,
            category: "test".to_string(),
            input: "input".to_string(),
            country_hint: None,
            locale_hint: None,
            expected_greeting: expected.map(str::to_string),
            expected_gender: Some(GenderHint::Female),
            notes: String::new(),
        }
    }

    fn prediction(greeting: Option<&str>, confidence: f64) -> RawInference {
        RawInference {
            greeting_candidate: greeting.map(str::to_string),
            confidence,
            gender_hint: Some(GenderHint::Female),
            gender_confidence: 1.0,
        }
    }

    #[test]
    fn wrong_and_abstained_are_distinct() {
        assert_eq!(
            outcome(&case(Some("Alice")), &prediction(Some("Bob"), 1.0), 0.5),
            CaseOutcome::Wrong
        );
        assert_eq!(
            outcome(&case(Some("Alice")), &prediction(Some("Alice"), 0.1), 0.5),
            CaseOutcome::Abstained
        );
    }

    #[test]
    fn organization_abstention_is_correct() {
        assert_eq!(
            outcome(&case(None), &prediction(Some("Alice"), 0.1), 0.5),
            CaseOutcome::Correct
        );
    }
}
