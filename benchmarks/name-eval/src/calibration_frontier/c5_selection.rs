use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use sha2::{Digest, Sha256};

use super::*;

const DENSE_TARGETS: [f64; 20] = [
    0.9950, 0.9940, 0.9930, 0.9920, 0.9910, 0.9900, 0.9890, 0.9880, 0.9870, 0.9860, 0.9850, 0.9840,
    0.9830, 0.9820, 0.9810, 0.9800, 0.9775, 0.9750, 0.9725, 0.9700,
];
const PRODUCT_FLOORS: [(&str, f64); 3] = [
    ("conservative", 0.990),
    ("balanced", 0.985),
    ("permissive", 0.980),
];
const C5_NAME: &str = "C5-balanced-controlled-calibration-v1";
const C5_TRAINING_TARGET: f64 = 0.986;
const C5_QUALITY_MIN: f64 = 0.70;
const C5_RELIABILITY_MIN: f64 = 0.0;
const C5_ROLE_MIN: f64 = 0.0;
const C5_MARGIN_MIN: f64 = 0.50;
const C5_CONFIG_SHA256: &str = "427a15afb5c79846f80506f29b8d138a8c6969a8513c1d1dacf0ae1e491678b6";

#[derive(Clone)]
struct DenseFold {
    held_out: Population,
    family: Family,
    target: f64,
    policy: Policy,
    training_metrics: EmissionMetrics,
    held_out_metrics: EmissionMetrics,
}

#[derive(Clone)]
struct OofPoint {
    family: Family,
    target: f64,
    metrics: EmissionMetrics,
    signature: Vec<u64>,
    folds: Vec<DenseFold>,
}

#[derive(Clone)]
struct ProductCandidate {
    label: &'static str,
    precision_floor: f64,
    oof: OofPoint,
    full_development: OperatingPoint,
    validation_metrics: EmissionMetrics,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct C5OnlyAggregate {
    emitted: usize,
    correct: usize,
    wrong: usize,
    null_false_emissions: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SealedC4C5Comparison {
    c4: EmissionMetrics,
    c5: EmissionMetrics,
    c5_only: C5OnlyAggregate,
}

pub(crate) fn run_c5_selection(
    output: &Path,
    corpus: &impl EvidenceSource,
    holdouts: Vec<FrozenHoldout>,
    fixtures: &Path,
) -> Result<String> {
    let holdouts = validate_and_order_holdouts(holdouts)?;
    let proxy_rows = build_proxy_rows(corpus, &holdouts);
    let validation_rows = build_validation_rows(corpus, fixtures)?;
    assert_dataset_counts(&proxy_rows)?;
    assert_historical_checkpoints(&proxy_rows)?;
    assert_coarse_frontier(&proxy_rows)?;

    let folds = dense_logo_frontier(&proxy_rows)?;
    let points = aggregate_oof_points(&proxy_rows, &folds)?;
    let pareto = pareto_frontier(&points);
    let candidates = product_candidates(&proxy_rows, &validation_rows, &pareto)?;
    let selected = select_product_recommendation(&candidates)
        .ok_or("dense frontier did not establish the frozen C5 candidate")?;
    assert_frozen_c5(selected, &proxy_rows, &validation_rows)?;
    let outputs = build_selection_outputs(
        &holdouts,
        &proxy_rows,
        &validation_rows,
        &points,
        &pareto,
        &candidates,
        selected,
    )?;
    let repeated = build_selection_outputs(
        &holdouts,
        &proxy_rows,
        &validation_rows,
        &points,
        &pareto,
        &candidates,
        selected,
    )?;
    if outputs != repeated {
        return Err("C5 selection output is not deterministic".into());
    }
    assert_aggregate_only_outputs(&outputs)?;
    for (name, bytes) in &outputs {
        fs::write(output.join(name), bytes)?;
    }
    let report = outputs.get("report.md").ok_or("C5 report missing")?;
    Ok(String::from_utf8(report.clone())?)
}

pub(crate) fn run_sealed_c4_c5_comparison(
    output: &Path,
    corpus: &impl EvidenceSource,
    holdout: FrozenHoldout,
) -> Result<String> {
    assert_frozen_c5_configuration()?;
    let rows = holdout
        .cases
        .iter()
        .filter(|case| case.is_evaluable())
        .enumerate()
        .map(|(ordinal, case)| {
            feature_row(
                corpus,
                Population::Validation,
                ordinal,
                &case.display_name,
                case.expected_greeting(),
                nonempty(&case.country_hint),
                nonempty(&case.locale_hint),
            )
        })
        .collect::<Vec<_>>();
    let comparison = compare_c4_c5(&rows)?;
    let outputs = sealed_comparison_outputs(&holdout, comparison)?;
    let repeated = sealed_comparison_outputs(&holdout, comparison)?;
    if outputs != repeated {
        return Err("sealed C4/C5 aggregate output is not deterministic".into());
    }
    assert_sealed_aggregate_only(&outputs)?;
    fs::write(
        output.join("sealed_c4_c5_summary.csv"),
        outputs
            .get("sealed_c4_c5_summary.csv")
            .ok_or("sealed C4/C5 summary is missing")?,
    )?;
    let report = outputs
        .get("report.md")
        .ok_or("sealed C4/C5 report is missing")?;
    Ok(String::from_utf8(report.clone())?)
}

fn compare_c4_c5(rows: &[FeatureRow]) -> Result<SealedC4C5Comparison> {
    let mut c4 = EmissionMetrics::default();
    let mut c5 = EmissionMetrics::default();
    let mut c5_only = C5OnlyAggregate::default();
    for row in rows {
        let c4_emits = Policy::C4.emits(row);
        let c5_emits = frozen_c5_emits(row);
        if c4_emits && !c5_emits {
            return Err("frozen C5 violated its additive C4 invariant".into());
        }
        c4.observe(row, c4_emits);
        c5.observe(row, c5_emits);
        if !c4_emits && c5_emits {
            c5_only.observe(row);
        }
    }
    if c5.emitted != c4.emitted + c5_only.emitted
        || c5.correct != c4.correct + c5_only.correct
        || c5.wrong != c4.wrong + c5_only.wrong
        || c5.null_false_emissions != c4.null_false_emissions + c5_only.null_false_emissions
    {
        return Err("frozen C5 aggregate is not the exact additive C4 delta".into());
    }
    Ok(SealedC4C5Comparison { c4, c5, c5_only })
}

impl C5OnlyAggregate {
    fn observe(&mut self, row: &FeatureRow) {
        self.emitted += 1;
        if row.expected_greeting && row.selected_matches {
            self.correct += 1;
        } else {
            self.wrong += 1;
            if !row.expected_greeting {
                self.null_false_emissions += 1;
            }
        }
    }
}

fn assert_coarse_frontier(rows: &[FeatureRow]) -> Result<()> {
    let folds = logo_frontier(rows)?;
    let points = best_cross_validated_families(&folds);
    let expected: [(f64, usize, usize); 6] = [
        (0.995, 901, 3),
        (0.990, 2_139, 25),
        (0.980, 3_179, 68),
        (0.970, 3_436, 109),
        (0.950, 3_867, 175),
        (0.900, 4_917, 548),
    ];
    for (target, correct, wrong) in expected {
        let point = points
            .iter()
            .find(|point| point.target.to_bits() == target.to_bits())
            .ok_or_else(|| format!("historical C5 frontier target {target} is missing"))?;
        if (point.metrics.correct, point.metrics.wrong) != (correct, wrong) {
            return Err(format!(
                "historical C5 frontier target {target} changed: expected ({correct}, {wrong}), got ({}, {})",
                point.metrics.correct, point.metrics.wrong
            )
            .into());
        }
    }
    Ok(())
}

fn dense_logo_frontier(rows: &[FeatureRow]) -> Result<Vec<DenseFold>> {
    let mut results = Vec::new();
    for held_out in Population::PROXIES {
        let training = rows
            .iter()
            .filter(|row| row.population != held_out)
            .cloned()
            .collect::<Vec<_>>();
        let held_out_rows = rows
            .iter()
            .filter(|row| row.population == held_out)
            .collect::<Vec<_>>();
        if training.iter().any(|row| row.population == held_out) {
            return Err(format!("C5 LOGO leakage detected for {}", held_out.as_str()).into());
        }
        let score = score_frontier(&training);
        let controlled = controlled_frontier(&training);
        let model = fit_logistic(&training)?;
        let logistic = logistic_frontier(&training, &model, false);
        let additive = logistic_frontier(&training, &model, true);
        for (family, frontier) in [
            (Family::ScoreOnly, score.as_slice()),
            (Family::ControlledC4, controlled.as_slice()),
            (Family::Logistic, logistic.as_slice()),
            (Family::C4PlusLogistic, additive.as_slice()),
        ] {
            for target in DENSE_TARGETS {
                let Some(point) = select_point(frontier, target) else {
                    continue;
                };
                results.push(DenseFold {
                    held_out,
                    family,
                    target,
                    policy: point.policy.clone(),
                    training_metrics: point.metrics,
                    held_out_metrics: evaluate_policy(held_out_rows.iter().copied(), &point.policy),
                });
            }
        }
    }
    Ok(results)
}

fn aggregate_oof_points(rows: &[FeatureRow], folds: &[DenseFold]) -> Result<Vec<OofPoint>> {
    let mut points = Vec::new();
    for family in Family::ALL {
        for target in DENSE_TARGETS {
            let matching = folds
                .iter()
                .filter(|fold| fold.family == family && fold.target.to_bits() == target.to_bits())
                .cloned()
                .collect::<Vec<_>>();
            if matching.is_empty() {
                continue;
            }
            if matching.len() != Population::PROXIES.len() {
                continue;
            }
            let populations = matching
                .iter()
                .map(|fold| fold.held_out)
                .collect::<BTreeSet<_>>();
            if populations.len() != Population::PROXIES.len() {
                return Err(format!(
                    "C5 OOF point has duplicate held-out generations for {} at {target:.4}",
                    family.as_str()
                )
                .into());
            }
            let mut metrics = EmissionMetrics::default();
            for fold in &matching {
                metrics.add(fold.held_out_metrics);
            }
            points.push(OofPoint {
                family,
                target,
                metrics,
                signature: oof_signature(rows, &matching)?,
                folds: matching,
            });
        }
    }
    Ok(points)
}

fn oof_signature(rows: &[FeatureRow], folds: &[DenseFold]) -> Result<Vec<u64>> {
    let policies = folds
        .iter()
        .map(|fold| (fold.held_out, &fold.policy))
        .collect::<BTreeMap<_, _>>();
    if policies.len() != Population::PROXIES.len() {
        return Err("C5 OOF signature requires one policy per generation".into());
    }
    let mut signature = vec![0_u64; rows.len().div_ceil(64)];
    for (index, row) in rows.iter().enumerate() {
        let policy = policies
            .get(&row.population)
            .ok_or("C5 OOF row has no held-out policy")?;
        if policy.emits(row) {
            signature[index / 64] |= 1_u64 << (index % 64);
        }
    }
    Ok(signature)
}

fn pareto_frontier(points: &[OofPoint]) -> Vec<OofPoint> {
    let mut unique = BTreeMap::<Vec<u64>, OofPoint>::new();
    for point in points {
        match unique.get(&point.signature) {
            Some(current) if compare_representatives(current, point) != Ordering::Less => {}
            _ => {
                unique.insert(point.signature.clone(), point.clone());
            }
        }
    }
    let unique = unique.into_values().collect::<Vec<_>>();
    let mut pareto = unique
        .iter()
        .filter(|candidate| {
            !unique
                .iter()
                .any(|other| !std::ptr::eq(*candidate, other) && dominates(other, candidate))
        })
        .cloned()
        .collect::<Vec<_>>();
    pareto.sort_by(|left, right| {
        precision_cmp(right.metrics, left.metrics)
            .then_with(|| left.metrics.correct.cmp(&right.metrics.correct))
            .then_with(|| left.family.cmp(&right.family))
            .then_with(|| right.target.total_cmp(&left.target))
    });
    pareto
}

fn compare_representatives(left: &OofPoint, right: &OofPoint) -> Ordering {
    family_complexity(right.family)
        .cmp(&family_complexity(left.family))
        .then_with(|| left.target.total_cmp(&right.target))
        .then_with(|| right.family.cmp(&left.family))
}

fn dominates(left: &OofPoint, right: &OofPoint) -> bool {
    let precision = precision_cmp(left.metrics, right.metrics);
    let recall = left.metrics.correct.cmp(&right.metrics.correct);
    precision != Ordering::Less
        && recall != Ordering::Less
        && (precision == Ordering::Greater || recall == Ordering::Greater)
}

fn precision_cmp(left: EmissionMetrics, right: EmissionMetrics) -> Ordering {
    match (left.emitted, right.emitted) {
        (0, 0) => Ordering::Equal,
        (0, _) => Ordering::Less,
        (_, 0) => Ordering::Greater,
        _ => ((left.correct as u128) * (right.emitted as u128))
            .cmp(&((right.correct as u128) * (left.emitted as u128))),
    }
}

fn family_complexity(family: Family) -> usize {
    match family {
        Family::ScoreOnly => 1,
        Family::ControlledC4 => 4,
        Family::Logistic => FEATURE_COUNT + 2,
        Family::C4PlusLogistic => FEATURE_COUNT + 3,
    }
}

fn product_candidates(
    proxy_rows: &[FeatureRow],
    validation_rows: &[FeatureRow],
    pareto: &[OofPoint],
) -> Result<Vec<ProductCandidate>> {
    let mut candidates = Vec::new();
    let mut signatures = BTreeSet::new();
    for (label, precision_floor) in PRODUCT_FLOORS {
        let Some(oof) = product_point_at_floor(pareto, precision_floor).cloned() else {
            continue;
        };
        if !signatures.insert(oof.signature.clone()) {
            continue;
        }
        let full_development = full_development_point(proxy_rows, oof.family, oof.target)?;
        let validation_metrics = evaluate_policy(validation_rows.iter(), &full_development.policy);
        candidates.push(ProductCandidate {
            label,
            precision_floor,
            oof,
            full_development,
            validation_metrics,
        });
    }
    Ok(candidates)
}

fn product_point_at_floor(pareto: &[OofPoint], precision_floor: f64) -> Option<&OofPoint> {
    pareto
        .iter()
        .filter(|point| {
            point
                .metrics
                .precision()
                .is_some_and(|precision| precision >= precision_floor)
        })
        .max_by(|left, right| compare_product_points(left, right))
}

fn compare_product_points(left: &OofPoint, right: &OofPoint) -> Ordering {
    left.metrics
        .correct
        .cmp(&right.metrics.correct)
        .then_with(|| right.metrics.wrong.cmp(&left.metrics.wrong))
        .then_with(|| precision_cmp(left.metrics, right.metrics))
        .then_with(|| family_complexity(right.family).cmp(&family_complexity(left.family)))
        .then_with(|| left.target.total_cmp(&right.target))
}

fn full_development_point(
    rows: &[FeatureRow],
    family: Family,
    target: f64,
) -> Result<OperatingPoint> {
    let score = score_frontier(rows);
    let controlled = controlled_frontier(rows);
    let model = fit_logistic(rows)?;
    let logistic = logistic_frontier(rows, &model, false);
    let additive = logistic_frontier(rows, &model, true);
    let frontier = match family {
        Family::ScoreOnly => &score,
        Family::ControlledC4 => &controlled,
        Family::Logistic => &logistic,
        Family::C4PlusLogistic => &additive,
    };
    select_point(frontier, target).cloned().ok_or_else(|| {
        format!(
            "no full-development {} point at {target:.4}",
            family.as_str()
        )
        .into()
    })
}

fn select_product_recommendation(candidates: &[ProductCandidate]) -> Option<&ProductCandidate> {
    candidates
        .iter()
        .find(|candidate| candidate.label == "balanced")
}

fn frozen_c5_policy() -> Policy {
    Policy::Controlled {
        quality: C5_QUALITY_MIN,
        reliability: C5_RELIABILITY_MIN,
        role: C5_ROLE_MIN,
        margin: C5_MARGIN_MIN,
    }
}

pub(super) fn frozen_c5_emits(row: &FeatureRow) -> bool {
    frozen_c5_policy().emits(row)
}

fn frozen_c5_config() -> String {
    format!(
        "schema=1;name={C5_NAME};family=controlled_c4;training_target={C5_TRAINING_TARGET:.17};quality={C5_QUALITY_MIN:.17};reliability={C5_RELIABILITY_MIN:.17};role={C5_ROLE_MIN:.17};margin={C5_MARGIN_MIN:.17}"
    )
}

fn frozen_c5_digest() -> String {
    format!("{:x}", Sha256::digest(frozen_c5_config().as_bytes()))
}

fn assert_frozen_c5_configuration() -> Result<()> {
    let parameters = frozen_c5_policy().parameters();
    if parameters != "quality=0.70;reliability=0.00;role=0.00;margin=0.50" {
        return Err(format!("frozen C5 parameters changed: {parameters}").into());
    }
    let digest = frozen_c5_digest();
    if digest != C5_CONFIG_SHA256 {
        return Err(format!(
            "frozen C5 configuration digest changed: expected {C5_CONFIG_SHA256}, got {digest}"
        )
        .into());
    }
    Ok(())
}

fn assert_frozen_c5(
    selected: &ProductCandidate,
    proxy_rows: &[FeatureRow],
    validation_rows: &[FeatureRow],
) -> Result<()> {
    if selected.label != "balanced"
        || selected.oof.family != Family::ControlledC4
        || selected.oof.target.to_bits() != C5_TRAINING_TARGET.to_bits()
        || selected.full_development.policy.parameters() != frozen_c5_policy().parameters()
    {
        return Err(format!(
            "selected C5 no longer matches the frozen product decision: label={}, family={}, target={:.17}, parameters={}",
            selected.label,
            selected.oof.family.as_str(),
            selected.oof.target,
            selected.full_development.policy.parameters()
        )
        .into());
    }
    let oof = selected.oof.metrics;
    if (
        oof.emitted,
        oof.correct,
        oof.wrong,
        oof.null_false_emissions,
    ) != (2_629, 2_591, 38, 11)
    {
        return Err(format!(
            "frozen C5 OOF checkpoint changed: got ({}, {}, {}, {})",
            oof.emitted, oof.correct, oof.wrong, oof.null_false_emissions
        )
        .into());
    }
    let validation = selected.validation_metrics;
    if (
        validation.emitted,
        validation.correct,
        validation.wrong,
        validation.null_false_emissions,
    ) != (24_243, 23_596, 647, 0)
    {
        return Err(format!(
            "frozen C5 VALIDATION checkpoint changed: got ({}, {}, {}, {})",
            validation.emitted,
            validation.correct,
            validation.wrong,
            validation.null_false_emissions
        )
        .into());
    }
    let development = selected.full_development.metrics;
    if (
        development.emitted,
        development.correct,
        development.wrong,
        development.null_false_emissions,
    ) != (2_560, 2_527, 33, 10)
    {
        return Err(format!(
            "frozen C5 full-development checkpoint changed: got ({}, {}, {}, {})",
            development.emitted,
            development.correct,
            development.wrong,
            development.null_false_emissions
        )
        .into());
    }
    if evaluate_frozen_c5(proxy_rows) != development
        || evaluate_frozen_c5(validation_rows) != validation
    {
        return Err("frozen C5 implementation does not reproduce the selected policy".into());
    }
    assert_frozen_c5_configuration()
}

fn evaluate_frozen_c5(rows: &[FeatureRow]) -> EmissionMetrics {
    let mut metrics = EmissionMetrics::default();
    for row in rows {
        metrics.observe(row, frozen_c5_emits(row));
    }
    metrics
}

fn sealed_comparison_outputs(
    holdout: &FrozenHoldout,
    comparison: SealedC4C5Comparison,
) -> Result<BTreeMap<String, Vec<u8>>> {
    let mut outputs = BTreeMap::new();
    outputs.insert(
        "sealed_c4_c5_summary.csv".to_string(),
        sealed_comparison_csv(comparison)?,
    );
    outputs.insert(
        "report.md".to_string(),
        sealed_comparison_report(holdout, comparison).into_bytes(),
    );
    Ok(outputs)
}

fn sealed_comparison_csv(comparison: SealedC4C5Comparison) -> Result<Vec<u8>> {
    csv_bytes(|writer| {
        writer.write_record([
            "record",
            "emitted",
            "correct",
            "wrong",
            "null_fp",
            "precision",
            "wilson_95_lower",
            "wilson_95_upper",
            "recall",
            "abstention_rate",
            "false_abstentions",
            "winner_correct_but_abstained",
            "correct_per_wrong",
        ])?;
        write_sealed_policy_row(writer, "c4", comparison.c4)?;
        write_sealed_policy_row(writer, "c5", comparison.c5)?;
        writer.write_record([
            "c5_only",
            &comparison.c5_only.emitted.to_string(),
            &comparison.c5_only.correct.to_string(),
            &comparison.c5_only.wrong.to_string(),
            &comparison.c5_only.null_false_emissions.to_string(),
            &optional_float(ratio(
                comparison.c5_only.correct,
                comparison.c5_only.emitted,
            )),
            "",
            "",
            "",
            "",
            &(comparison.c4.false_abstentions - comparison.c5.false_abstentions).to_string(),
            "",
            &correct_per_wrong(comparison.c5_only.correct, comparison.c5_only.wrong),
        ])?;
        Ok(())
    })
}

fn write_sealed_policy_row(
    writer: &mut csv::Writer<Vec<u8>>,
    name: &str,
    metrics: EmissionMetrics,
) -> Result<()> {
    let interval = wilson_interval(metrics.correct, metrics.emitted);
    writer.write_record([
        name,
        &metrics.emitted.to_string(),
        &metrics.correct.to_string(),
        &metrics.wrong.to_string(),
        &metrics.null_false_emissions.to_string(),
        &optional_float(metrics.precision()),
        &interval.map_or_else(String::new, |value| float(value.lower)),
        &interval.map_or_else(String::new, |value| float(value.upper)),
        &optional_float(metrics.recall()),
        &optional_float(metrics.abstention_rate()),
        &metrics.false_abstentions.to_string(),
        &metrics.winner_correct_but_abstained.to_string(),
        &correct_per_wrong(metrics.correct, metrics.wrong),
    ])?;
    Ok(())
}

fn sealed_comparison_report(holdout: &FrozenHoldout, comparison: SealedC4C5Comparison) -> String {
    let mut report = String::new();
    writeln!(report, "# Sealed C4/C5 REAL_PROXY_V6 comparison\n").unwrap();
    writeln!(
        report,
        "The holdout was checksum-verified as `{}` before the sole classifier invocation. It contains {} source rows: {} evaluable and {} skipped, with {} expected greetings and {} expected abstentions. Only aggregate counts were produced.\n",
        holdout.manifest.holdout_sha256,
        holdout.manifest.total_cases,
        holdout.manifest.evaluable_cases,
        holdout.manifest.skipped_cases,
        holdout.manifest.expected_greetings,
        holdout.manifest.expected_abstentions,
    )
    .unwrap();
    writeln!(
        report,
        "| Classifier | Emitted | Correct | Wrong | NULL FP | Precision | Wilson 95% | Recall | Abstention | False abstentions | Correct veto-free winner rejected | Correct per wrong |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    .unwrap();
    for (name, metrics) in [("C4", comparison.c4), ("C5", comparison.c5)] {
        let interval = wilson_interval(metrics.correct, metrics.emitted).map_or_else(
            || "n/a".to_string(),
            |value| {
                format!(
                    "{}–{}",
                    percent(Some(value.lower)),
                    percent(Some(value.upper))
                )
            },
        );
        writeln!(
            report,
            "| {name} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            metrics.emitted,
            metrics.correct,
            metrics.wrong,
            metrics.null_false_emissions,
            percent(metrics.precision()),
            interval,
            percent(metrics.recall()),
            percent(metrics.abstention_rate()),
            metrics.false_abstentions,
            metrics.winner_correct_but_abstained,
            correct_per_wrong(metrics.correct, metrics.wrong),
        )
        .unwrap();
    }
    let reduction = comparison.c4.false_abstentions - comparison.c5.false_abstentions;
    writeln!(report, "\n## C5-only delta\n").unwrap();
    writeln!(
        report,
        "C5 added {} emissions: {} correct, {} wrong, and {} expected-NULL false emissions. That is {} additional correct greetings per additional wrong greeting. False abstentions fell by {}.\n",
        comparison.c5_only.emitted,
        comparison.c5_only.correct,
        comparison.c5_only.wrong,
        comparison.c5_only.null_false_emissions,
        correct_per_wrong(comparison.c5_only.correct, comparison.c5_only.wrong),
        reduction,
    )
    .unwrap();
    writeln!(report, "## Interpretation boundary\n").unwrap();
    writeln!(report, "This is one-shot machine-consensus proxy evidence, not a worldwide population-precision claim. The aggregate checkpoint must be interpreted before any row is inspected. Production remains on C4; promotion requires a separate change.\n").unwrap();
    report
}

fn correct_per_wrong(correct: usize, wrong: usize) -> String {
    if wrong == 0 {
        "no observed wrong emissions".to_string()
    } else {
        format!("{:.2}", correct as f64 / wrong as f64)
    }
}

fn assert_sealed_aggregate_only(outputs: &BTreeMap<String, Vec<u8>>) -> Result<()> {
    if outputs.keys().map(String::as_str).collect::<Vec<_>>()
        != ["report.md", "sealed_c4_c5_summary.csv"]
    {
        return Err("sealed C4/C5 comparator produced an unexpected output".into());
    }
    for (name, bytes) in outputs {
        let text = std::str::from_utf8(bytes)?;
        for forbidden in [
            "display_name",
            "source_id",
            "source_row",
            "user_name",
            "expected_greeting",
            "candidate_quality",
            "winner_margin",
            "decision_score",
        ] {
            if text.contains(forbidden) {
                return Err(
                    format!("sealed C4/C5 aggregate output {name} contains {forbidden}").into(),
                );
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_selection_outputs(
    holdouts: &[FrozenHoldout],
    proxy_rows: &[FeatureRow],
    validation_rows: &[FeatureRow],
    points: &[OofPoint],
    pareto: &[OofPoint],
    candidates: &[ProductCandidate],
    selected: &ProductCandidate,
) -> Result<BTreeMap<String, Vec<u8>>> {
    let mut outputs = BTreeMap::new();
    outputs.insert(
        "dense_oof_frontier.csv".to_string(),
        oof_frontier_csv(points)?,
    );
    outputs.insert("pareto_frontier.csv".to_string(), oof_frontier_csv(pareto)?);
    outputs.insert(
        "candidate_operating_points.csv".to_string(),
        candidate_points_csv(candidates)?,
    );
    outputs.insert(
        "per_generation_stability.csv".to_string(),
        per_generation_csv(candidates)?,
    );
    outputs.insert(
        "cost_sensitive_points.csv".to_string(),
        cost_sensitive_csv(proxy_rows, points)?,
    );
    outputs.insert(
        "synthetic_validation.csv".to_string(),
        validation_csv(validation_rows, candidates)?,
    );
    outputs.insert("frozen_c5_config.csv".to_string(), frozen_c5_csv(selected)?);
    outputs.insert(
        "report.md".to_string(),
        build_selection_report(
            holdouts,
            proxy_rows,
            validation_rows,
            pareto,
            candidates,
            selected,
        )?
        .into_bytes(),
    );
    Ok(outputs)
}

fn assert_aggregate_only_outputs(outputs: &BTreeMap<String, Vec<u8>>) -> Result<()> {
    for (name, bytes) in outputs {
        let text = std::str::from_utf8(bytes)?;
        for forbidden in ["display_name", "source_id", "source_row", "user_name"] {
            if text.contains(forbidden) {
                return Err(format!("C5 aggregate output {name} contains {forbidden}").into());
            }
        }
    }
    Ok(())
}

fn oof_frontier_csv(points: &[OofPoint]) -> Result<Vec<u8>> {
    csv_bytes(|writer| {
        write_c5_metrics_header(writer, &["family", "training_target", "parameters_by_fold"])?;
        for point in points {
            let parameters = point
                .folds
                .iter()
                .map(|fold| format!("{}={}", fold.held_out.as_str(), fold.policy.parameters()))
                .collect::<Vec<_>>()
                .join("|");
            write_c5_metrics(
                writer,
                &[point.family.as_str(), &float(point.target), &parameters],
                point.metrics,
            )?;
        }
        Ok(())
    })
}

fn candidate_points_csv(candidates: &[ProductCandidate]) -> Result<Vec<u8>> {
    csv_bytes(|writer| {
        write_c5_metrics_header(
            writer,
            &[
                "label",
                "observed_precision_floor",
                "family",
                "training_target",
                "full_development_parameters",
            ],
        )?;
        for candidate in candidates {
            write_c5_metrics(
                writer,
                &[
                    candidate.label,
                    &float(candidate.precision_floor),
                    candidate.oof.family.as_str(),
                    &float(candidate.oof.target),
                    &candidate.full_development.policy.parameters(),
                ],
                candidate.oof.metrics,
            )?;
        }
        Ok(())
    })
}

fn per_generation_csv(candidates: &[ProductCandidate]) -> Result<Vec<u8>> {
    csv_bytes(|writer| {
        write_c5_metrics_header(
            writer,
            &[
                "label",
                "family",
                "training_target",
                "held_out",
                "training_precision",
                "training_recall",
            ],
        )?;
        for candidate in candidates {
            for fold in &candidate.oof.folds {
                write_c5_metrics(
                    writer,
                    &[
                        candidate.label,
                        candidate.oof.family.as_str(),
                        &float(candidate.oof.target),
                        fold.held_out.as_str(),
                        &optional_float(fold.training_metrics.precision()),
                        &optional_float(fold.training_metrics.recall()),
                    ],
                    fold.held_out_metrics,
                )?;
            }
        }
        Ok(())
    })
}

fn cost_sensitive_csv(rows: &[FeatureRow], points: &[OofPoint]) -> Result<Vec<u8>> {
    csv_bytes(|writer| {
        write_c5_metrics_header(writer, &["wrong_cost", "policy", "training_target", "loss"])?;
        let c4 = evaluate_policy(rows.iter(), &Policy::C4);
        for cost in COSTS {
            let selected = points.iter().min_by(|left, right| {
                product_loss(left.metrics, cost)
                    .cmp(&product_loss(right.metrics, cost))
                    .then_with(|| right.metrics.correct.cmp(&left.metrics.correct))
                    .then_with(|| left.metrics.wrong.cmp(&right.metrics.wrong))
            });
            let c4_loss = product_loss(c4, cost);
            if let Some(point) =
                selected.filter(|point| product_loss(point.metrics, cost) < c4_loss)
            {
                write_c5_metrics(
                    writer,
                    &[
                        &cost.to_string(),
                        point.family.as_str(),
                        &float(point.target),
                        &product_loss(point.metrics, cost).to_string(),
                    ],
                    point.metrics,
                )?;
            } else {
                write_c5_metrics(
                    writer,
                    &[&cost.to_string(), "c4", "", &c4_loss.to_string()],
                    c4,
                )?;
            }
        }
        Ok(())
    })
}

fn product_loss(metrics: EmissionMetrics, wrong_cost: usize) -> usize {
    metrics.false_abstentions + wrong_cost * metrics.wrong
}

fn validation_csv(rows: &[FeatureRow], candidates: &[ProductCandidate]) -> Result<Vec<u8>> {
    csv_bytes(|writer| {
        write_c5_metrics_header(writer, &["policy", "parameters"])?;
        write_c5_metrics(
            writer,
            &["c4", "frozen_c4"],
            evaluate_policy(rows.iter(), &Policy::C4),
        )?;
        for candidate in candidates {
            write_c5_metrics(
                writer,
                &[
                    candidate.label,
                    &candidate.full_development.policy.parameters(),
                ],
                candidate.validation_metrics,
            )?;
        }
        Ok(())
    })
}

fn frozen_c5_csv(selected: &ProductCandidate) -> Result<Vec<u8>> {
    csv_bytes(|writer| {
        write_c5_metrics_header(
            writer,
            &[
                "name",
                "family",
                "training_target",
                "parameters",
                "config_sha256",
                "evidence_scope",
                "status",
            ],
        )?;
        write_c5_metrics(
            writer,
            &[
                C5_NAME,
                selected.oof.family.as_str(),
                &float(C5_TRAINING_TARGET),
                &frozen_c5_policy().parameters(),
                &frozen_c5_digest(),
                "all_spent_v1_v2_v3_v4_v5",
                "development_candidate_requires_v6",
            ],
            selected.full_development.metrics,
        )?;
        Ok(())
    })
}

fn write_c5_metrics_header(writer: &mut csv::Writer<Vec<u8>>, prefix: &[&str]) -> Result<()> {
    let mut header = prefix.iter().map(ToString::to_string).collect::<Vec<_>>();
    header.extend(
        [
            "rows",
            "expected_greetings",
            "expected_nulls",
            "emitted",
            "correct",
            "wrong",
            "null_fp",
            "precision",
            "wilson_95_lower",
            "wilson_95_upper",
            "recall",
            "abstention_rate",
            "false_abstentions",
            "winner_correct_but_abstained",
            "wrong_per_100_correct",
            "correct_per_wrong",
        ]
        .into_iter()
        .map(str::to_string),
    );
    writer.write_record(header)?;
    Ok(())
}

fn write_c5_metrics(
    writer: &mut csv::Writer<Vec<u8>>,
    prefix: &[&str],
    metrics: EmissionMetrics,
) -> Result<()> {
    let interval = wilson_interval(metrics.correct, metrics.emitted);
    let mut record = prefix.iter().map(ToString::to_string).collect::<Vec<_>>();
    record.extend([
        metrics.rows.to_string(),
        metrics.expected_greetings.to_string(),
        metrics.expected_nulls.to_string(),
        metrics.emitted.to_string(),
        metrics.correct.to_string(),
        metrics.wrong.to_string(),
        metrics.null_false_emissions.to_string(),
        optional_float(metrics.precision()),
        interval.map_or_else(String::new, |value| float(value.lower)),
        interval.map_or_else(String::new, |value| float(value.upper)),
        optional_float(metrics.recall()),
        optional_float(metrics.abstention_rate()),
        metrics.false_abstentions.to_string(),
        metrics.winner_correct_but_abstained.to_string(),
        optional_float(ratio(100 * metrics.wrong, metrics.correct)),
        if metrics.wrong == 0 {
            "infinite".to_string()
        } else {
            float(metrics.correct as f64 / metrics.wrong as f64)
        },
    ]);
    writer.write_record(record)?;
    Ok(())
}

fn build_selection_report(
    holdouts: &[FrozenHoldout],
    rows: &[FeatureRow],
    validation_rows: &[FeatureRow],
    pareto: &[OofPoint],
    candidates: &[ProductCandidate],
    selected: &ProductCandidate,
) -> Result<String> {
    let mut report = String::new();
    writeln!(report, "# C5 dense operating-point selection\n")?;
    writeln!(
        report,
        "C4 remains frozen production behavior. This development-only study reuses the existing seven-feature calibration families over spent REAL_PROXY_V1-V5. It does not use position, capitalization, morphology, TEST, or V6.\n"
    )?;
    writeln!(
        report,
        "Earlier iterations required zero observed new errors. This selection instead treats false abstention as a real UX loss and measures the dense empirical tradeoff. Generic position was marginal and not promoted; capitalization and character morphology were harmful/no-value in their tested forms; locale-aware ordering remains unresolved. None enters C5.\n"
    )?;
    writeln!(report, "## Development population\n")?;
    writeln!(
        report,
        "| Population | Evaluable | Expected greeting | Expected NULL |\n| --- | ---: | ---: | ---: |"
    )?;
    for holdout in holdouts {
        let population = Population::from_digest(&holdout.manifest.holdout_sha256)
            .expect("validated proxy digest");
        writeln!(
            report,
            "| {} | {} | {} | {} |",
            population.as_str(),
            holdout.manifest.evaluable_cases,
            holdout.manifest.expected_greetings,
            holdout.manifest.expected_abstentions,
        )?;
    }
    writeln!(
        report,
        "| **Combined** | **7,808** | **6,478** | **1,330** |\n"
    )?;

    let c4 = evaluate_policy(rows.iter(), &Policy::C4);
    writeln!(report, "## C4 reference\n")?;
    writeln!(
        report,
        "C4 emits {} correct and {} wrong greetings at {} precision and {} recall. It falsely abstains on {} expected greetings, including {} ({}) whose selected winner is already correct and veto-free.\n",
        c4.correct,
        c4.wrong,
        percent(c4.precision()),
        percent(c4.recall()),
        c4.false_abstentions,
        c4.winner_correct_but_abstained,
        percent(ratio(
            c4.winner_correct_but_abstained,
            c4.expected_greetings
        )),
    )?;

    writeln!(report, "## Dense OOF Pareto frontier\n")?;
    writeln!(
        report,
        "Each point aggregates disjoint generation-held-out predictions. The target is the training-fold selection target; observed OOF precision is reported without interpolation.\n"
    )?;
    writeln!(
        report,
        "| Family | Training target | Precision | Wilson 95% | Recall | Correct | Wrong | NULL FP | False abstentions | Correct winner rejected | Correct / wrong |\n| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )?;
    for point in pareto {
        let interval = wilson_interval(point.metrics.correct, point.metrics.emitted)
            .ok_or("Pareto point has no emissions")?;
        writeln!(
            report,
            "| {} | {} | {} | {}-{} | {} | {} | {} | {} | {} | {} | {} |",
            point.family.as_str(),
            percent(Some(point.target)),
            percent(point.metrics.precision()),
            percent(Some(interval.lower)),
            percent(Some(interval.upper)),
            percent(point.metrics.recall()),
            point.metrics.correct,
            point.metrics.wrong,
            point.metrics.null_false_emissions,
            point.metrics.false_abstentions,
            point.metrics.winner_correct_but_abstained,
            if point.metrics.wrong == 0 {
                "infinite".to_string()
            } else {
                format!(
                    "{:.2}",
                    point.metrics.correct as f64 / point.metrics.wrong as f64
                )
            },
        )?;
    }

    writeln!(report, "\n## Product candidates\n")?;
    writeln!(
        report,
        "| Label | Observed floor | Family | Training target | OOF precision | Wilson 95% | Recall | Correct | Wrong | NULL FP | False abstentions | Correct winner rejected | Minimum generation precision | Maximum generation wrong | Recall range | VALIDATION precision | VALIDATION recall |\n| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |"
    )?;
    for candidate in candidates {
        let interval =
            wilson_interval(candidate.oof.metrics.correct, candidate.oof.metrics.emitted)
                .ok_or("candidate point has no emissions")?;
        let min_precision = candidate
            .oof
            .folds
            .iter()
            .filter_map(|fold| fold.held_out_metrics.precision())
            .min_by(f64::total_cmp);
        let min_recall = candidate
            .oof
            .folds
            .iter()
            .filter_map(|fold| fold.held_out_metrics.recall())
            .min_by(f64::total_cmp);
        let max_recall = candidate
            .oof
            .folds
            .iter()
            .filter_map(|fold| fold.held_out_metrics.recall())
            .max_by(f64::total_cmp);
        let maximum_wrong = candidate
            .oof
            .folds
            .iter()
            .map(|fold| fold.held_out_metrics.wrong)
            .max()
            .unwrap_or(0);
        writeln!(
            report,
            "| {} | {} | {} | {} | {} | {}-{} | {} | {} | {} | {} | {} | {} | {} | {} | {}-{} | {} | {} |",
            candidate.label,
            percent(Some(candidate.precision_floor)),
            candidate.oof.family.as_str(),
            percent(Some(candidate.oof.target)),
            percent(candidate.oof.metrics.precision()),
            percent(Some(interval.lower)),
            percent(Some(interval.upper)),
            percent(candidate.oof.metrics.recall()),
            candidate.oof.metrics.correct,
            candidate.oof.metrics.wrong,
            candidate.oof.metrics.null_false_emissions,
            candidate.oof.metrics.false_abstentions,
            candidate.oof.metrics.winner_correct_but_abstained,
            percent(min_precision),
            maximum_wrong,
            percent(min_recall),
            percent(max_recall),
            percent(candidate.validation_metrics.precision()),
            percent(candidate.validation_metrics.recall()),
        )?;
    }

    writeln!(report, "\n## Per-generation candidate stability\n")?;
    writeln!(
        report,
        "| Candidate | Held out | Emitted | Correct | Wrong | NULL FP | Precision | Recall |\n| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    )?;
    for candidate in candidates {
        for fold in &candidate.oof.folds {
            writeln!(
                report,
                "| {} | {} | {} | {} | {} | {} | {} | {} |",
                candidate.label,
                fold.held_out.as_str(),
                fold.held_out_metrics.emitted,
                fold.held_out_metrics.correct,
                fold.held_out_metrics.wrong,
                fold.held_out_metrics.null_false_emissions,
                percent(fold.held_out_metrics.precision()),
                percent(fold.held_out_metrics.recall()),
            )?;
        }
    }

    writeln!(report, "\n## Synthetic VALIDATION\n")?;
    writeln!(
        report,
        "| Policy | Emitted | Correct | Wrong | NULL FP | Precision | Recall |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    )?;
    let c4_validation = evaluate_policy(validation_rows.iter(), &Policy::C4);
    writeln!(
        report,
        "| C4 | {} | {} | {} | {} | {} | {} |",
        c4_validation.emitted,
        c4_validation.correct,
        c4_validation.wrong,
        c4_validation.null_false_emissions,
        percent(c4_validation.precision()),
        percent(c4_validation.recall()),
    )?;
    for candidate in candidates {
        writeln!(
            report,
            "| {} | {} | {} | {} | {} | {} | {} |",
            candidate.label,
            candidate.validation_metrics.emitted,
            candidate.validation_metrics.correct,
            candidate.validation_metrics.wrong,
            candidate.validation_metrics.null_false_emissions,
            percent(candidate.validation_metrics.precision()),
            percent(candidate.validation_metrics.recall()),
        )?;
    }

    writeln!(report, "\n## Cost-sensitive view\n")?;
    writeln!(
        report,
        "| Wrong cost | Preferred OOF policy | Loss | Correct | Wrong | Recall |\n| ---: | --- | ---: | ---: | ---: | ---: |"
    )?;
    for cost in COSTS {
        let selected_cost = pareto.iter().min_by(|left, right| {
            product_loss(left.metrics, cost)
                .cmp(&product_loss(right.metrics, cost))
                .then_with(|| right.metrics.correct.cmp(&left.metrics.correct))
        });
        let (name, metrics) = selected_cost
            .filter(|point| product_loss(point.metrics, cost) < product_loss(c4, cost))
            .map_or(("c4".to_string(), c4), |point| {
                (
                    format!(
                        "{} @ {}",
                        point.family.as_str(),
                        percent(Some(point.target))
                    ),
                    point.metrics,
                )
            });
        writeln!(
            report,
            "| {}x | {} | {} | {} | {} | {} |",
            cost,
            name,
            product_loss(metrics, cost),
            metrics.correct,
            metrics.wrong,
            percent(metrics.recall()),
        )?;
    }

    writeln!(report, "\n## Frozen C5 development candidate\n")?;
    let c5 = selected.full_development.metrics;
    let c5_interval = wilson_interval(selected.oof.metrics.correct, selected.oof.metrics.emitted)
        .ok_or("selected C5 has no emissions")?;
    let permissive = candidates
        .iter()
        .find(|candidate| candidate.label == "permissive")
        .ok_or("permissive comparison point is missing")?;
    writeln!(
        report,
        "The selected product point is **{}**: {} at a {} training target, with {} OOF precision, {} OOF recall, and a {}-{} Wilson interval. Relative to C4 it adds {} correct emissions and {} wrong emissions in the OOF development evidence, while reducing correct veto-free winner rejections from {} to {}.\n",
        selected.label,
        selected.oof.family.as_str(),
        percent(Some(selected.oof.target)),
        percent(selected.oof.metrics.precision()),
        percent(selected.oof.metrics.recall()),
        percent(Some(c5_interval.lower)),
        percent(Some(c5_interval.upper)),
        selected.oof.metrics.correct.saturating_sub(c4.correct),
        selected.oof.metrics.wrong.saturating_sub(c4.wrong),
        c4.winner_correct_but_abstained,
        selected.oof.metrics.winner_correct_but_abstained,
    )?;
    writeln!(
        report,
        "The permissive point adds only {} correct emissions beyond balanced while adding {} wrong emissions. On separate synthetic VALIDATION it is also dominated by balanced: balanced emits {} correct / {} wrong at {} precision and {} recall. This makes balanced the empirical knee rather than merely the middle label.\n",
        permissive
            .oof
            .metrics
            .correct
            .saturating_sub(selected.oof.metrics.correct),
        permissive
            .oof
            .metrics
            .wrong
            .saturating_sub(selected.oof.metrics.wrong),
        selected.validation_metrics.correct,
        selected.validation_metrics.wrong,
        percent(selected.validation_metrics.precision()),
        percent(selected.validation_metrics.recall()),
    )?;
    writeln!(
        report,
        "The full-development frozen policy is `{}`. Its canonical configuration digest is `{}`. On all spent proxy rows it emits {} correct / {} wrong at {} precision and {} recall. These are development metrics, not validation. Production remains on C4 until untouched V6 compares C4 and C5 once.\n",
        frozen_c5_policy().parameters(),
        frozen_c5_digest(),
        c5.correct,
        c5.wrong,
        percent(c5.precision()),
        percent(c5.recall()),
    )?;
    writeln!(
        report,
        "V1 has different annotation provenance from V2-V5. Wilson intervals describe case-level binomial uncertainty only and are not worldwide precision guarantees. V6 remains untouched.\n"
    )?;
    writeln!(report, "## Post-selection qualitative smoke tests\n")?;
    writeln!(
        report,
        "The four motivating examples were run locally only after selecting and freezing the three product points. Their raw identities were removed before the repository-visible implementation; each selected its intended first span, shown here as `REDACTED`.\n"
    )?;
    writeln!(
        report,
        "| Case | Selected candidate | Conservative | Balanced | Permissive | Relevant evidence |\n| --- | --- | --- | --- | --- | --- |\n| `REDACTED` 1 | `REDACTED` | abstain | abstain | emit | quality 0.6964; margin 1.0000 |\n| `REDACTED` 2 | `REDACTED` | abstain | abstain | abstain | quality 0.7097; margin 0.3053 |\n| `REDACTED` 3 | `REDACTED` | abstain | abstain | abstain | quality 0.6446; margin 0.3075 |\n| `REDACTED` 4 | `REDACTED` | abstain | abstain | abstain | quality 0.7695; margin 0.3248 |\n"
    )?;
    writeln!(
        report,
        "The conservative logistic scores were respectively 0.9075, 0.8175, 0.7471, and 0.8775 against its frozen 0.9304 threshold. These examples did not influence model fitting, frontier construction, candidate selection, or the balanced recommendation.\n"
    )?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics(correct: usize, wrong: usize) -> EmissionMetrics {
        EmissionMetrics {
            rows: 100,
            expected_greetings: 100,
            emitted: correct + wrong,
            correct,
            wrong,
            false_abstentions: 100 - correct,
            ..EmissionMetrics::default()
        }
    }

    fn point(family: Family, target: f64, correct: usize, wrong: usize) -> OofPoint {
        OofPoint {
            family,
            target,
            metrics: metrics(correct, wrong),
            signature: vec![target.to_bits()],
            folds: Vec::new(),
        }
    }

    fn feature_row() -> FeatureRow {
        FeatureRow {
            population: Population::V1,
            ordinal: 0,
            expected_greeting: true,
            selected_matches: true,
            winner_present: true,
            vetoes_pass: true,
            decision_score: 0.5,
            candidate_quality: C5_QUALITY_MIN,
            candidate_count: 1,
            winner_margin: C5_MARGIN_MIN,
            margin_signal: 1.0,
            role_llr: 0.0,
            role_signal: C5_ROLE_MIN,
            reliability: C5_RELIABILITY_MIN,
            alphabetic_length: 5,
            native: true,
            segmentation_mechanism: None,
            hard_organization_marker: false,
            generic_organization_marker: false,
            ampersand: false,
            candidate_too_short: false,
            country_hint_present: false,
            locale_hint_present: false,
            c31_emits: false,
            c4_emits: false,
            c4_source: C4EmissionSource::Abstain,
            unhinted: None,
        }
    }

    #[test]
    fn dense_targets_are_strictly_descending_and_cover_requested_region() {
        assert_eq!(DENSE_TARGETS.first(), Some(&0.995));
        assert_eq!(DENSE_TARGETS.last(), Some(&0.970));
        assert!(DENSE_TARGETS.windows(2).all(|pair| pair[0] > pair[1]));
        assert_eq!(
            DENSE_TARGETS
                .iter()
                .map(|target| target.to_bits())
                .collect::<BTreeSet<_>>()
                .len(),
            DENSE_TARGETS.len()
        );
    }

    #[test]
    fn pareto_frontier_removes_dominated_points() {
        let dominant = point(Family::ControlledC4, 0.99, 80, 1);
        let dominated = point(Family::ScoreOnly, 0.98, 70, 2);
        let tradeoff = point(Family::Logistic, 0.995, 60, 0);
        let pareto = pareto_frontier(&[dominant.clone(), dominated, tradeoff.clone()]);
        assert_eq!(pareto.len(), 2);
        assert!(pareto.iter().any(|point| point.target == dominant.target));
        assert!(pareto.iter().any(|point| point.target == tradeoff.target));
    }

    #[test]
    fn equivalent_emissions_keep_the_simpler_representative() {
        let mut simple = point(Family::ScoreOnly, 0.99, 80, 1);
        let mut complex = point(Family::ControlledC4, 0.99, 80, 1);
        simple.signature = vec![7];
        complex.signature = vec![7];
        let pareto = pareto_frontier(&[complex, simple]);
        assert_eq!(pareto.len(), 1);
        assert_eq!(pareto[0].family, Family::ScoreOnly);
    }

    #[test]
    fn product_floor_selects_an_observed_point_without_interpolation() {
        let below = point(Family::ControlledC4, 0.985, 90, 2);
        let at_floor = point(Family::Logistic, 0.990, 80, 0);
        let points = [below, at_floor.clone()];
        let selected = product_point_at_floor(&points, 0.990).unwrap();
        assert_eq!(selected.family, at_floor.family);
        assert_eq!(selected.target.to_bits(), at_floor.target.to_bits());
    }

    #[test]
    fn aggregate_oof_rejects_duplicate_held_out_generations() {
        let folds = (0..Population::PROXIES.len())
            .map(|_| DenseFold {
                held_out: Population::V1,
                family: Family::ScoreOnly,
                target: DENSE_TARGETS[0],
                policy: Policy::Score { threshold: 0.5 },
                training_metrics: metrics(80, 1),
                held_out_metrics: metrics(10, 1),
            })
            .collect::<Vec<_>>();
        assert!(aggregate_oof_points(&[], &folds).is_err());
    }

    #[test]
    fn aggregate_output_privacy_rejects_row_level_columns() {
        let mut outputs = BTreeMap::new();
        outputs.insert(
            "unsafe.csv".to_string(),
            b"display_name,decision\nREDACTED,emit\n".to_vec(),
        );
        assert!(assert_aggregate_only_outputs(&outputs).is_err());
    }

    #[test]
    fn frozen_c5_uses_inclusive_boundaries_and_preserves_vetoes() {
        let policy = frozen_c5_policy();
        let boundary = feature_row();
        assert!(policy.emits(&boundary));

        let mut low_quality = boundary.clone();
        low_quality.candidate_quality = f64::from_bits(C5_QUALITY_MIN.to_bits() - 1);
        assert!(!policy.emits(&low_quality));

        let mut competing = boundary.clone();
        competing.candidate_count = 2;
        competing.winner_margin = C5_MARGIN_MIN;
        assert!(policy.emits(&competing));
        competing.winner_margin = f64::from_bits(C5_MARGIN_MIN.to_bits() - 1);
        assert!(!policy.emits(&competing));

        let mut vetoed = boundary.clone();
        vetoed.vetoes_pass = false;
        vetoed.hard_organization_marker = true;
        assert!(!policy.emits(&vetoed));

        let mut segmented = boundary;
        segmented.native = false;
        segmented.segmentation_mechanism = Some("digit");
        assert!(!policy.emits(&segmented));
    }

    #[test]
    fn frozen_c5_configuration_digest_is_exact() {
        assert_eq!(frozen_c5_digest(), C5_CONFIG_SHA256);
        assert_eq!(
            frozen_c5_policy().parameters(),
            "quality=0.70;reliability=0.00;role=0.00;margin=0.50"
        );
    }

    #[test]
    fn sealed_comparison_is_additive_and_accounts_for_every_delta_outcome() {
        let mut c4_correct = feature_row();
        c4_correct.c4_emits = true;

        let c5_only_correct = feature_row();

        let mut c5_only_wrong = feature_row();
        c5_only_wrong.selected_matches = false;

        let mut c5_only_null = feature_row();
        c5_only_null.expected_greeting = false;
        c5_only_null.selected_matches = false;

        let mut abstained_correct = feature_row();
        abstained_correct.candidate_quality = f64::from_bits(C5_QUALITY_MIN.to_bits() - 1);

        let comparison = compare_c4_c5(&[
            c4_correct,
            c5_only_correct,
            c5_only_wrong,
            c5_only_null,
            abstained_correct,
        ])
        .unwrap();
        assert_eq!(comparison.c4.emitted, 1);
        assert_eq!(comparison.c4.correct, 1);
        assert_eq!(comparison.c4.false_abstentions, 3);
        assert_eq!(comparison.c4.winner_correct_but_abstained, 2);
        assert_eq!(comparison.c5.emitted, 4);
        assert_eq!(comparison.c5.correct, 2);
        assert_eq!(comparison.c5.wrong, 2);
        assert_eq!(comparison.c5.null_false_emissions, 1);
        assert_eq!(comparison.c5.false_abstentions, 1);
        assert_eq!(comparison.c5.winner_correct_but_abstained, 1);
        assert_eq!(
            comparison.c5_only,
            C5OnlyAggregate {
                emitted: 3,
                correct: 1,
                wrong: 2,
                null_false_emissions: 1,
            }
        );
    }

    #[test]
    fn sealed_comparison_outputs_are_deterministic_and_aggregate_only() {
        let holdout = FrozenHoldout {
            cases: Vec::new(),
            manifest: name_eval::holdout::HoldoutManifest {
                format_version: 1,
                holdout_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    .to_string(),
                total_cases: 2_000,
                evaluable_cases: 1_500,
                skipped_cases: 500,
                expected_greetings: 1_200,
                expected_abstentions: 300,
                person_cases: 1_200,
                non_person_cases: 100,
                unknown_kind_cases: 700,
                provenance: "fresh blind proxy agreement".to_string(),
            },
        };
        let comparison = SealedC4C5Comparison {
            c4: EmissionMetrics {
                rows: 1_500,
                expected_greetings: 1_200,
                expected_nulls: 300,
                emitted: 101,
                correct: 100,
                wrong: 1,
                false_abstentions: 1_099,
                winner_correct_but_abstained: 900,
                expected_null_correct_abstentions: 300,
                ..EmissionMetrics::default()
            },
            c5: EmissionMetrics {
                rows: 1_500,
                expected_greetings: 1_200,
                expected_nulls: 300,
                emitted: 122,
                correct: 120,
                wrong: 2,
                false_abstentions: 1_078,
                winner_correct_but_abstained: 880,
                expected_null_correct_abstentions: 300,
                ..EmissionMetrics::default()
            },
            c5_only: C5OnlyAggregate {
                emitted: 21,
                correct: 20,
                wrong: 1,
                null_false_emissions: 0,
            },
        };
        let outputs = sealed_comparison_outputs(&holdout, comparison).unwrap();
        assert_eq!(
            outputs,
            sealed_comparison_outputs(&holdout, comparison).unwrap()
        );
        assert!(assert_sealed_aggregate_only(&outputs).is_ok());
        let combined = outputs
            .values()
            .map(|bytes| std::str::from_utf8(bytes).unwrap())
            .collect::<String>();
        for forbidden in [
            "Private Display Name",
            "case-private",
            "display_name",
            "expected_greeting",
            "decision_score",
        ] {
            assert!(!combined.contains(forbidden));
        }
    }

    #[test]
    fn correct_per_wrong_handles_zero_and_finite_counts() {
        assert_eq!(correct_per_wrong(44, 0), "no observed wrong emissions");
        assert_eq!(correct_per_wrong(44, 2), "22.00");
    }

    #[test]
    fn product_loss_counts_wrong_and_false_abstention() {
        let value = EmissionMetrics {
            false_abstentions: 40,
            wrong: 3,
            null_false_emissions: 2,
            ..EmissionMetrics::default()
        };
        assert_eq!(product_loss(value, 20), 100);
    }

    #[test]
    fn zero_wrong_ratio_is_explicit() {
        let mut writer = csv::Writer::from_writer(Vec::new());
        write_c5_metrics_header(&mut writer, &["policy"]).unwrap();
        write_c5_metrics(&mut writer, &["test"], metrics(10, 0)).unwrap();
        writer.flush().unwrap();
        let text = String::from_utf8(writer.into_inner().unwrap()).unwrap();
        assert!(text.contains("infinite"));
    }
}
