import unittest

from morphology_nn.metrics import (
    average_precision,
    binary_metrics,
    constrained_threshold,
    pearson,
    percentiles,
    roc_auc,
    spearman,
)


class MetricsTests(unittest.TestCase):
    def test_perfect_ranking(self):
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.2, 0.8, 0.9]
        self.assertEqual(roc_auc(labels, scores), 1.0)
        self.assertEqual(average_precision(labels, scores), 1.0)

    def test_ties_are_order_independent(self):
        self.assertEqual(roc_auc([0, 1], [0.5, 0.5]), 0.5)
        self.assertEqual(average_precision([0, 1], [0.5, 0.5]), 0.5)

    def test_constrained_threshold_keeps_ties_together(self):
        threshold, metrics = constrained_threshold(
            [1, 1, 0, 0], [0.9, 0.8, 0.8, 0.1], maximum_fpr=0.0
        )
        self.assertEqual(threshold, 0.9)
        self.assertEqual(metrics.true_positives, 1)
        self.assertEqual(metrics.false_positives, 0)

    def test_operating_point_counts_and_percentiles(self):
        metrics = binary_metrics([1, 1, 0, 0], [0.9, 0.4, 0.8, 0.1], 0.5)
        self.assertEqual(metrics.true_positives, 1)
        self.assertEqual(metrics.false_positives, 1)
        self.assertEqual(metrics.precision, 0.5)
        self.assertEqual(metrics.recall, 0.5)
        self.assertEqual(metrics.false_positive_rate, 0.5)
        self.assertEqual(percentiles([1, 2, 3, 4, 5])["p50"], 3)

    def test_precision_constrained_threshold(self):
        threshold, metrics = constrained_threshold(
            [1, 0, 1, 0],
            [0.9, 0.8, 0.7, 0.1],
            minimum_precision=0.6,
        )
        self.assertEqual(threshold, 0.7)
        self.assertAlmostEqual(metrics.precision, 2 / 3)

    def test_correlations(self):
        self.assertAlmostEqual(pearson([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(spearman([1, 3, 2], [10, 30, 20]), 1.0)


if __name__ == "__main__":
    unittest.main()
