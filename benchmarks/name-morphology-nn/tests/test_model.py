import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from morphology_nn.cli import SEED, train_epoch
from morphology_nn.data import encode_utf8
from morphology_nn.model import (
    CONFIGURATIONS,
    MorphologyModel,
    export_model,
    load_export,
    parameter_count,
)


class ModelTests(unittest.TestCase):
    def test_shapes_and_parameter_ceiling(self):
        expected = {"tiny_string": 3809, "small_string": 12993}
        ids, mask, _ = encode_utf8("Olivier")
        for config in CONFIGURATIONS:
            model = MorphologyModel(config, 3)
            output = model(
                torch.tensor([ids]),
                torch.tensor([mask]),
                torch.tensor([0]),
            )
            self.assertEqual(tuple(output.shape), (1,))
            self.assertLess(parameter_count(model), 500_000)
            if config.name in expected:
                self.assertEqual(parameter_count(model), expected[config.name])

    def test_float32_export_round_trip(self):
        config = CONFIGURATIONS[0]
        model = MorphologyModel(config, 1).eval()
        ids, mask, _ = encode_utf8("Baris")
        inputs = (
            torch.tensor([ids]),
            torch.tensor([mask]),
            torch.tensor([0]),
        )
        expected = model(*inputs)
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            export_model(
                model,
                {"countries": ["UNKNOWN"]},
                directory / "model.json",
                directory / "model.bin",
            )
            loaded, metadata = load_export(directory / "model.bin")
            self.assertEqual(metadata["config"]["name"], "tiny_string")
            torch.testing.assert_close(expected, loaded(*inputs), rtol=0, atol=1e-6)

    def test_country_unknown_embedding_stays_zero_after_update(self):
        config = CONFIGURATIONS[2]
        model = MorphologyModel(config, 3)
        self.assertEqual(parameter_count(model), 13_133)
        torch.testing.assert_close(
            model.country_embedding.weight[0],
            torch.zeros(config.country_embedding),
        )

        ids, mask, _ = encode_utf8("Maria")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        output = model(
            torch.tensor([ids, ids]),
            torch.tensor([mask, mask]),
            torch.tensor([0, 1]),
        ).sum()
        output.backward()
        optimizer.step()
        torch.testing.assert_close(
            model.country_embedding.weight[0],
            torch.zeros(config.country_embedding),
        )

    def test_seeded_batch_training_is_deterministic(self):
        config = CONFIGURATIONS[0]
        encoded = [
            encode_utf8(value) for value in ("Maria", "Martin", "Élodie", "GmbH")
        ]
        data = SimpleNamespace(
            examples=tuple(range(len(encoded))),
            ids=torch.tensor([value[0] for value in encoded]),
            mask=torch.tensor([value[1] for value in encoded]),
            countries=torch.zeros(len(encoded), dtype=torch.long),
            labels=torch.tensor([1.0, 0.0, 1.0, 0.0]),
        )
        trained = []
        for _ in range(2):
            torch.manual_seed(SEED)
            model = MorphologyModel(config, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
            train_epoch(model, optimizer, data, torch.ones(2), config, epoch=1)
            trained.append(model.state_dict())
        for name in trained[0]:
            torch.testing.assert_close(
                trained[0][name], trained[1][name], rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
