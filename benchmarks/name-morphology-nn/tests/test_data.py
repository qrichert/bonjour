import unittest

from morphology_nn.data import (
    BOS_ID,
    BYTE_OFFSET,
    EOS_ID,
    MAX_IDS,
    PAD_ID,
    TRUNCATED_MIDDLE_ID,
    encode_utf8,
)


class EncodingTests(unittest.TestCase):
    def test_boundaries_padding_and_utf8_bytes(self):
        ids, mask, truncated = encode_utf8("é")
        self.assertEqual(
            ids[:4], [BOS_ID, 0xC3 + BYTE_OFFSET, 0xA9 + BYTE_OFFSET, EOS_ID]
        )
        self.assertTrue(all(mask[:4]))
        self.assertEqual(ids[4:], [PAD_ID] * (MAX_IDS - 4))
        self.assertFalse(any(mask[4:]))
        self.assertFalse(truncated)

    def test_middle_truncation_preserves_ends(self):
        value = "a" * 50 + "b" * 50
        ids, mask, truncated = encode_utf8(value)
        self.assertTrue(truncated)
        self.assertEqual(len(ids), MAX_IDS)
        self.assertTrue(all(mask))
        self.assertEqual(ids[0], BOS_ID)
        self.assertIn(TRUNCATED_MIDDLE_ID, ids)
        self.assertEqual(ids[-1], EOS_ID)


if __name__ == "__main__":
    unittest.main()
