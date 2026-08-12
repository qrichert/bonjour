# External data provenance

The source corpus and derived bulk outputs are external to Git. This
file pins the snapshots used for the recorded benchmark results.

## Raw source tree

- 105 country CSV files under `name_dataset/data`.
- Raw tree size reported by `du -sb`: 10,684,317,420 bytes.
- Raw person rows scanned: 491,655,925.
- Tree SHA-256:
  `32be4c54402f06ee0c7f195cf63649d1a478ced5764cf474fcd20cb91941788a`.

The tree digest is computed by sorting relative `*.csv` paths, SHA-256
hashing each file, retaining each `sha256sum` line
(`digest  ./relative-path`), then SHA-256 hashing that concatenated
manifest.

## Normalized first-name counts

The committed `scripts/extract_name_counts.py` is the generator
definition.

- `normalized_name_dataset.csv`: 962,898,914 bytes.
- SHA-1: `cd3e7dd9b0ba7aa9ffb5207bb2601ada78edfcc4`.
- SHA-256:
  `d6fc4b1409ddaa7a917d16b772f44f83e94159cf61f7a3ab6a9c06d98092381a`.
- Rows: 50,308,485.
- Distinct exact names: 30,895,021.
- Observations: 490,678,049.

## Selected derived snapshots

- `clean-v1.csv`: 139,510,716 bytes; SHA-256
  `57a82801894facf883769403271e68094bcedb563d31c81f853192dc05e66b47`.
- Global-surname C32+q8 archive: SHA-256
  `a5f2d839f659fbe9f3a72dbedc1ecd68f69048d08ec276d11fa9281f2c6bb702`.
- Fresh generated TEST snapshot: SHA-256
  `1be896d0febaade25d6c6f8ac8f9b55c382600df1a25f70c135f84fa7425d9ff`.
- Former inspected 120,000-case TEST snapshot: SHA-256
  `2233794897ba69c3e9f8ffb9bdecd376545856d9f1bfa508793235cb8e74f962`.

The archive itself is not committed. Its uncompressed constituent arrays
are pinned by size and SHA-256 in
`name-eval/fixtures/artifact-manifest.csv` and
`name-eval/fixtures/surname-artifact-manifest.csv`; those are the
manifests validated before evaluation.
