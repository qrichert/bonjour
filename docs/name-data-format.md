# Bonjour name-data artifact

This document describes the maintainer-owned pipeline and fixed binary
snapshot used by `bonjour` 0.1.0. It is a reproduction guide for the
official artifact, not a public custom-corpus interface: the 0.1.0
loader accepts only the exact manifest and bytes pinned in
`data/name-v1/manifest.json`.

Only the twelve derived binary constituents, `manifest.json`,
`README.md`, and `NOTICE` are distributed. Source CSVs and intermediate
rows are not. The artifact contains no name strings: it contains a
minimal perfect hash function (MPHF), independent 32-bit membership
fingerprints, and coarse aggregate metadata. It therefore cannot
enumerate names, although a caller can test a guessed string for
approximate membership. An unknown lookup has an assumed
false-acceptance probability of approximately `1 / 2^32`.

In the repository, the twelve binary constituents live under
`data/name-v1/files/` so Cargo can exclude that directory from the
crates.io package with one stable rule. The release archive remains
flat: its root contains the manifest, documents, and twelve binary
constituents together.

## Build and distribution matrix

| Context                        | Cargo features | Data source                  |
| ------------------------------ | -------------- | ---------------------------- |
| GitHub release binary          | `standalone`   | Embedded production artifact |
| Repository `just build`        | `standalone`   | `data/name-v1/files/`        |
| Default crates.io installation | default        | Extracted runtime artifact   |
| docs.rs                        | default        | No artifact is loaded        |

Repository linting, tests, and local documentation use all features
because the versioned constituents are available. `cargo package` and
docs.rs use default features because the crates.io package excludes
`data/name-v1/files/`.

`BONJOUR_DATA_DIR` is an override, not the normal repository path. It
selects a nonstandard extracted artifact directory at runtime, or the
exact pinned artifact to embed when building `standalone` from packaged
crate sources.

## External inputs

The normalized given-name input is UTF-8 CSV with this exact header:

```text
name,country,gender,count
```

`name` is preserved as one complete value, including Unicode, spaces,
apostrophes, and hyphens. `country` is an uppercase two-letter code,
`gender` is empty, `F`, or `M`, and `count` is a positive integer.
Duplicate `(name,country,gender)` tuples are aggregated.

Surname-role evidence is scanned from headerless country files named
`CC.csv`. Each row has exactly four columns:

```text
first_name,last_name,gender,country
```

The row country must match the filename. Every non-empty `last_name`
contributes to the global surname denominator, but a per-key surname
count is retained only when that exact UTF-8 string is already a
retained given-name key. Surname-only strings never become MPHF keys.

These inputs are maintained outside Git. Their contents and applicable
source terms remain the responsibility of the maintainer running the
pipeline.

## Corpus policy

`name-clean-v1` applies deliberately conservative sanitation: ASCII
digits, Unicode controls, URL/email markers, and the configured strong
legal-form tokens are rejected. Whitespace, multiple tokens, accents,
Unicode letters and marks, apostrophes, and hyphens remain valid.

Thresholds are applied in this exact order:

1. Aggregate duplicate `(name,country,gender)` tuples.
2. Remove names whose aggregated global count is below 5.
3. Remove individual rows whose count is below 2.
4. Recompute each name's total over surviving rows.
5. Remove names whose recomputed total is below 5.
6. Compute `given_total_observations` from only the final retained rows.

For example, row counts `2,1,1,1` initially total 5 but leave only 2
after row pruning, so the name is removed by the second global check.

## Reproduction commands

All output paths must be new; the tools intentionally refuse to
overwrite their inputs or previous results.

```console
uv run scripts/extract_name_counts.py \
  /external/name-data/CC-files \
  _wip/normalized-by-country \
  _wip/normalized_name_dataset.csv

cargo run --release --manifest-path benchmarks/name-clean-v1/Cargo.toml -- \
  _wip/normalized_name_dataset.csv \
  _wip/name-clean-v1

cargo run --release --manifest-path benchmarks/name-corpus-audit/Cargo.toml -- \
  _wip/name-clean-v1/clean-v1.csv \
  _wip/name-corpus-audit

cargo run --release --manifest-path benchmarks/name-surname-v2/Cargo.toml -- \
  _wip/name-clean-v1/clean-v1.csv \
  /external/name-data/CC-files \
  _wip/name-corpus-audit/c32/min-global-001 \
  _wip/name-clean-v2

uv run scripts/package_name_data.py \
  _wip/name-clean-v2/c32-q8-surname-global \
  --output-directory dist
```

Before packaging an externally generated artifact, copy
`data/name-v1/manifest.json`, `README.md`, and `NOTICE` beside the
selected twelve files. The repository snapshot may instead keep the
twelve files in `data/name-v1/files/`; `package_name_data.py` accepts
both layouts and always emits the same flat archive. It requires exact
manifest bytes, checks all sizes and SHA-256 digests, rejects symlinks,
and emits a deterministic `bonjour-name-data-v1.tar.zst` plus checksum.

The clean-v1 and surname-v2 tools hash their read-only inputs before and
after generation. The producer rejects a collision in the seeded 64-bit
routing hashes and exhaustively validates every known key before
discarding its builder-side name strings. The runtime loader cannot
repeat that collision check because names are intentionally absent from
the artifact.

## Deterministic construction

- Names are sorted by bytewise UTF-8 order before MPHF construction.
- Countries are sorted by bytewise uppercase order.
- Metadata rows are packed by MPHF slot, country, then gender.
- The routing seed is `0x6e61_6d65_2d72_6f75`.
- The fingerprint seed is `0x6e61_6d65_2d66_7033`.
- The MPHF call is `Mphf::new_parallel(1.7, &routing_hashes, None)`.
- Logically equivalent shuffled input rows must produce byte-identical
  output.

The q8 count encoder is:

```text
count == 0       -> 0
max_count <= 1   -> 1
otherwise        -> round(1 + ln(count) / ln(max_count) * 254), clamped 1..255
```

The decoder is:

```text
value == 0       -> 0
max_count <= 1   -> 1
otherwise        -> round(exp((value - 1) / 254 * ln(max_count))),
                    clamped 1..max_count
```

The given-name encoder's degenerate branch was normalized to match the
surname encoder and decoder. Both pinned maxima exceed one, and the
normalization must leave every pinned production digest unchanged.

## Binary constituents

All integers are little-endian. The production snapshot contains
1,803,175 keys and 8,722,920 metadata rows.

| File                                 | Representation and relationship                                       |
| ------------------------------------ | --------------------------------------------------------------------- |
| `names.mphf`                         | `bincode`-serialized `boomphf::Mphf<u64>` routing hashes to slots     |
| `fingerprints.u32`                   | one independent 32-bit fingerprint per MPHF slot                      |
| `row_offsets.u32`                    | `key_count + 1` row offsets; slot `s` owns `[offset[s], offset[s+1])` |
| `countries.dict`                     | concatenated two-byte uppercase country codes                         |
| `country_ids.u8`                     | one index into `countries.dict` per metadata row                      |
| `genders.2bit`                       | four rows per byte; `1` is female and `2` is male                     |
| `counts.q8`                          | one quantized given-name count per metadata row                       |
| `quantization_max_count.u32`         | given-name q8 maximum                                                 |
| `surname_counts.q8`                  | one quantized global surname count per MPHF slot                      |
| `surname_quantization_max_count.u32` | surname q8 maximum                                                    |
| `clean_given_total_observations.u64` | denominator for given-name role likelihood                            |
| `surname_total_observations.u64`     | all non-empty surname observations, not merely overlapping keys       |

Cross-file invariants require matching key/row lengths, monotonic
offsets that start at zero and end at the row count, valid country IDs,
nonzero q8 maxima, and nonzero denominators equal to the pinned
manifest. Generation rejects any aggregated count or q8 maximum above
`u32::MAX`, more than `u32::MAX` metadata rows/offsets, more than 256
countries, denominator overflow above `u64::MAX`, or an empty final
artifact.

The current MPHF serialization is tied to the pinned `boomphf` and
`bincode` versions. Any incompatible producer change requires a new
format version and a new pinned manifest rather than silently replacing
v1 bytes.

## Packaging and release approval

The archive uses one `bonjour-name-data-v1/` root, bytewise member
ordering, USTAR headers, timestamp/UID/GID zero, empty owner names,
`0755`/`0644` modes, and zstandard 0.25.0 at level 19 with one thread,
content size, and checksum.

The artifact was derived from a separately maintained corpus assembled
primarily from public data and subsequently cleaned, aggregated, and
quantized. It contains no original rows or name strings. Publishing or
redistributing the snapshot remains gated on explicit approval of the
artifact decision and the exact `data/name-v1/NOTICE` text.
