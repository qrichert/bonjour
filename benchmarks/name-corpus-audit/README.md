# Name corpus audit

This isolated tool audits the normalized name corpus without modifying
it. It aggregates global exact-name frequencies, records
country/gender-row frequency bands, creates deterministic samples of the
long tail and unusual string features, and evaluates a matrix of global
and per-row frequency thresholds.

It also builds directly queryable C32 prototypes for minimum global
counts of `1`, `2`, `5`, `10`, `20`, `50`, `100`, and `500`. Each
prototype uses:

- a seeded 64-bit routing hash as the MPHF key;
- an independently seeded 32-bit membership fingerprint;
- `u32` per-name row offsets;
- one-byte country IDs and logarithmically quantized counts;
- two-bit gender values.

The generator rejects a 64-bit routing collision. Duplicate 32-bit
membership fingerprints are expected and safe because lookup compares
only the MPHF-selected slot.

Run it with a new output path:

```console
cargo run --release --manifest-path benchmarks/name-corpus-audit/Cargo.toml -- \
  data/normalized_name_dataset.csv _wip/name-corpus-audit
```

The input is opened read-only and the program refuses to overwrite its
output. Generated CSV reports are structural coverage measurements, not
classifier precision or recall; the latter requires an independent
labeled corpus.

## Recorded frequency result

The raw exact-name vocabulary was dominated by statistically weak
values: 23,502,128 of 30,895,021 names (76.07%) occurred once,
representing only 4.79% of observations. Frequencies below 5 comprised
92.30% of keys but only 7.32% of observations.

| Global minimum |       Keys | Observation coverage | Direct C32 + q8 |
| -------------: | ---------: | -------------------: | --------------: |
|              1 | 30,895,021 |              100.00% |      356.34 MiB |
|              2 |  7,392,893 |               95.21% |      116.96 MiB |
|              5 |  2,378,464 |               92.68% |       58.52 MiB |
|             10 |  1,258,058 |               91.21% |       41.66 MiB |
|             20 |    701,067 |               89.69% |       30.93 MiB |
|             50 |    335,746 |               87.43% |       21.26 MiB |
|            100 |    195,264 |               85.45% |       15.98 MiB |
|            500 |     57,576 |               79.56% |        7.96 MiB |

This established frequency pruning as an evidence policy, not merely a
storage optimization, and led to clean-v1's conservative global/row
`5/2` thresholds.
