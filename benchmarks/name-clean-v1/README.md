# clean-v1 corpus generator

This isolated tool reads `normalized_name_dataset.csv` without modifying
it and generates a conservative `clean-v1.csv` plus deterministic audit
samples and a report.

The sanitation policy rejects only ASCII digits, Unicode control
characters, high-confidence URL/email markers, and the ten configured
legal-form tokens. It preserves whitespace, hyphens, apostrophes,
accents, Unicode letters and marks, and multi-token values. Rows are
aggregated by exact `(name,country,gender)`, the global minimum is 5,
and the per-row minimum is 2. The final global sum is checked again
after row pruning.

Run it with a new output directory:

```console
cargo run --release --manifest-path benchmarks/name-clean-v1/Cargo.toml -- \
  data/normalized_name_dataset.csv _wip/name-clean-v1
```

The generator refuses to overwrite its output, hashes the original with
SHA-1 and SHA-256 before and after processing, validates the finished
CSV independently, and verifies its deterministic gzip and zstd
archives.

## Recorded clean-v1 result

Input integrity was unchanged before/after processing: 962,898,914
bytes, SHA-1 `cd3e7dd9b0ba7aa9ffb5207bb2601ada78edfcc4`, and SHA-256
`d6fc4b1409ddaa7a917d16b772f44f83e94159cf61f7a3ab6a9c06d98092381a`.

| Measure               |                               Result |
| --------------------- | -----------------------------------: |
| Final distinct names  |                            1,803,175 |
| Final metadata rows   |                            8,722,920 |
| Retained observations | 444,154,759 / 490,678,049 (90.5186%) |
| Exact CSV             |                           133.05 MiB |
| gzip-9 CSV            |                            32.81 MiB |
| zstd-19 CSV           |                            27.28 MiB |
| Direct C32 + q8       |                            33.22 MiB |
| zstd-19 artifact      |                            19.60 MiB |

The source had 30 URL/email-like and 610 strong-organization-marker
keys. After global `<5`, row `<2`, and the post-row global recheck,
exhaustive output validation and all 1,803,175 known-key C32 lookups
passed. The exact clean-v1 CSV SHA-256 is
`57a82801894facf883769403271e68094bcedb563d31c81f853192dc05e66b47`.

The C32 artifact used independent 64-bit MPHF routing and a 32-bit
membership fingerprint. Routing collisions were zero; 347 duplicate
stored fingerprint values are safe because lookup compares only the
routed slot. q8 p99 row error was 3.33%, maximum relative error 4.17%,
and aggregate bias +0.0702%.
