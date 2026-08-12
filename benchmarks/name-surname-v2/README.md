# clean-v2 surname-evidence generator

This isolated benchmark enriches the retained `clean-v1` first-name key
set with evidence that the same exact strings occur in the raw surname
column. It never adds surname-only keys and never modifies its inputs.

The primary output has one row per retained-name/country pair:

```text
name,country,given_unknown_count,given_female_count,given_male_count,as_surname_count
```

Grouping the three given-name gender counts avoids duplicating a
country-level surname count across gender rows. Exact counts remain in
the CSV. Separate global-only and global-plus-sparse-country C32+q8
artifact variants measure the binary cost without changing application
behavior.

Run with a new output directory and an extracted clean-v1 C32 artifact:

```console
cargo run --release --manifest-path benchmarks/name-surname-v2/Cargo.toml -- \
  clean-v1.csv name_dataset/data c32/min-global-001 new-clean-v2-directory
```

Matching is exact UTF-8 byte equality, consistent with `clean-v1`
aggregation. All non-empty raw surnames contribute to global and
per-country denominator totals, but per-key counts are retained only
when the exact surname is already a `clean-v1` key.

## Recorded surname-evidence result

- 1,424,492 / 1,803,175 retained first-name keys (78.9991%) also
  occurred as surnames.
- 364,386,816 surname observations matched retained keys, while the
  correct surname likelihood denominator was all 489,631,377 non-empty
  surname observations.
- Global surname q8 added only 1.72 MiB: 33.22 MiB became 34.94 MiB
  direct (20.81 MiB at zstd-19).
- Global plus country surname evidence cost 61.00 MiB direct, so it was
  not selected without evaluator evidence that justified the extra 26.06
  MiB.
- Global surname q8 had 3.70% p99 row-relative error, 4.17% maximum
  relative error, and +0.0271% signed aggregate bias.

The selected artifact retains exactly the clean-v1 MPHF key set and adds
only global q8 surname counts plus the full given/surname denominators.
The exact artifact files are pinned by
`../name-eval/fixtures/artifact-manifest.csv` and
`../name-eval/fixtures/surname-artifact-manifest.csv`. The disposable
archive used for evaluation had SHA-256
`a5f2d839f659fbe9f3a72dbedc1ecd68f69048d08ec276d11fa9281f2c6bb702`.
