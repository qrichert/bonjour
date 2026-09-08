# Tiny byte-level name-morphology experiment

This directory contains an offline research experiment that asks whether
a small byte-level CNN can recognize morphology associated with
plausible given names. It does not change the Bonjour classifier, C3.1,
C4, C5, the evaluator, or any corpus artifact.

The sigmoid values produced here are uncalibrated scores. They are
morphological/statistical evidence, not identity inference and not
greeting decisions.

## Frozen design

The data builder aggregates the local name totals by canonicalized,
Unicode-casefolded NFC string before labeling. It uses these
conservative labels:

- positive: valid lexical candidate, `given_count >= 100`, aggregate
  `role_llr >= 2.0`, and no frozen organization marker;
- strong-surname negative: valid lexical candidate,
  `surname_count >= 100`, and aggregate `role_llr <= -2.0`;
- organization negative: one of the 31 frozen production
  organization/legal tokens in `fixtures/non_name_tokens.csv`;
- malformed negative: an observed source key rejected by the production
  lexical candidate gate.

No random corruption is used. The raw surname-only source and a clean
dictionary-negative source are not present locally, so those populations
are reported as unavailable.

Exact strings and accent/case/separator morphology families share one
deterministic 80/10/10 split. The predeclared qualitative probes and
their families are quarantined before splitting. `select` accepts only
TRAIN and VALIDATION; MORPH_TEST is read only by `evaluate-test` after
model and threshold selection.

The input is at most 96 IDs: PAD, BOS, EOS, a middle-truncation marker,
and raw UTF-8 bytes. Three fixed configurations are compared:

- `tiny_string`: 3,809 parameters;
- `small_string`: 12,993 parameters;
- `small_country`: 13,505 parameters for the frozen country vocabulary.

All use two 1D convolutions, masked global max/mean pooling, a small
MLP, and one logit. The country-aware configuration has a learned
four-value embedding with index zero reserved for UNKNOWN. There is no
transformer, pretrained model, external embedding, or gender input.

## Environment and tests

Run commands below from the repository root. Python 3.12 and PyTorch
2.2.2 are locked in this isolated project because newer PyTorch releases
do not publish Intel macOS wheels.

```sh
cargo fmt --manifest-path benchmarks/name-morphology-nn/Cargo.toml -- --check
cargo test --locked --manifest-path benchmarks/name-morphology-nn/Cargo.toml
cargo clippy --locked --all-targets \
  --manifest-path benchmarks/name-morphology-nn/Cargo.toml -- -D warnings

uv run --project benchmarks/name-morphology-nn \
  --directory benchmarks/name-morphology-nn --python 3.12 \
  python -m unittest discover -s tests -p 'test_*.py' -v
```

## Full experiment

The following commands use only the acknowledged local inputs. Each
stage publishes transactionally to a new directory and refuses to
overwrite an existing result.

```sh
cargo build --release --locked \
  --manifest-path benchmarks/name-morphology-nn/Cargo.toml

benchmarks/name-morphology-nn/target/release/name-morphology-nn-data prepare \
  _wip/name-eval-artifact-c/c32-q8-surname-global \
  _wip/name-clean-v2/name-totals.csv \
  _wip/name-clean-v1/clean-v1.csv \
  _wip/name-morphology-nn-v1/data

uv run --project benchmarks/name-morphology-nn \
  --directory benchmarks/name-morphology-nn --python 3.12 \
  python -m morphology_nn select \
  --train ../../_wip/name-morphology-nn-v1/data/train.csv \
  --validation ../../_wip/name-morphology-nn-v1/data/validation.csv \
  --manifest ../../_wip/name-morphology-nn-v1/data/dataset_manifest.json \
  --output ../../_wip/name-morphology-nn-v1/selection

uv run --project benchmarks/name-morphology-nn \
  --directory benchmarks/name-morphology-nn --python 3.12 \
  python -m morphology_nn evaluate-test \
  --test ../../_wip/name-morphology-nn-v1/data/morph_test.csv \
  --manifest ../../_wip/name-morphology-nn-v1/data/dataset_manifest.json \
  --selection ../../_wip/name-morphology-nn-v1/selection \
  --output ../../_wip/name-morphology-nn-v1/test
```

Spent-proxy rows are streamed directly from the authenticated V1/V3/V4
holdouts into the scorer. Raw proxy candidates are not persisted by the
experiment:

```sh
set -o pipefail
benchmarks/name-morphology-nn/target/release/name-morphology-nn-data proxy \
  _wip/name-eval-artifact-c/c32-q8-surname-global \
  --sealed=_wip/real-proxy-v1/sealed.csv \
  --manifest=_wip/real-proxy-v1/sealed.manifest.csv \
  --sealed=_wip/real-proxy-v3/sealed.csv \
  --manifest=_wip/real-proxy-v3/sealed.manifest.csv \
  --sealed=_wip/real-proxy-v4/sealed.csv \
  --manifest=_wip/real-proxy-v4/sealed.manifest.csv | \
uv run --project benchmarks/name-morphology-nn \
  --directory benchmarks/name-morphology-nn --python 3.12 \
  python -m morphology_nn evaluate-proxy \
  --selection ../../_wip/name-morphology-nn-v1/selection \
  --test ../../_wip/name-morphology-nn-v1/test \
  --output ../../_wip/name-morphology-nn-v1/proxy

uv run --project benchmarks/name-morphology-nn \
  --directory benchmarks/name-morphology-nn --python 3.12 \
  python -m morphology_nn report \
  --data ../../_wip/name-morphology-nn-v1/data \
  --selection ../../_wip/name-morphology-nn-v1/selection \
  --test ../../_wip/name-morphology-nn-v1/test \
  --proxy ../../_wip/name-morphology-nn-v1/proxy \
  --probes fixtures/qualitative_probes.csv \
  --output ../../_wip/name-morphology-nn-v1/report
```

The final narrative is `_wip/name-morphology-nn-v1/report/report.md`.
Detailed CSV outputs preserve separate negative populations,
fixed-threshold operating points, handcrafted-signal comparisons,
score/role grids, and bounded morphology-source error examples. The
selected float32 weights are exported in a deterministic,
documented-by-metadata binary suitable for a later custom Rust reader;
this experiment does not implement that reader or quantization.
