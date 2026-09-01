# Greeting-name classifier evaluation harness

This crate evaluates greeting-name inference against labels that are
independent of the clean-v1 statistical corpus. The production crate and
evaluator now consume the same artifact, lexical, and classifier
implementation through the root crate's hidden `benchmark-internals`
feature. Corpus sanitation and C32 encoding remain separate benchmark
stages.

The harness contains five candidate-generation algorithms:

- `A-frequency-v1`, a weak frequency-led comparator;
- `B-simple-signals-v1`, which additionally uses country support, token
  position, single/multi-token structure, competing candidates, and
  strong and generic organization evidence.
- `C-global-role-v1`, which compares normalized global given-name and
  surname evidence for first-name-index candidates, supports directly
  evidenced compound candidates, and hard-abstains on strong legal
  markers. It does not add surname-only keys or use country-level
  surname metadata.
- `C1-compositional-role-v1`, the frozen C0 successor, which can compose
  an unsupported whitespace or hyphenated greeting span from two
  independently given-like components. Unsupported whitespace
  composition requires a remainder token; an otherwise unsupported
  two-token input remains ambiguous and is not combined.
- `C3-conservative-handle-candidates-v1`, which preserves every C1
  candidate and adds only corpus-backed substrings exposed by ASCII
  digit runs, `_`/`.` separators, or safe Unicode lower-to-upper case
  transitions. C3 is evaluated through the permanently frozen C2
  emission policy rather than a new score or threshold.

All return an unthresholded score. The evaluator applies configurable
thresholds afterward and writes split/category metrics, threshold
curves, precision- and error-budget-constrained operating points, DEV
failure samples, A/B/C0/C1 changes, and aggregate role-LLR
distributions.

Before any algorithm queries the corpus, a shared Unicode lexical gate
requires at least one alphabetic character and permits only Unicode
alphabetic characters, Unicode mark categories, whitespace,
apostrophe-like separators, and hyphen-like separators. Other
punctuation and symbols make that candidate ineligible. This is a
runtime candidate rule; it does not remove corpus rows.

## Independent data

`fixtures/given_names.csv`, `fixtures/surnames.csv`, and
`fixtures/organization_words.csv` are small, manually curated evaluation
labels. They were not extracted from clean-v1. DEV, VALIDATION,
LEGACY_TEST, INSPECTED_TEST, C0_TEST, and TEST assignment occurs at the
seed-name level before string generation. The loader rejects a
normalized given-name or surname atom used in more than one partition.

The expanded generator creates exactly 60,000 DEV and 60,000 VALIDATION
cases with SplitMix64 seed `0x6576616c2d763032`. The previously
inspected 120,000-case TEST is retained as `INSPECTED_TEST` under that
seed. C0's 120,000-case TEST is retained as `C0_TEST`, uses seed
`0x6576616c2d763033`, and is guarded by SHA-256
`1be896d0febaade25d6c6f8ac8f9b55c382600df1a25f70c135f84fa7425d9ff`.
After C1 and its threshold were frozen, a new 120,000-case TEST was
created with seed `0x6576616c2d763034` and snapshotted before classifier
evaluation as SHA-256
`403528ab491a2552308729df6b0a984fc864cc99c8438ca23bc1c122d8b772ba`. The
generator combines legitimate person
order/casing/whitespace/accent/compound/separator forms with difficult
organization negatives that all contain independently labeled person
atoms. The older inspected 116-case TEST is frozen as `LEGACY_TEST`,
guarded by a snapshot digest, and excluded from primary quality claims.
It retains its original seed `0x6576616c2d763031`.

`fixtures/regression.csv` is inspectable and may be optimized against.
Its metrics are never pooled into generated or sealed quality metrics.

The real-world holdout is a separate layer. It is never pooled into
synthetic metrics or exposed to Algorithms A/B/C0. The complete labeling
and freezing workflow is documented below.

Both generators use SplitMix64 with the fixed seeds documented above,
with additional domain separation by split.

## Fixed artifact baseline

C4 is the frozen production algorithm for bonjour 0.1.0. C2, C3, and
C3.1 remain available here as historical comparison baselines;
production does not expose algorithm selection or tuning controls.

Before the benchmark-local implementation was retired, same-process
comparisons ran old and shared C3.1 over regression, DEV, and
VALIDATION:

| Target/toolchain                        |   Cases | Maximum decision difference | Maximum gender difference | Decision mismatches |
| --------------------------------------- | ------: | --------------------------: | ------------------------: | ------------------: |
| `x86_64-apple-darwin`, Rust 1.93.0      | 120,014 |                           0 |                         0 |                   0 |
| `x86_64-unknown-linux-gnu`, Rust 1.93.0 | 120,014 |                           0 |                         0 |                   0 |

Both targets produced the frozen behavior digest
`9fd21be0cdc49b9f5e5e6f82f5c286514cf34ea9164d732b6d8a252d9111eab7`. The
committed `check-c31-parity` command reproduces that digest on the
pinned Linux target and toolchain. Other targets use semantic tests
rather than an assumed portable exact-bit guarantee for floating-point
transcendental functions.

```console
cargo +1.93.0 run --locked --release \
  --bin check-c31-parity -- \
  ../../_wip/name-eval-artifact-c/c32-q8-surname-global
```

`fixtures/artifact-manifest.csv` pins the eight files in the existing
clean-v1 C32 + q8 artifact by byte length and SHA-256.
`fixtures/surname-artifact-manifest.csv` separately pins the four global
surname q8 sidecars, including the full given/surname observation
denominators. Evaluation stops before loading if any file differs.

Extract the retained archive into a disposable directory:

```console
mkdir -p _wip/name-eval-artifact-c
zstd -dc _wip/name-clean-v2/c32-q8-surname-global.tar.zst \
  | tar -xf - -C _wip/name-eval-artifact-c
```

Then run into a new output directory:

```console
cargo run --release --manifest-path benchmarks/name-eval/Cargo.toml -- \
  _wip/name-eval-artifact-c/c32-q8-surname-global \
  _wip/name-clean-v1/clean-v1.csv \
  _wip/name-eval-v2
```

Optional arguments:

```text
--reference-threshold=0.80
--sealed=/path/to/frozen-holdout.csv
--sealed-manifest=/path/to/frozen-holdout.manifest.csv
--development-only
```

The reference threshold exists only to make A/B/C0/C1 decision diffs
concrete. C1's synthetic operating threshold is frozen at `0.93`, chosen
on VALIDATION; it is not yet a production recommendation. The
`--development-only` mode omits the fresh TEST entirely.

The clean-v1 CSV is opened read-only to report how many exact keys and
observations would fail the lexical rule. It is never rewritten.

## Sealed real-world holdout workflow

The labeling CLI is artifact-independent: it neither loads nor links to
classifier code, clean-v1, clean-v2, or the public-data-derived corpus.
It cannot display C1 output, confidence, frequency, membership, or role
LLR. Do not run the evaluator until labeling is complete and the holdout
has been frozen.

Prepare a local UTF-8 CSV with this header:

```csv
display_name,country_hint,locale_hint
```

Only `display_name` is required. Preserve source spelling and casing. Do
not include email addresses, phone numbers, account identifiers,
addresses, or other user attributes. Use inputs whose collection and
review are authorized; the loader rejects any column other than the
three shown above. This task provides no scraper or personal-data
ingestion. A header-only template is checked in as
`fixtures/holdout-source.example.csv`.

Keep private source, draft, frozen, and manifest files under `_wip/` or
another access-controlled, ignored location. Start or resume labeling:

For a first external proxy holdout, the official Meta Kaggle `Users.csv`
can be reduced to this format without retaining account IDs or other
profile attributes:

```console
python3 benchmarks/name-eval/scripts/prepare_meta_kaggle_holdout.py \
  _wip/meta-kaggle/Users.csv \
  _wip/name-holdout/source.csv \
  _wip/name-holdout/source.provenance.json
```

The script uniformly samples 2,000 user rows with fixed seed
`0x5245414C`, preserves duplicate population weight and original display
name text, and excludes only empty or whitespace-only values that the
labeler cannot accept. It refuses to overwrite outputs and verifies the
source checksum before and after processing. It leaves both hints empty:
Meta Kaggle provides full country names rather than the ISO alpha-2
codes C1 accepts, and it provides no locale. Neither `Users.csv` nor the
generated source/provenance files belong in Git. This is a platform-
specific real-world proxy; it does not replace a future random sample
from the actual product population.

Start or resume labeling:

```console
cargo run --manifest-path benchmarks/name-eval/Cargo.toml \
  --bin name-holdout -- \
  label \
  _wip/name-holdout/source.csv \
  _wip/name-holdout/draft.csv
```

The CLI presents one original display name at a time. It offers
contiguous spans delimited by whitespace and non-name punctuation, while
retaining apostrophes, hyphens, Unicode letters, and combining marks
inside a span. Multi-token spans are also offered. Choose a span,
`NULL`, `SKIP`, or save and quit. For a `NULL` label, optionally mark
the case as person, organization/non-person, or unknown; this supports a
non-person false-positive rate without forcing an uncertain type.
Progress is written after every decision. Opaque IDs are deterministic
ordinals and do not encode an account identity.

The draft schema is:

```csv
id,display_name,country_hint,locale_hint,label_status,expected_greeting,span_start,span_end,case_kind
```

Greeting labels store exact UTF-8 byte offsets into the original
`display_name`; validation rejects any label whose text is not exactly
that original span. Empty `expected_greeting` with `abstain` means an
intentional `NULL`. `skip` means undecidable and is excluded from metric
denominators.

Once all rows are labeled or skipped, freeze the holdout before any
classifier evaluation:

```console
cargo run --manifest-path benchmarks/name-eval/Cargo.toml \
  --bin name-holdout -- \
  freeze \
  _wip/name-holdout/draft.csv \
  _wip/name-holdout/sealed.csv \
  _wip/name-holdout/sealed.manifest.csv \
  --provenance="brief non-identifying description of the authorized source"
```

Freeze refuses to overwrite existing outputs. It sorts and serializes
the labeled rows deterministically, records SHA-256 and counts, and
writes a one-row manifest containing provenance, total/evaluable/skipped
cases, greeting/abstention labels, and optional case-kind counts. The
evaluator rejects a changed checksum, non-canonical serialization, or
counts that differ from the manifest.

Run the first sealed evaluation with both frozen files:

```console
cargo run --release --manifest-path benchmarks/name-eval/Cargo.toml \
  --bin name-eval -- \
  _wip/name-eval-artifact-c/c32-q8-surname-global \
  _wip/name-eval-sealed-first \
  --sealed-only \
  --sealed=_wip/name-holdout/sealed.csv \
  --sealed-manifest=_wip/name-holdout/sealed.manifest.csv
```

Supplying only one sealed argument is an error. Sealed evaluation uses
exactly frozen `C1-compositional-role-v1` at `0.93`; the reference
threshold and development options are rejected in `--sealed-only` mode.
This mode does not load clean-v1 or generate synthetic cases. It writes
only `sealed_summary_metrics.csv`, `sealed_confidence_buckets.csv`, and
the same aggregate tables in `report.md`. It never writes sealed cases,
failures, comparisons, traces, predictions, or threshold sweeps. The
confidence buckets are `0.93–0.95`, `0.95–0.97`, `0.97–0.99`, and
`0.99–1.00` and must not be used to retune C1.

If an individual sealed row is later inspected to design a classifier
change, it is no longer sealed evidence. Move it into DEV or regression
before evaluating a future algorithm. Do not report the same inspected
holdout as an unbiased test.

### REAL_PROXY_V1 checkpoint and diagnostic funnel

`REAL_PROXY_V1` was a uniform 2,000-row Meta Kaggle display-name sample
whose labels were generated by ChatGPT independently of C1. It is proxy
agreement evidence, not manually adjudicated ground truth. Its frozen
SHA-256 is
`de95213f27fc1849032ee6788c8f16d7d515c1a991ae8b2e8414b7b155814c4e`. The
first aggregate-only C1 evaluation at the frozen `0.93` threshold
produced:

| Evaluable | Expected greetings | Emitted | Correct | Wrong | Precision | Recall | False emissions on expected NULL |
| --------: | -----------------: | ------: | ------: | ----: | --------: | -----: | -------------------------------: |
|     1,957 |              1,616 |      36 |      34 |     2 |    94.44% |  2.10% |                                0 |

That checkpoint was preserved before any row was inspected. V1 was then
deliberately spent for diagnosis with an explicit checksum
acknowledgement:

```console
cargo run --release --manifest-path benchmarks/name-eval/Cargo.toml \
  --bin name-eval -- \
  _wip/name-eval-artifact-c/c32-q8-surname-global \
  _wip/name-eval-real-proxy-v1-diagnostic \
  --diagnose-spent-holdout-sha256=de95213f27fc1849032ee6788c8f16d7d515c1a991ae8b2e8414b7b155814c4e \
  --sealed=_wip/real-proxy-v1/sealed.csv \
  --sealed-manifest=_wip/real-proxy-v1/sealed.manifest.csv
```

The diagnostic command accepts no threshold or development options. It
checksum-verifies the frozen input, runs only unchanged C1 at `0.93`,
and writes row-level material only to its explicit local output
directory. The ordinary `--sealed-only` mode remains aggregate-only.

The resulting funnel was:

| Stage                                      | Count | Share of expected greetings |
| ------------------------------------------ | ----: | --------------------------: |
| Lexically eligible                         | 1,615 |                      99.94% |
| Direct normalized artifact lookup          | 1,500 |                      92.82% |
| Direct accent-folded lookup                |     0 |                       0.00% |
| Corpus coverage ceiling                    | 1,507 |                      93.25% |
| Candidate-generation ceiling               | 1,208 |                      74.75% |
| Production-reachable generation            | 1,208 |                      74.75% |
| Correct candidate ranked first             | 1,098 |                      67.95% |
| Correct candidate reached threshold `0.93` |    34 |                       2.10% |

Mutually exclusive terminal reasons were 1 lexical rejection, 108
without usable corpus evidence, 299 with evidence but no generated
matching candidate, 110 ranking losses, 1,064 correct winners below
threshold, and 34 correct emissions. No expected greeting triggered the
hard-organization abstention. Median final confidence among the 1,098
correctly ranked cases was `0.746279`; its 90th percentile was only
`0.886698`.

Manual review covered all 36 emissions and the deterministic miss sample
(the sole lexical miss plus 50 rows from each other non-empty miss
category). The two emitted disagreements included one clear credential
selection and one culturally ambiguous abbreviation that was unsafe as a
standalone greeting. The 299 candidate-generation misses were dominated
by names embedded in whitespace-free handles with digits, underscores,
concatenated components, suffixes, or repeated letters. Ranking misses
mixed genuine competition errors with unresolved
exact-span/compound-label policy.

These results diagnose a major score-distribution shift, but they do not
justify lowering `0.93`: noisy handles remain present, the labels are
not human ground truth, and V1 is now inspected development evidence.
Any C2 configuration must be selected elsewhere and evaluated on a new
disjoint `REAL_PROXY_V2`. The original sealed CSV and manifest remain
unchanged.

Diagnostic row files and manual notes stay ignored under `_wip/`; only
these aggregate conclusions belong in Git. The proxy's 341 expected
abstentions all have unknown case type, so V1 cannot establish an
organization/non- person false-positive rate.

### C2 proxy-calibrated emission baseline

C2 changes only the decision to emit C1's already-selected winner.
Candidate generation, role scoring, ranking, greeting text, and gender
remain C1. C2's decision score combines normalized signals as follows:

```text
0.00 * C1 winner score
+ 0.10 * normalized winner margin (scale 0.50)
+ 0.70 * role signal
+ 0.20 * count reliability
```

It additionally requires at least three Unicode alphabetic characters
and hard-abstains for C1 hard-organization evidence, a generic
organization marker, or ampersand evidence. The frozen development
threshold is `0.78975882405736963`. This is an uncalibrated decision
score, not a probability.

The configuration was selected by a deterministic finite search over
convex weights in increments of `0.10`, five margin scales, minimum
candidate lengths 1–3, and every threshold that changed a development
emission set. Every feasible row had to produce zero wrong emissions on
both inspected REAL_PROXY_V1_DEV and synthetic VALIDATION, plus zero
VALIDATION organization false positives. The search found 918 feasible
configurations and selected the one maximizing correct proxy emissions,
with VALIDATION correct emissions and the documented parameter ordering
as tie-breaks.

Run and reproduce the selection with:

```console
cargo run --release --manifest-path benchmarks/name-eval/Cargo.toml \
  --bin name-eval -- \
  _wip/name-eval-artifact-c/c32-q8-surname-global \
  _wip/name-eval-c2-calibration \
  --develop-c2-from-spent-holdout-sha256=de95213f27fc1849032ee6788c8f16d7d515c1a991ae8b2e8414b7b155814c4e \
  --sealed=_wip/real-proxy-v1/sealed.csv \
  --sealed-manifest=_wip/real-proxy-v1/sealed.manifest.csv
```

The command refuses threshold overrides and other evaluation modes,
verifies the spent digest, evaluates only REAL_PROXY_V1_DEV and
synthetic VALIDATION, and asserts that deterministic selection exactly
reproduces the frozen C2 constants. Synthetic TEST splits are not
inferred.

Development results were:

| Population        | Algorithm | Emitted | Correct | Wrong | Observed precision | Recall | One-sided 95% Wilson lower bound |
| ----------------- | --------- | ------: | ------: | ----: | -----------------: | -----: | -------------------------------: |
| REAL_PROXY_V1_DEV | C1        |      36 |      34 |     2 |             94.44% |  2.10% |                           84.53% |
| REAL_PROXY_V1_DEV | C2        |     207 |     207 |     0 |            100.00% | 12.81% |                           98.71% |
| VALIDATION        | C1        |  10,626 |  10,626 |     0 |            100.00% | 27.09% |                           99.97% |
| VALIDATION        | C2        |  14,686 |  14,686 |     0 |            100.00% | 37.44% |                           99.98% |

These are development agreement figures, not real-world quality claims.
V1 was used to select C2, its labels are ChatGPT-generated rather than
manually adjudicated, and synthetic transformations are correlated.
Wilson bounds are therefore reported separately and are not pooled.

The V1 winner populations explain why the new score helped:

| Population                     | Cases | C1 score p10 | Median |   p90 | Margin p10 | Median |   p90 | Role LLR p10 | Median |   p90 |
| ------------------------------ | ----: | -----------: | -----: | ----: | ---------: | -----: | ----: | -----------: | -----: | ----: |
| Correct winner                 | 1,098 |        0.545 |  0.746 | 0.887 |      0.161 |  0.635 | 1.000 |        0.363 |  2.042 | 3.384 |
| Wrong winner on positive label |   196 |        0.397 |  0.569 | 0.791 |      0.025 |  0.178 | 1.000 |       -1.277 |  0.795 | 2.463 |
| Candidate on expected NULL     |    61 |        0.334 |  0.445 | 0.632 |      0.038 |  1.000 | 1.000 |       -2.744 | -0.562 | 1.217 |

Here “C1 score” is C1's final pre-threshold confidence. The
expected-NULL population demonstrates why margin alone is insufficient:
many NULL cases have no competitor and therefore unit margin, while
their role evidence is substantially weaker.

The 299 evidence-covered but ungenerated V1 labels were categorized
without changing the candidate generator. All 299 are embedded in a
larger whitespace-free token; 140 containing tokens have ASCII digits,
54 have ineligible punctuation/symbols, and 100 look concatenated or
camel-case-like. Only four labels are two-token whitespace compounds and
two contain hyphens. This points to handle segmentation rather than
ordinary compound generation, but subjective substring labels do not yet
justify such a feature.

C2 is now frozen development state. It must be evaluated once on a new,
independently labeled, disjoint REAL_PROXY_V2 before any generalization
or precision claim. Candidate generation, ranking, ordering,
punctuation, and handle segmentation remain separate future experiments.

### REAL_PROXY_V2 blind agreement workflow

REAL_PROXY_V2 uses another fixed 2,000-row Meta Kaggle sample. It is
value-disjoint from V1: every source row whose exact `DisplayName`
occurs in the V1 source is excluded before reservoir sampling. This is
stronger than row-level disjointness and avoids retaining Kaggle account
IDs, although removing every repeated V1 value slightly changes the
remaining population.

Generate V2 with fixed seed `0x5245414C5F5632`:

```console
python3 benchmarks/name-eval/scripts/prepare_meta_kaggle_holdout.py \
  /path/to/Users.csv \
  _wip/real-proxy-v2/source.csv \
  _wip/real-proxy-v2/source.provenance.json \
  --seed=0x5245414C5F5632 \
  --exclude-source=_wip/source.csv
```

The aggregate provenance records the source and exclusion-file hashes,
unique excluded values, excluded source-row occurrences, remaining
eligible population, RNG seed, and output hash. It contains no display
names or account identifiers. The source and exclusion files are
checksum-verified and never modified.

Export two classifier-blind annotation templates:

```console
cargo run --manifest-path benchmarks/name-eval/Cargo.toml \
  --bin name-holdout -- \
  export-blind \
  _wip/real-proxy-v2/source.csv \
  _wip/real-proxy-v2/annotation-a.csv

cargo run --manifest-path benchmarks/name-eval/Cargo.toml \
  --bin name-holdout -- \
  export-blind \
  _wip/real-proxy-v2/source.csv \
  _wip/real-proxy-v2/annotation-b.csv
```

Each annotator receives only:

```csv
id,display_name,country_hint,locale_hint,decision,expected_greeting
```

`decision` must be exactly `GREETING`, `NULL`, or `SKIP`.
`expected_greeting` must be empty for `NULL`/`SKIP`; for `GREETING` it
must reproduce an exact contiguous UTF-8 span of the original display
name. The two files must be completed independently without classifier
results, confidence, corpus membership, frequencies, or role evidence.

Merge them mechanically:

```console
cargo run --manifest-path benchmarks/name-eval/Cargo.toml \
  --bin name-holdout -- \
  consensus \
  _wip/real-proxy-v2/source.csv \
  _wip/real-proxy-v2/annotation-a.csv \
  _wip/real-proxy-v2/annotation-b.csv \
  _wip/real-proxy-v2/consensus-draft.csv \
  _wip/real-proxy-v2/consensus-summary.csv
```

Identical greeting spans become greeting labels and `NULL` + `NULL`
becomes expected abstention. Any `SKIP` or disagreement becomes `SKIP`,
with no manual guess. The tool rejects missing/duplicate IDs,
source-field mutation, unsupported decisions, non-spans, and existing
outputs. The aggregate summary reports greeting agreement, NULL
agreement, annotator-skip, and disagreement counts. This agreement
filter can select an easier subset, and independent models can still
share cultural mistakes; V2 remains proxy evidence rather than human
ground truth.

Freeze the consensus before loading the classifier artifact:

```console
cargo run --manifest-path benchmarks/name-eval/Cargo.toml \
  --bin name-holdout -- \
  freeze \
  _wip/real-proxy-v2/consensus-draft.csv \
  _wip/real-proxy-v2/sealed.csv \
  _wip/real-proxy-v2/sealed.manifest.csv \
  --provenance="REAL_PROXY_V2: disjoint Meta Kaggle sample; exact agreement of two independent classifier-blind annotations"
```

Then substitute the printed frozen digest and run the paired comparison
once:

```console
cargo run --release --manifest-path benchmarks/name-eval/Cargo.toml \
  --bin name-eval -- \
  _wip/name-eval-artifact-c/c32-q8-surname-global \
  _wip/name-eval-real-proxy-v2 \
  --compare-sealed-c1-c2-sha256=FROZEN_SHA256 \
  --sealed=_wip/real-proxy-v2/sealed.csv \
  --sealed-manifest=_wip/real-proxy-v2/sealed.manifest.csv
```

The digest must match the verified manifest. The command evaluates
frozen C1 at `0.93` and frozen C2 at `0.78975882405736963` on exactly
the same evaluable cases. It writes only
`sealed_comparison_summary.csv`,
`sealed_comparison_confidence_buckets.csv`, and `report.md`; it does not
write cases, predictions, failures, traces, threshold sweeps, or
changed-row comparisons. C1 and C2 scores are different quantities, so
their coarse buckets are reported separately and are diagnostic only. Do
not inspect V2 failures or use its aggregates to retune either frozen
algorithm.

The first V2 sample was frozen before inference with SHA-256
`7d704a646b8dd9fa3820f88b9504d4397b676af9435532cf2da9befda7663a73`. Of
2,000 source rows, the two annotations agreed on 1,217 exact greeting
spans and 279 expected abstentions. Another 416 rows had an annotator
skip or unusable non-exact span and 88 had different usable labels, so
the frozen comparison evaluated 1,496 rows and skipped 504.

The single aggregate-only comparison produced:

| Algorithm                   | Emitted | Correct | Wrong | Expected-NULL emissions | Observed precision | Recall | Abstention |
| --------------------------- | ------: | ------: | ----: | ----------------------: | -----------------: | -----: | ---------: |
| C1 at `0.93`                |      43 |      39 |     4 |                       0 |             90.70% |  3.20% |     97.13% |
| C2 at `0.78975882405736963` |     208 |     206 |     2 |                       0 |             99.04% | 16.93% |     86.10% |

C2 therefore emitted about 4.8 times as often, recovered about 5.3 times
C1's greeting recall, and halved observed wrong emissions on the same
agreed subset. Of C2's 208 emissions, 144 scored `0.789759–0.85` (142
correct, 2 wrong) and 64 scored `0.85–0.90` (all 64 correct); none
scored above `0.90`. These data validate the direction of the C2
emission policy on this fresh proxy, but 208 emissions and
machine-agreed labels do not establish 99% worldwide or production
precision. V2 is now frozen comparison evidence and must not be used to
retune C2.

### C3 conservative handle candidates

C3 addresses only the 299 spent-V1 labels whose corpus evidence existed
but whose expected span was embedded in a larger whitespace-free token.
It preserves C1 candidates, role scoring, ranking, organization and
gender evidence, then adds maximal corpus-backed segments exposed by:

- an ASCII digit run;
- `_` or `.`;
- a Unicode lowercase-to-uppercase transition.

Tokens containing another non-name punctuation or symbol do not produce
handle-derived candidates, so `/`, `\\`, `@`, `:`, and similar URL,
email, or arbitrary-symbol forms are not split. Camel-like parts are
also rejected when any resulting component lacks a lowercase character;
this prevents an acronym-like suffix such as `PrincessFC` from exposing
`Princess`. C3 does not scan arbitrary prefixes, split all-lowercase or
all-uppercase concatenations, remove repeated letters, infer
script-specific boundaries, or repair misspellings. Derived candidates
receive the unchanged C1 role score with no origin bonus or penalty, and
the winner is gated by frozen C2 at exactly `0.78975882405736963`.

Reproduce the development experiment with spent REAL_PROXY_V1_DEV only:

```console
cargo run --release --manifest-path benchmarks/name-eval/Cargo.toml \
  --bin name-eval -- \
  _wip/name-eval-artifact-c/c32-q8-surname-global \
  _wip/name-eval-c3-development \
  --develop-c3-from-spent-holdout-sha256=de95213f27fc1849032ee6788c8f16d7d515c1a991ae8b2e8414b7b155814c4e \
  --sealed=_wip/real-proxy-v1/sealed.csv \
  --sealed-manifest=_wip/real-proxy-v1/sealed.manifest.csv
```

The command verifies the spent digest, reproduces the historical C2
checkpoints, and evaluates only V1 plus synthetic VALIDATION. It never
loads V2 or a synthetic TEST split. Local changed-case diagnostics are
development material under `_wip` and must not be committed.

Development results were:

| Population        | Algorithm | Emitted | Correct | Wrong | Expected-NULL emissions | Observed precision | Recall |
| ----------------- | --------- | ------: | ------: | ----: | ----------------------: | -----------------: | -----: |
| REAL_PROXY_V1_DEV | C2        |     207 |     207 |     0 |                       0 |            100.00% | 12.81% |
| REAL_PROXY_V1_DEV | C3        |     234 |     234 |     0 |                       0 |            100.00% | 14.48% |
| VALIDATION        | C2        |  14,686 |  14,686 |     0 |                       0 |            100.00% | 37.44% |
| VALIDATION        | C3        |  14,686 |  14,686 |     0 |                       0 |            100.00% | 37.44% |

On V1's 1,616 expected greetings, matching-candidate generation rose
from 1,208 (74.75%) to 1,387 (85.83%), and correct pre-threshold winner
selection rose from 1,098 (67.95%) to 1,268 (78.47%). C3 generated 179
previously missing matching candidates, selected 172 of them, and
emitted 27. The first, broader camel rule emitted `Princess` from the
expected-NULL handle `PrincessFC`; rejecting acronym-like camel parts
removed that unsafe behavior before C3 was frozen.

These are selection results on machine-labeled spent V1 plus synthetic
VALIDATION, not held-out quality evidence. The observed zero-error rows
do not establish precision. C3 is frozen as a development candidate and
requires a fresh, disjoint, independently annotated REAL_PROXY_V3 for a
one-shot comparison against frozen C2. V2 remains untouched and cannot
validate or retune C3.

### REAL_PROXY_V3 frozen C2/C3 comparison

V3 was drawn from the same external Meta Kaggle `Users.csv` with fixed
seed `0x5245414C5F5633`, after excluding every exact display-name value
in both V1 and V2:

```console
python3 benchmarks/name-eval/scripts/prepare_meta_kaggle_holdout.py \
  /path/to/Users.csv \
  _wip/real-proxy-v3/source.csv \
  _wip/real-proxy-v3/source.provenance.json \
  --seed=0x5245414C5F5633 \
  --exclude-source=_wip/source.csv \
  --exclude-source=_wip/real-proxy-v2/source.csv
```

The resulting 2,000-row source has SHA-256
`9deefa258a64c873d833357e8f242f18fab01ca2eedfa8d2442a56d931d361e7`. Its
1,999 unique values have zero exact overlap with either earlier proxy.
The 2,587,424,211-byte source `Users.csv` remained unchanged at SHA-256
`30b95ff7d079289fe76a0fada39ebbb174f15f6f85a2e09f7a208c6fdf57dd82`.

Both annotations were produced without classifier output or corpus
evidence. Their returned schemas were normalized mechanically using the
same policy as V2: exact original-text spans become `GREETING`, explicit
`NULL` and `SKIP` are preserved, non-exact or unsupported labels become
`SKIP`, and annotator confidence/notes are ignored. Raw annotation files
remain unchanged. Annotator A supplied 1,662 exact spans, 274 NULLs, 52
explicit skips, and 12 unusable labels. Annotator B supplied 1,289 exact
spans, 331 NULLs, 28 explicit skips, and 352 unusable labels.

Mechanical consensus yielded:

| Source rows | Greeting agreements | NULL agreements | Annotator-skip cases | Other disagreements | Evaluable | Skipped |
| ----------: | ------------------: | --------------: | -------------------: | ------------------: | --------: | ------: |
|       2,000 |               1,232 |             242 |                  424 |                 102 |     1,474 |     526 |

The consensus was frozen before the artifact was loaded, with SHA-256
`d70e4d4b2ed7e49bed09dc1e8d2ba60ade8a752e3b86c772e964bd64883ee6fe`. The
only V3 classifier invocation was:

```console
cargo run --release --manifest-path benchmarks/name-eval/Cargo.toml \
  --bin name-eval -- \
  _wip/name-eval-artifact-c/c32-q8-surname-global \
  _wip/name-eval-real-proxy-v3 \
  --compare-sealed-c2-c3-sha256=d70e4d4b2ed7e49bed09dc1e8d2ba60ade8a752e3b86c772e964bd64883ee6fe \
  --sealed=_wip/real-proxy-v3/sealed.csv \
  --sealed-manifest=_wip/real-proxy-v3/sealed.manifest.csv
```

The comparison uses `ALGORITHM_C1` candidate generation for C2 and
`ALGORITHM_C3` candidate generation for C3. Both then use the identical
frozen C2 score and threshold `0.78975882405736963`:

| Algorithm | Emitted | Correct | Wrong | Expected-NULL emissions | Observed precision | Recall | Abstention |
| --------- | ------: | ------: | ----: | ----------------------: | -----------------: | -----: | ---------: |
| C2        |     205 |     200 |     5 |                       2 |             97.56% | 16.23% |     86.09% |
| C3        |     223 |     217 |     6 |                       3 |             97.31% | 17.61% |     84.87% |

C3 added 18 emissions, of which 17 matched the proxy labels. Relative to
C2 it gained `1.38` recall points while adding one wrong greeting and
one expected-NULL emission. The handle candidates therefore generalized
a modest reachability benefit, but the fresh comparison does not show
unchanged safety and does not make C3 an unambiguous replacement for C2.

Of C2's emissions, the `0.789759–0.85` bucket contained 147 correct and
4 wrong; `0.85–0.90` contained 47 correct and none wrong; `0.90–0.95`
contained 6 correct and 1 wrong. C3's corresponding buckets contained
155/5, 56/0, and 6/1 correct/wrong. These aggregates are diagnostic only
and must not be used to retune the shared threshold.

No row-level V3 predictions, failures, traces, or changed-case
comparisons were written or inspected. V3 is now spent comparison
evidence and cannot be used to change C2 or C3. Exact agreement can
select an easier subset, the two machine annotators can share cultural
mistakes, and Meta Kaggle is not guaranteed to match the product
population. These figures are relative proxy evidence, not worldwide or
production-quality claims.

### C3.1 segmented-candidate provenance gate

After the aggregate V3 checkpoint above was preserved, V3 was
deliberately spent only on the C2-to-C3 delta. The diagnostic reproduces
the frozen C2/C3 V1, V3, and VALIDATION metrics before writing any
row-level development output. It records the original input, expected
span, both winners and emissions, handle-boundary mechanism, candidate
length, candidate and emission scores, role LLR, winner margin,
reliability, counts, and candidate count.

On spent V3, C2 abstained while C3 emitted in 18 cases. Seventeen
matched the proxy labels and one digit-boundary candidate was an
expected-NULL emission. The 18 new emissions comprised:

| Segmentation mechanism | Correct | Wrong greeting | Expected-NULL emission |
| ---------------------- | ------: | -------------: | ---------------------: |
| Lower-to-upper         |      13 |              0 |                      0 |
| Digit                  |       2 |              0 |                      1 |
| Dot                    |       1 |              0 |                      0 |
| Underscore             |       1 |              0 |                      0 |

The pre-threshold winner changed in 175 V3 cases, but 157 of those new
winners still remained below the frozen C2 gate. On spent V1, all 27
C3-only emissions matched the proxy labels: 13 lower-to-upper, 9 digit,
4 underscore, and 1 mixed. Because only one unsafe delta exists and V1
contains correct digit-derived emissions, the data do not support
separate mechanism-specific constants.

C3.1 freezes one simple provenance rule:

```text
native C3 winner:
    emission_score = frozen C2 score

handle-segment C3 winner:
    emission_score = frozen C2 score - 0.025
```

Candidate generation, ranking, role evidence, organization vetoes, and
gender evidence are unchanged. The public threshold remains
`0.78975882405736963`; the effective segmented-winner requirement is
`0.8147588240573696`. The penalty is independent of boundary mechanism
and does not affect exact or compositional winners.

Reproduce the spent diagnostics separately for V3 and V1:

```console
cargo run --release --manifest-path benchmarks/name-eval/Cargo.toml \
  --bin name-eval -- \
  _wip/name-eval-artifact-c/c32-q8-surname-global \
  _wip/name-eval-c31-v3-diagnostic \
  --develop-c31-from-spent-holdout-sha256=d70e4d4b2ed7e49bed09dc1e8d2ba60ade8a752e3b86c772e964bd64883ee6fe \
  --sealed=_wip/real-proxy-v3/sealed.csv \
  --sealed-manifest=_wip/real-proxy-v3/sealed.manifest.csv

cargo run --release --manifest-path benchmarks/name-eval/Cargo.toml \
  --bin name-eval -- \
  _wip/name-eval-artifact-c/c32-q8-surname-global \
  _wip/name-eval-c31-v1-diagnostic \
  --develop-c31-from-spent-holdout-sha256=de95213f27fc1849032ee6788c8f16d7d515c1a991ae8b2e8414b7b155814c4e \
  --sealed=_wip/real-proxy-v1/sealed.csv \
  --sealed-manifest=_wip/real-proxy-v1/sealed.manifest.csv
```

The selected development checkpoints are:

| Population        | Algorithm | Emitted | Correct | Wrong | Expected-NULL emissions | Observed precision | Recall |
| ----------------- | --------- | ------: | ------: | ----: | ----------------------: | -----------------: | -----: |
| REAL_PROXY_V1_DEV | C2        |     207 |     207 |     0 |                       0 |            100.00% | 12.81% |
| REAL_PROXY_V1_DEV | C3        |     234 |     234 |     0 |                       0 |            100.00% | 14.48% |
| REAL_PROXY_V1_DEV | C3.1      |     226 |     226 |     0 |                       0 |            100.00% | 13.99% |
| REAL_PROXY_V3_DEV | C2        |     205 |     200 |     5 |                       2 |             97.56% | 16.23% |
| REAL_PROXY_V3_DEV | C3        |     223 |     217 |     6 |                       3 |             97.31% | 17.61% |
| REAL_PROXY_V3_DEV | C3.1      |     219 |     214 |     5 |                       2 |             97.72% | 17.37% |
| VALIDATION        | C2        |  14,686 |  14,686 |     0 |                       0 |            100.00% | 37.44% |
| VALIDATION        | C3        |  14,686 |  14,686 |     0 |                       0 |            100.00% | 37.44% |
| VALIDATION        | C3.1      |  14,686 |  14,686 |     0 |                       0 |            100.00% | 37.44% |

C3.1 retains 14 of C3's 17 additional correct V3 emissions while
returning aggregate wrong and expected-NULL emission counts to C2's V3
checkpoint. That is a spent-data selection result, not evidence that
C3.1 generalizes or matches C2's safety. C2 remains the current
production candidate; C3 and C3.1 remain experimental. C3.1 is frozen
and requires a fresh, disjoint REAL_PROXY_V4 comparison before any
promotion claim. REAL_PROXY_V2 remains untouched by this development
pass.

### REAL_PROXY_V4 frozen C2/C3/C3.1 comparison

REAL_PROXY_V4 was sampled independently from Meta Kaggle `Users.csv`
with fixed seed `0x5245414C5F5634`. Sampling excluded every exact
display-name value in V1, V2, or V3. The 2,000-row source sample has
SHA-256
`234857bb418ddd3fe6b812b998ad514adf63569e81d62a873dfa4c6c5dc99a46`; the
2.59 GB source checksum remained
`30b95ff7d079289fe76a0fada39ebbb174f15f6f85a2e09f7a208c6fdf57dd82`
before and after sampling. Explicit set checks found zero exact V4
display-name overlap with V1, V2, or V3.

Both annotations were produced without classifier output or corpus
evidence. Their raw files were preserved and normalized mechanically
under the V2/V3 policy. Annotator A supplied 1,653 exact spans, 259
NULLs, 67 explicit skips, and 21 non-exact labels mapped to `SKIP`.
Annotator B supplied 1,319 exact spans, 306 NULLs, 27 explicit skips,
and 348 unusable or non-exact labels mapped to `SKIP`. Confidence and
notes were ignored.

Mechanical consensus yielded:

| Source rows | Greeting agreements | NULL agreements | Annotator-skip cases | Other disagreements | Evaluable | Skipped |
| ----------: | ------------------: | --------------: | -------------------: | ------------------: | --------: | ------: |
|       2,000 |               1,220 |             221 |                  439 |                 120 |     1,441 |     559 |

The deterministic holdout serialization was frozen before loading the
artifact, with SHA-256
`d95c589bec836faaeecaeda85b146989d2936914bff0209934f289ccb9446c7f`.
After a non-evaluating release build, the sole V4 classifier invocation
was:

```console
benchmarks/name-eval/target/release/name-eval \
  _wip/name-eval-artifact-c/c32-q8-surname-global \
  _wip/name-eval-real-proxy-v4 \
  --compare-sealed-c2-c3-c31-sha256=d95c589bec836faaeecaeda85b146989d2936914bff0209934f289ccb9446c7f \
  --sealed=_wip/real-proxy-v4/sealed.csv \
  --sealed-manifest=_wip/real-proxy-v4/sealed.manifest.csv
```

All algorithms used the frozen public threshold `0.78975882405736963`.
C2 used C1 candidate generation and the frozen C2 emission score; C3
used its frozen handle candidate generation and the same score; C3.1
applied its frozen `0.025` penalty only when C3's winner came from
handle segmentation:

| Algorithm | Emitted | Correct | Wrong | Expected-NULL emissions | Observed precision | Recall | Abstention |
| --------- | ------: | ------: | ----: | ----------------------: | -----------------: | -----: | ---------: |
| C2        |     213 |     210 |     3 |                       0 |             98.59% | 17.21% |     85.22% |
| C3        |     237 |     233 |     4 |                       0 |             98.31% | 19.10% |     83.55% |
| C3.1      |     227 |     224 |     3 |                       0 |             98.68% | 18.36% |     84.25% |

Relative to C2, C3 added 23 correct greetings and one wrong greeting,
raising recall by `1.89` points. C3.1 retained 14 of those additional
correct greetings, matched C2's three observed wrong greetings and zero
expected-NULL emissions, and raised recall by `1.15` points. Relative to
C3, its provenance gate withheld 10 emissions: 9 correct and the one
additional wrong emission.

The emitted-score buckets were:

| Algorithm | `0.789759–0.85` correct/wrong | `0.85–0.90` correct/wrong | `0.90–0.95` correct/wrong | `0.95–1.00` correct/wrong |
| --------- | ----------------------------: | ------------------------: | ------------------------: | ------------------------: |
| C2        |                         164/2 |                      45/1 |                       1/0 |                       0/0 |
| C3        |                         184/3 |                      47/1 |                       2/0 |                       0/0 |
| C3.1      |                         177/2 |                      46/1 |                       1/0 |                       0/0 |

V4 therefore independently reproduced the aggregate tradeoff C3.1 was
selected to provide on spent V3. C3.1 is promoted to the leading
classifier candidate; C2 and C3 remain permanently frozen baselines. The
tiny error counts do not establish population-level safety equivalence,
the agreement filter selects clearer labels, the machine annotators can
share cultural mistakes, and Meta Kaggle may not match the product
population. No row-level V4 predictions, failures, traces, or
changed-case comparisons were written or inspected. V4 is now spent
comparison evidence and must not tune C2, C3, or C3.1.

### Relational emission diagnosis before C4

After V4 became spent development evidence, a checksum-gated diagnostic
combined V1, V3, and V4 with synthetic VALIDATION. It did not load TEST,
create V5, change C3.1, or implement C4. The exact command was:

```console
benchmarks/name-eval/target/release/name-eval \
  _wip/name-eval-artifact-c/c32-q8-surname-global \
  _wip/name-eval-relational-c4-diagnostic-final \
  --diagnose-relational-emission \
  --spent-holdout=_wip/real-proxy-v1/sealed.csv \
  --spent-manifest=_wip/real-proxy-v1/sealed.manifest.csv \
  --spent-sha256=de95213f27fc1849032ee6788c8f16d7d515c1a991ae8b2e8414b7b155814c4e \
  --spent-holdout=_wip/real-proxy-v3/sealed.csv \
  --spent-manifest=_wip/real-proxy-v3/sealed.manifest.csv \
  --spent-sha256=d70e4d4b2ed7e49bed09dc1e8d2ba60ade8a752e3b86c772e964bd64883ee6fe \
  --spent-holdout=_wip/real-proxy-v4/sealed.csv \
  --spent-manifest=_wip/real-proxy-v4/sealed.manifest.csv \
  --spent-sha256=d95c589bec836faaeecaeda85b146989d2936914bff0209934f289ccb9446c7f
```

The command revalidated these frozen C3.1 checkpoints before searching:

| Population        | Emitted | Correct | Wrong | Expected-NULL emissions |
| ----------------- | ------: | ------: | ----: | ----------------------: |
| REAL_PROXY_V1_DEV |     226 |     226 |     0 |                       0 |
| REAL_PROXY_V3_DEV |     219 |     214 |     5 |                       2 |
| REAL_PROXY_V4_DEV |     227 |     224 |     3 |                       0 |
| COMBINED_SPENT    |     672 |     664 |     8 |                       2 |
| VALIDATION        |  14,686 |  14,686 |     0 |                       0 |

Sole-candidate status was not independently safe. Among native spent
proxy winners, the sole-candidate bucket contained 891 correct winners,
122 wrong winners, and 78 expected-NULL winners; C3.1 currently
abstained on 672 of the correct winners. A strict sole path could
recover 17 additional correct greetings with no observed new errors, but
it added nothing on VALIDATION.

Large-margin competition was substantially more useful. The best
zero-error operating point on the documented monotonic grid was:

```text
native candidate
candidate_count >= 2
winner_margin >= 0.50
candidate_quality >= 0.70
reliability >= 0.75
role_signal >= 0.40
all frozen C3.1 vetoes pass
```

As an additive path over C3.1, it recovered 124 correct COMBINED_SPENT
greetings and 1,609 VALIDATION greetings with zero observed new wrong or
expected-NULL emissions in either population. Lowering only reliability
to `0.70` recovered 160 spent-proxy greetings but introduced one new
wrong emission. The shared-threshold combined sole-or-dominant family
recovered 79 spent-proxy greetings at its best zero-error point, so the
dominant-only rule is the C4 development candidate.

Candidate quality did not independently define that safe region. With
margin, reliability, and role floors fixed, lowering quality from the
selected conservative tie-break of `0.70` to the searched floor `0.40`
changed no emission or outcome. The diagnostic therefore supports the
relational margin and reliability conditions, but not a claim that
candidate quality becomes useful conditionally within this grid.

Country hints could not be measured on the real proxies because all
three sealed samples have empty country and locale fields. On synthetic
VALIDATION, 37,801 of 40,047 hint-bearing rows allowed the same
candidate to be compared with and without hints. Candidate quality
changed by at least `0.05` in 23,582 rows, including 6,998 currently
abstained correct winners, while the median final-score change was zero.
This confirms that the frozen zero weight discards country-aware quality
on the synthetic distribution; it does not establish the same effect on
real proxy data.

The deterministic qualitative sample also explains the safety limits.
Wrong sole winners frequently arose when an unfamiliar expected given
name lacked usable corpus support and another token became the only
recognized candidate. Large-margin correct abstentions commonly looked
like ordinary person names, while sampled large-margin errors included
culturally diverse expected given names losing to another recognized
token. Handle segmentation included both useful names and unsafe
fragments, so relational alternatives remain native-only and do not
weaken C3.1's `0.025` provenance penalty.

Generated outputs include topology outcomes, percentile and categorical
feature summaries, the country audit, every operating point, selected
zero-error and one-error points, and a separate spent-only qualitative
review sample. These are development diagnostics over machine-generated
or machine-consensus proxy labels, not worldwide precision estimates.
The proposed operating point still requires untouched REAL_PROXY_V5
validation before any C4 promotion.

### C4 relational-emission development freeze

C4 implements the two selected relational paths as explicit additions
over frozen C3.1. It does not change candidate generation, ranking, the
selected winner, the C3.1 decision score or threshold, the segmented
candidate penalty, or any existing veto. The checksum-gated freeze run
used only spent V1/V3/V4 and synthetic VALIDATION:

```console
benchmarks/name-eval/target/release/name-eval \
  _wip/name-eval-artifact-c/c32-q8-surname-global \
  _wip/name-eval-c4-development-freeze \
  --freeze-c4-relational-emission \
  --spent-holdout=_wip/real-proxy-v1/sealed.csv \
  --spent-manifest=_wip/real-proxy-v1/sealed.manifest.csv \
  --spent-sha256=de95213f27fc1849032ee6788c8f16d7d515c1a991ae8b2e8414b7b155814c4e \
  --spent-holdout=_wip/real-proxy-v3/sealed.csv \
  --spent-manifest=_wip/real-proxy-v3/sealed.manifest.csv \
  --spent-sha256=d70e4d4b2ed7e49bed09dc1e8d2ba60ade8a752e3b86c772e964bd64883ee6fe \
  --spent-holdout=_wip/real-proxy-v4/sealed.csv \
  --spent-manifest=_wip/real-proxy-v4/sealed.manifest.csv \
  --spent-sha256=d95c589bec836faaeecaeda85b146989d2936914bff0209934f289ccb9446c7f
```

The frozen rules are:

```text
sole native:
    candidate_count == 1
    candidate_quality >= 0.75
    reliability >= 0.40
    role_signal >= 0.80

dominant native winner:
    candidate_count >= 2
    raw winner_margin >= 0.50
    candidate_quality >= 0.40
    reliability >= 0.75
    role_signal >= 0.40

both:
    native / non-segmented winner
    all frozen C3.1 vetoes pass
```

The dominant `0.40` quality floor is the lower bound of the completed
search grid. Re-evaluating the branch at `0.70` selected exactly the
same rows, so candidate quality did not establish independent
conditional discrimination for this rule.

| Population        | Branch          | Correct added | Wrong added | NULL FP added |
| ----------------- | --------------- | ------------: | ----------: | ------------: |
| REAL_PROXY_V1_DEV | sole native     |             9 |           0 |             0 |
| REAL_PROXY_V1_DEV | dominant winner |            36 |           0 |             0 |
| REAL_PROXY_V3_DEV | sole native     |             4 |           0 |             0 |
| REAL_PROXY_V3_DEV | dominant winner |            42 |           0 |             0 |
| REAL_PROXY_V4_DEV | sole native     |             4 |           0 |             0 |
| REAL_PROXY_V4_DEV | dominant winner |            46 |           0 |             0 |
| COMBINED_SPENT    | sole native     |            17 |           0 |             0 |
| COMBINED_SPENT    | dominant winner |           124 |           0 |             0 |
| COMBINED_SPENT    | combined unique |           141 |           0 |             0 |
| VALIDATION        | sole native     |             0 |           0 |             0 |
| VALIDATION        | dominant winner |         1,609 |           0 |             0 |
| VALIDATION        | combined unique |         1,609 |           0 |             0 |

The combined spent checkpoint is 813 emitted / 805 correct / 8 wrong / 2
expected-NULL emissions. VALIDATION is 16,295 emitted / 16,295 correct /
0 wrong / 0 expected-NULL emissions. Candidate-count conditions make the
two paths mutually exclusive, and the freeze run also checks this case
by case.

After the thresholds and reproduction assertions were fixed, the two
non-selecting qualitative examples both remained abstentions:
<!-- Redacted: Names that selected their given names but failed both relational paths. -->

`Olivier REDACTED` selected `Olivier` at the unchanged C3.1 score, and
`Baris REDACTED` selected `Baris`, but neither relational path passed.
The generated `c4_qualitative_diagnostics.json` records the emission
source, winner features, provenance, vetoes, and each branch condition.

C4 is frozen only as a development candidate. C3.1 remains the leading
independently validated classifier and the application runtime remains
unchanged. C4 requires a one-shot untouched REAL_PROXY_V5 comparison
before any promotion.

### REAL_PROXY_V5 frozen pre-inference checkpoint

REAL_PROXY_V5 was sampled independently from the same checksum-pinned
Meta Kaggle `Users.csv` source with seed `0x5245414C5F5635`. Sampling
excluded every exact display-name value from V1, V2, V3, and V4. The
33,084,108-row source retained its SHA-256
`30b95ff7d079289fe76a0fada39ebbb174f15f6f85a2e09f7a208c6fdf57dd82`
before and after sampling. The 2,000-row V5 source has SHA-256
`e26c1e45c51ec87da4285110fd740a50b319149e4cfc5035862d3356d1a73c89`;
explicit set checks found zero exact value overlap with every prior
proxy source.

Both raw annotations were produced without classifier output or corpus
evidence and retained unchanged. Mechanical normalization accepted only
exact original-text greeting spans, NULL, and SKIP. It mapped 399 unique
cases with an unusable or non-exact value from at least one annotator to
SKIP. Consensus yielded:

| Source rows | Greeting agreements | NULL agreements | Annotator-skip cases | Other disagreements | Evaluable | Skipped |
| ----------: | ------------------: | --------------: | -------------------: | ------------------: | --------: | ------: |
|       2,000 |               1,193 |             247 |                  476 |                  84 |     1,440 |     560 |

The deterministic holdout serialization was frozen before loading the
artifact, with SHA-256
`69070614fee68401b896d6c5bfb4c22c55cca9744237f66213a9dd04291db6c7`. It
contains 1,193 expected greetings, 247 expected abstentions, and 560
skipped cases. No C3.1 or C4 inference had occurred when this digest and
the preceding counts were recorded.

The reviewed release evaluator was then invoked exactly once:

```console
benchmarks/name-eval/target/release/name-eval \
  _wip/name-eval-artifact-c/c32-q8-surname-global \
  _wip/name-eval-real-proxy-v5 \
  --compare-sealed-c31-c4-sha256=69070614fee68401b896d6c5bfb4c22c55cca9744237f66213a9dd04291db6c7 \
  --sealed=_wip/real-proxy-v5/sealed.csv \
  --sealed-manifest=_wip/real-proxy-v5/sealed.manifest.csv
```

It wrote only the aggregate report and summary CSV:

| Classifier | Emitted | Correct | Wrong | NULL FP | Precision | Recall | Abstention |
| ---------- | ------: | ------: | ----: | ------: | --------: | -----: | ---------: |
| C3.1       |     191 |     189 |     2 |       1 |    98.95% | 15.84% |     86.74% |
| C4         |     235 |     233 |     2 |       1 |    99.15% | 19.53% |     83.68% |

The exact additive C4-only delta was:

| Branch          | Additional emissions | Correct | Wrong | NULL FP | Incremental recall |
| --------------- | -------------------: | ------: | ----: | ------: | -----------------: |
| sole native     |                   11 |      11 |     0 |       0 |              0.92% |
| dominant winner |                   33 |      33 |     0 |       0 |              2.77% |
| combined        |                   44 |      44 |     0 |       0 |              3.69% |

C4 therefore receives classification **A — validated** as a classifier
candidate: both relational branches recovered unseen correct greetings,
and the combined 44 additional emissions introduced no observed wrong or
expected-NULL emission. This does not establish worldwide precision or
safety equivalence from small counts. It also does not change the
application runtime, which remains on C3.1 until a separate explicit
promotion change.

No V5 row-level prediction, failure, trace, changed case, or confidence
bucket was generated or inspected. The aggregate report has SHA-256
`af38ed1dfa815c21d9325c36c4acf66b9c7f45cde2447a09f05c3fb9fc5d166d`; the
aggregate CSV has SHA-256
`cbd6ae4c70ce040bc942dd33de8952d2a0ad2ff52bd3be143cf09a5f50613048`. V5
remains sealed unless a later task explicitly declares it spent.

### Production promotion

After the preceding aggregate-only V5 result was frozen, a separate
promotion changed the application default from C3.1 to the
already-frozen C4 implementation. It did not inspect V5 rows, alter
candidate generation or ranking, change the C3.1 score, modify either
relational rule, or change the artifact.

The public `greeting()` decision and plain CLI now use C4. Explicit
`greeting_at(...)` and `--threshold` calls retain their C3.1 score-only
semantics, so applying the C3.1 default threshold can legitimately
differ from the C4 default result. C2, C3, C3.1, and C4 remain
reproducible as frozen benchmark modes.

### C5 calibration-frontier diagnosis

After preserving the sealed V5 checkpoint above, the calibration study
explicitly marked V5 as spent and combined it with spent V1-V4. It did
not load TEST or V6 and did not change C4, candidate generation,
ranking, vetoes, the corpus, the artifact, or production behavior.

The development population is:

| Population        | Evaluable | Expected greeting | Expected NULL | Label provenance                        |
| ----------------- | --------: | ----------------: | ------------: | --------------------------------------- |
| REAL_PROXY_V1_DEV |     1,957 |             1,616 |           341 | one classifier-blind machine annotation |
| REAL_PROXY_V2_DEV |     1,496 |             1,217 |           279 | exact two-annotation machine consensus  |
| REAL_PROXY_V3_DEV |     1,474 |             1,232 |           242 | exact two-annotation machine consensus  |
| REAL_PROXY_V4_DEV |     1,441 |             1,220 |           221 | exact two-annotation machine consensus  |
| REAL_PROXY_V5_DEV |     1,440 |             1,193 |           247 | exact two-annotation machine consensus  |
| **Combined**      | **7,808** |         **6,478** |     **1,330** | generation-balanced fitting             |

V1's different label provenance remains visible throughout the report.
The proxy population contains no country or locale hints. Synthetic
VALIDATION remains a separate sanity population and never contributes to
fitting, proxy precision, or confidence intervals.

The combined frozen baselines are:

| Policy | Emitted | Correct | Wrong | NULL FP | Precision | Recall | False abstentions | Correct winner rejected |
| ------ | ------: | ------: | ----: | ------: | --------: | -----: | ----------------: | ----------------------: |
| C3.1   |   1,093 |   1,081 |    12 |       3 |    98.90% | 16.69% |             5,388 |                   4,217 |
| C4     |   1,318 |   1,305 |    13 |       3 |    99.01% | 20.15% |             5,163 |                   3,993 |

`Correct winner rejected` means that the row expects a greeting, the
already-selected winner matches that greeting, every frozen veto passes,
and the policy abstains. It therefore isolates calibration loss from
candidate-generation and ranking loss.

C4 rejects **3,993 / 6,478 expected greetings (61.64%)** for which the
winner is already correct and veto-free. This is the principal result:
C4 is a conservative reference point that discards substantial usable
ranking signal. The rejected-correct-winner bucket breaks down as:

| Feature           | Notable bucket | Count | Share of rejected correct winners |
| ----------------- | -------------- | ----: | --------------------------------: |
| Winner margin     | `0.50-1.00`    | 2,175 |                            54.47% |
| Candidate quality | `0.60-0.80`    | 2,274 |                            56.95% |
| Role signal       | `0.60-0.80`    | 1,825 |                            45.70% |
| Reliability       | `0.40-0.60`    | 1,302 |                            32.61% |
| Candidate count   | `1`            | 1,409 |                            35.29% |
| Candidate count   | `2`            | 2,171 |                            54.37% |

The generated report retains every fixed bin, combined and by proxy
generation. These bins are descriptive; they did not select model
thresholds.

Three deterministic calibration families were compared with frozen C4:

- the unchanged C3.1 scalar score at every distinct threshold;
- frozen C4 plus one native-only monotonic relational branch;
- a small generation-balanced, regularized, nonnegative logistic model,
  both alone and additively over C4.

Every policy may only emit the existing winner or abstain. All frozen
vetoes remain mandatory. The primary frontier aggregates disjoint
leave-one-generation-out predictions: each fold selects parameters on
four generations and evaluates them on the fifth.

| Training precision target | LOGO family   | OOF precision | Target met OOF | Recall | Correct | Wrong | NULL FP | False abstentions | Correct winner rejected | Wilson 95% interval |
| ------------------------: | ------------- | ------------: | :------------: | -----: | ------: | ----: | ------: | ----------------: | ----------------------: | ------------------: |
|                     99.9% | logistic      |        98.98% |       no       |  1.50% |      97 |     1 |       1 |             6,381 |                   5,201 |       94.44%-99.82% |
|                     99.5% | logistic      |        99.67% |      yes       | 13.91% |     901 |     3 |       2 |             5,576 |                   4,397 |       99.03%-99.89% |
|                     99.0% | controlled C4 |        98.84% |       no       | 33.02% |   2,139 |    25 |       7 |             4,321 |                   3,159 |       98.30%-99.22% |
|                     98.0% | controlled C4 |        97.91% |       no       | 49.07% |   3,179 |    68 |      22 |             3,253 |                   2,119 |       97.35%-98.34% |
|                     97.0% | controlled C4 |        96.93% |       no       | 53.04% |   3,436 |   109 |      31 |             2,964 |                   1,862 |       96.30%-97.44% |
|                     95.0% | controlled C4 |        95.67% |      yes       | 59.69% |   3,867 |   175 |      44 |             2,480 |                   1,431 |       95.00%-96.26% |
|                     90.0% | score only    |        89.97% |       no       | 75.90% |   4,917 |   548 |      93 |             1,106 |                     381 |       89.15%-90.74% |

The `99.9%` row is unsupported: only 98 emissions occurred and its
Wilson interval is too broad. A training target is not a guarantee that
the independently aggregated held-out folds meet that target, which is
why the `99%`, `98%`, `97%`, and `90%` misses are reported rather than
rounded away.

The score-only pooled descriptive frontier reached 9.46%, 16.18%,
22.01%, 34.79%, 52.92%, and 75.83% recall at observed 99.5%, 99%, 98%,
97%, 95%, and 90% precision. The controlled relational family reached
34.01%, 49.20%, 53.16%, 59.69%, and 68.14% recall at pooled 99%, 98%,
97%, 95%, and 90%. These pooled figures describe curve shape only; they
are not held-out evidence.

Cross-generation stability remains imperfect. The selected LOGO families
produced these held-out-generation ranges:

| Target | Precision range |  Recall range |
| -----: | --------------: | ------------: |
|  99.5% |  98.93%-100.00% | 11.51%-15.37% |
|  99.0% |   98.63%-99.41% | 28.16%-36.32% |
|  98.0% |   97.60%-98.69% | 46.23%-51.05% |
|  97.0% |   96.14%-97.88% | 50.56%-55.22% |
|  95.0% |   95.25%-96.61% | 55.26%-62.46% |
|  90.0% |   87.53%-91.67% | 72.96%-78.12% |

The full per-generation emitted, correct, wrong, NULL-FP, false-
abstention, and correct-winner-rejected counts are reproduced by the
generated report and CSVs.

A cost view using `correct - cost * wrong` preferred controlled
relational policies at wrong-emission costs 5x through 50x, with recall
falling from 59.23% to 34.01% as the cost increased. At 100x, the
preferred family was logistic at 13.99% recall. This is diagnostic only;
it does not encode a product decision.

The controlled relational model exposes substantially more usable
ranking signal, but this study does not establish a policy that strictly
dominates C4 near its existing precision. The 99% and 98% LOGO points
missed their targets slightly. The logistic model was selected only at
99.9% and 99.5%, where it reduced recall below C4. No balanced C5 point
is therefore frozen.

Two development candidates remain for explicit product discussion:

- a very-conservative 99.5% pure-logistic point: 901 correct / 3 wrong,
  99.67% OOF precision, and 13.91% OOF recall;
- an aggressive 95% controlled-relational point: 3,867 correct / 175
  wrong, 95.67% OOF precision, and 59.69% OOF recall.

On separate synthetic VALIDATION, frozen C4 remained 16,295 correct / 0
wrong. The conservative candidate produced 13,605 correct / 53 wrong;
the aggressive candidate produced 27,161 correct / 1,390 wrong. The
proxy rows contain no hints, so they cannot validate country-sensitive
calibration. On hinted synthetic rows, country evidence changed median
candidate quality by `+0.06298` while the median C3.1 score change was
zero, confirming that useful country-aware quality evidence is mostly
discarded by the current score form.

Only after selection, the two non-selecting qualitative examples were
checked.
<!-- Redacted: Names that abstain at the conservative point and emit at the aggressive point. -->

`Olivier REDACTED` and `Baris REDACTED` both remain abstentions at the
conservative point and both emit their expected selected candidate at
the aggressive point. They did not participate in fitting or policy
selection.

The diagnostic writes aggregate reports plus a development-only feature
table that contains neither display names nor source IDs. Two
independent runs produced byte-identical output. The final generated
`report.md` has SHA-256
`30cc22cb416bb8a7c3300412aa020c4a43d2ec779c42c6a151989a31fea4ec24`. No
C5 policy was implemented or frozen. Any selected future C5 still
requires untouched REAL_PROXY_V6 one-shot validation.

### Ordering and position evidence diagnosis

The next benchmark-only experiment kept C2, C3, C3.1, C4, candidate
generation, the corpus, and the artifact frozen. It reused only spent
REAL_PROXY_V1-V5 and separate synthetic VALIDATION to test whether token
position can shift the existing calibration frontier. It did not create
or inspect V6 or TEST and did not change production behavior.

The experiment adds these interpretable features to each existing
candidate span:

- initial/final token position and token-span proportion;
- position before or after the strongest competing candidate;
- a strict comma-inversion shape;
- agreement or conflict with a tiny CLDR-derived name-order prior.

The locale prior is derived from Unicode CLDR 48 person-name ordering
and likely-subtag data. The experimental region tables total 408 bytes;
unknown or malformed hints are neutral. All 7,808 proxy rows lack
country and locale hints, so their frontier can evaluate only generic
position and the single observed comma-inversion case. Locale-aware
ordering remains a synthetic sanity check, not real-proxy evidence.

Direct proxy correlation is strong but distribution-specific:

| Winner population          |  Rows | Initial |  Final |
| -------------------------- | ----: | ------: | -----: |
| Correct selected winner    | 5,313 |  96.97% | 19.89% |
| Wrong selected winner      |   727 |  29.85% | 68.36% |
| Expected-NULL winner       |   452 |  81.19% | 79.20% |
| Correct winner C4 rejected | 3,993 |  96.52% | 22.51% |

Two bounded ranking families were searched. The flat control adds a
small position adjustment; the confirmatory form multiplies that same
adjustment by frozen candidate quality, so weak candidates receive less
help. Total adjustment is clamped to `±0.06`. The full-development grid
selected the flat `+0.03` generic first-position adjustment and no
locale or comma adjustment. It raised the frozen ranking ceiling from
82.02% to 82.88% without changing the 5,742-case candidate-generation
ceiling. Leave-one-generation-out ranking selections remain recorded
separately.

Calibration then compared the existing C5 features with:

- additive position/order features;
- eight predeclared confirmatory interactions, including quality ×
  position, quality × margin, quality × reliability, and margin ×
  reliability;
- the same interaction model after the bounded ranking adjustment.

Every coefficient is nonnegative and L2-regularized. Position never
creates a candidate or bypasses a veto. The generation-held-out proxy
frontier is:

| Training target | Best ordering variant | OOF precision | Recall | Δ recall vs old frontier | Correct winner rejected |
| --------------: | --------------------- | ------------: | -----: | -----------------------: | ----------------------: |
|           99.9% | additive ordering     |        99.86% | 10.85% |                 +9.35 pp |                   4,595 |
|           99.5% | reranked interaction  |        99.34% | 37.03% |                +23.12 pp |                   2,955 |
|           99.0% | interaction ordering  |        98.94% | 53.54% |                +20.52 pp |                   1,830 |
|           98.0% | reranked interaction  |        97.96% | 65.27% |                +16.19 pp |                   1,126 |
|           97.0% | reranked interaction  |        96.83% | 71.26% |                +18.22 pp |                     738 |
|           95.0% | reranked interaction  |        94.85% | 77.09% |                +17.40 pp |                     360 |
|           90.0% | reranked interaction  |        89.92% | 81.24% |                 +5.34 pp |                      91 |

The target is selected on four generations, then reported on the omitted
generation. None of the aggregated held-out rows reaches its nominal
target exactly, so the targets must not be mistaken for achieved proxy
precision. Per-generation results and Wilson intervals remain explicit
in the generated CSVs.

The proxy gain does not transfer safely to synthetic structure. At the
99.5% selection target, the matching old frontier policy obtains 99.61%
precision on VALIDATION, while the ordering-enabled policy obtains
95.78%. Category metrics show severe errors on family-name-first and
surname-given cases. Generic first-position evidence is therefore
classified as **marginal** despite the large proxy gain: retain the
experiment for a future culture-aware interaction study, but do not
promote it or freeze C5.

After selection, qualitative smoke tests showed that:
<!-- Redacted: Names that first emit at the documented candidate points. -->

- `Olivier REDACTED` and `Baris REDACTED` emit at the 99% candidate
  point;
- `Alexandre REDACTED` emits at 99.5%;
- `Ngoc Lam REDACTED` first emits at 98%.

These examples did not participate in fitting or threshold selection.

Run the diagnostic with `--diagnose-ordering-evidence` and the same five
acknowledged spent-holdout triplets used by the C5 frontier command. It
writes aggregate feature correlations, the complete ranking grid,
leave-one-generation-out policies, matched frontier and synthetic
comparisons, model coefficients, hint/comma accounting, qualitative
smoke tests, complexity accounting, and `report.md`. No C5 policy was
implemented or frozen. Two independent release-mode runs produced
byte-identical output. The final generated `report.md` has SHA-256
`b0c895a7836c455071ab20d0a2a1ce28f8e93f67fe09a6bf1adc8e400c3c2425`.

## Metric definitions

- Greeting precision: correct emitted greetings / all emitted greetings.
- Greeting recall: correct emitted greetings / labeled person cases.
- Organization false-positive rate: organization cases with any emitted
  greeting / organization cases.
- Person false-negative rate: person cases where the classifier abstains
  / person cases. Wrong emitted greetings are counted separately and
  also reduce recall.
- Gender precision: correct gender emissions / gender emissions on
  gender-labeled cases. A gender attached to a wrong greeting is not
  correct.
- Gender coverage: gender emissions / gender-labeled cases.

Precision-target output includes emitted/correct/wrong counts and a
one-sided 95% Wilson lower bound. For each target, the evaluator
maximizes correct emissions only among observed thresholds whose lower
bound reaches the target; a small zero-error sample is not presented as
99%, 99.5%, or 99.9% quality.

Regression results are behavior checks, generated TEST is a held-out
synthetic name partition, and only a sealed real-world holdout can
support a final real-world quality claim.

## Recorded C0 and C1 results

The frozen C0 result remains the original baseline:

| Algorithm                 | C0_TEST emissions | Correct | Wrong | Precision | Recall | Organization FPR |
| ------------------------- | ----------------: | ------: | ----: | --------: | -----: | ---------------: |
| A frequency baseline      |            12,482 |   7,774 | 4,708 |    62.28% |  9.79% |            0.00% |
| B simple-signals baseline |            13,357 |  10,621 | 2,736 |    79.52% | 13.38% |            0.00% |
| C global-role baseline    |            38,384 |  38,376 |     8 |   99.979% | 48.34% |            0.00% |

For C, the median global role LLR was +2.203 for independently labeled
given candidates and -2.888 for disjoint competing first-name-index
candidates. The role statistic uses all 444,154,759 retained given
observations and all 489,631,377 non-empty surname observations as
denominators with add-0.5 smoothing.

That C0_TEST also establishes important unresolved limits.
Compound-given recall was 0% because its independently partitioned
compound fixtures lacked direct corpus support; hyphenated recall was
21.83%. At `0.85`, C emitted 33,207 correct greetings with no errors
(41.83% recall) on this synthetic holdout. These figures are not claims
about real-world quality.

On the expanded VALIDATION set, C1 at the selected threshold `0.93`
emitted 10,626 greetings, all correct, for 27.09% recall and zero
organization false positives. On the newly generated, one-shot TEST it
emitted 24,185 greetings, all correct, for 30.45% recall and zero
organization false positives. This TEST was easier than VALIDATION: its
observed zero-error threshold could be lowered to `0.784923` for 67.71%
recall, but that held-out result was not used to change the frozen
threshold.

At the common diagnostic threshold `0.80` on C0_TEST, C1 raised recall
from 48.34% to 53.26%, with 42,285 correct and 13 wrong emissions
(99.969% precision). Compound-given recall rose from 0% to 48.75%, and
hyphenated recall rose from 21.83% to 22.68%. The new TEST happened not
to distinguish C0 from C1 at `0.80`; it therefore confirms the shared
role baseline's precision on new fixture atoms but is not independent
evidence for C1's marginal compositional improvement.

The new TEST's aggregate categories still show weak coverage at the
selected threshold, especially apostrophe forms (0% recall) and
surname-comma-given forms (10.58%). These held-out findings are recorded
without tuning against their rows. No synthetic result substitutes for a
sealed real-world holdout.

The runtime lexical gate made 6,959 clean-v1 keys (0.386%) ineligible,
representing only 172,614 observations (0.039%). No additional
sanitation pass was warranted.
