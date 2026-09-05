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

## Capitalization evidence diagnostic

Capitalization was evaluated independently of the marginal ordering
experiment. C4, candidate generation and ranking, the artifact, and all
hard vetoes remained frozen. The diagnostic used the same 7,808 spent
V1-V5 proxy rows: 6,478 expected greetings and 1,330 expected NULLs.
Synthetic VALIDATION remained a separate regression population.

Each candidate and its strongest contextual competitor are classified
with Unicode properties as `all_upper`, `all_lower`, `title_like`,
`mixed_internal`, `uncased`, or `other`. Combining marks and accepted
name separators are neutral. Uncased scripts and spans mixing cased and
uncased scripts receive no casing support rather than a penalty.

| Availability                             | Count |   Rate |
| ---------------------------------------- | ----: | -----: |
| Any usable cased display token           | 7,742 | 99.15% |
| Usable selected candidate                | 6,482 | 83.02% |
| Usable selected candidate and competitor | 5,572 | 71.36% |
| Nonzero candidate/competitor contrast    |   493 |  6.31% |
| Alphabetic input entirely uncased        |    53 |  0.68% |

Contrast is uncommon and does not separate outcomes cleanly. A
title-like winner against an uppercase competitor occurs in 3.65% of
correct winners, 2.75% of wrong winners, 3.32% of expected-NULL winners,
and 3.98% of correct veto-free winners rejected by C4. Any nonzero
contrast occurs in 7.40%, 7.98%, 9.07%, and 8.26% respectively.

The ranking grid tested only bounded, evidence-modulated adjustments up
to `±0.04`; it never applied a flat case-class bonus. Leave-one-
generation-out selection retained a quality-gated ranker, while the
full-development grid selected `gate=quality;weight=0.04`. It changed
the frozen ranking result by only one case:

| Ranking        | Correct winner | Wrong winner | NULL winner | Ceiling |
| -------------- | -------------: | -----------: | ----------: | ------: |
| Frozen         |          5,313 |          727 |         453 |  82.02% |
| Capitalization |          5,314 |          726 |         453 |  82.03% |

Calibration compared additive casing features, four explicit
interactions with existing evidence, and the interaction model after the
bounded reranking. Every fitted coefficient is nonnegative and
L2-regularized. The generation-held-out frontier, compared with the best
previously established policy at each target, is:

| Target | Old precision | Old recall | Casing precision | Casing recall |     Delta |
| -----: | ------------: | ---------: | ---------------: | ------------: | --------: |
|  99.5% |        99.67% |     13.91% |           99.33% |        16.01% |  +2.10 pp |
|  99.0% |        98.84% |     33.02% |           99.07% |        19.79% | -13.23 pp |
|  98.0% |        97.91% |     49.07% |           97.68% |        31.80% | -17.27 pp |
|  97.0% |        96.93% |     53.04% |           96.88% |        39.77% | -13.28 pp |
|  95.0% |        95.67% |     59.69% |           94.84% |        57.58% |  -2.11 pp |
|  90.0% |        89.97% |     75.90% |           89.93% |        76.83% |  +0.93 pp |

The 99.5% casing point does not achieve its nominal target. At 99%, 98%,
97%, and 95%, casing does not improve the established frontier.
Interaction terms do improve a like-for-like logistic model, but that
comparison is insufficient because the established frontier also
contains controlled C4 relaxations.

Per-generation leave-one-generation-out results remain explicit in
`capitalization_logo_results.csv`. They are unstable at the strict end:
for example, the interaction model selected at the 99.5% training target
ranges from 98.61% observed precision on V1 to 100% on V5. These proxy
results are not worldwide population estimates.

The capitalization-only structural suite derives 119,132 evaluable rows
from synthetic VALIDATION without fitting on them. It exposes the
decisive regressions:

| Target | Policy | Transformation                   | Correct | Wrong | Recall |
| -----: | ------ | -------------------------------- | ------: | ----: | -----: |
|  99.5% | old    | all uppercase                    |  12,676 |    47 | 42.56% |
|  99.5% | casing | all uppercase                    |       0 |     0 |  0.00% |
|  99.5% | casing | expected title / remainder upper |  18,823 |    12 | 63.20% |
|  99.5% | casing | expected upper / remainder title |       0 |   314 |  0.00% |
|  99.0% | old    | all lowercase                    |  18,585 |   173 | 62.40% |
|  99.0% | casing | all lowercase                    |       0 |     0 |  0.00% |
|  99.0% | casing | expected title / remainder upper |  20,575 |    30 | 69.08% |
|  99.0% | casing | expected upper / remainder title |       0 |   495 |  0.00% |

Capitalization is therefore classified as **harmful / no value** for the
future C5 feature set at this stage. It adds no runtime data, and all
extraction and model allocations remain benchmark-only. C4 stays
unchanged; no C5 policy is implemented or frozen, and V6 remains
untouched.

The original qualitative examples were exercised locally after model
selection, then their identifying remainders were replaced with literal
`REDACTED` before the commit boundary. The local run confirmed that an
uppercase remainder can raise contrastive support, but those examples
did not affect fitting, selection, or the negative recommendation.

Run the diagnostic with `--diagnose-capitalization-evidence` and the
same five acknowledged spent-holdout triplets used by the C5 frontier
command. It writes the feature table, direct correlations, complete
ranking grid, model coefficients, generation-held-out results,
structural regressions, redacted qualitative output, and `report.md`.
Two independent release-mode runs produced byte-identical output. The
final generated `report.md` has SHA-256
`63e4e4aae3b5031ce100df65ab916e72ade15b9c49a3951f7156ca9ad807ad09`.

## Morphological role-evidence diagnostic

The morphology experiment remained separate from generic position and
capitalization. It did not change candidate generation, frozen ranking,
C4, the artifact, or production. It used exact aggregate counts from
`name-totals.csv`, pinned at SHA-256
`e43e8661261b2762d3d4f2581ebb803af94abb7505409873f46041be1470ff62`.

Training labels were deliberately conservative. A lexically eligible,
single-token retained key was given-like only when its exact given count
was at least 100 and its normalized role LLR was at least +2.0. It was
surname-like only when its exact overlapping-surname count was at least
100 and role LLR was at most -2.0. Ambiguous keys were excluded. Case-
folded, accent-folded groups stayed in one deterministic split; 41
conflicting groups were discarded rather than assigned a label.

| Split      | Given-like | Surname-like |   Total |
| ---------- | ---------: | -----------: | ------: |
| TRAIN      |     33,578 |       79,989 | 113,567 |
| VALIDATION |      4,274 |       10,062 |  14,336 |
| TEST       |      4,101 |        9,971 |  14,072 |

The deterministic grid covered Unicode-scalar character 2-3, 2-4, and
2-5-grams with boundary markers and 16K, 32K, 64K, or 128K signed-hash
buckets. A second locked grid varied FTRL alpha and L2 only after the
representation was selected on corpus-derived VALIDATION. The selected
model used 2-5-grams, 128K buckets, alpha 0.10, L2 0.1, and five epochs.

| Corpus-derived split | Accuracy | Balanced accuracy | ROC AUC |
| -------------------- | -------: | ----------------: | ------: |
| TRAIN                |   93.73% |            94.01% |  0.9855 |
| VALIDATION           |   86.28% |            85.19% |  0.9333 |
| TEST                 |   86.28% |            85.33% |  0.9347 |

The standalone TEST ROC AUC was 0.9266 for Latin, 0.9875 for Cyrillic,
and 0.9411 for Arabic. Greek had only 11 one-class TEST rows, while
other scripts had small or heterogeneous samples; those figures are
insufficient for broader international claims. Missing or unsupported
script evidence degrades to neutral. The attempted morphology-
reliability feature nevertheless saturated at 1.0 for nearly every proxy
winner and therefore provided no useful proxy discrimination.

The selected TRAIN vocabulary contained 365,922 unique 64-bit n-gram
hashes across 123,002 occupied buckets. This is a 66.39% feature-bucket
collision rate, with no observed primary 64-bit hash collision under an
independent secondary hash. The complete serialized f32 diagnostic model
is 540,829 bytes.

| Weights | Runtime payload | TEST ROC AUC | Signal p99 error |
| ------- | --------------: | -----------: | ---------------: |
| f32     |       540,676 B |       0.9347 |         0.000000 |
| int16   |       278,536 B |       0.9347 |         0.000080 |
| int8    |       147,464 B |       0.9348 |         0.020151 |

Quantization was not the limiting factor. Morphology itself separated
the aggregate proxy populations, but not well enough to improve the
existing classifier. Median morphology signals were +0.5603 for correct
selected winners, +0.1342 for wrong winners, -0.3322 for expected-NULL
winners, and +0.4710 for correct veto-free winners rejected by C4.

A bounded morphology ranking adjustment was selected independently in
each leave-one-generation-out fold. V2 and V5 selected zero weight; V1,
V3, and V4 selected 0.02. Across all held-out rows it reduced correct
winners from 5,313 to 5,311 and increased wrong winners from 727 to 729,
without changing the 5,742-case candidate-generation ceiling.

The generation-held-out calibration frontier compared the established
frontier with a morphology main effect and four predeclared interactions
with quality, role, reliability, and margin:

| Target | Existing precision | Existing recall | Morph precision | Morph recall |     Delta |
| -----: | -----------------: | --------------: | --------------: | -----------: | --------: |
|  99.5% |             99.67% |          13.91% |          99.67% |       13.91% |  +0.00 pp |
|  99.0% |             98.84% |          33.02% |          98.75% |       19.48% | -13.54 pp |
|  98.0% |             97.91% |          49.07% |          97.75% |       34.18% | -14.90 pp |
|  97.0% |             96.93% |          53.04% |          96.85% |       41.34% | -11.70 pp |
|  95.0% |             95.67% |          59.69% |          94.91% |       54.68% |  -5.02 pp |
|  90.0% |             89.97% |          75.90% |          89.85% |       75.81% |  -0.09 pp |

Interactions helped only relative to the weaker morphology main-effect
model: at the 99% target they raised recall from 17.24% to 19.48%, and
at 98% from 26.29% to 34.18%. Both remained far behind the established
frontier. The saturated reliability feature added no practical
uncertainty information. Exact generation-held-out coefficients and
model-form comparisons are preserved in the generated aggregate CSVs.

At 99%, the morphology interaction model ranged from 97.36% held-out
precision on V3 to 100% on V5, with recall from 15.26% to 23.94% across
generations. At 98%, held-out precision ranged from 96.15% to 99.49%.
The result is neither an outward frontier shift nor stable evidence of a
strict operating point. On separate synthetic VALIDATION, the full-
development morphology interaction selected at the 99% proxy target made
465 wrong emissions and reached only 95.92% precision.

The four motivating real-name examples were exercised locally only after
model and policy selection. Their repository-visible identities are
`REDACTED`:

- `REDACTED` case 1 selected the intended first span, but morphology
  scored it -0.5257; it emitted only at the existing 95%/90% points;
- `REDACTED` case 2 selected the intended first span, but morphology
  scored it -0.6862; it emitted only at the existing 90% point;
- `REDACTED` case 3 selected the intended first span; all three
  candidate signals were negative and it emitted only at 90%;
- `REDACTED` case 4 selected the intended first span with +0.9450
  morphology versus +0.5467 for its competitor; it emitted at the
  morphology 97% point and the existing 95%/90% points.

These examples did not train or select anything, and the temporary raw
inputs and test harness were removed after the local run.

Morphology is therefore classified as **harmful / no value** for the
future C5 feature set in this tested form. A role-spelling signal
exists, but it is substantially redundant with the artifact's direct
role evidence and degrades both held-out ranking and calibration. No
model is promoted, C4 stays production behavior, C5 is not frozen, and
V6 remains untouched.

Run the diagnostic with `--diagnose-morphology-evidence`, the pinned
`--morphology-name-totals=FILE`, and the same five acknowledged spent-
holdout triplets used by the C5 frontier command. It writes aggregate
label, grid, standalone, script, collision, quantization, proxy,
ranking, calibration, and synthetic reports plus the benchmark-only
model. Machine-dependent timing is isolated from deterministic output;
the observed release-mode cost was approximately 4.49 microseconds per
pre-normalized token on this machine. Two independent from-scratch runs
produced byte-identical deterministic output. The generated model has
SHA-256
`fed8c2e19a778931c7a090900991cbf29c76b0751bf36df3dcfbc60c75e01541`, and
the final generated `report.md` has SHA-256
`43d4f33a0b6ee43be914b83df9542b1cf853652e328b611bedf377630d403594`.

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

## C5 product operating-point selection

The feature-search phase ended with generic position classified as
marginal and generic capitalization and character morphology classified
as harmful or no-value. Locale-aware ordering remains unresolved. None
of those experimental features enters C5.

The selection study therefore reused only the established seven
calibration inputs: C3.1 decision score, candidate quality, winner
margin, role signal, reliability, sole-candidate status, and native
provenance. Candidate generation, ranking, normalization, hard vetoes,
the corpus, the artifact, and production C4 behavior remained frozen. V6
and TEST were not created, read, or evaluated.

The primary evidence is generation-held-out. For every target and
existing model family, each fold selected a policy using four spent
proxy generations and applied it only to the omitted generation. The
five disjoint held-out predictions form the OOF frontier. V1-V5 contain
7,808 evaluable rows: 6,478 expected greetings and 1,330 expected NULLs.

Frozen C4 emits 1,305 correct and 13 wrong greetings for 99.01%
precision and 20.15% recall on that development population. It falsely
abstains on 5,163 expected greetings, including 3,993 cases (61.64% of
all expected greetings) where the winner is already correct and every
hard veto passes.

### Dense OOF Pareto frontier

The training target is the constraint applied to each fold's four-
generation training population. Observed OOF precision is reported
without interpolation.

| Family        | Training target | OOF precision | Recall | Correct | Wrong | NULL FP | Correct winner rejected |
| ------------- | --------------: | ------------: | -----: | ------: | ----: | ------: | ----------------------: |
| logistic      |          99.50% |        99.67% | 13.91% |     901 |     3 |       2 |                   4,397 |
| logistic      |          99.30% |        99.29% | 15.11% |     979 |     7 |       3 |                   4,319 |
| logistic      |          99.10% |        99.08% | 16.67% |   1,080 |    10 |       4 |                   4,218 |
| logistic      |          99.00% |        98.98% | 18.05% |   1,169 |    12 |       4 |                   4,129 |
| controlled C4 |          99.00% |        98.84% | 33.02% |   2,139 |    25 |       7 |                   3,159 |
| controlled C4 |          98.80% |        98.80% | 36.92% |   2,392 |    29 |       6 |                   2,906 |
| controlled C4 |          98.70% |        98.58% | 38.68% |   2,506 |    36 |      11 |                   2,792 |
| controlled C4 |          98.60% |        98.55% | 40.00% |   2,591 |    38 |      11 |                   2,707 |
| controlled C4 |          98.50% |        98.36% | 41.65% |   2,698 |    45 |      11 |                   2,600 |
| controlled C4 |          98.40% |        98.32% | 43.32% |   2,806 |    48 |      13 |                   2,492 |
| controlled C4 |          98.30% |        98.27% | 43.73% |   2,833 |    50 |      13 |                   2,465 |
| controlled C4 |          98.20% |        98.16% | 45.28% |   2,933 |    55 |      15 |                   2,365 |
| controlled C4 |          98.10% |        97.99% | 47.33% |   3,066 |    63 |      20 |                   2,232 |
| controlled C4 |          98.00% |        97.91% | 49.07% |   3,179 |    68 |      22 |                   2,119 |
| controlled C4 |          97.75% |        97.90% | 50.26% |   3,256 |    70 |      24 |                   2,042 |
| controlled C4 |          97.50% |        97.63% | 50.93% |   3,299 |    80 |      31 |                   1,999 |
| controlled C4 |          97.25% |        97.43% | 51.59% |   3,342 |    88 |      35 |                   1,956 |
| controlled C4 |          97.00% |        96.93% | 53.04% |   3,436 |   109 |      31 |                   1,862 |

The complete dense sweep retains every requested training target. The
table above contains only nondominated, distinct emission signatures;
missing target rows were empirically dominated rather than interpolated.

### Product points

Three actual OOF points were retained for product comparison:

| Candidate    | Family / target     | Precision | Wilson 95%    | Recall | Correct / wrong | Wrong per 100 correct | Correct winner rejected |
| ------------ | ------------------- | --------: | ------------- | -----: | --------------: | --------------------: | ----------------------: |
| conservative | logistic / 99.10%   |    99.08% | 98.32%-99.50% | 16.67% |      1,080 / 10 |                 0.926 |                   4,218 |
| balanced     | controlled / 98.60% |    98.55% | 98.02%-98.95% | 40.00% |      2,591 / 38 |                 1.467 |                   2,707 |
| permissive   | controlled / 98.20% |    98.16% | 97.61%-98.58% | 45.28% |      2,933 / 55 |                 1.875 |                   2,365 |

The balanced point is selected as the C5 development candidate. It adds
1,286 correct and 25 wrong OOF emissions over C4 while reducing the
correct-veto-free-winner rejection count from 3,993 to 2,707. The
permissive point adds only another 342 correct emissions while adding 17
wrong emissions. On separate synthetic VALIDATION, permissive is also
dominated by balanced: it emits fewer correct and more wrong greetings.

Balanced OOF stability by omitted generation is:

| Held out | Emitted | Correct | Wrong | NULL FP | Precision | Recall |
| -------- | ------: | ------: | ----: | ------: | --------: | -----: |
| V1       |     590 |     582 |     8 |       4 |    98.64% | 36.01% |
| V2       |     550 |     538 |    12 |       2 |    97.82% | 44.21% |
| V3       |     528 |     518 |    10 |       2 |    98.11% | 42.05% |
| V4       |     503 |     498 |     5 |       1 |    99.01% | 40.82% |
| V5       |     458 |     455 |     3 |       2 |    99.34% | 38.14% |

The minimum generation precision is 97.82%, the maximum generation wrong
count is 12, and recall ranges from 36.01% to 44.21%. V1's label
provenance differs from V2-V5, so the per-generation rows remain more
important than a pooled precision claim.

Separate synthetic VALIDATION results are:

| Policy       | Emitted | Correct | Wrong | NULL FP | Precision | Recall |
| ------------ | ------: | ------: | ----: | ------: | --------: | -----: |
| C4           |  16,295 |  16,295 |     0 |       0 |   100.00% | 41.54% |
| conservative |  16,072 |  16,019 |    53 |       0 |    99.67% | 40.84% |
| balanced     |  24,243 |  23,596 |   647 |       0 |    97.33% | 60.16% |
| permissive   |  23,673 |  22,926 |   747 |       0 |    96.84% | 58.45% |

A loss of `false_abstentions + cost * wrong_emissions` selects 53.04%
recall at 5x, 50.26% at 10x and 20x, 36.92% at 50x, and 13.91% at 100x.
This table is decision support, not the selection mechanism.

### Frozen development configuration

The all-development refit corresponding to the selected OOF point is:

```text
schema=1
name=C5-balanced-controlled-calibration-v1
family=controlled_c4
training_target=0.98599999999999999
quality=0.69999999999999996
reliability=0.00000000000000000
role=0.00000000000000000
margin=0.50000000000000000
```

Its canonical configuration SHA-256 is
`427a15afb5c79846f80506f29b8d138a8c6969a8513c1d1dacf0ae1e491678b6`.
Operationally, C5 keeps every C4 emission and otherwise emits only a
native, veto-free selected winner with candidate quality at least 0.70
and, when multiple candidates exist, winner margin at least 0.50. The
reliability and role floors are zero because they did not constrain the
selected empirical point; they remain explicit frozen fields.

On all spent V1-V5 rows, this final single policy emits 2,527 correct
and 33 wrong greetings at 98.71% precision and 39.01% recall. Those are
development-fit metrics, not the OOF selection evidence and not fresh
validation. C4 remains production behavior. C5 requires a single
untouched REAL_PROXY_V6 comparison before any promotion.

The four motivating qualitative examples were exercised locally only
after selection. Repository-visible identities are literal `REDACTED`.
Every case selected its intended first span. The conservative and
balanced points abstained on all four; permissive emitted only
`REDACTED` case 1. That case had quality 0.6964 and margin 1.0000. The
other cases had margins 0.3053, 0.3075, and 0.3248. These examples did
not influence fitting or selection.

Run the selection with:

```console
cargo run --release --manifest-path benchmarks/name-eval/Cargo.toml -- \
  ARTIFACT OUTPUT --select-freeze-c5-operating-point \
  --spent-holdout=V1.csv --spent-manifest=V1.manifest.csv \
  --spent-sha256=de95213f27fc1849032ee6788c8f16d7d515c1a991ae8b2e8414b7b155814c4e \
  --spent-holdout=V2.csv --spent-manifest=V2.manifest.csv \
  --spent-sha256=7d704a646b8dd9fa3820f88b9504d4397b676af9435532cf2da9befda7663a73 \
  --spent-holdout=V3.csv --spent-manifest=V3.manifest.csv \
  --spent-sha256=d70e4d4b2ed7e49bed09dc1e8d2ba60ade8a752e3b86c772e964bd64883ee6fe \
  --spent-holdout=V4.csv --spent-manifest=V4.manifest.csv \
  --spent-sha256=d95c589bec836faaeecaeda85b146989d2936914bff0209934f289ccb9446c7f \
  --spent-holdout=V5.csv --spent-manifest=V5.manifest.csv \
  --spent-sha256=69070614fee68401b896d6c5bfb4c22c55cca9744237f66213a9dd04291db6c7
```

Two independent release-mode runs produced byte-identical output. The
final generated `report.md` has SHA-256
`d19a1acfbf1a58fdadc87df55f29d95f3de41167a108bc26ec0cb48ddf7fff48`.
Generated reports and proxy rows remain local and ignored.

## REAL_PROXY_V6 pre-inference freeze checkpoint

REAL_PROXY_V6 was sampled from the checksum-pinned Meta Kaggle
`Users.csv` using Python `random.Random` reservoir sampling with seed
`0x5245414C5F5636`. The source contained 33,084,108 rows. After 204
blank or whitespace-only rows and 118,341 rows whose exact display-name
value occurred in V1-V5 were excluded, 32,965,563 rows were eligible.
The resulting sample contains 2,000 unique display-name values and has
SHA-256
`82977f38c728e3c5f93b644522942720cf50025f4e9e0830f9bf373a682ed7ea`. Its
exact-value overlap with each of V1, V2, V3, V4, and V5 is zero.

The 2,587,424,211-byte source had SHA-256
`30b95ff7d079289fe76a0fada39ebbb174f15f6f85a2e09f7a208c6fdf57dd82` both
before and after sampling. The five exclusion-source SHA-256 values
were, in version order:

```text
V1  ccf7f2776355888c9f3c9d79cbb20a3ab9c3d354fb216a06ab75f814fa5bf182
V2  e658d2262c9f639a703be6e521d81e273c79d187694c9ed1a02da0fa4532879e
V3  9deefa258a64c873d833357e8f242f18fab01ca2eedfa8d2442a56d931d361e7
V4  234857bb418ddd3fe6b812b998ad514adf63569e81d62a873dfa4c6c5dc99a46
V5  e26c1e45c51ec87da4285110fd740a50b319149e4cfc5035862d3356d1a73c89
```

Two independent classifier-blind machine annotations were normalized
mechanically. The raw files remained unchanged at SHA-256
`0a3d406c1393f8abe87f4d6fbbc19b6b58bf1cbae0a284186b0139bb0592ee9f` and
`2772416f91c8338ce948e6498b4013bba2f6cc2efbaaf7c331873a9c814995c0`.
Annotator A supplied 1,579 exact greetings, 310 NULLs, 95 explicit
skips, and 16 unusable labels mapped to SKIP. Annotator B supplied 1,272
exact greetings, 346 NULLs, 27 explicit skips, and 355 unusable labels
mapped to SKIP.

Exact consensus produced 1,172 greeting agreements, 271 NULL agreements,
462 annotator-skip cases, and 95 other disagreements. The frozen holdout
therefore contains 2,000 cases: 1,443 evaluable and 557 skipped. Its
canonical sealed SHA-256, recorded before any classifier inference, is:

```text
a02d7105ea4f084e9d4ee94b3633e5068eb35e076dd7413b58b6d65549e734b1
```

No C4 or C5 inference had occurred when this checkpoint was written.

### One-shot C4/C5 result

The release comparator authenticated the frozen digest and invoked C4
and C5 exactly once on the same 1,443 evaluable cases. It wrote only an
aggregate summary and aggregate report; no case IDs, display names,
labels, predictions, failures, scores, traces, confidence buckets, or
changed-case rows were produced or inspected.

| Classifier | Emitted | Correct | Wrong | NULL FP | Precision | Wilson 95%    | Recall | Abstention | False abstentions | Correct winner rejected | Correct / wrong |
| ---------- | ------: | ------: | ----: | ------: | --------: | ------------- | -----: | ---------: | ----------------: | ----------------------: | --------------: |
| C4         |     263 |     259 |     4 |       1 |    98.48% | 96.16%-99.41% | 22.10% |     81.77% |               910 |                     703 |           64.75 |
| C5         |     491 |     484 |     7 |       3 |    98.57% | 97.09%-99.31% | 41.30% |     65.97% |               684 |                     478 |           69.14 |

C5 added 228 emissions over C4: 225 correct, three wrong, and two
expected-NULL false emissions. Expected-NULL false emissions are a
subset of wrong emissions. This is 75 additional correct greetings per
additional wrong greeting. C5 reduced false abstentions by 226 and
correct, veto-free winner rejections by 225.

**A - C5 validated.** The unseen recall gain is large and directionally
consistent with development, fresh precision remains in the intended
high-98% regime, and the added errors do not indicate an unexpected
safety collapse. C5 is now the leading classifier candidate. Production
remains on frozen C4 until a separate promotion change.

The case counts are still too small to claim worldwide precision, and
machine consensus preferentially retains cases on which both annotators
agree. In particular, C5's 98.57% observed precision has a 97.09%-99.31%
case-level Wilson interval. The three added wrong emissions and two
added expected-NULL emissions remain material product costs rather than
being hidden by the recall improvement.

The aggregate `report.md` has SHA-256
`86bf5c60222cfbdea744894c2a7fb8edd84d2eff26f19f5e225eabbd33aa4886`. The
aggregate `sealed_c4_c5_summary.csv` has SHA-256
`4613e4f2933043e1a90f4af224045b6b56db83c2bb6b27a156f50c9966e282f2`. V6
remains sealed and uninspected. Any future row-level diagnosis would
require a separate explicit decision to spend it.

### C5 production promotion

After the sealed V6 checkpoint above was committed, a separate change
promoted the already-frozen C5 policy to the default library and CLI
behavior. Production and this evaluator now use the same canonical C5
decision implementation. Existing C4 emissions retain their historical
`c3_1`, `sole_native`, or `dominant_winner` provenance; `c5` identifies
only an additional emission admitted by C5.

The promotion did not inspect V6 rows, alter C5's configuration, change
candidate generation or ranking, modify vetoes, or change the artifact.
C2, C3, C3.1, and C4 remain frozen historical benchmark modes.
