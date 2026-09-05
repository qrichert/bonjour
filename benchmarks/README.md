# Compact greeting-name corpus research

This directory records the experiments and decisions behind the
candidate embedded greeting-name classifier. Generated datasets,
archives, binary artifacts, reports, and build directories are
deliberately excluded from Git.

## Pipeline

```text
external names-dataset country CSVs
  -> scripts/extract_name_counts.py
  -> normalized_name_dataset.csv
  -> name-indexes / name-corpus-audit      representation and tail experiments
  -> name-clean-v1                         selected sanitation and 5/2 policy
  -> name-surname-v2                       selected global surname role evidence
  -> name-eval                             independent A/B/C/C1/C2/C3/C3.1/C4/C5 evaluation
  -> root crate                            frozen C5 production implementation
  -> name-runtime                          load/startup/inference benchmark
```

`scripts/extract_name_counts.py` validates the raw four-column country
files, drops empty first names and the surname field, and counts exact
`(first_name, gender)` tuples per country. It deliberately preserves
spelling, case, whitespace, accents, and separators.

`name-indexes` compared full-corpus storage representations.
`name-corpus-audit` then showed that most keys belonged to a
statistically weak long tail. `name-clean-v1` established a conservative
reusable corpus policy. `name-surname-v2` augmented only retained
first-name keys with surname usage. `name-eval` tests greeting inference
against labels independent of the corpus and provides an
artifact-independent workflow for manually labeling, checksum-freezing,
and aggregate-only evaluation of a locally retained real-world holdout.

## What was tried

### Full vocabulary and first-name evidence only

The normalized input contained 30,895,021 exact keys and 50,308,485
metadata rows. The original FST/MPHF comparison produced:

| Representation                                | Direct size |
| --------------------------------------------- | ----------: |
| FST + lossless packed metadata                |  613.46 MiB |
| MPHF + 64-bit fingerprint + lossless metadata |  592.32 MiB |
| MPHF + 64-bit fingerprint + q8 counts         |  474.20 MiB |

Switching membership fingerprints from 64 to 32 bits reduced the
unpruned C32+q8 layout to 356.34 MiB. The dominant problem was therefore
the corpus, not the string index. Of all exact keys, 23,502,128 (76.07%)
occurred once and represented only 4.79% of observations.

Threshold experiments showed that a global minimum of 5 retained 92.68%
of raw observations in 2,378,464 keys before sanitation/row pruning.
This made frequency pruning an evidence policy: extremely rare
observations are weak evidence that a value is a usable greeting name.

### clean-v1

The selected sanitation pass rejects only high-confidence contamination:
ASCII digits, controls, narrow URL/email markers, and strong legal-form
tokens. It preserves whitespace, compounds, Unicode letters/marks,
apostrophes, and hyphens. It applies global count `>=5`, row count
`>=2`, then rechecks the global threshold after row pruning.

| Measure          |                      clean-v1 result |
| ---------------- | -----------------------------------: |
| Retained names   |                            1,803,175 |
| Metadata rows    |                            8,722,920 |
| Observations     | 444,154,759 / 490,678,049 (90.5186%) |
| Direct C32 + q8  |                            33.22 MiB |
| zstd-19 artifact |                            19.60 MiB |

All 1,803,175 known-key lookups validated. No broader heuristic
sanitation was added. The later runtime lexical gate excludes only 6,959
retained keys (0.386%), carrying 172,614 observations (0.039%).

### First-name-only inference (Algorithms A and B)

Algorithm A ranked candidates primarily by global given-name frequency.
Algorithm B added country support, token position, compound/single-token
bonuses, candidate competition, and organization penalties. These
baselines showed that structural signals helped, but absolute first-name
frequency could still prefer `Martin` over `Elodie` in `Elodie Martin`.
They also exposed pathological compound and hyphenated categories and
score miscalibration.

### Global and country surname evidence

Surname data is not used to extract or index arbitrary surnames. It is
used only to answer whether a retained first-name candidate is also
strongly surname-shaped. Surname-only strings never become greeting
candidates.

Of the clean-v1 keys, 1,424,492 (78.9991%) occurred as surnames. Global
surname q8 metadata cost only 1.72 MiB, taking the direct artifact from
33.22 to 34.94 MiB. Country surname metadata increased it to 61.00 MiB,
so that variant was rejected for now: country-specific given evidence
remains available, while the evaluator has not justified another 26.06
MiB for country surnames.

The surname probability denominator is all 489,631,377 non-empty surname
observations, not merely the 364,386,816 observations whose strings
overlap a retained first-name key.

## Role-scoring foundation: Algorithms C0 and C1

Algorithms C0 and C1 remain evaluation baselines and form the ranking
foundation inherited by production C4 through its frozen C3.1 base.
Their input is conceptually:

```text
infer(display_name, country_hint, locale_hint)
```

It tokenizes the display name and generates contiguous one- and
two-token candidates. A candidate is considered only if it exists in the
clean-v1 first-name MPHF and consists of Unicode alphabetic
characters/marks, whitespace, apostrophe-like separators, or hyphen-like
separators.

For each candidate it decodes:

- global q8 given-name evidence;
- country-specific q8 given-name evidence when a country hint is
  available;
- global q8 surname evidence for the same retained first-name key;
- gender evidence.

Its central role feature is a smoothed log-likelihood ratio:

```text
role_llr(name)
  = ln((given_count + 0.5) / all_retained_given_observations)
  - ln((surname_count + 0.5) / all_nonempty_surname_observations)
```

The score combines this role signal with count reliability, country
given-name support, direct evidence for a two-token compound relative to
its component tokens, and the role margin against disjoint competing
first-name candidates. It does not try to identify or return the
surname. Unknown remainder tokens are ignored unless they provide
organization evidence.

The current pre-competition candidate score is exactly:

```text
role_signal = sigmoid((role_llr - 1.0) / 1.4)
reliability = clamp(log-scale(global_given_count, 5 .. 1,500,000))
country_support = ln(country_given_count + 1) / ln(global_given_count + 1)

score = 0.28
      + 0.56 * role_signal
      + 0.10 * reliability
      + 0.08 * country_support
```

A single-token display receives `+0.04`. A directly supported two-token
span receives up to `+0.18`, scaled by its role signal and its direct
count relative to the strongest component. Disjoint candidate role
competition contributes between `-0.12` and `+0.12`. Scores are clamped
to `[0, 1]`, ranked without a hardcoded name-order preference, then the
winner loses up to `0.08` when the runner-up is close.

Strong legal markers hard-abstain in C0/C1. Generic organization markers
and `&` multiply the final score by `0.12`. Unicode NFC, whitespace,
apostrophe-like, and hyphen-like punctuation are canonicalized. Lookup
tries canonical, title-case, lowercase, and accent-stripped forms.
Country hints take precedence over locale-region hints. Gender uses the
selected candidate's country counts when present, otherwise global
counts, and is emitted only when the dominant gender share is at least
`0.80`.

The classifier returns this uncalibrated score before thresholding. C0
is frozen as the direct-evidence role baseline. C1 preserves all of C0's
evidence and scoring, then adds two conservative compositional candidate
forms when the full span is absent from the first-name index:

- two adjacent whitespace-separated components, only when both have
  `role_llr >= 0.75` and the input has at least one remainder token;
- one hyphenated token whose two components both meet the same role
  floor.

For either form, the synthesized candidate uses the weaker component's
role signal, geometric-mean reliability/country support, and up to
`0.20` compositional evidence. A synthesized hyphenated token receives
an additional `0.04` structural bonus. Unsupported two-token whitespace
inputs are deliberately not combined: without direct phrase evidence,
`Mary Jane` cannot safely be distinguished from given + surname.

C1's historical synthetic operating threshold is frozen at `0.93`,
selected on VALIDATION. Production C4 uses the later proxy-calibrated
C3.1 decision policy as its base.

## Current evaluation result

C0 was selected using DEV and VALIDATION before its TEST was generated
and evaluated. That snapshot is now named `C0_TEST`. At the shared
diagnostic threshold `0.80`:

| Algorithm                 | C0_TEST emissions | Correct | Wrong | Precision | Recall | Organization FPR |
| ------------------------- | ----------------: | ------: | ----: | --------: | -----: | ---------------: |
| A frequency baseline      |            12,482 |   7,774 | 4,708 |    62.28% |  9.79% |            0.00% |
| B simple-signals baseline |            13,357 |  10,621 | 2,736 |    79.52% | 13.38% |            0.00% |
| C global-role baseline    |            38,384 |  38,376 |     8 |   99.979% | 48.34% |            0.00% |

The median C0_TEST role LLR was +2.203 for independently labeled given
candidates and -2.888 for disjoint competing candidates. At threshold
`0.85`, C emitted 33,207 correct greetings and no errors, with 41.83%
recall on this synthetic holdout.

The C0_TEST also records unresolved limits. Compound-given recall was 0%
because its independently partitioned compound fixtures lacked direct
corpus support. Hyphenated recall was 21.83%. These results are
synthetic and cannot replace a sealed, manually labeled real-world
holdout.

C1 was then developed against expanded DEV/VALIDATION compound and
hyphen fixtures. At its VALIDATION-selected threshold `0.93`, it
produced:

| Split      | Emitted | Correct | Wrong | Precision | Recall | Organization FPR |
| ---------- | ------: | ------: | ----: | --------: | -----: | ---------------: |
| VALIDATION |  10,626 |  10,626 |     0 |   100.00% | 27.09% |            0.00% |
| fresh TEST |  24,185 |  24,185 |     0 |   100.00% | 30.45% |            0.00% |

The fresh TEST used new atom-isolated fixtures, seed
`0x6576616c2d763034`, and a checksum frozen before its single
evaluation. It was easier than VALIDATION: its observed zero-error curve
reached 67.71% recall at `0.784923`, but the held-out result was not
used to change C1's threshold.

At the common `0.80` diagnostic threshold on C0_TEST, C1 increased
overall recall from 48.34% to 53.26%. It emitted 42,285 correct and 13
wrong greetings (99.969% precision); compound-given recall increased
from 0% to 48.75%. On the new TEST, C0 and C1 made identical decisions
at `0.80`, so that snapshot confirms the shared role baseline on new
atoms but does not independently establish C1's marginal improvement.

The selected-threshold fresh TEST still records coverage gaps:
apostrophe-form recall was 0%, surname-comma-given recall was 10.58%,
compound recall was 29.69%, and hyphenated recall was 31.49%. These
aggregate findings were preserved without inspecting or tuning against
TEST failure rows.

An independent 2,000-row Meta Kaggle display-name proxy then exposed a
large distribution shift. Frozen C1 at `0.93` emitted 36 of 1,616
expected greetings: 34 matched the proxy labels and 2 did not (94.44%
observed precision, 2.10% recall). After preserving that aggregate
checkpoint, the proxy was deliberately spent for diagnosis. Corpus
evidence covered 93.25% of expected greetings, C1 generated 74.75%, and
the correct candidate ranked first for 67.95%, but 1,064 correctly
ranked cases remained below `0.93`. Another 299 evidence-supported
labels were embedded in forms C1 does not generate, predominantly
whitespace-free handles. Details are in
[`name-eval/README.md`](name-eval/README.md). This inspected proxy is
now development evidence and cannot validate C2; the next frozen
algorithm requires a disjoint REAL_PROXY_V2.

C2 freezes a new emission score over unchanged C1 winners. It weights
role signal at 0.70, count reliability at 0.20, and normalized winner
margin at 0.10, with a three-letter safety floor and hard
generic-organization/ ampersand veto. On development evidence it
increased REAL_PROXY_V1_DEV from 34 correct / 2 wrong emissions to 207
correct / 0 wrong, and synthetic VALIDATION from 10,626 to 14,686
correct emissions with no observed errors. These are selection results,
not held-out quality evidence; C2 now requires a fresh disjoint
REAL_PROXY_V2. Full methodology and limitations are recorded in
[`name-eval/README.md`](name-eval/README.md).

The next evaluation layer draws a fixed 2,000-row REAL_PROXY_V2 after
excluding every exact V1 display-name value. Two independent,
classifier-blind annotations are merged mechanically: exact span or NULL
agreement is evaluable, while every disagreement or annotator skip
remains skipped. After checksum freezing, a digest-acknowledged command
compares frozen C1 and C2 on the identical agreed subset and writes only
aggregate metrics and algorithm-specific score buckets. This can test
fresh proxy generalization, but model agreement is not human-validated
worldwide ground truth and may select an easier subset.

The first frozen V2 comparison retained 1,496 exact-agreement cases and
skipped 504 disagreements, annotator skips, or unusable spans. C1
emitted 43 greetings (39 correct, 4 wrong; 3.20% recall), while frozen
C2 emitted 208 (206 correct, 2 wrong; 16.93% recall). Neither emitted on
an expected-NULL case. This supports C2's relative improvement on fresh
proxy evidence, not a 99% worldwide-production precision claim; V2 is
not threshold-tuning data.

The holdout workflow accepts only display name plus optional country and
locale hints, stores exact original-text spans, supports intentional
abstention and skipped labels, and freezes deterministic serialization
with a SHA-256 manifest before inference. Labeling cannot access
classifier or corpus evidence. The original sealed mode remains frozen
to C1 at `0.93`; the explicit V2 comparison mode evaluates frozen C1 and
C2 together. Both emit aggregate metrics and coarse score buckets only.
No real holdout or personal-data source is included in the repository.

C3 is the next candidate-generation-only experiment. It keeps C1
scoring/ranking and the permanently frozen C2 emission configuration,
but adds maximal corpus-backed handle segments exposed by ASCII digits,
`_`/`.`, or conservative Unicode lower-to-upper case boundaries. It does
not scan arbitrary substrings, split all-lower/all-uppercase
concatenations, repair repeated letters, or parse URL/email punctuation.
A camel-like part containing an all-uppercase component is rejected to
avoid unsafe acronym or credential suffix extraction.

On spent REAL_PROXY_V1_DEV, C3 increased matching-candidate generation
from 74.75% to 85.83%, correct pre-threshold selection from 67.95% to
78.47%, and emitted recall from 12.81% to 14.48%. It emitted 234/234
proxy greetings correctly with no expected-NULL emission; synthetic
VALIDATION remained exactly 14,686/14,686 correct emissions at 37.44%
recall. These are development-selection results, not held-out quality
evidence. C3 is frozen pending a fresh disjoint REAL_PROXY_V3; V2 was
not loaded, inspected, or used to tune it.

REAL_PROXY_V3 was then sampled with fixed seed `0x5245414C5F5633` after
excluding every exact V1 and V2 display-name value. Its two blind
machine annotations produced 1,232 exact greeting agreements, 242 NULL
agreements, and 526 skipped cases. The holdout was frozen at SHA-256
`d70e4d4b2ed7e49bed09dc1e8d2ba60ade8a752e3b86c772e964bd64883ee6fe`
before the single aggregate-only comparison.

On the 1,474 agreed V3 cases, frozen C2 emitted 205 greetings: 200
correct, 5 wrong, and 2 on expected-NULL cases (97.56% observed
precision, 16.23% recall). Frozen C3 emitted 223: 217 correct, 6 wrong,
and 3 on expected-NULL cases (97.31% observed precision, 17.61% recall).
C3 therefore generalized a modest `+1.38`-point recall gain and 17
additional correct greetings, but also added one wrong selection and one
expected-NULL emission. This supports the handle segmentation's
reachability benefit but does not establish unchanged safety or make C3
an unambiguous replacement for C2. No V3 failure rows were inspected,
and V3 cannot be used to change either frozen algorithm.

After preserving that checkpoint, V3 was spent specifically on the C3
delta. C2 abstained while C3 emitted in 18 cases: 17 matched the proxy
labels and one digit-boundary candidate was an expected-NULL emission.
The new emissions comprised 13 lower-to-upper, 3 digit, 1 dot, and 1
underscore segment. V1's corresponding 27 emissions were all correct, so
the single unsafe digit example does not justify mechanism-specific
weights.

C3.1 therefore keeps C3's candidate generation and ranking but subtracts
a uniform `0.025` from the frozen C2 emission score only when the winner
is a handle segment. Native winners retain their exact C2 score. This
raises the effective handle threshold to `0.8147588240573696` without
changing the shared public threshold. On spent V1, C3.1 emitted 226
correct / 0 wrong; on spent V3, it emitted 214 correct / 5 wrong with 2
expected-NULL emissions (97.72% observed precision, 17.37% recall); and
synthetic VALIDATION remained 14,686 correct / 0 wrong. C3.1 preserves
14 of C3's 17 additional V3 matches while returning aggregate error and
NULL-emission counts to C2's checkpoint. These are selection results,
not held-out validation. At that historical checkpoint C2 remained the
leading candidate and C3.1 still required fresh REAL_PROXY_V4 evidence.

REAL_PROXY_V4 was then drawn with fixed seed `0x5245414C5F5634` after
excluding every exact V1, V2, and V3 display-name value. Its two blind
machine annotations produced 1,220 exact greeting agreements, 221 NULL
agreements, and 559 skipped cases. The holdout was frozen at SHA-256
`d95c589bec836faaeecaeda85b146989d2936914bff0209934f289ccb9446c7f`
before the sole aggregate-only comparison.

On the 1,441 agreed V4 cases, frozen C2 emitted 213 greetings: 210
correct, 3 wrong, and none on expected-NULL cases (98.59% observed
precision, 17.21% recall). C3 emitted 237: 233 correct, 4 wrong, and no
expected-NULL emissions (98.31%, 19.10%). C3.1 emitted 227: 224 correct,
3 wrong, and no expected-NULL emissions (98.68%, 18.36%). Relative to
C2, C3.1 therefore added 14 correct greetings and `1.15` recall points
with the same observed wrong and NULL-emission counts. Relative to C3,
the provenance penalty withheld 10 emissions, including the one
additional wrong emission, while giving back 9 correct greetings.

This fresh comparison independently reproduces the aggregate behavior
C3.1 was selected to provide on spent V3, so C3.1 is promoted to the
leading classifier candidate and C2/C3 remain frozen comparison
baselines. Three errors versus three errors do not establish equal
population precision, and exact machine-annotation consensus selects a
clearer subset. No V4 row-level prediction or failure was written or
inspected; V4 is now spent comparison evidence and cannot tune any of
the three frozen algorithms.

C4 then added two explicit relational emission paths over the unchanged
C3.1 winner: a strict sole-native rule and a dominant multi-candidate
winner rule. Their thresholds were frozen on spent V1/V3/V4 and
VALIDATION before REAL_PROXY_V5 was sampled. V5 used seed
`0x5245414C5F5635`, excluded every exact V1-V4 display-name value, and
was frozen at SHA-256
`69070614fee68401b896d6c5bfb4c22c55cca9744237f66213a9dd04291db6c7`
before inference. Consensus retained 1,440 evaluable cases and
skipped 560.

The sole aggregate-only V5 comparison produced:

| Classifier | Emitted | Correct | Wrong | NULL FP | Precision | Recall |
| ---------- | ------: | ------: | ----: | ------: | --------: | -----: |
| C3.1       |     191 |     189 |     2 |       1 |    98.95% | 15.84% |
| C4         |     235 |     233 |     2 |       1 |    99.15% | 19.53% |

C4's 44 additional emissions all matched the proxy labels, with no
additional wrong or expected-NULL emission. Sole-native contributed 11
and dominant-winner contributed 33. C4 is therefore validated as the
leading classifier candidate on this one-shot proxy experiment. The
small error counts do not establish worldwide precision or safety
equivalence, and no V5 row-level result was generated or inspected. The
subsequent explicit production promotion made frozen C4 the default
application behavior without changing those rules. Detailed provenance
and aggregate hashes are recorded in
[`name-eval/README.md`](name-eval/README.md).

V5 was subsequently marked spent for a development-only C5 calibration
frontier study together with V1-V4. Across 7,808 proxy rows, frozen C4
emitted 1,305 correct and 13 wrong greetings for 20.15% recall. More
importantly, it abstained on 3,993 rows where the selected winner
already matched the expected greeting and every frozen veto passed:
61.64% of all expected greetings were therefore lost specifically at
emission calibration rather than candidate generation or ranking.

Generation-level leave-one-out analysis found a large but non-free
frontier. A controlled relational relaxation reached 33.02% recall at
98.84% out-of-fold precision and 59.69% recall at 95.67% precision. The
former narrowly missed its 99% target; the latter met its 95% target. A
monotonic logistic model met the 99.5% target at 99.67% precision but
only 13.91% recall, below C4. Consequently no balanced C5 policy was
frozen: C4 remains production behavior, and any later product-selected
point requires untouched REAL_PROXY_V6 validation. The detailed
frontier, Wilson intervals, false-abstention counts, per-generation
stability, cost analysis, and deterministic report hash are recorded in
[`name-eval/README.md`](name-eval/README.md).

An isolated ordering/position experiment then tested the same spent
V1-V5 rows without changing C4 or production. Initial position strongly
correlates with correct proxy winners, and small regularized interaction
models moved the held-out proxy frontier substantially: the best 99.5%
training-target point reached 37.03% recall at 99.34% observed OOF
precision, versus 13.91% recall for the previous frontier. A bounded
generic ranking adjustment also raised the correct-winner ceiling from
82.02% to 82.88%.

That gain did not generalize safely to synthetic ordering structures. At
the same 99.5% selection target, VALIDATION precision fell from 99.61%
for the matching old policy to 95.78% with ordering, with clear
family-name-first and surname-given regressions. The proxy population
has no country/locale hints and therefore cannot validate the tiny
CLDR-derived culture-aware prior. Ordering is retained only as a
marginal benchmark experiment; generic first-position evidence is not
promoted and no C5 policy is frozen. Detailed feature definitions,
per-generation results, synthetic categories, and reproducibility
instructions are recorded in
[`name-eval/README.md`](name-eval/README.md).

## Capitalization evidence experiment

The next isolated experiment tested Unicode-aware casing as contrastive
context, including regularized interactions with candidate quality, role
evidence, winner margin, and reliability. It used the same 7,808 spent
V1-V5 proxy rows and kept generic ordering out of the model.

Casing was broadly present but rarely contrastive: only 493 rows (6.31%)
had nonzero candidate/competitor casing contrast. A bounded casing
ranker moved the frozen ranking ceiling from 82.02% to only 82.03%.
Calibration did not improve the established frontier at the 99%, 98%,
97%, or 95% targets.

More importantly, the synthetic structural suite found severe
distribution-sensitive behavior. Uniform uppercase and lowercase
variants could collapse to zero recall, while reversing which span was
uppercase produced hundreds of wrong emissions. Capitalization is
therefore classified as **harmful / no value** for the future C5 feature
set. It remains benchmark-only; C4 production behavior is unchanged and
V6 remains untouched. Detailed results and the reproducibility digest
are recorded in [`name-eval/README.md`](name-eval/README.md).

## Morphological evidence experiment

A final isolated structural experiment trained a deterministic hashed
Unicode-scalar character n-gram model from exact aggregate given and
overlapping-surname counts. Conservative role labels excluded ambiguous
keys, and accent/case groups remained in disjoint TRAIN, VALIDATION, and
TEST partitions. The selected 128K-bucket 2-5-gram model achieved 0.9347
ROC AUC on its corpus-derived TEST set and serialized to about 528 KiB.

That standalone signal did not improve greeting inference. Bounded
morphology reranking lost two correct proxy winners. Morphology-aware
leave-one-generation-out calibration matched the established frontier
only at 99.5%; it reduced recall by 13.54 points at the 99% target and
14.90 points at 98%. Int16 quantization was effectively exact, so model
representation was not the limiting factor.

Morphology is therefore classified as **harmful / no value** for the
future C5 feature set in this tested form. The aggregate role-spelling
signal is real but redundant with direct corpus role evidence and
unstable for emission calibration. The model remains benchmark-only; C4
production behavior is unchanged, C5 is not frozen, and V6 remains
untouched. Full label construction, international breakdowns,
quantization, proxy distributions, per-generation results, redacted
qualitative observations, and reproducibility hashes are recorded in
[`name-eval/README.md`](name-eval/README.md).

## Current production model: C5

Production inference now uses exactly the evaluator's frozen C5 code.
The display name is NFC/punctuation/whitespace-canonicalized for lookup
while its original UTF-8 text is retained for output. Candidate lookup
tries canonical, title-case, lowercase, and accent-folded variants,
subject to the Unicode lexical eligibility gate.

Candidate generation consists of:

- exact contiguous one- and two-token spans present in the first-name
  MPHF;
- conservative whitespace/hyphen compounds whose two components are both
  statistically given-like;
- corpus-backed handle segments exposed only by ASCII digit runs, `_`,
  `.`, or safe Unicode lower-to-upper transitions.

The candidate ranker uses the C1 role model documented above. It
compares global given-name likelihood against global surname likelihood,
adds count reliability and country-specific given evidence, and uses
direct/compositional compound and competing-candidate evidence. It does
not parse, index, or return arbitrary surnames.

The winner is then passed through frozen C2 emission calibration:

```text
margin_signal = clamp(winner_margin / 0.5, 0, 1)

decision_score = 0.10 * margin_signal
               + 0.70 * role_signal
               + 0.20 * count_reliability
```

Generic organization evidence, an ampersand, or a candidate shorter than
three alphabetic characters vetoes this score. Strong legal markers
hard-abstain earlier. C3.1 subtracts `0.025` only when the winning
candidate came from handle segmentation and emits at
`0.78975882405736963`.

C4 keeps every C3.1 emission and adds two native-candidate paths:

```text
sole native:
    candidate_count == 1
    candidate_quality >= 0.75
    reliability >= 0.40
    role_signal >= 0.80

dominant native winner:
    candidate_count >= 2
    winner_margin >= 0.50
    candidate_quality >= 0.40
    reliability >= 0.75
    role_signal >= 0.40
```

Both paths retain every frozen C3.1 veto and emit the already-selected
winner.

C5 keeps every C4 emission and adds one native, veto-free path:

```text
candidate_quality >= 0.70

if candidate_count >= 2:
    winner_margin >= 0.50
```

`greeting()` uses C5. The returned greeting is the corresponding
contiguous span of the original input—not the canonical lookup string.
Gender is returned only alongside a default C5 emission and only when
its majority share meets the frozen `0.80` threshold. Otherwise the
classifier returns `None`, and the caller safely retains the complete
display name.

## Current storage choice

The selected production artifact is:

```text
1,803,175 clean-v1 first-name keys
+ global and country given-name q8 evidence
+ global surname q8 evidence for those same keys
+ gender evidence
+ C32 membership fingerprint
= 34.94 MiB direct
```

The root crate now consumes this fixed format in runtime-loaded and
standalone modes. Its twelve no-name-string constituents total
36,632,687 bytes; the small trusted production manifest is committed
under `data/name-v1/`. The large constituents remain outside the current
change until redistribution and the exact NOTICE are explicitly
approved. External bulk-input checksums are recorded in
[SOURCE_DATA.md](SOURCE_DATA.md), and the complete format/pipeline is
documented in [`docs/name-data-format.md`](../docs/name-data-format.md).

## C5 operating-point selection

After the generic position, capitalization, and character-morphology
experiments failed to earn promotion, a dense generation-held-out study
selected a C5 operating point using only the existing calibration
features. No corpus, artifact, candidate-generation, ranking, veto, or
production behavior changed.

The selected OOF point uses the controlled-C4 family at a 98.6% training
target. Across held-out V1-V5 predictions it emitted 2,591 correct and
38 wrong greetings for 98.55% precision and 40.00% recall, versus C4's
1,305 correct / 13 wrong and 20.15% recall. Its Wilson 95% precision
interval is 98.02%-98.95%; proxy precision is not a worldwide population
guarantee.

The final all-development C5 rule keeps C4 and adds a native, veto-free
path requiring candidate quality `>= 0.70` and, for multiple candidates,
winner margin `>= 0.50`. The frozen configuration SHA-256 is
`427a15afb5c79846f80506f29b8d138a8c6969a8513c1d1dacf0ae1e491678b6`. It
is a development candidate only. Production remains on C4 until an
untouched REAL_PROXY_V6 one-shot comparison validates or rejects C5.

The dense Pareto frontier, per-generation results, cost view, synthetic
VALIDATION results, exact configuration, reproduction command, and
deterministic report hash are recorded in
[`name-eval/README.md`](name-eval/README.md).

## REAL_PROXY_V6 C5 validation

The frozen balanced C5 development candidate was compared once against
production C4 on a fresh, value-disjoint, machine-consensus proxy
holdout. Its sealed SHA-256 is
`a02d7105ea4f084e9d4ee94b3633e5068eb35e076dd7413b58b6d65549e734b1`. Of
2,000 sampled rows, 1,443 were evaluable after exact annotation
consensus and 557 were skipped.

| Classifier | Correct | Wrong | NULL FP | Precision | Recall |
| ---------- | ------: | ----: | ------: | --------: | -----: |
| C4         |     259 |     4 |       1 |    98.48% | 22.10% |
| C5         |     484 |     7 |       3 |    98.57% | 41.30% |

C5 added 225 correct and three wrong greetings, including two added
expected-NULL false emissions. That is 75 additional correct greetings
per additional wrong greeting, and false abstentions fell from 910
to 684. The result validates C5 as the leading candidate, but does not
claim worldwide precision: C5's case-level Wilson 95% interval is
97.09%-99.31%, and exact machine consensus excludes ambiguous or
disagreed labels. Production remains on C4 until a separate promotion
change. No V6 row-level result was produced or inspected.

## C5 production promotion

After the aggregate-only V6 checkpoint was committed, the default
library and CLI policy was changed from frozen C4 to the already-frozen
C5 implementation. The production and benchmark paths share that
implementation. Candidate generation, ranking, evidence, vetoes,
artifact bytes, and every historical classifier remain unchanged.
