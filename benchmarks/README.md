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
  -> name-eval                             independent A/B/C0/C1 evaluation
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
against labels independent of the corpus.

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

## Current model: Algorithms C0 and C1

Algorithms C0 and C1 exist only in the evaluation harness; application
runtime behavior has not changed. Their input is conceptually:

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

C1's synthetic operating threshold is frozen at `0.93`, selected on
VALIDATION. It is not yet a production threshold.

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
TEST failure rows. A sealed, representative real-world holdout remains
the next evidence milestone.

## Current storage choice

The leading experimental artifact is therefore:

```text
1,803,175 clean-v1 first-name keys
+ global and country given-name q8 evidence
+ global surname q8 evidence for those same keys
+ gender evidence
+ C32 membership fingerprint
= 34.94 MiB direct
```

No generated artifact is committed because the application does not
consume this format yet and its binary layout may still change. Exact
evaluator inputs are pinned by the manifests under `name-eval/fixtures`;
external bulk-input checksums are recorded in
[SOURCE_DATA.md](SOURCE_DATA.md).
