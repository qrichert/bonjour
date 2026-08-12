# Greeting-name classifier evaluation harness

This isolated crate evaluates greeting-name inference against labels
that are independent of the clean-v1 statistical corpus. It does not
change corpus sanitation, C32 encoding, or application runtime behavior.

The harness contains four algorithms:

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

`fixtures/sealed-holdout.example.csv` defines the manually labeled
holdout format. Keep the real file outside the repository if its
contents must remain sealed. The default report emits aggregate sealed
metrics only: no sealed row is written to result, failure, or comparison
files. If a row is inspected in order to alter an algorithm, remove it
from the sealed file and add it to DEV or the regression corpus before
the next evaluation.

Both generators use SplitMix64 with the fixed seeds documented above,
with additional domain separation by split.

## Fixed artifact baseline

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
--sealed=/path/to/manually-labeled-holdout.csv
--development-only
```

The reference threshold exists only to make A/B/C0/C1 decision diffs
concrete. C1's synthetic operating threshold is frozen at `0.93`, chosen
on VALIDATION; it is not yet a production recommendation. The
`--development-only` mode omits the fresh TEST entirely.

The clean-v1 CSV is opened read-only to report how many exact keys and
observations would fail the lexical rule. It is never rewritten.

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
