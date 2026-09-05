# bonjour

> [!CAUTION]
>
> Writing code wasn't gonna be the fun part of this project, so
> delegated that entirely to AI. The fun part was designing the
> inferance models, data packing, and the overall architecture. But I
> didn't write a single line of code. Maybe someday I'll dig into the
> code and clean it up, but for now that's that.

Experimental crate to extract first names from display names.

```console
$ bonjour "Quentin Richert"
Bonjour Quentin !
```

## Use Case

In social apps, it's often nicer UX to store display names instead of
separate first and last names.

The reason being, if you're a business or an association, filling-in
"first name" and "last name" is awkward, and if you have OCD like me,
this would be a major annoyance:

```
> First Name: Motorcycle
> Last Name: Club
```

Especially if the UI later greeted you like:

```
Hello Motorcycle!
```

### The Theoretical "Better" Way

Instead you could store a display name, which solves the greeting
problem for entities:

```
> Display Name: Motorcycle Club
> Hello Motorcycle Club!
```

But creates a new one for people:

```
> Display Name: Quentin Richert
```

You could greet people with their full name, but it's a bit unnatural
and less warm.

Instead, you'd want to greet them by their first name if you could
_identify their first name with high confidence_; which is exactly the
point of this project.

```console
$ bonjour --json "Quentin Richert"
{
  "input": "Quentin Richert",
  "best_candidate": "Quentin",
  "greeting_name": "Quentin",
  "decision_score": 0.8258187425766436,
  "decision": {
    "emission_source": "c3_1",
    "candidate_count": 2,
    "candidate_quality": 0.9341785978125992,
    "winner_margin": 0.7342375610072307,
    "margin_signal": 1.0,
    "role_llr": 3.1788968086994007,
    "role_signal": 0.8258296772199872,
    "reliability": 0.7386898426132628,
    "alphabetic_length": 7,
    "minimum_alphabetic_length": 3,
    "contributions": {
      "candidate_quality": 0.0,
      "winner_margin": 0.1,
      "role": 0.578080774053991,
      "reliability": 0.14773796852265256
    },
    "pre_veto_score": 0.8258187425766436,
    "post_veto_score": 0.8258187425766436,
    "segmented_candidate": false,
    "segmentation_mechanism": null,
    "segmented_candidate_penalty": 0.0,
    "vetoes": {
      "strong_organization_marker": false,
      "generic_organization_marker": false,
      "ampersand": false,
      "candidate_too_short": false
    },
    "sole_native": {
      "c3_1_abstained": false,
      "native_candidate": true,
      "candidate_count_pass": false,
      "candidate_quality_min": 0.75,
      "candidate_quality_pass": true,
      "winner_margin_min": null,
      "winner_margin_pass": true,
      "reliability_min": 0.4,
      "reliability_pass": true,
      "role_signal_min": 0.8,
      "role_signal_pass": true,
      "vetoes_pass": true,
      "passed": false
    },
    "dominant_winner": {
      "c3_1_abstained": false,
      "native_candidate": true,
      "candidate_count_pass": true,
      "candidate_quality_min": 0.4,
      "candidate_quality_pass": true,
      "winner_margin_min": 0.5,
      "winner_margin_pass": true,
      "reliability_min": 0.75,
      "reliability_pass": false,
      "role_signal_min": 0.4,
      "role_signal_pass": true,
      "vetoes_pass": true,
      "passed": false
    },
    "c5": {
      "c4_abstained": false,
      "native_candidate": true,
      "candidate_count": 2,
      "candidate_count_pass": true,
      "candidate_quality_min": 0.7,
      "candidate_quality_pass": true,
      "winner_margin_min": 0.5,
      "winner_margin_pass": true,
      "reliability_min": 0.0,
      "reliability_pass": true,
      "role_signal_min": 0.0,
      "role_signal_pass": true,
      "vetoes_pass": true,
      "passed": false
    }
  },
  "candidates": [
    {
      "candidate": "Quentin",
      "ranking_score": 0.9341785978125992,
      "signals": {
        "corpus_score": 0.9341785978125992
      }
    },
    {
      "candidate": "Richert",
      "ranking_score": 0.1999410368053685,
      "signals": {
        "corpus_score": 0.1999410368053685
      }
    },
    {
      "candidate": "Quentin Richert",
      "ranking_score": null,
      "signals": {
        "corpus_score": null
      }
    }
  ],
  "gender_hint": "male",
  "gender_confidence": 0.9170640418908462
}
```

`best_candidate` is the highest-ranked candidate before C5 decides
whether it is safe to greet. `greeting_name` is the candidate emitted by
C5, or `null` when C5 abstains. `decision_score` is the frozen C3.1
diagnostic score. The default C5 decision is recorded by
`emission_source`: `c3_1`, `sole_native`, and `dominant_winner` preserve
the historical C4 path, while `c5` identifies a candidate newly emitted
by C5. Each `ranking_score` orders competing candidates. These are
different model quantities, and none is a calibrated probability.

The `decision` object shows how the winning candidate reaches the C3.1
score: the winner margin, role evidence, and reliability feed the
weighted `contributions`; vetoes can then reduce the score to zero, and
segmented handle candidates can receive a provenance penalty. It also
shows every condition in C4's two additive relational paths and C5's
controlled-calibration path. Candidate length has no numeric bonus:
`alphabetic_length` is only checked against `minimum_alphabetic_length`
as a safety veto.

`candidates` also includes eligible source spans that the current corpus
cannot score. Their `ranking_score` and `signals.corpus_score` are
`null`, rather than a fabricated zero. The `signals` object leaves room
for future scorers, for example a first-name-ness model, without making
candidate enumeration depend on any one evidence source.

There's a very high chance "Quentin" is the first name here, so it's
overwhelmingly fine to write:

```
Hello Quentin!
```

## Why it's not that simple

- `display_name.split_whitespace()[0]` Well, some countries don't follow
  the _first-name-last-name_ convention, and even in countries that _do_
  follow it, some people may not. Moreover, it risks extracting "The"
  from "The Motorcycle Club".
- **A dictionary of first names.** First names can also be last names.
  In France, for instance, Martin is both a popular first name and a
  popular last name. So in "Jean Martin", which one is the first name?
- Another caveat is that in certain languages, the greeting agrees in
  gender and in number, which means you also have to know the gender of
  the name for a proper greeting. However, the same name can have a
  different gender based on the country the person is in. In France,
  "Simone" is unequivocally a woman's name, but, in Italy, "Simone" is a
  man's name (Simone (FR) = Simona (IT) and Simone (IT) = Simon (FR)).

... and many other things that can only be statistically answered.
That's why `bonjour` returns decision scores, and it's up to the user to
determine if the score is high enough for the use case. The decision and
ranking scores are not calibrated probabilities.

To help guide the detection, `bonjour` accepts country and locale hints.
If you know the country, it can massively increase the confidence in its
detections and improve the associated gender hint.

## Usage

Expected output may be something like this:

For readability, the following examples omit the production-decision
`emission_source`, `candidate_count`, `sole_native`, and
`dominant_winner` fields, plus the `c5` trace shown in the complete
diagnostic above.

```console
$ bonjour --json "Quentin Richert"
{
  "input": "Quentin Richert",
  "best_candidate": "Quentin",
  "greeting_name": "Quentin",
  "decision_score": 0.8258187425766436,
  "decision": {
    "candidate_quality": 0.9341785978125992,
    "winner_margin": 0.7342375610072307,
    "margin_signal": 1.0,
    "role_llr": 3.1788968086994007,
    "role_signal": 0.8258296772199872,
    "reliability": 0.7386898426132628,
    "alphabetic_length": 7,
    "minimum_alphabetic_length": 3,
    "contributions": {
      "candidate_quality": 0.0,
      "winner_margin": 0.1,
      "role": 0.578080774053991,
      "reliability": 0.14773796852265256
    },
    "pre_veto_score": 0.8258187425766436,
    "post_veto_score": 0.8258187425766436,
    "segmented_candidate": false,
    "segmentation_mechanism": null,
    "segmented_candidate_penalty": 0.0,
    "vetoes": {
      "strong_organization_marker": false,
      "generic_organization_marker": false,
      "ampersand": false,
      "candidate_too_short": false
    }
  },
  "candidates": [
    {
      "candidate": "Quentin",
      "ranking_score": 0.9341785978125992,
      "signals": {
        "corpus_score": 0.9341785978125992
      }
    },
    {
      "candidate": "Richert",
      "ranking_score": 0.1999410368053685,
      "signals": {
        "corpus_score": 0.1999410368053685
      }
    },
    {
      "candidate": "Quentin Richert",
      "ranking_score": null,
      "signals": {
        "corpus_score": null
      }
    }
  ],
  "gender_hint": "male",
  "gender_confidence": 0.9170640418908462
}
```

The idea is that is also "detects", or at lease significantly reduces
scores in company names, for instance:

```console
# The company marker 'SAS' reduces the decision score to zero.
$ bonjour --json "Quentin Richert SAS"
{
  "input": "Quentin Richert SAS",
  "best_candidate": null,
  "greeting_name": null,
  "decision_score": 0.0,
  "decision": {
    "candidate_quality": null,
    "winner_margin": null,
    "margin_signal": null,
    "role_llr": null,
    "role_signal": null,
    "reliability": null,
    "alphabetic_length": null,
    "minimum_alphabetic_length": 3,
    "contributions": null,
    "pre_veto_score": null,
    "post_veto_score": 0.0,
    "segmented_candidate": null,
    "segmentation_mechanism": null,
    "segmented_candidate_penalty": 0.0,
    "vetoes": {
      "strong_organization_marker": true,
      "generic_organization_marker": false,
      "ampersand": false,
      "candidate_too_short": false
    }
  },
  "candidates": [
    {
      "candidate": "Quentin",
      "ranking_score": 0.9237220385528407,
      "signals": {
        "corpus_score": 0.9237220385528407
      }
    },
    {
      "candidate": "SAS",
      "ranking_score": 0.33597018983124183,
      "signals": {
        "corpus_score": 0.33597018983124183
      }
    },
    {
      "candidate": "Richert",
      "ranking_score": 0.1999410368053685,
      "signals": {
        "corpus_score": 0.1999410368053685
      }
    },
    {
      "candidate": "Quentin Richert",
      "ranking_score": null,
      "signals": {
        "corpus_score": null
      }
    },
    {
      "candidate": "Richert SAS",
      "ranking_score": null,
      "signals": {
        "corpus_score": null
      }
    }
  ],
  "gender_hint": null,
  "gender_confidence": 0.0
}
```

Unknown, unsafe, or ambiguous input can still produce a low-scoring
candidate in JSON:

```console
$ bonjour --json "Les Motards d'Alsace"
{
  "input": "Les Motards d'Alsace",
  "best_candidate": "Les",
  "greeting_name": null,
  "decision_score": 0.5695974113878561,
  "decision": {
    "candidate_quality": 0.7024874196505136,
    "winner_margin": 0.22899529952191633,
    "margin_signal": 0.45799059904383266,
    "role_llr": 1.4333677703393182,
    "role_signal": 0.5767750286718791,
    "reliability": 0.6002791570657873,
    "alphabetic_length": 3,
    "minimum_alphabetic_length": 3,
    "contributions": {
      "candidate_quality": 0.0,
      "winner_margin": 0.045799059904383266,
      "role": 0.40374252007031536,
      "reliability": 0.12005583141315745
    },
    "pre_veto_score": 0.5695974113878561,
    "post_veto_score": 0.5695974113878561,
    "segmented_candidate": false,
    "segmentation_mechanism": null,
    "segmented_candidate_penalty": 0.0,
    "vetoes": {
      "strong_organization_marker": false,
      "generic_organization_marker": false,
      "ampersand": false,
      "candidate_too_short": false
    }
  },
  "candidates": [
    {
      "candidate": "Les",
      "ranking_score": 0.7024874196505136,
      "signals": {
        "corpus_score": 0.7024874196505136
      }
    },
    {
      "candidate": "Motards",
      "ranking_score": 0.4734921201285973,
      "signals": {
        "corpus_score": 0.4734921201285973
      }
    },
    {
      "candidate": "Les Motards",
      "ranking_score": null,
      "signals": {
        "corpus_score": null
      }
    },
    {
      "candidate": "Motards d'Alsace",
      "ranking_score": null,
      "signals": {
        "corpus_score": null
      }
    },
    {
      "candidate": "d'Alsace",
      "ranking_score": null,
      "signals": {
        "corpus_score": null
      }
    }
  ],
  "gender_hint": null,
  "gender_confidence": 0.0
}
```

The plain greeting applies frozen C5 and therefore keeps using the
complete display name here.

## Country and gender hints

Gender is not a property of a name alone — `Simone` is female in France,
male in Italy. Pass the user's country and/or gender as hints and they
resolve each other: a country pins the gender, a gender pins the
country.

```console
$ bonjour --json --country=IT "Simone Veil"
{
  "input": "Simone Veil",
  "best_candidate": "Simone",
  "greeting_name": "Simone",
  "decision_score": 0.8100985093918445,
  "decision": {
    "candidate_quality": 0.9827567736643067,
    "winner_margin": 0.676638131101677,
    "margin_signal": 1.0,
    "role_llr": 2.685054369563951,
    "role_signal": 0.7691664066737898,
    "reliability": 0.858410123600958,
    "alphabetic_length": 6,
    "minimum_alphabetic_length": 3,
    "contributions": {
      "candidate_quality": 0.0,
      "winner_margin": 0.1,
      "role": 0.5384164846716528,
      "reliability": 0.17168202472019162
    },
    "pre_veto_score": 0.8100985093918445,
    "post_veto_score": 0.8100985093918445,
    "segmented_candidate": false,
    "segmentation_mechanism": null,
    "segmented_candidate_penalty": 0.0,
    "vetoes": {
      "strong_organization_marker": false,
      "generic_organization_marker": false,
      "ampersand": false,
      "candidate_too_short": false
    }
  },
  "candidates": [
    {
      "candidate": "Simone",
      "ranking_score": 0.9827567736643067,
      "signals": {
        "corpus_score": 0.9827567736643067
      }
    },
    {
      "candidate": "Veil",
      "ranking_score": 0.30611864256262966,
      "signals": {
        "corpus_score": 0.30611864256262966
      }
    },
    {
      "candidate": "Simone Veil",
      "ranking_score": null,
      "signals": {
        "corpus_score": null
      }
    }
  ],
  "gender_hint": "male",
  "gender_confidence": 0.9422276178609842
}

$ bonjour --json --country=FR "Simone Veil"
{
  "input": "Simone Veil",
  "best_candidate": "Simone",
  "greeting_name": "Simone",
  "decision_score": 0.8100985093918445,
  "decision": {
    "candidate_quality": 0.9608364679342599,
    "winner_margin": 0.6647400631716429,
    "margin_signal": 1.0,
    "role_llr": 2.685054369563951,
    "role_signal": 0.7691664066737898,
    "reliability": 0.858410123600958,
    "alphabetic_length": 6,
    "minimum_alphabetic_length": 3,
    "contributions": {
      "candidate_quality": 0.0,
      "winner_margin": 0.1,
      "role": 0.5384164846716528,
      "reliability": 0.17168202472019162
    },
    "pre_veto_score": 0.8100985093918445,
    "post_veto_score": 0.8100985093918445,
    "segmented_candidate": false,
    "segmentation_mechanism": null,
    "segmented_candidate_penalty": 0.0,
    "vetoes": {
      "strong_organization_marker": false,
      "generic_organization_marker": false,
      "ampersand": false,
      "candidate_too_short": false
    }
  },
  "candidates": [
    {
      "candidate": "Simone",
      "ranking_score": 0.9608364679342599,
      "signals": {
        "corpus_score": 0.9608364679342599
      }
    },
    {
      "candidate": "Veil",
      "ranking_score": 0.296096404762617,
      "signals": {
        "corpus_score": 0.296096404762617
      }
    },
    {
      "candidate": "Simone Veil",
      "ranking_score": null,
      "signals": {
        "corpus_score": null
      }
    }
  ],
  "gender_hint": "female",
  "gender_confidence": 0.9226812409580454
}
```

With no hint and a name whose gender differs by country, `gender_hint`
is left `null` rather than guessed:

```console
$ bonjour --json Simone
{
  "input": "Simone",
  "best_candidate": "Simone",
  "greeting_name": "Simone",
  "decision_score": 0.8100985093918445,
  "decision": {
    "candidate_quality": 0.8365742000974182,
    "winner_margin": 1.0,
    "margin_signal": 1.0,
    "role_llr": 2.685054369563951,
    "role_signal": 0.7691664066737898,
    "reliability": 0.858410123600958,
    "alphabetic_length": 6,
    "minimum_alphabetic_length": 3,
    "contributions": {
      "candidate_quality": 0.0,
      "winner_margin": 0.1,
      "role": 0.5384164846716528,
      "reliability": 0.17168202472019162
    },
    "pre_veto_score": 0.8100985093918445,
    "post_veto_score": 0.8100985093918445,
    "segmented_candidate": false,
    "segmentation_mechanism": null,
    "segmented_candidate_penalty": 0.0,
    "vetoes": {
      "strong_organization_marker": false,
      "generic_organization_marker": false,
      "ampersand": false,
      "candidate_too_short": false
    }
  },
  "candidates": [
    {
      "candidate": "Simone",
      "ranking_score": 0.8365742000974182,
      "signals": {
        "corpus_score": 0.8365742000974182
      }
    }
  ],
  "gender_hint": null,
  "gender_confidence": 0.0
}
```

The gender hint accepts `F` or `M`:

```console
$ bonjour --json --gender=M Simone
{
  "input": "Simone",
  "best_candidate": "Simone",
  "greeting_name": "Simone",
  "decision_score": 0.8100985093918445,
  "decision": {
    "candidate_quality": 0.8365742000974182,
    "winner_margin": 1.0,
    "margin_signal": 1.0,
    "role_llr": 2.685054369563951,
    "role_signal": 0.7691664066737898,
    "reliability": 0.858410123600958,
    "alphabetic_length": 6,
    "minimum_alphabetic_length": 3,
    "contributions": {
      "candidate_quality": 0.0,
      "winner_margin": 0.1,
      "role": 0.5384164846716528,
      "reliability": 0.17168202472019162
    },
    "pre_veto_score": 0.8100985093918445,
    "post_veto_score": 0.8100985093918445,
    "segmented_candidate": false,
    "segmentation_mechanism": null,
    "segmented_candidate_penalty": 0.0,
    "vetoes": {
      "strong_organization_marker": false,
      "generic_organization_marker": false,
      "ampersand": false,
      "candidate_too_short": false
    }
  },
  "candidates": [
    {
      "candidate": "Simone",
      "ranking_score": 0.8365742000974182,
      "signals": {
        "corpus_score": 0.8365742000974182
      }
    }
  ],
  "gender_hint": "male",
  "gender_confidence": 0.714385674755892
}
```

`--locale=fr-FR` can provide the country fallback when no valid explicit
country hint is supplied. Hints guide the statistical evidence; they do
not force a greeting.

```text
usage: bonjour [--data-dir=PATH] [--country=XX] [--gender=F|M] [--locale=LOCALE] [--json] <display name>
```

Plain greetings use frozen C5. `--json` reports the selected candidate,
C3.1 diagnostic score, C5 emission source, and rule traces.

## Installation

There are two ways to install `bonjour`.

### Self-contained binaries

The binaries on the [latest GitHub release] include the name-data
artifact and work without a separate data installation. Download the
binary for your platform and run it directly:

```console
$ bonjour "Quentin Richert"
```

A repository build is self-contained too. `just build` deliberately
builds with the `standalone` feature and embeds the pinned artifact from
`data/name-v1/files/`:

```console
$ just build
$ ./target/release/bonjour "Quentin Richert"
```

The build fails with a clear error if the artifact is absent; it never
emits a supposedly standalone binary that can only fail at runtime.

### crates.io plus a separate artifact

Install the `bonjour` command from [crates.io] with Cargo. The package
deliberately omits the roughly 35 MiB artifact:

```console
$ cargo install bonjour
```

crates.io provides normal Cargo versioning, dependency resolution, API
documentation, and reproducible source builds. The data is distributed
separately because even its compressed release archive is larger than
crates.io's package limit. Runtime loading also lets multiple
applications share one installed copy instead of embedding the same data
in every binary.

Download `bonjour-name-data-v1.tar.zst` from the matching GitHub release
and extract it into the platform data directory checked by the CLI:

- macOS: `~/Library/Application Support/bonjour/name-v1`
- Linux: `${XDG_DATA_HOME:-~/.local/share}/bonjour/name-v1`
- Windows: `%LOCALAPPDATA%\bonjour\name-v1`, or `%APPDATA%` when
  `LOCALAPPDATA` is unavailable

If the artifact is missing or does not match the version pinned by the
crate, the CLI exits with an actionable error instead of silently
changing behavior.

The complete build and data matrix is:

| Context                         | Cargo features | Name data                               |
| ------------------------------- | -------------- | --------------------------------------- |
| GitHub release binary           | `standalone`   | Embedded; no setup                      |
| Repository `just build`         | `standalone`   | Embedded from `data/name-v1/files/`     |
| Default `cargo install bonjour` | default        | Platform data directory                 |
| docs.rs                         | default        | Not loaded while building documentation |

For a nonstandard runtime location, use `--data-dir` for one invocation
or set `BONJOUR_DATA_DIR`. The same environment variable can supply the
pinned artifact when explicitly building the crates.io package with
`standalone`:

```console
$ bonjour --data-dir=/path/to/bonjour-name-data-v1 "Quentin Richert"

$ BONJOUR_DATA_DIR=/path/to/bonjour-name-data-v1 \
    cargo install bonjour --features standalone
```

Repository linting, tests, and local documentation enable all features.
Crates.io package verification and docs.rs use default features because
the registry package intentionally excludes the embedded data.

[Documentation] is available on docs.rs. See the [distribution guide]
for complete runtime-loaded and standalone library and binary examples.

## Rust API

For runtime-loaded applications and reusable libraries, use the default
dependency:

```toml
[dependencies]
bonjour = "0.1"
```

Runtime-loaded mode uses the same separately downloaded artifact:

```rust,no_run
use bonjour::Classifier;

let classifier = Classifier::from_dir("/path/to/bonjour-name-data-v1")?;
let inference = classifier.infer("Quentin Richert", Some("FR"), Some("fr-FR"));

if let Some(name) = inference.greeting() {
    assert_eq!(name, "Quentin");
}

let simone = classifier.infer_with_gender(
    "Simone",
    None,
    None,
    Some(bonjour::GenderHint::Male),
);
assert_eq!(simone.gender_hint, Some(bonjour::GenderHint::Male));
# Ok::<(), Box<dyn std::error::Error>>(())
```

For a self-contained binary or library, enable `standalone`:

```toml
[dependencies]
bonjour = { version = "0.1", features = ["standalone"] }
```

Then provide the extracted artifact while Cargo compiles `bonjour`:

```console
$ BONJOUR_DATA_DIR=/path/to/bonjour-name-data-v1 cargo build --release
```

The resulting binary does not need `BONJOUR_DATA_DIR` or external data
at runtime:

```rust,no_run
use bonjour::Classifier;

let classifier = Classifier::standalone()?;
let inference = classifier.infer("Quentin Richert", Some("FR"), None);
assert_eq!(inference.greeting(), Some("Quentin"));
# Ok::<(), Box<dyn std::error::Error>>(())
```

The same feature works when `bonjour` is used from a library crate.
Reusable libraries should generally keep the default runtime-loaded
dependency and let the final application choose where data lives; a
library that enables `standalone` must document the build-time artifact
requirement to its users.

`inference.greeting_name` is the selected candidate before the emission
decision; `greeting()` applies frozen C5. Every candidate is a non-empty
contiguous span of the original input, preserving its spelling, casing,
accents, punctuation, normalization form, and internal whitespace.
`Classifier` is immutable, reusable, and `Send + Sync`.

## Name data

The artifact was derived from a separately maintained corpus assembled
primarily from public data and subsequently cleaned, aggregated, and
quantized. It contains no original rows or name strings.

The production artifact uses an MPHF, 32-bit fingerprints, and quantized
aggregate metadata, and cannot enumerate its key set. See [the artifact
format and maintainer pipeline] for its schema, deterministic
generation, limits, and packaging.

The corpus and binary artifact are statistical evidence, not identity
records or ground truth. Benchmark methodology and frozen classifier
history live in
[`benchmarks/name-eval`](benchmarks/name-eval/README.md).

Production uses frozen C5. On untouched REAL_PROXY_V6, C5 added 225
correct proxy-label matches over C4 for three additional wrong
greetings, including two additional expected-NULL emissions. Recall
increased from 22.10% to 41.30%. This machine-consensus proxy result is
not a claim of worldwide population precision; ambiguous annotator
disagreements were excluded.

## TODO

- Evaluate gender inference independently of greeting selection. When
  greeting selection abstains but every plausible name candidate agrees
  on sufficiently strong gender evidence, a future model could still
  emit a gender hint. It must use candidate consensus rather than
  exposing the rejected greeting winner's gender. Current C5 behavior
  intentionally returns `gender_hint: null` and `gender_confidence: 0.0`
  whenever the default greeting decision abstains.

## License

The source code is available under the [0BSD license](LICENSE).

Datasets distributed with or used to build this project are compiled
from publicly available information. They are not covered by the 0BSD
license; their contents remain subject to any applicable rights and
source terms.

The separately distributed name-data artifact is not covered by the
source-code license. Its exact notice ships inside the data archive.

[latest GitHub release]:
  https://github.com/qrichert/bonjour/releases/latest
[crates.io]: https://crates.io/crates/bonjour
[Documentation]: https://docs.rs/bonjour
[distribution guide]: docs/distribution.md
[the artifact format and maintainer pipeline]: docs/name-data-format.md
