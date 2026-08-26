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
  "greeting_name": "Quentin",
  "confidence": 0.8258187425766436,
  "gender_hint": "male",
  "gender_confidence": 0.9170640418908462
}
```

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
That's why `bonjour` returns confidence levels, and it's up to the user
to determine if the confidence is high enough for the use case. The
confidence is a decision score, not a calibrated probability.

To help guide the detection, `bonjour` accepts country and locale hints.
If you know the country, it can massively increase the confidence in its
detections and improve the associated gender hint.

## Usage

Expected output may be something like this:

```console
$ bonjour --json "Quentin Richert"
{
  "input": "Quentin Richert",
  "greeting_name": "Quentin",
  "confidence": 0.8258187425766436,
  "gender_hint": "male",
  "gender_confidence": 0.9170640418908462
}
```

The idea is that is also "detects", or at lease significantly reduces
confidences in company names, for instance:

```console
# The company marker 'SAS' significantly reduces confidence.
$ bonjour --json "Quentin Richert SAS"
{
  "input": "Quentin Richert SAS",
  "greeting_name": null,
  "confidence": 0.0,
  "gender_hint": null,
  "gender_confidence": 0.0
}
```

Unknown, unsafe, or ambiguous input can still produce a low-confidence
candidate in JSON:

```console
$ bonjour --json "Les Motards d'Alsace"
{
  "input": "Les Motards d'Alsace",
  "greeting_name": "Les",
  "confidence": 0.5695974113878561,
  "gender_hint": null,
  "gender_confidence": 0.0
}
```

The plain greeting applies the configured threshold and therefore keeps
using the complete display name here.

## Country and gender hints

Gender is not a property of a name alone — `Simone` is female in France,
male in Italy. Pass the user's country and/or gender as hints and they
resolve each other: a country pins the gender, a gender pins the
country.

```console
$ bonjour --json --country=IT "Simone Veil"
{
  "input": "Simone Veil",
  "greeting_name": "Simone",
  "confidence": 0.8100985093918445,
  "gender_hint": "male",
  "gender_confidence": 0.9422276178609842
}

$ bonjour --json --country=FR "Simone Veil"
{
  "input": "Simone Veil",
  "greeting_name": "Simone",
  "confidence": 0.8100985093918445,
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
  "greeting_name": "Simone",
  "confidence": 0.8100985093918445,
  "gender_hint": null,
  "gender_confidence": 0.0
}
```

The gender hint accepts `F` or `M`:

```console
$ bonjour --json --gender=M Simone
{
  "input": "Simone",
  "greeting_name": "Simone",
  "confidence": 0.8100985093918445,
  "gender_hint": "male",
  "gender_confidence": 0.714385674755892
}
```

`--locale=fr-FR` can provide the country fallback when no valid explicit
country hint is supplied. Hints guide the statistical evidence; they do
not force a greeting.

```text
usage: bonjour [--data-dir=PATH] [--country=XX] [--gender=F|M] [--locale=LOCALE] [--threshold=FLOAT | --json] <display name>
```

Plain greetings use a default threshold of `0.7897588240573696`. You can
choose a different operating point for your use case:

```console
$ bonjour --threshold=0.83 "Quentin Richert"
Bonjour Quentin Richert !
```

Lowering the threshold increases recall at the cost of potentially
unsafe greetings. `--json` reports the pre-threshold candidate and
confidence, so it is mutually exclusive with `--threshold`.

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

[Documentation] is available on docs.rs.

## Rust API

Runtime-loaded mode uses the same separately downloaded artifact:

```rust,no_run
use bonjour::Classifier;

let classifier = Classifier::from_dir("/path/to/bonjour-name-data-v1")?;
let inference = classifier.infer("Quentin Richert", Some("FR"), Some("fr-FR"));

if let Some(name) = inference.greeting() {
    assert_eq!(name, "Quentin");
}

let greeting = inference.greeting_at(0.83)?;
assert_eq!(greeting, None);

let simone = classifier.infer_with_gender(
    "Simone",
    None,
    None,
    Some(bonjour::GenderHint::Male),
);
assert_eq!(simone.gender_hint, Some(bonjour::GenderHint::Male));
# Ok::<(), Box<dyn std::error::Error>>(())
```

With the `standalone` feature, use `Classifier::standalone()` instead.
`inference.greeting_name` is the pre-threshold candidate; `greeting()`
applies the default, while `greeting_at(...)` applies your setting.
Every candidate is a non-empty contiguous span of the original input,
preserving its spelling, casing, accents, punctuation, normalization
form, and internal whitespace. `Classifier` is immutable, reusable, and
`Send + Sync`.

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

## TODO

- Evaluate gender inference independently of greeting selection. When
  greeting selection abstains but every plausible name candidate agrees
  on sufficiently strong gender evidence, a future model could still
  emit a gender hint. It must use candidate consensus rather than
  exposing the rejected greeting winner's gender. Current C3.1 behavior
  intentionally returns `gender_hint: null` and `gender_confidence: 0.0`
  whenever `greeting_name` is null.

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
[the artifact format and maintainer pipeline]: docs/name-data-format.md
