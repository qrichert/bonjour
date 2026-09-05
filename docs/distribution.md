# Rust distribution and name data

`bonjour` supports two Rust distribution modes. Both use the same
versioned, authenticated name-data artifact and frozen classifier. They
differ only in whether the data is loaded at runtime or embedded while
compiling.

The default crates.io package compiles without name data. It there
provides ordinary Cargo dependency resolution, version selection, source
verification, and docs.rs API documentation.

The data is a separate GitHub release asset because the artifact is
about 35 MiB uncompressed and its compressed archive is still larger
than crates.io's package limit. Keeping it separate also allows several
installed applications to share one copy instead of embedding the same
data in each binary.

The separate-data mode is analogous to dynamic linking at a distribution
level, but it is not an operating-system shared library. The standalone
mode is analogous to static linking: Cargo embeds the authenticated
bytes in the compiled program.

## Distribution matrix

| Use case                    | Cargo feature | Build-time data       | Runtime data           |
| --------------------------- | ------------- | --------------------- | ---------------------- |
| GitHub release CLI          | `standalone`  | Release artifact      | None                   |
| Repository `just build`     | `standalone`  | `data/name-v1/files/` | None                   |
| `cargo install bonjour`     | default       | None                  | Extracted artifact     |
| Rust library, shared data   | default       | None                  | `Classifier::from_dir` |
| Rust binary, embedded data  | `standalone`  | `BONJOUR_DATA_DIR`    | None                   |
| Rust library, embedded data | `standalone`  | `BONJOUR_DATA_DIR`    | None                   |
| docs.rs                     | default       | None                  | Not loaded             |

## Name-data release

Download these matching assets from the same GitHub release as the crate
version:

```text
bonjour-name-data-v1.tar.zst
bonjour-name-data-v1.tar.zst.sha256
```

Verify the checksum before extracting the archive. The archive contains
one `bonjour-name-data-v1/` directory. Pass that directory itself to
`Classifier::from_dir` or `BONJOUR_DATA_DIR`.

Do not combine a crate version with an artifact from an unrelated
release. The loader and build script authenticate the exact manifest,
documentation, notice, constituent sizes, and SHA-256 digests pinned by
the crate.

## Runtime-loaded CLI

Install the small default package:

```console
$ cargo install bonjour --version 0.1.0
```

Extract `bonjour-name-data-v1.tar.zst` into the platform data location:

```text
macOS    ~/Library/Application Support/bonjour/name-v1
Linux    ${XDG_DATA_HOME:-~/.local/share}/bonjour/name-v1
Windows  %LOCALAPPDATA%\bonjour\name-v1
```

`%APPDATA%\bonjour\name-v1` is the Windows fallback when `LOCALAPPDATA`
is unavailable. The directory contents must be the archive root
contents, not an extra nested copy of `bonjour-name-data-v1/`.

For a nonstandard location, select it for one command:

```console
$ bonjour --data-dir=/path/to/bonjour-name-data-v1 "Quentin Richert"
```

Or set the runtime override:

```console
$ BONJOUR_DATA_DIR=/path/to/bonjour-name-data-v1 \
    bonjour "Quentin Richert"
```

The CLI reports every automatically searched location when the data is
missing. A malformed, modified, or mismatched artifact produces a typed,
actionable load error instead of silently changing classifier behavior.

## Runtime-loaded Rust library

Use the default dependency:

```toml
[dependencies]
bonjour = "0.1"
```

Load the artifact once and reuse the immutable classifier:

```rust,no_run
use bonjour::Classifier;

let classifier = Classifier::from_dir("/path/to/bonjour-name-data-v1")?;
let inference = classifier.infer("Quentin Richert", Some("FR"), None);
let greeting = inference.greeting().unwrap_or("Quentin Richert");
assert_eq!(greeting, "Quentin");
# Ok::<(), Box<dyn std::error::Error>>(())
```

The runnable equivalent is [`examples/runtime_loaded.rs`]:

```console
$ cargo run --example runtime_loaded -- \
    /path/to/bonjour-name-data-v1 "Quentin Richert"
Bonjour Quentin !
```

Reusable libraries should normally accept a `&Classifier` from their
caller or expose a constructor that accepts the data directory. This
leaves the final application in control of installation and avoids
embedding another copy of the artifact.

## Standalone GitHub binaries

The easiest self-contained installation is a binary from the matching
GitHub release. It already contains the name data and needs no
environment variable, data directory, or extraction step.

```console
$ bonjour "Quentin Richert"
Bonjour Quentin !
```

## Standalone Rust binary

Enable the feature in the binary crate:

```toml
[dependencies]
bonjour = { version = "0.1", features = ["standalone"] }
```

When building from crates.io, point Cargo at the extracted artifact:

```console
$ BONJOUR_DATA_DIR=/path/to/bonjour-name-data-v1 cargo build --release
```

Then initialize the embedded classifier:

```rust,no_run
use bonjour::Classifier;

let classifier = Classifier::standalone()?;
let inference = classifier.infer("Quentin Richert", Some("FR"), None);
assert_eq!(inference.greeting(), Some("Quentin"));
# Ok::<(), Box<dyn std::error::Error>>(())
```

`BONJOUR_DATA_DIR` is only a build input in this mode. After
compilation, the binary works when moved away from both the repository
and extracted artifact. The build fails clearly if data is missing or
does not match the manifest.

The runnable repository example is [`examples/standalone.rs`]:

```console
$ cargo run --features standalone --example standalone -- \
    "Quentin Richert"
Bonjour Quentin !
```

The repository example finds the versioned files in
`data/name-v1/files/`. Supplying `BONJOUR_DATA_DIR` overrides that
normal checkout location.

## Standalone Rust library

A Rust library can enable `standalone` in exactly the same dependency
entry:

```toml
[dependencies]
bonjour = { version = "0.1", features = ["standalone"] }
```

Its application or CI build must provide the artifact when Cargo
compiles the dependency:

```console
$ BONJOUR_DATA_DIR=/path/to/bonjour-name-data-v1 cargo build --release
```

Library code can then expose or reuse the embedded classifier:

```rust,no_run
use bonjour::{Classifier, LoadError};

pub fn load_classifier() -> Result<Classifier, LoadError> {
    Classifier::standalone()
}
```

Cargo features are additive across a dependency graph. Consequently, if
a library enables `bonjour/standalone`, the final consumer also inherits
its build-time data requirement. Library authors should enable it
deliberately and document that requirement; runtime loading is the more
flexible default for a general-purpose library.

## Building the standalone CLI from crates.io

The published CLI can also be compiled with embedded data directly
through `cargo install`:

```console
$ BONJOUR_DATA_DIR=/path/to/bonjour-name-data-v1 \
    cargo install bonjour --version 0.1.0 --features standalone
```

The resulting `bonjour` executable is self-contained.

## Python wheels

`pyjour` is the Python distribution, import package, and command name:

```console
$ pip install pyjour
```

Or, with uv:

```console
$ uv add pyjour
```

Every published wheel embeds the same authenticated artifact as a
standalone Rust build. Python users do not install the separate data
archive and do not set `BONJOUR_DATA_DIR`.

```python
import pyjour

inference = pyjour.infer("Quentin Richert", country_hint="FR")
print(inference.greeting_name)
```

`pyjour.infer()` returns a frozen summary containing the best candidate,
the greeting actually emitted by C5, the diagnostic decision score,
emission source, and gated gender evidence. `pyjour.infer_detailed()`
returns the same diagnostic structure as `bonjour --json`.

The command-line interface has the same greeting fallback and hint
semantics:

```console
$ pyjour "Quentin Richert"
Bonjour Quentin !

$ pyjour --json --country=FR "Quentin Richert"
```

Version 0.1.0 targets ordinary GIL-enabled CPython 3.12 and newer.
Wheels are built for x86-64 and Arm64 Linux, Intel and Apple Silicon
macOS, and x86-64 Windows. PyPy, free-threaded CPython, and source
distributions are not part of the initial release.

## Maintainer-owned artifact generation

The artifact schema, exact source columns, sanitation thresholds,
deterministic producer commands, binary constituents, and packaging
procedure are documented in [the name-data format guide]. Custom
artifacts are not a supported 0.1.0 runtime interface: the loader
accepts only the exact artifact pinned by this release.

[`examples/runtime_loaded.rs`]: ../examples/runtime_loaded.rs
[`examples/standalone.rs`]: ../examples/standalone.rs
[the name-data format guide]: name-data-format.md
