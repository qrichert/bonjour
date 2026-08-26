# Production runtime benchmark

This small benchmark exercises the public production `Classifier` with
the pinned artifact. It reports runtime-directory and standalone load
time, post-warmup inference throughput, a deterministic emission
checksum, and the benchmark binary size. It does not use
platform-specific memory APIs.

Build and run with the same artifact available at build time:

```console
BONJOUR_DATA_DIR="$PWD/_wip/name-eval-artifact-c/c32-q8-surname-global" \
  cargo run --locked --release \
  --manifest-path benchmarks/name-runtime/Cargo.toml -- \
  _wip/name-eval-artifact-c/c32-q8-surname-global 100000
```

Measure peak resident memory externally:

```console
# Linux
/usr/bin/time -v benchmarks/name-runtime/target/release/name-runtime \
  _wip/name-eval-artifact-c/c32-q8-surname-global 100000

# macOS
/usr/bin/time -l benchmarks/name-runtime/target/release/name-runtime \
  _wip/name-eval-artifact-c/c32-q8-surname-global 100000
```

Record the exact target, `rustc -vV`, optimization profile, artifact
manifest digest, command, load times, throughput, binary size, and
external peak RSS when publishing a benchmark result.

## Initial production result

Measured on 2026-08-20 with Rust 1.93.0, target `x86_64-apple-darwin`,
release profile, manifest SHA-256
`3dc5d96d7d4ea5e52821c442f5086865d760f0820349f6f090a307fe8c743474`, and
10,000 iterations over eight inputs:

| Measurement        |     Runtime-loaded |         Standalone |
| ------------------ | -----------------: | -----------------: |
| Load time          |            0.156 s |            0.156 s |
| Lookups/second     |             43,679 |             43,304 |
| Nanoseconds/lookup |             22,894 |             23,093 |
| Emission checksum  | `30a68334aafcd025` | `30a68334aafcd025` |

The standalone benchmark binary was 37,664,848 bytes. `/usr/bin/time -l`
reported maximum resident set size of 75,935,744 bytes and peak memory
footprint of 38,752,256 bytes.

The matching `x86_64-unknown-linux-gnu` Rust 1.93.0 run produced:

| Measurement        |     Runtime-loaded |         Standalone |
| ------------------ | -----------------: | -----------------: |
| Load time          |            0.309 s |            0.290 s |
| Lookups/second     |             32,879 |             34,731 |
| Nanoseconds/lookup |             30,414 |             28,793 |
| Emission checksum  | `30a68334aafcd025` | `30a68334aafcd025` |

Its standalone benchmark binary was 37,676,936 bytes. Sampling `/proc`
during a separate identical run observed 75,272 KiB peak resident
memory. These are single-machine engineering measurements, not portable
performance guarantees.
