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

## C4 production promotion result

Measured before and after the C4 production promotion on 2026-08-31 with
Rust 1.93.0, target `x86_64-apple-darwin`, release profile, manifest
SHA-256
`6e5864efc224bf31aaa849c2acce780f7790fe76a42e56af2361ab1c7efcaf2a`, and
50,000 iterations over the same eight inputs:

| Measurement                  | Runtime C3.1 | Runtime C4 | Standalone C3.1 | Standalone C4 |
| ---------------------------- | -----------: | ---------: | --------------: | ------------: |
| Load time                    |      0.206 s |    0.158 s |         0.164 s |       0.152 s |
| Lookups/second               |       37,657 |     36,654 |          40,082 |        38,411 |
| Nanoseconds/lookup           |       26,555 |     27,282 |          24,949 |        26,034 |
| Emission checksum            |   `8cc2e086` | `8cc2e086` |      `8cc2e086` |    `8cc2e086` |
| Allocation calls/lookup (C4) |            — |        129 |               — |           129 |
| Allocated bytes/lookup (C4)  |            — |  6,750.375 |               — |     6,750.375 |

The full emission checksum was `8cc2e086fc208425` in all four runs. The
single-run C4 throughput was 2.7% lower in runtime-loaded mode and 4.2%
lower in standalone mode; this does not distinguish the small relational
checks from ordinary measurement noise. The promoted inference path does
allocate: a separate 80,000-lookup counting pass observed 129 allocation
calls and 6,750.375 allocated bytes per lookup in both modes. Allocation
counting was disabled during the timed pass.

The artifact remained unchanged at 36,632,687 bytes. The standalone C4
benchmark binary was 37,665,008 bytes. This task records the existing
allocation behavior and does not optimize it.
