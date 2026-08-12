# Name index representation benchmark

This isolated prototype compares three immutable exact-name lookup
layouts over the normalized four-column dataset:

1. an FST mapped to lossless packed metadata;
2. an MPHF plus a 64-bit membership fingerprint and the same lossless
   metadata;
3. the MPHF layout with logarithmically quantized `u8` counts.

The metadata is columnar. Country IDs use one byte, gender uses two
bits, and per-name row ranges use `u32` offsets. Lossless counts use
unsigned varints plus per-name byte offsets. Quantized counts use one
byte and store the maximum source count needed to decode the logarithmic
scale.

Run it with a new output path:

```console
cargo run --release --manifest-path benchmarks/name-indexes/Cargo.toml -- \
  data/normalized_name_dataset.csv _wip/name-index-benchmark
```

The program refuses to overwrite its output. It parses the CSV twice but
does not create a sorted copy, keeping peak disk usage limited to the
three finished prototype directories. The generated `report.txt`
contains exact byte sizes, timings, and count-quantization error.

## Recorded full-corpus result

The input had 50,308,485 metadata rows, 30,895,021 exact-name keys, 105
countries, and maximum row count 1,584,172.

| Prototype                                     | Direct size |
| --------------------------------------------- | ----------: |
| FST + lossless packed metadata                |  613.46 MiB |
| MPHF + 64-bit fingerprint + lossless metadata |  592.32 MiB |
| MPHF + 64-bit fingerprint + q8 counts         |  474.20 MiB |

For q8 counts, p99 row-relative error was 1.54%, maximum relative error
was 4.17%, and signed aggregate bias was +0.0635%. These results
motivated C32, q8, and long-tail policy work; they are not the selected
production artifact.
