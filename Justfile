PREFIX := "/usr/local"

@_default:
    just --list --list-prefix '  '

# Clean project files
clean:
    cargo clean

alias r := run
# Build and run program
run *args:
    cargo run --quiet --features standalone -- {{ args }}

alias b := build
# Make optimized release build
build:
    cargo build --release --features standalone

alias l := lint
# Run various linting tools
lint:
    prek run --all-files

# Most stringent checks (includes checks still in development)
check:
    -rustup update
    cargo fmt
    cargo doc --no-deps --all-features
    cargo check --all-features
    cargo clippy --all-targets --all-features -- -D warnings -W clippy::all -W clippy::cargo -W clippy::complexity -W clippy::correctness -W clippy::nursery -W clippy::pedantic -W clippy::perf -W clippy::style -W clippy::suspicious -A clippy::missing-const-for-fn -A clippy::option_if_let_else
    just test
    just coverage-pct

alias t := test
# Run unit tests
test *args:
    cargo test --all-features -- {{ args }}
    uv run python -m unittest discover -s tests -p 'test_*.py'
    uv run python -m unittest discover -s benchmarks/name-eval/tests -p 'test_*.py'

# Build documentation
doc:
    cargo doc --document-private-items --all-features
    @echo file://`pwd`/target/doc/`basename \`pwd\` | sed 's/-/_/g'`/index.html

alias c := coverage
# Unit tests coverage report
coverage:
    cargo tarpaulin --engine Llvm --timeout 120 --skip-clean --out Html --output-dir target/ --all-features --exclude-files 'benchmarks/*'
    @echo file://`pwd`/target/tarpaulin-report.html

alias cpc := coverage-pct
# Ensure code coverage minimum %
coverage-pct:
    cargo tarpaulin --engine Llvm --timeout 120 --out Stdout --all-features --exclude-files 'benchmarks/*' --fail-under 95

# Install `bonjour`
install:
    install -d "{{ PREFIX }}/bin/"
    install ./target/release/bonjour "{{ PREFIX }}/bin/bonjour"

# Output binary name for use in CI
@ci-bin-name:
    echo "bonjour"
