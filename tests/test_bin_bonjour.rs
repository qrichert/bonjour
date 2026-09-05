use std::process::Command;

const BONJOUR: &str = env!("CARGO_BIN_EXE_bonjour");

#[test]
fn help() {
    for flag in ["-h", "--help"] {
        let output = Command::new(BONJOUR).arg(flag).output().unwrap();

        assert!(output.status.success());
        assert_eq!(
            String::from_utf8_lossy(&output.stdout),
            "\
usage: bonjour [--data-dir=PATH] [--country=XX] [--gender=F|M] [--locale=LOCALE] [--json] <display name>

Arguments:
  <display name>        Display name to inspect.

Options:
  --data-dir=<PATH>     Exact bonjour-name-data-v1 directory.
  --country=<XX>        Two-letter country hint.
  --gender=<F|M>        Gender hint.
  --locale=<LOCALE>     Locale used as country fallback.
  --json                Print detailed inference as JSON.
  -h, --help            Show this message and exit.
  -V, --version         Show the version and exit.
"
        );
        assert!(output.stderr.is_empty());
    }
}

#[test]
fn version() {
    for flag in ["-V", "--version"] {
        let output = Command::new(BONJOUR).arg(flag).output().unwrap();

        assert!(output.status.success());
        assert_eq!(
            String::from_utf8_lossy(&output.stdout),
            format!("{} {}\n", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"))
        );
        assert!(output.stderr.is_empty());
    }
}

#[test]
fn malformed_invocation_uses_exit_two() {
    for arguments in [vec![], vec!["--unknown"]] {
        let output = Command::new(BONJOUR).args(arguments).output().unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(output.stdout.is_empty());
        assert!(String::from_utf8_lossy(&output.stderr).contains("usage: bonjour"));
    }
}

#[test]
fn removed_threshold_option_is_rejected() {
    let output = Command::new(BONJOUR)
        .args(["--threshold=0.8", "Quentin Richert"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("unknown option"));
}

#[cfg(not(feature = "standalone"))]
#[test]
fn explicit_missing_data_uses_exit_one() {
    let output = Command::new(BONJOUR)
        .args([
            "--data-dir=definitely-missing-bonjour-name-data",
            "Quentin Richert",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("MissingData"));
    assert!(stderr.contains("qrichert/bonjour/releases"));
}
