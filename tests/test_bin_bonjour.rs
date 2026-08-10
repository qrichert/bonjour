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
usage: bonjour [<options>] <display name>

Arguments:
  <display name>        Display name to extract a probable first name from.

Options:
  --country=<XX>        Hint with an ISO 3166-1 alpha-2 country code.
  --gender=<F|M>        Hint with a gender.
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
