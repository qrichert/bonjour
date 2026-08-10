//! Thin CLI over [`bonjour::extract`]: reads a display name (plus optional
//! `--country=`/`--gender=` hints), prints the extraction as JSON.

use std::process::ExitCode;

// Thin argv-parsing shell over the library; exercised end-to-end, not by unit
// tests. Excluded from coverage (see the `tarpaulin_include` check-cfg in
// `Cargo.toml`) so the gate reflects library coverage.
#[cfg(not(tarpaulin_include))]
fn main() -> ExitCode {
    let mut country: Option<String> = None;
    let mut gender: Option<String> = None;
    let mut words: Vec<String> = Vec::new();

    // Hints are `--key=value` flags; everything else joins into the name, so
    // spaces in the display name are preserved.
    for arg in std::env::args().skip(1) {
        if let Some(value) = arg.strip_prefix("--country=") {
            country = Some(value.to_string());
        } else if let Some(value) = arg.strip_prefix("--gender=") {
            gender = Some(value.to_string());
        } else {
            words.push(arg);
        }
    }
    let name = words.join(" ");

    if name.is_empty() {
        eprintln!("usage: bonjour [--country=XX] [--gender=F|M] <display name>");
        return ExitCode::from(2);
    }

    let extraction = bonjour::extract(&name, country.as_deref(), gender.as_deref());
    match serde_json::to_string_pretty(&extraction) {
        Ok(json) => {
            println!("{json}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("bonjour: {err}");
            ExitCode::FAILURE
        }
    }
}
