//! Thin CLI over [`bonjour::extract`]: reads a display name, prints the
//! extraction as JSON.

use std::process::ExitCode;

fn main() -> ExitCode {
    // The name comes from the arguments, joined to preserve spaces.
    let args: Vec<String> = std::env::args().skip(1).collect();
    let name = args.join(" ");

    if name.is_empty() {
        eprintln!("usage: bonjour <display name>");
        return ExitCode::from(2);
    }

    match serde_json::to_string_pretty(&bonjour::extract(&name)) {
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
