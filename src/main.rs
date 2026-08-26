//! Command-line interface for frozen C3.1 greeting-name inference.

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use bonjour::Classifier;
#[cfg(not(feature = "standalone"))]
use bonjour::LoadErrorKind;
use serde::Serialize;

#[derive(Default)]
struct Arguments {
    data_dir: Option<PathBuf>,
    country: Option<String>,
    locale: Option<String>,
    gender: Option<bonjour::GenderHint>,
    threshold: Option<f64>,
    json: bool,
    display_parts: Vec<String>,
}

#[derive(Serialize)]
struct Output<'a> {
    input: &'a str,
    greeting_name: Option<&'a str>,
    confidence: f64,
    gender_hint: Option<bonjour::GenderHint>,
    gender_confidence: f64,
}

#[cfg(not(tarpaulin_include))]
fn main() -> ExitCode {
    match run(env::args().skip(1)) {
        Ok(()) => ExitCode::SUCCESS,
        Err((code, message)) => {
            eprintln!("bonjour: {message}");
            ExitCode::from(code)
        }
    }
}

#[cfg(not(tarpaulin_include))]
fn run(arguments: impl Iterator<Item = String>) -> Result<(), (u8, String)> {
    let Some(arguments) = parse_arguments(arguments)? else {
        return Ok(());
    };
    let display_name = arguments.display_parts.join(" ");
    if display_name.is_empty() {
        return Err((2, usage_line()));
    }
    let classifier = load_classifier(arguments.data_dir)?;
    let inference = classifier.infer_with_gender(
        &display_name,
        arguments.country.as_deref(),
        arguments.locale.as_deref(),
        arguments.gender,
    );
    if arguments.json {
        let output = Output {
            input: &display_name,
            greeting_name: inference.greeting_name,
            confidence: inference.confidence,
            gender_hint: inference.gender_hint,
            gender_confidence: inference.gender_confidence,
        };
        let json = serde_json::to_string_pretty(&output)
            .map_err(|error| (1, format!("cannot serialize inference: {error}")))?;
        println!("{json}");
    } else {
        let greeting_name = match arguments.threshold {
            Some(threshold) => inference
                .greeting_at(threshold)
                .map_err(|error| (2, error.to_string()))?,
            None => inference.greeting(),
        };
        println!(
            "Bonjour {} !",
            greeting_name.unwrap_or(display_name.as_str())
        );
    }
    Ok(())
}

#[cfg(not(tarpaulin_include))]
fn parse_arguments(
    arguments: impl Iterator<Item = String>,
) -> Result<Option<Arguments>, (u8, String)> {
    let mut parsed = Arguments::default();
    for argument in arguments {
        if matches!(argument.as_str(), "-h" | "--help") {
            help();
            return Ok(None);
        }
        if matches!(argument.as_str(), "-V" | "--version") {
            version();
            return Ok(None);
        }
        if let Some(value) = argument.strip_prefix("--data-dir=") {
            parsed.data_dir = Some(PathBuf::from(value));
        } else if let Some(value) = argument.strip_prefix("--country=") {
            parsed.country = Some(value.to_string());
        } else if let Some(value) = argument.strip_prefix("--locale=") {
            parsed.locale = Some(value.to_string());
        } else if let Some(value) = argument.strip_prefix("--gender=") {
            parsed.gender = bonjour::GenderHint::parse(value);
        } else if let Some(value) = argument.strip_prefix("--threshold=") {
            if parsed.threshold.is_some() {
                return Err((2, "--threshold may only be supplied once".to_string()));
            }
            let threshold = value
                .parse::<f64>()
                .map_err(|_| (2, format!("invalid greeting threshold {value:?}")))?;
            if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
                return Err((
                    2,
                    format!(
                        "greeting threshold must be a finite value in 0.0..=1.0, got {value:?}"
                    ),
                ));
            }
            parsed.threshold = Some(threshold);
        } else if argument == "--json" {
            parsed.json = true;
        } else if argument.starts_with('-') {
            return Err((2, format!("unknown option {argument:?}\n{}", usage_line())));
        } else {
            parsed.display_parts.push(argument);
        }
    }
    if parsed.json && parsed.threshold.is_some() {
        return Err((
            2,
            "--json and --threshold are mutually exclusive".to_string(),
        ));
    }
    Ok(Some(parsed))
}

#[cfg(all(not(tarpaulin_include), feature = "standalone"))]
fn load_classifier(_explicit: Option<PathBuf>) -> Result<Classifier, (u8, String)> {
    Classifier::standalone().map_err(|error| (1, standalone_error(&error)))
}

#[cfg(all(not(tarpaulin_include), not(feature = "standalone")))]
fn load_classifier(explicit: Option<PathBuf>) -> Result<Classifier, (u8, String)> {
    if let Some(path) = explicit {
        return Classifier::from_dir(&path).map_err(|error| (1, load_error(&error)));
    }
    if let Some(path) = env::var_os("BONJOUR_DATA_DIR") {
        return Classifier::from_dir(PathBuf::from(path)).map_err(|error| (1, load_error(&error)));
    }

    let paths = automatic_data_paths();
    for path in &paths {
        match Classifier::from_dir(path) {
            Ok(classifier) => return Ok(classifier),
            Err(error) if error.kind() == LoadErrorKind::MissingData => {}
            Err(error) => return Err((1, load_error(&error))),
        }
    }
    let searched = paths
        .iter()
        .map(|path| format!("  {}", path.display()))
        .collect::<Vec<_>>()
        .join("\n");
    Err((
        1,
        format!(
            "missing bonjour-name-data-v1; download it from \
             https://github.com/qrichert/bonjour/releases and use --data-dir or \
             BONJOUR_DATA_DIR\nsearched:\n{searched}"
        ),
    ))
}

#[cfg(all(not(tarpaulin_include), not(feature = "standalone")))]
fn automatic_data_paths() -> Vec<PathBuf> {
    #[cfg(target_os = "macos")]
    {
        env::var_os("HOME")
            .map(|home| PathBuf::from(home).join("Library/Application Support/bonjour/name-v1"))
            .into_iter()
            .collect()
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share")))
            .map(|root| root.join("bonjour/name-v1"))
            .into_iter()
            .collect()
    }
    #[cfg(windows)]
    {
        env::var_os("LOCALAPPDATA")
            .or_else(|| env::var_os("APPDATA"))
            .map(PathBuf::from)
            .map(|root| root.join("bonjour/name-v1"))
            .into_iter()
            .collect()
    }
    #[cfg(not(any(unix, windows)))]
    {
        Vec::new()
    }
}

#[cfg(all(not(tarpaulin_include), not(feature = "standalone")))]
fn load_error(error: &bonjour::LoadError) -> String {
    format!(
        "cannot load bonjour-name-data-v1 ({:?}): {error}; check \
         https://github.com/qrichert/bonjour/releases, --data-dir, or BONJOUR_DATA_DIR",
        error.kind()
    )
}

#[cfg(feature = "standalone")]
fn standalone_error(error: &bonjour::LoadError) -> String {
    format!(
        "standalone data unavailable ({:?}): {error}; build with a valid \
         BONJOUR_DATA_DIR or use a self-contained binary from \
         https://github.com/qrichert/bonjour/releases",
        error.kind()
    )
}

#[cfg(not(tarpaulin_include))]
fn usage_line() -> String {
    "usage: bonjour [--data-dir=PATH] [--country=XX] [--gender=F|M] [--locale=LOCALE] \
     [--threshold=FLOAT | --json] <display name>"
        .to_string()
}

#[cfg(not(tarpaulin_include))]
fn help() {
    println!(
        "{usage}\n\nArguments:\n  <display name>        Display name to inspect.\n\nOptions:\n  --data-dir=<PATH>     Exact bonjour-name-data-v1 directory.\n  --country=<XX>        Two-letter country hint.\n  --gender=<F|M>        Gender hint.\n  --locale=<LOCALE>     Locale used as country fallback.\n  --threshold=<FLOAT>   Greeting threshold (default: {default}).\n  --json                Print the unthresholded inference as JSON.\n  -h, --help            Show this message and exit.\n  -V, --version         Show the version and exit.",
        usage = usage_line(),
        default = bonjour::DEFAULT_GREETING_THRESHOLD,
    );
}

#[cfg(not(tarpaulin_include))]
fn version() {
    println!("{} {}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_contract_covers_emission_abstention_and_absent_gender() {
        let emission = Output {
            input: "Quentin Richert",
            greeting_name: Some("Quentin"),
            confidence: 0.9,
            gender_hint: Some(bonjour::GenderHint::Male),
            gender_confidence: 0.95,
        };
        assert_eq!(
            serde_json::to_string(&emission).unwrap(),
            r#"{"input":"Quentin Richert","greeting_name":"Quentin","confidence":0.9,"gender_hint":"male","gender_confidence":0.95}"#
        );

        let abstention = Output {
            input: "Baris Kebab",
            greeting_name: None,
            confidence: 0.0,
            gender_hint: None,
            gender_confidence: 0.0,
        };
        assert_eq!(
            serde_json::to_string_pretty(&abstention).unwrap(),
            "{\n  \"input\": \"Baris Kebab\",\n  \"greeting_name\": null,\n  \"confidence\": 0.0,\n  \"gender_hint\": null,\n  \"gender_confidence\": 0.0\n}"
        );

        let absent_gender = Output {
            input: "Example Person",
            greeting_name: Some("Example"),
            confidence: 0.8,
            gender_hint: None,
            gender_confidence: 0.6,
        };
        assert_eq!(
            serde_json::to_string(&absent_gender).unwrap(),
            r#"{"input":"Example Person","greeting_name":"Example","confidence":0.8,"gender_hint":null,"gender_confidence":0.6}"#
        );
    }

    #[cfg(feature = "standalone")]
    #[test]
    fn standalone_error_message_is_actionable() {
        let error = Classifier::from_dir("definitely-missing-name-data").unwrap_err();
        let message = standalone_error(&error);
        assert!(message.contains("standalone data unavailable"));
        assert!(message.contains("github.com/qrichert/bonjour/releases"));
    }
}
