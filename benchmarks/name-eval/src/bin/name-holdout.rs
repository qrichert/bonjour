use std::error::Error;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use name_eval::holdout::{
    CaseKind, LabelStatus, freeze, load_or_initialize_draft, render_label_prompt, save_draft,
    span_candidates,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let mut arguments = std::env::args_os().skip(1);
    let Some(command) = arguments.next() else {
        return Err(usage().into());
    };
    match command.to_string_lossy().as_ref() {
        "label" => {
            let source = next_path(&mut arguments, "missing source CSV")?;
            let draft = next_path(&mut arguments, "missing draft CSV")?;
            if arguments.next().is_some() {
                return Err(usage().into());
            }
            if source == draft {
                return Err("source and draft paths must differ".into());
            }
            label(&source, &draft)
        }
        "freeze" => {
            let draft = next_path(&mut arguments, "missing draft CSV")?;
            let sealed = next_path(&mut arguments, "missing sealed CSV")?;
            let manifest = next_path(&mut arguments, "missing manifest CSV")?;
            let provenance = arguments
                .next()
                .and_then(|argument| {
                    argument
                        .to_string_lossy()
                        .strip_prefix("--provenance=")
                        .map(str::to_string)
                })
                .ok_or("freeze requires --provenance=DESCRIPTION")?;
            if arguments.next().is_some() {
                return Err(usage().into());
            }
            if draft == sealed || draft == manifest || sealed == manifest {
                return Err("draft, sealed, and manifest paths must differ".into());
            }
            let frozen = freeze(&draft, &sealed, &manifest, &provenance)?;
            println!("Frozen holdout: {}", sealed.display());
            println!("Manifest: {}", manifest.display());
            println!("SHA-256: {}", frozen.holdout_sha256);
            println!("Cases: {}", frozen.total_cases);
            println!("Evaluable: {}", frozen.evaluable_cases);
            println!("Skipped: {}", frozen.skipped_cases);
            println!("Expected greetings: {}", frozen.expected_greetings);
            println!("Expected abstentions: {}", frozen.expected_abstentions);
            Ok(())
        }
        "help" | "--help" | "-h" => {
            println!("{}", usage());
            Ok(())
        }
        _ => Err(usage().into()),
    }
}

fn label(source: &Path, draft: &Path) -> Result<()> {
    let mut cases = load_or_initialize_draft(source, draft)?;
    let total = cases.len();
    let mut input = String::new();
    while let Some(index) = cases
        .iter()
        .position(|case| case.label_status == LabelStatus::Unlabeled)
    {
        let completed = cases
            .iter()
            .filter(|case| case.label_status != LabelStatus::Unlabeled)
            .count();
        println!("\nCase {} of {}", completed + 1, total);
        print!("{}", render_label_prompt(&cases[index]));
        io::stdout().flush()?;
        input.clear();
        if io::stdin().read_line(&mut input)? == 0 {
            println!("\nInput closed; progress is saved in {}", draft.display());
            return Ok(());
        }
        let selection = input.trim();
        if selection.eq_ignore_ascii_case("q") {
            println!("Progress saved in {}", draft.display());
            return Ok(());
        }
        if selection.eq_ignore_ascii_case("s") {
            cases[index].select_skip();
            save_draft(draft, &cases)?;
            continue;
        }
        if selection.eq_ignore_ascii_case("n") {
            let kind = prompt_abstention_kind(&mut input)?;
            cases[index].select_abstention(kind);
            save_draft(draft, &cases)?;
            continue;
        }
        let Ok(candidate_index) = selection.parse::<usize>() else {
            println!("Invalid selection; enter a candidate number, N, S, or Q.");
            continue;
        };
        let candidates = span_candidates(&cases[index].display_name);
        let Some(candidate) = candidate_index
            .checked_sub(1)
            .and_then(|candidate_index| candidates.get(candidate_index))
        else {
            println!("Candidate number is out of range.");
            continue;
        };
        cases[index].select_greeting(candidate)?;
        save_draft(draft, &cases)?;
    }
    let skipped = cases
        .iter()
        .filter(|case| case.label_status == LabelStatus::Skip)
        .count();
    println!("\nLabeling complete: {total} cases ({skipped} skipped).");
    println!("Draft: {}", draft.display());
    println!("Freeze it before any classifier evaluation.");
    Ok(())
}

fn prompt_abstention_kind(input: &mut String) -> Result<CaseKind> {
    loop {
        print!(
            "Optional case type: [P] person, [O] organization/non-person, [U] unknown (default U)\n> "
        );
        io::stdout().flush()?;
        input.clear();
        if io::stdin().read_line(input)? == 0 {
            return Ok(CaseKind::Unknown);
        }
        match input.trim().to_ascii_lowercase().as_str() {
            "" | "u" => return Ok(CaseKind::Unknown),
            "p" => return Ok(CaseKind::Person),
            "o" => return Ok(CaseKind::NonPerson),
            _ => println!("Invalid case type; enter P, O, or U."),
        }
    }
}

fn next_path(
    arguments: &mut impl Iterator<Item = std::ffi::OsString>,
    message: &str,
) -> Result<PathBuf> {
    arguments
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| message.into())
}

fn usage() -> &'static str {
    "usage:\n  name-holdout label <source.csv> <draft.csv>\n  name-holdout freeze <draft.csv> <sealed.csv> <manifest.csv> --provenance=DESCRIPTION"
}
