use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use bonjour::Classifier;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const INPUTS: [(&str, Option<&str>, Option<&str>); 8] = [
    ("Quentin Richert", Some("FR"), Some("fr-FR")),
    ("Richert Quentin", Some("FR"), None),
    ("Jean Martin", Some("FR"), None),
    ("Anne Marie Dupont", Some("FR"), None),
    ("Quentin42", None, None),
    ("Quentin Richert GmbH", Some("DE"), None),
    ("Association Jean Moulin", Some("FR"), None),
    ("unknown display", None, Some("en-US")),
];

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let mut arguments = std::env::args_os().skip(1);
    let directory = arguments
        .next()
        .map(PathBuf::from)
        .ok_or("usage: name-runtime ARTIFACT_DIRECTORY [ITERATIONS]")?;
    let iterations = arguments
        .next()
        .map(|value| value.to_string_lossy().parse::<usize>())
        .transpose()?
        .unwrap_or(100_000);
    if arguments.next().is_some() || iterations == 0 {
        return Err("usage: name-runtime ARTIFACT_DIRECTORY [ITERATIONS]".into());
    }

    let runtime_started = Instant::now();
    let runtime = Classifier::from_dir(&directory)?;
    let runtime_load = runtime_started.elapsed();
    let standalone_started = Instant::now();
    let standalone = Classifier::standalone();
    let standalone_load = standalone_started.elapsed();

    println!("runtime_load_seconds={:.6}", runtime_load.as_secs_f64());
    match &standalone {
        Ok(_) => println!(
            "standalone_load_seconds={:.6}",
            standalone_load.as_secs_f64()
        ),
        Err(error) => println!("standalone_unavailable={:?}", error.kind()),
    }
    benchmark_inference("runtime", &runtime, iterations);
    if let Ok(standalone) = &standalone {
        benchmark_inference("standalone", standalone, iterations);
    }
    let binary_bytes = std::env::current_exe()?.metadata()?.len();
    println!("binary_bytes={binary_bytes}");
    Ok(())
}

fn benchmark_inference(label: &str, classifier: &Classifier, iterations: usize) {
    for _ in 0..1_000 {
        for (input, country, locale) in INPUTS {
            std::hint::black_box(classifier.infer(input, country, locale));
        }
    }

    let started = Instant::now();
    let mut checksum = 0xcbf2_9ce4_8422_2325_u64;
    for _ in 0..iterations {
        for (input, country, locale) in INPUTS {
            let inference = classifier.infer(input, country, locale);
            checksum = fold_bytes(checksum, input.as_bytes());
            checksum = fold_bytes(checksum, inference.greeting().unwrap_or("").as_bytes());
            checksum = fold_bytes(checksum, &inference.confidence.to_bits().to_le_bytes());
        }
    }
    let elapsed = started.elapsed();
    let lookups = iterations * INPUTS.len();
    println!("{label}_lookups={lookups}");
    println!(
        "{label}_nanoseconds_per_lookup={:.3}",
        elapsed.as_nanos() as f64 / lookups as f64
    );
    println!(
        "{label}_lookups_per_second={:.3}",
        lookups as f64 / elapsed.as_secs_f64()
    );
    println!("{label}_emission_checksum={checksum:016x}");
}

fn fold_bytes(mut state: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        state ^= u64::from(*byte);
        state = state.wrapping_mul(0x0000_0100_0000_01b3);
    }
    state
}
