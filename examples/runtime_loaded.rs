use std::env;
use std::error::Error;
use std::io;

use bonjour::Classifier;

fn main() -> Result<(), Box<dyn Error>> {
    let (data_directory, display_name) = arguments()?;
    let classifier = Classifier::from_dir(data_directory)?;
    let inference = classifier.infer(&display_name, None, None);

    println!(
        "Bonjour {} !",
        inference.greeting().unwrap_or(display_name.as_str())
    );
    Ok(())
}

fn arguments() -> Result<(String, String), io::Error> {
    let mut arguments = env::args().skip(1);
    let data_directory = arguments
        .next()
        .ok_or_else(|| invalid_input("usage: runtime_loaded <data directory> <display name>"))?;
    let display_name = arguments.collect::<Vec<_>>().join(" ");
    if display_name.is_empty() {
        return Err(invalid_input(
            "usage: runtime_loaded <data directory> <display name>",
        ));
    }
    Ok((data_directory, display_name))
}

fn invalid_input(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}
