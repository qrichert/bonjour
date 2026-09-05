use std::env;
use std::error::Error;
use std::io;

use bonjour::Classifier;

fn main() -> Result<(), Box<dyn Error>> {
    let display_name = display_name()?;
    let classifier = Classifier::standalone()?;
    let inference = classifier.infer(&display_name, None, None);

    println!(
        "Bonjour {} !",
        inference.greeting().unwrap_or(display_name.as_str())
    );
    Ok(())
}

fn display_name() -> Result<String, io::Error> {
    let display_name = env::args().skip(1).collect::<Vec<_>>().join(" ");
    if display_name.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "usage: standalone <display name>",
        ));
    }
    Ok(display_name)
}
