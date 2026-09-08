use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

pub(crate) type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub(crate) fn publish_directory(
    output: &Path,
    write: impl FnOnce(&Path) -> Result<()>,
) -> Result<()> {
    if output.exists() {
        return Err(format!("refusing to overwrite: {}", output.display()).into());
    }
    let parent = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let temporary = temporary_path(output)?;
    if temporary.exists() {
        return Err(format!("refusing to overwrite: {}", temporary.display()).into());
    }
    fs::create_dir(&temporary)?;

    match write(&temporary) {
        Ok(()) => {
            fs::rename(&temporary, output)?;
            Ok(())
        }
        Err(error) => {
            if let Err(cleanup) = fs::remove_dir_all(&temporary) {
                eprintln!(
                    "warning: failed to clean temporary output {}: {cleanup}",
                    temporary.display()
                );
            }
            Err(error)
        }
    }
}

pub(crate) fn file_sha256(path: &Path) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}

fn temporary_path(output: &Path) -> Result<PathBuf> {
    let name = output
        .file_name()
        .ok_or_else(|| format!("output has no final component: {}", output.display()))?;
    Ok(output.with_file_name(format!(
        ".{}.tmp-{}",
        name.to_string_lossy(),
        std::process::id()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_to_overwrite_published_output() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("output");
        fs::create_dir(&output).unwrap();
        let error = publish_directory(&output, |_| Ok(())).unwrap_err();
        assert!(error.to_string().contains("refusing to overwrite"));
    }

    #[test]
    fn publishes_completed_directory() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("output");
        publish_directory(&output, |temporary| {
            fs::write(temporary.join("done"), b"yes")?;
            Ok(())
        })
        .unwrap();
        assert_eq!(fs::read(output.join("done")).unwrap(), b"yes");
    }
}
