use std::error::Error;
use std::path::Path;

use crate::lexical::candidate_is_eligible;

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, Debug, Default)]
pub struct LexicalAudit {
    pub total_keys: u64,
    pub total_observations: u128,
    pub ineligible_keys: u64,
    pub ineligible_observations: u128,
}

pub fn audit_clean_v1(path: &Path) -> Result<LexicalAudit> {
    let mut reader = csv::Reader::from_path(path)?;
    let mut audit = LexicalAudit::default();
    let mut current_name = None::<String>;
    let mut current_observations = 0_u128;

    for (index, result) in reader.records().enumerate() {
        let record = result?;
        if record.len() != 4 {
            return Err(format!("clean-v1 line {} has {} fields", index + 2, record.len()).into());
        }
        let name = record.get(0).ok_or("clean-v1 row has no name")?;
        let count = record
            .get(3)
            .ok_or("clean-v1 row has no count")?
            .parse::<u64>()?;
        audit.total_observations += u128::from(count);

        match current_name.as_deref() {
            Some(current) if current == name => {
                current_observations += u128::from(count);
            }
            Some(current) => {
                if current > name {
                    return Err("clean-v1 is not sorted by exact name".into());
                }
                observe_key(&mut audit, current, current_observations);
                current_name = Some(name.to_string());
                current_observations = u128::from(count);
            }
            None => {
                current_name = Some(name.to_string());
                current_observations = u128::from(count);
            }
        }
    }
    if let Some(name) = current_name {
        observe_key(&mut audit, &name, current_observations);
    }
    Ok(audit)
}

fn observe_key(audit: &mut LexicalAudit, name: &str, observations: u128) {
    audit.total_keys += 1;
    if !candidate_is_eligible(name) {
        audit.ineligible_keys += 1;
        audit.ineligible_observations += observations;
    }
}
