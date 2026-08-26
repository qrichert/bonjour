use crate::artifact::{C32Artifact, EmbeddedArtifact};
use crate::{LoadError, LoadErrorKind};

include!(concat!(env!("OUT_DIR"), "/embedded_data.rs"));

pub fn artifact() -> Result<C32Artifact, LoadError> {
    EMBEDDED_ARTIFACT
        .as_ref()
        .map_or_else(unavailable, C32Artifact::from_embedded)
}

fn unavailable() -> Result<C32Artifact, LoadError> {
    Err(LoadError::new(
        LoadErrorKind::StandaloneDataUnavailable,
        None,
        "standalone name data was unavailable when this crate was built",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unavailable_branch_has_a_typed_error() {
        let error = unavailable().unwrap_err();
        assert_eq!(error.kind(), LoadErrorKind::StandaloneDataUnavailable);
        assert_eq!(error.path(), None);
    }
}
