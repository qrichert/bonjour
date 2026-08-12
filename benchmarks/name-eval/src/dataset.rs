use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::path::Path;

use serde::Deserialize;
use sha2::{Digest, Sha256};
use unicode_normalization::{UnicodeNormalization, char::is_combining_mark};

use crate::artifact::GenderHint;

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub const GENERATOR_SEED: u64 = 0x6576_616c_2d76_3031;
pub const LARGE_GENERATOR_SEED: u64 = 0x6576_616c_2d76_3032;
pub const C0_TEST_GENERATOR_SEED: u64 = 0x6576_616c_2d76_3033;
pub const FRESH_TEST_GENERATOR_SEED: u64 = 0x6576_616c_2d76_3034;
pub const DEV_TARGET: usize = 60_000;
pub const VALIDATION_TARGET: usize = 60_000;
pub const TEST_TARGET: usize = 120_000;
const LEGACY_TEST_SHA256: &str = "56e047a7232f75f8ef717b2580b6eabc2fea036bb8f3bef3f123466796a91168";
pub const INSPECTED_TEST_SHA256: &str =
    "2233794897ba69c3e9f8ffb9bdecd376545856d9f1bfa508793235cb8e74f962";
pub const C0_TEST_SHA256: &str = "1be896d0febaade25d6c6f8ac8f9b55c382600df1a25f70c135f84fa7425d9ff";
pub const FRESH_TEST_SHA256: &str =
    "403528ab491a2552308729df6b0a984fc864cc99c8438ca23bc1c122d8b772ba";

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Split {
    Regression,
    Dev,
    Validation,
    LegacyTest,
    InspectedTest,
    C0Test,
    Test,
    Sealed,
}

impl Split {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Regression => "REGRESSION",
            Self::Dev => "DEV",
            Self::Validation => "VALIDATION",
            Self::LegacyTest => "LEGACY_TEST",
            Self::InspectedTest => "INSPECTED_TEST",
            Self::C0Test => "C0_TEST",
            Self::Test => "TEST",
            Self::Sealed => "SEALED",
        }
    }

    fn seed_tag(self) -> u64 {
        match self {
            Self::Regression => 0,
            Self::Dev => 1,
            Self::Validation => 2,
            Self::LegacyTest => 3,
            Self::InspectedTest => 4,
            Self::C0Test => 5,
            Self::Test => 6,
            Self::Sealed => 7,
        }
    }
}

impl std::str::FromStr for Split {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim().to_ascii_uppercase().as_str() {
            "DEV" => Ok(Self::Dev),
            "VALIDATION" => Ok(Self::Validation),
            "LEGACY_TEST" => Ok(Self::LegacyTest),
            "INSPECTED_TEST" => Ok(Self::InspectedTest),
            "C0_TEST" => Ok(Self::C0Test),
            "TEST" => Ok(Self::Test),
            other => Err(format!("unknown split {other:?}")),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Case {
    pub id: String,
    pub split: Split,
    pub category: String,
    pub input: String,
    pub country_hint: Option<String>,
    pub locale_hint: Option<String>,
    pub expected_greeting: Option<String>,
    pub expected_gender: Option<GenderHint>,
    pub notes: String,
}

#[derive(Clone, Copy, Debug)]
pub struct SeedStats {
    pub split: Split,
    pub given_names: usize,
    pub surnames: usize,
    pub generic_organization_words: usize,
    pub legal_markers: usize,
}

impl Case {
    pub fn is_person(&self) -> bool {
        self.expected_greeting.is_some()
    }
}

#[derive(Deserialize)]
struct LabeledRow {
    id: String,
    input: String,
    country_hint: String,
    locale_hint: String,
    expected_greeting: String,
    expected_gender: String,
    category: String,
    notes: String,
}

#[derive(Clone, Deserialize)]
struct GivenRow {
    split: String,
    name: String,
    gender: String,
    country: String,
    compound: bool,
}

#[derive(Clone, Deserialize)]
struct SurnameRow {
    split: String,
    name: String,
}

#[derive(Clone, Deserialize)]
struct OrganizationRow {
    split: String,
    word: String,
    kind: String,
}

pub fn load_regression(path: &Path) -> Result<Vec<Case>> {
    load_labeled(path, Split::Regression)
}

pub fn load_sealed(path: &Path) -> Result<Vec<Case>> {
    load_labeled(path, Split::Sealed)
}

fn load_labeled(path: &Path, split: Split) -> Result<Vec<Case>> {
    let mut reader = csv::Reader::from_path(path)?;
    let mut cases = Vec::new();
    let mut ids = HashSet::new();
    for row in reader.deserialize::<LabeledRow>() {
        let row = row?;
        if !ids.insert(row.id.clone()) {
            return Err(format!("duplicate {} ID {}", split.as_str(), row.id).into());
        }
        let expected_greeting = nonempty(row.expected_greeting);
        let expected_gender = parse_gender(&row.expected_gender)?;
        if expected_greeting.is_none() && expected_gender.is_some() {
            return Err(format!("{} supplies gender without a greeting", row.id).into());
        }
        cases.push(Case {
            id: row.id,
            split,
            category: row.category,
            input: row.input,
            country_hint: nonempty(row.country_hint),
            locale_hint: nonempty(row.locale_hint),
            expected_greeting,
            expected_gender,
            notes: row.notes,
        });
    }
    Ok(cases)
}

pub fn generate_cases(fixtures: &Path, include_fresh_test: bool) -> Result<Vec<Case>> {
    let given = read_rows::<GivenRow>(&fixtures.join("given_names.csv"))?;
    let surnames = read_rows::<SurnameRow>(&fixtures.join("surnames.csv"))?;
    let organizations = read_rows::<OrganizationRow>(&fixtures.join("organization_words.csv"))?;
    validate_seed_splits(&given, &surnames)?;

    let mut cases = generate_legacy_test(&given, &surnames, &organizations)?;
    validate_legacy_test(&cases)?;
    for (split, target) in [
        (Split::Dev, DEV_TARGET),
        (Split::Validation, VALIDATION_TARGET),
        (Split::InspectedTest, TEST_TARGET),
    ] {
        generate_large_split(
            split,
            target,
            LARGE_GENERATOR_SEED,
            &given,
            &surnames,
            &organizations,
            &mut cases,
        )?;
    }
    validate_inspected_test(&cases)?;
    generate_large_split(
        Split::C0Test,
        TEST_TARGET,
        C0_TEST_GENERATOR_SEED,
        &given,
        &surnames,
        &organizations,
        &mut cases,
    )?;
    validate_c0_test(&cases)?;
    if include_fresh_test {
        generate_large_split(
            Split::Test,
            TEST_TARGET,
            FRESH_TEST_GENERATOR_SEED,
            &given,
            &surnames,
            &organizations,
            &mut cases,
        )?;
        validate_fresh_test(&cases)?;
    }

    let mut ids = HashSet::new();
    for case in &cases {
        if !ids.insert(case.id.clone()) {
            return Err(format!("duplicate generated case ID {}", case.id).into());
        }
    }
    Ok(cases)
}

pub fn seed_stats(fixtures: &Path) -> Result<Vec<SeedStats>> {
    let given = read_rows::<GivenRow>(&fixtures.join("given_names.csv"))?;
    let surnames = read_rows::<SurnameRow>(&fixtures.join("surnames.csv"))?;
    let organizations = read_rows::<OrganizationRow>(&fixtures.join("organization_words.csv"))?;
    validate_seed_splits(&given, &surnames)?;
    [
        Split::Dev,
        Split::Validation,
        Split::LegacyTest,
        Split::InspectedTest,
        Split::C0Test,
        Split::Test,
    ]
    .into_iter()
    .map(|split| {
        Ok(SeedStats {
            split,
            given_names: rows_for_split(&given, split, |row| &row.split).len(),
            surnames: rows_for_split(&surnames, split, |row| &row.split).len(),
            generic_organization_words: organizations
                .iter()
                .filter(|row| {
                    row.split.parse::<Split>().ok() == Some(split) && row.kind == "generic"
                })
                .count(),
            legal_markers: organizations
                .iter()
                .filter(|row| row.split.parse::<Split>().ok() == Some(split) && row.kind == "legal")
                .count(),
        })
    })
    .collect()
}

fn generate_legacy_test(
    given: &[GivenRow],
    surnames: &[SurnameRow],
    organizations: &[OrganizationRow],
) -> Result<Vec<Case>> {
    let split = Split::LegacyTest;
    let split_given = rows_for_split(given, split, |row| &row.split);
    let split_surnames = rows_for_split(surnames, split, |row| &row.split);
    let split_organizations = rows_for_split(organizations, split, |row| &row.split);
    let generic = split_organizations
        .iter()
        .filter(|row| row.kind == "generic")
        .copied()
        .collect::<Vec<_>>();
    let legal = split_organizations
        .iter()
        .filter(|row| row.kind == "legal")
        .copied()
        .collect::<Vec<_>>();
    if split_given.is_empty() || split_surnames.is_empty() || generic.is_empty() || legal.is_empty()
    {
        return Err("LEGACY_TEST fixture partition is incomplete".into());
    }

    let mut cases = Vec::new();
    let mut rng = SplitMix64::new(GENERATOR_SEED ^ split.seed_tag());
    for (index, given) in split_given.iter().enumerate() {
        let surname = split_surnames[rng.index(split_surnames.len())];
        let gender = parse_gender(&given.gender)?;
        let (country_hint, locale_hint) = hints_for(index, &given.country);
        let base_category = if given.compound {
            "person_compound_given"
        } else {
            "person_given_surname"
        };
        push_case(
            &mut cases,
            split,
            format!("{index:03}-given-surname"),
            base_category,
            format!("{} {}", given.name, surname.name),
            Some(given.name.clone()),
            gender,
            country_hint.clone(),
            locale_hint.clone(),
        );
        push_case(
            &mut cases,
            split,
            format!("{index:03}-surname-given"),
            "person_surname_given",
            format!("{} {}", surname.name, given.name),
            Some(given.name.clone()),
            gender,
            country_hint.clone(),
            locale_hint.clone(),
        );

        let cased_given = if index % 2 == 0 {
            given.name.to_uppercase()
        } else {
            given.name.to_lowercase()
        };
        let cased_surname = if index % 2 == 0 {
            surname.name.to_uppercase()
        } else {
            surname.name.to_lowercase()
        };
        push_case(
            &mut cases,
            split,
            format!("{index:03}-casing"),
            "person_casing",
            format!("{cased_given} {cased_surname}"),
            Some(cased_given),
            gender,
            country_hint.clone(),
            locale_hint.clone(),
        );
        push_case(
            &mut cases,
            split,
            format!("{index:03}-whitespace"),
            "person_whitespace",
            format!("  {}   {}  ", given.name, surname.name),
            Some(given.name.clone()),
            gender,
            country_hint.clone(),
            locale_hint.clone(),
        );
        push_case(
            &mut cases,
            split,
            format!("{index:03}-punctuation"),
            "person_punctuation",
            format!("{}, {}", surname.name, given.name),
            Some(given.name.clone()),
            gender,
            country_hint.clone(),
            locale_hint.clone(),
        );

        let stripped = strip_accents(&given.name);
        if stripped != given.name {
            push_case(
                &mut cases,
                split,
                format!("{index:03}-accent-stripped"),
                "person_accent_stripped",
                format!("{stripped} {}", surname.name),
                Some(stripped),
                gender,
                country_hint.clone(),
                locale_hint.clone(),
            );
        }
        let generic_word = generic[rng.index(generic.len())];
        push_case(
            &mut cases,
            split,
            format!("{index:03}-organization-generic"),
            "organization_name_collision",
            format!("{} {} {}", generic_word.word, given.name, surname.name),
            None,
            None,
            country_hint.clone(),
            locale_hint.clone(),
        );
        let legal_word = legal[rng.index(legal.len())];
        push_case(
            &mut cases,
            split,
            format!("{index:03}-organization-legal"),
            "organization_legal_suffix",
            format!("{} {}", given.name, legal_word.word),
            None,
            None,
            country_hint,
            locale_hint,
        );
    }

    let ambiguous = split_given
        .iter()
        .filter(|given| {
            split_surnames
                .iter()
                .any(|surname| canonical_label(&surname.name) == canonical_label(&given.name))
        })
        .collect::<Vec<_>>();
    for (index, surname_like_given) in ambiguous.iter().enumerate() {
        let given = split_given
            .iter()
            .find(|given| canonical_label(&given.name) != canonical_label(&surname_like_given.name))
            .ok_or("ambiguous generation needs another given name")?;
        push_case(
            &mut cases,
            split,
            format!("ambiguous-{index:03}"),
            "person_ambiguous_roles",
            format!("{} {}", given.name, surname_like_given.name),
            Some(given.name.clone()),
            parse_gender(&given.gender)?,
            Some(given.country.clone()),
            None,
        );
    }
    Ok(cases)
}

fn validate_legacy_test(cases: &[Case]) -> Result<()> {
    validate_snapshot(cases, Split::LegacyTest, 116, LEGACY_TEST_SHA256)
}

fn validate_inspected_test(cases: &[Case]) -> Result<()> {
    validate_snapshot(
        cases,
        Split::InspectedTest,
        TEST_TARGET,
        INSPECTED_TEST_SHA256,
    )
}

fn validate_c0_test(cases: &[Case]) -> Result<()> {
    validate_snapshot(cases, Split::C0Test, TEST_TARGET, C0_TEST_SHA256)
}

fn validate_fresh_test(cases: &[Case]) -> Result<()> {
    validate_snapshot(cases, Split::Test, TEST_TARGET, FRESH_TEST_SHA256)
}

fn validate_snapshot(
    cases: &[Case],
    split: Split,
    expected_len: usize,
    expected_sha256: &str,
) -> Result<()> {
    let selected = cases
        .iter()
        .filter(|case| case.split == split)
        .collect::<Vec<_>>();
    if selected.len() != expected_len {
        return Err(format!(
            "{} has {} cases, expected {expected_len}",
            split.as_str(),
            selected.len()
        )
        .into());
    }
    let mut hasher = Sha256::new();
    for case in selected {
        let fields = [
            case.category.as_str(),
            case.input.as_str(),
            case.country_hint.as_deref().unwrap_or(""),
            case.locale_hint.as_deref().unwrap_or(""),
            case.expected_greeting.as_deref().unwrap_or(""),
            case.expected_gender.map_or("", |gender| gender.as_str()),
        ];
        for (index, field) in fields.iter().enumerate() {
            if index != 0 {
                hasher.update([0x1f]);
            }
            hasher.update(field.as_bytes());
        }
        hasher.update(b"\n");
    }
    let actual = format!("{:x}", hasher.finalize());
    if actual != expected_sha256 {
        return Err(format!("{} snapshot changed: {actual}", split.as_str()).into());
    }
    Ok(())
}

fn generate_large_split(
    split: Split,
    target: usize,
    generator_seed: u64,
    given: &[GivenRow],
    surnames: &[SurnameRow],
    organizations: &[OrganizationRow],
    output: &mut Vec<Case>,
) -> Result<()> {
    let split_given = rows_for_split(given, split, |row| &row.split);
    let split_surnames = rows_for_split(surnames, split, |row| &row.split);
    let split_organizations = rows_for_split(organizations, split, |row| &row.split);
    let generic = split_organizations
        .iter()
        .filter(|row| row.kind == "generic")
        .copied()
        .collect::<Vec<_>>();
    let legal = split_organizations
        .iter()
        .filter(|row| row.kind == "legal")
        .copied()
        .collect::<Vec<_>>();
    if split_given.len() < 20 || split_surnames.len() < 20 || generic.len() < 7 || legal.len() < 3 {
        return Err(format!(
            "{} needs at least 20 given names, 20 surnames, 7 generic organization words, and 3 legal markers",
            split.as_str()
        )
        .into());
    }
    let compound = split_given
        .iter()
        .copied()
        .filter(|row| row.compound)
        .collect::<Vec<_>>();
    let accented = split_given
        .iter()
        .copied()
        .filter(|row| strip_accents(&row.name) != row.name)
        .collect::<Vec<_>>();
    let hyphen_given = split_given
        .iter()
        .copied()
        .filter(|row| row.name.contains('-'))
        .collect::<Vec<_>>();
    let hyphen_surnames = split_surnames
        .iter()
        .copied()
        .filter(|row| row.name.contains('-'))
        .collect::<Vec<_>>();
    let apostrophe_given = split_given
        .iter()
        .copied()
        .filter(|row| row.name.contains('\''))
        .collect::<Vec<_>>();
    let apostrophe_surnames = split_surnames
        .iter()
        .copied()
        .filter(|row| row.name.contains('\''))
        .collect::<Vec<_>>();
    let ambiguous_surnames = split_surnames
        .iter()
        .copied()
        .filter(|surname| {
            split_given
                .iter()
                .any(|given| canonical_label(&given.name) == canonical_label(&surname.name))
        })
        .collect::<Vec<_>>();
    if compound.is_empty()
        || accented.is_empty()
        || (hyphen_given.is_empty() && hyphen_surnames.is_empty())
        || (apostrophe_given.is_empty() && apostrophe_surnames.is_empty())
        || ambiguous_surnames.is_empty()
    {
        return Err(format!("{} lacks a required structured-name seed", split.as_str()).into());
    }

    let mut rng = SplitMix64::new(generator_seed ^ split.seed_tag());
    let mut unique = HashSet::<String>::with_capacity(target * 2);
    let mut generated = 0_usize;
    let mut attempts = 0_usize;
    while generated < target {
        attempts += 1;
        if attempts > target * 100 {
            return Err(format!("{} exhausted unique generated cases", split.as_str()).into());
        }
        let is_person = rng.index(20) < 13;
        let case = if is_person {
            generate_person_case(
                split,
                generated,
                &split_given,
                &split_surnames,
                &compound,
                &accented,
                &hyphen_given,
                &hyphen_surnames,
                &apostrophe_given,
                &apostrophe_surnames,
                &ambiguous_surnames,
                &mut rng,
            )?
        } else {
            generate_organization_case(
                split,
                generated,
                &split_given,
                &split_surnames,
                &generic,
                &legal,
                &mut rng,
            )
        };
        let unique_key = format!(
            "{}\0{}\0{}\0{}",
            case.input,
            case.country_hint.as_deref().unwrap_or(""),
            case.locale_hint.as_deref().unwrap_or(""),
            case.category
        );
        if unique.insert(unique_key) {
            output.push(case);
            generated += 1;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn generate_person_case(
    split: Split,
    index: usize,
    given_rows: &[&GivenRow],
    surnames: &[&SurnameRow],
    compound: &[&GivenRow],
    accented: &[&GivenRow],
    hyphen_given: &[&GivenRow],
    hyphen_surnames: &[&SurnameRow],
    apostrophe_given: &[&GivenRow],
    apostrophe_surnames: &[&SurnameRow],
    ambiguous_surnames: &[&SurnameRow],
    rng: &mut SplitMix64,
) -> Result<Case> {
    let variant = rng.index(14);
    let mut given = given_rows[rng.index(given_rows.len())];
    let mut surname = surnames[rng.index(surnames.len())];
    match variant {
        2 => given = compound[rng.index(compound.len())],
        4 => {
            if !hyphen_given.is_empty() && (hyphen_surnames.is_empty() || rng.index(2) == 0) {
                given = hyphen_given[rng.index(hyphen_given.len())];
            } else {
                surname = hyphen_surnames[rng.index(hyphen_surnames.len())];
            }
        }
        5 => {
            if !apostrophe_given.is_empty() && (apostrophe_surnames.is_empty() || rng.index(2) == 0)
            {
                given = apostrophe_given[rng.index(apostrophe_given.len())];
            } else {
                surname = apostrophe_surnames[rng.index(apostrophe_surnames.len())];
            }
        }
        6 | 7 => given = accented[rng.index(accented.len())],
        13 => {
            surname = ambiguous_surnames[rng.index(ambiguous_surnames.len())];
            if canonical_label(&given.name) == canonical_label(&surname.name) {
                given = given_rows
                    .iter()
                    .copied()
                    .find(|candidate| {
                        canonical_label(&candidate.name) != canonical_label(&surname.name)
                    })
                    .ok_or("ambiguous case needs a different given name")?;
            }
        }
        _ => {}
    }

    let style = if matches!(variant, 8) {
        1
    } else if matches!(variant, 9) {
        2
    } else {
        rng.index(3)
    };
    let base_given = if variant == 7 {
        strip_accents(&given.name)
    } else {
        given.name.clone()
    };
    let mut rendered_given = style_name(&base_given, style);
    let mut rendered_surname = style_name(&surname.name, style);
    if variant == 4 {
        let separator = ["-", "‐", "‑", "‒", "–"][rng.index(5)];
        rendered_given = rendered_given.replace('-', separator);
        rendered_surname = rendered_surname.replace('-', separator);
    } else if variant == 5 {
        let separator = ["'", "’", "ʼ", "ʻ"][rng.index(4)];
        rendered_given = rendered_given.replace('\'', separator);
        rendered_surname = rendered_surname.replace('\'', separator);
    }
    let expected = rendered_given
        .replace(['‐', '‑', '‒', '–'], "-")
        .replace(['’', 'ʼ', 'ʻ'], "'");
    let (category, input) = match variant {
        0 => (
            "person_given_surname",
            format!("{rendered_given} {rendered_surname}"),
        ),
        1 => (
            "person_surname_given",
            format!("{rendered_surname} {rendered_given}"),
        ),
        2 => (
            "person_compound_given",
            format!("{rendered_given} {rendered_surname}"),
        ),
        3 => (
            "person_repeated_whitespace",
            format!("{rendered_given} {rendered_surname}"),
        ),
        4 => (
            "person_hyphenated",
            format!("{rendered_given} {rendered_surname}"),
        ),
        5 => (
            "person_apostrophe",
            format!("{rendered_given} {rendered_surname}"),
        ),
        6 => (
            "person_accent_preserved",
            format!("{rendered_given} {rendered_surname}"),
        ),
        7 => (
            "person_accent_stripped",
            format!("{rendered_given} {rendered_surname}"),
        ),
        8 | 9 => (
            "person_casing",
            format!("{rendered_given} {rendered_surname}"),
        ),
        10 => (
            "person_surname_comma_given",
            format!("{rendered_surname}, {rendered_given}"),
        ),
        11 => (
            "person_family_name_first",
            format!("{rendered_surname} {rendered_given}"),
        ),
        12 => (
            "person_cultural_order",
            format!("{rendered_given} {rendered_surname}"),
        ),
        _ => (
            "person_ambiguous_roles",
            format!("{rendered_given} {rendered_surname}"),
        ),
    };
    let input = vary_whitespace(&input, rng);
    let (country_hint, locale_hint) = random_hints(rng, &given.country);
    Ok(Case {
        id: format!("{}-large-{index:06}", split.as_str().to_ascii_lowercase()),
        split,
        category: category.to_string(),
        input,
        country_hint,
        locale_hint,
        expected_greeting: Some(expected),
        expected_gender: parse_gender(&given.gender)?,
        notes: "large deterministic generation from independently curated person structure"
            .to_string(),
    })
}

fn generate_organization_case(
    split: Split,
    index: usize,
    given: &[&GivenRow],
    surnames: &[&SurnameRow],
    generic: &[&OrganizationRow],
    legal: &[&OrganizationRow],
    rng: &mut SplitMix64,
) -> Case {
    let given = given[rng.index(given.len())];
    let surname = surnames[rng.index(surnames.len())];
    let style = rng.index(3);
    let given_name = style_name(&given.name, style);
    let surname_name = style_name(&surname.name, style);
    let consulting = preferred_word(generic, "Consulting", rng);
    let fils = preferred_word(generic, "Fils", rng);
    let club = preferred_word(generic, "Club", rng);
    let association = preferred_word(generic, "Association", rng);
    let kebab = preferred_word(generic, "Kebab", rng);
    let group = preferred_word(generic, "Group", rng);
    let legal = &legal[rng.index(legal.len())].word;
    let (category, input) = match rng.index(7) {
        0 => (
            "organization_given_consulting",
            format!("{given_name} {consulting}"),
        ),
        1 => (
            "organization_given_ampersand",
            format!("{given_name} & {fils}"),
        ),
        2 => ("organization_club_given", format!("{club} {given_name}")),
        3 => (
            "organization_association_name",
            format!("{association} {given_name} {surname_name}"),
        ),
        4 => (
            "organization_legal_suffix",
            format!("{given_name} {surname_name} {legal}"),
        ),
        5 => ("organization_given_food", format!("{given_name} {kebab}")),
        _ => (
            "organization_name_group",
            format!("{given_name} {surname_name} {group}"),
        ),
    };
    let input = vary_whitespace(&input, rng);
    let (country_hint, locale_hint) = random_hints(rng, &given.country);
    Case {
        id: format!("{}-large-{index:06}", split.as_str().to_ascii_lowercase()),
        split,
        category: category.to_string(),
        input,
        country_hint,
        locale_hint,
        expected_greeting: None,
        expected_gender: None,
        notes: "difficult organization negative containing independently labeled person atoms"
            .to_string(),
    }
}

fn rows_for_split<T>(rows: &[T], split: Split, split_text: impl Fn(&T) -> &String) -> Vec<&T> {
    rows.iter()
        .filter(|row| split_text(row).parse::<Split>().ok() == Some(split))
        .collect()
}

fn preferred_word<'a>(
    words: &'a [&OrganizationRow],
    preferred: &str,
    rng: &mut SplitMix64,
) -> &'a str {
    words
        .iter()
        .find(|row| row.word.eq_ignore_ascii_case(preferred))
        .map_or_else(
            || words[rng.index(words.len())].word.as_str(),
            |row| row.word.as_str(),
        )
}

fn style_name(value: &str, style: usize) -> String {
    match style {
        1 => value.to_uppercase(),
        2 => value.to_lowercase(),
        _ => value.to_string(),
    }
}

fn vary_whitespace(value: &str, rng: &mut SplitMix64) -> String {
    let mut output = " ".repeat(rng.index(3));
    let mut words = value.split(' ').peekable();
    while let Some(word) = words.next() {
        output.push_str(word);
        if words.peek().is_some() {
            output.push_str(&" ".repeat(rng.index(5) + 1));
        }
    }
    output.push_str(&" ".repeat(rng.index(3)));
    output
}

fn random_hints(rng: &mut SplitMix64, country: &str) -> (Option<String>, Option<String>) {
    match rng.index(3) {
        0 => (Some(country.to_string()), None),
        1 => (None, Some(format!("und-{country}"))),
        _ => (None, None),
    }
}

fn validate_seed_splits(given: &[GivenRow], surnames: &[SurnameRow]) -> Result<()> {
    let mut atom_splits = BTreeMap::<String, Split>::new();
    for (split_text, name, source) in given
        .iter()
        .map(|row| (&row.split, &row.name, "given name"))
        .chain(
            surnames
                .iter()
                .map(|row| (&row.split, &row.name, "surname")),
        )
    {
        let split = split_text.parse::<Split>()?;
        for atom in label_atoms(name) {
            if let Some(previous) = atom_splits.insert(atom.clone(), split)
                && previous != split
            {
                return Err(format!(
                    "seed atom {atom:?} leaks from {} into {} ({source})",
                    previous.as_str(),
                    split.as_str()
                )
                .into());
            }
        }
    }
    Ok(())
}

fn label_atoms(name: &str) -> Vec<String> {
    let canonical = canonical_label(name);
    let mut atoms = canonical
        .split(|character: char| !character.is_alphanumeric())
        .filter(|atom| !atom.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    atoms.push(canonical);
    atoms.sort();
    atoms.dedup();
    atoms
}

fn canonical_label(name: &str) -> String {
    strip_accents(name).to_lowercase()
}

fn hints_for(index: usize, country: &str) -> (Option<String>, Option<String>) {
    match index % 3 {
        0 => (Some(country.to_string()), None),
        1 => (None, Some(format!("und-{country}"))),
        _ => (None, None),
    }
}

#[allow(clippy::too_many_arguments)]
fn push_case(
    cases: &mut Vec<Case>,
    split: Split,
    suffix: String,
    category: &str,
    input: String,
    expected_greeting: Option<String>,
    expected_gender: Option<GenderHint>,
    country_hint: Option<String>,
    locale_hint: Option<String>,
) {
    cases.push(Case {
        id: format!("{}-{suffix}", split.as_str().to_ascii_lowercase()),
        split,
        category: category.to_string(),
        input,
        country_hint,
        locale_hint,
        expected_greeting,
        expected_gender,
        notes: "deterministically generated from independently curated labels".to_string(),
    });
}

fn parse_gender(value: &str) -> Result<Option<GenderHint>> {
    match value.trim().to_ascii_uppercase().as_str() {
        "" => Ok(None),
        "F" => Ok(Some(GenderHint::Female)),
        "M" => Ok(Some(GenderHint::Male)),
        other => Err(format!("invalid gender label {other:?}").into()),
    }
}

fn nonempty(value: String) -> Option<String> {
    (!value.trim().is_empty()).then_some(value)
}

fn strip_accents(value: &str) -> String {
    value
        .nfd()
        .filter(|character| !is_combining_mark(*character))
        .nfc()
        .collect()
}

fn read_rows<T>(path: &Path) -> Result<Vec<T>>
where
    T: for<'de> Deserialize<'de>,
{
    csv::Reader::from_path(path)?
        .deserialize()
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Into::into)
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn index(&mut self, length: usize) -> usize {
        self.next() as usize % length
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_is_fixed() {
        assert_eq!(GENERATOR_SEED, 0x6576_616c_2d76_3031);
        assert_eq!(LARGE_GENERATOR_SEED, 0x6576_616c_2d76_3032);
        assert_eq!(C0_TEST_GENERATOR_SEED, 0x6576_616c_2d76_3033);
        assert_eq!(FRESH_TEST_GENERATOR_SEED, 0x6576_616c_2d76_3034);
    }

    #[test]
    fn canonical_atoms_remove_accents() {
        assert_eq!(label_atoms("María José"), ["jose", "maria", "maria jose"]);
    }

    #[test]
    fn checked_in_generation_is_valid_and_deterministic() {
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let first = generate_cases(&fixtures, true).unwrap();
        let second = generate_cases(&fixtures, true).unwrap();
        assert_eq!(
            first.iter().filter(|case| case.split == Split::Dev).count(),
            DEV_TARGET
        );
        assert_eq!(
            first
                .iter()
                .filter(|case| case.split == Split::Validation)
                .count(),
            VALIDATION_TARGET
        );
        assert_eq!(
            first
                .iter()
                .filter(|case| case.split == Split::InspectedTest)
                .count(),
            TEST_TARGET
        );
        assert_eq!(
            first
                .iter()
                .filter(|case| case.split == Split::C0Test)
                .count(),
            TEST_TARGET
        );
        assert_eq!(
            first
                .iter()
                .filter(|case| case.split == Split::Test)
                .count(),
            TEST_TARGET
        );
        assert_eq!(
            first
                .iter()
                .filter(|case| case.split == Split::LegacyTest)
                .count(),
            116
        );
        let summarize = |cases: &[Case]| {
            cases
                .iter()
                .map(|case| {
                    (
                        case.id.clone(),
                        case.input.clone(),
                        case.expected_greeting.clone(),
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(summarize(&first), summarize(&second));
    }
}
