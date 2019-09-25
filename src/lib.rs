//! This crate attempts to solve the problem of finding min cost
//! cover of posting lists for a given query.
//! More info to come.

#![warn(
    missing_docs,
    trivial_casts,
    trivial_numeric_casts,
    unused_import_braces,
    unused_qualifications
)]
#![deny(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

use std::iter::Iterator;
use std::ops::Deref;

mod power_set;
pub use power_set::{power_set_iter, PowerSetIter};

mod term_bitset;
pub use term_bitset::{Intersection, ResultClass, TermMask};

mod index;
pub use index::{Index, OptimizeMethod};

mod set_cover;
pub use set_cover::set_cover;

pub mod graph;
pub use graph::Graph;

/// The maximum length of a query that can be used for the algorithm.
pub const MAX_QUERY_LEN: usize = std::mem::size_of::<TermMask>() * 8 - 1;

/// The maximum number of posting lists supported by the algorithm.
/// Depends on the maximum query length.
pub const MAX_LIST_COUNT: usize = 1_usize << MAX_QUERY_LEN;

const QUERY_LEN_EXCEEDED: &str = &"Max query len exceeded";

/// Numerical term representation.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Term(pub u32);

/// Represents a cost of processing a certain intersection of terms.
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Cost(pub f32);

impl std::iter::Sum<Self> for Cost {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        Self(iter.map(|Self(cost)| cost).sum())
    }
}

impl std::ops::Add for Cost {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl std::ops::Sub for Cost {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
}

/// Score representation
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Score(pub f32);

impl std::iter::Sum<Self> for Score {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        Self(iter.map(|Self(cost)| cost).sum())
    }
}

impl std::ops::Add for Score {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl std::ops::Sub for Score {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
}

/// Query representation
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Query {
    /// Term IDs
    pub terms: Vec<Term>,
}

impl Query {
    /// Create a new query.
    pub fn new(terms: Vec<Term>) -> Self {
        assert!(terms.len() <= MAX_QUERY_LEN, QUERY_LEN_EXCEEDED);
        Self { terms }
    }
}
impl Deref for Query {
    type Target = Vec<Term>;

    fn deref(&self) -> &Self::Target {
        &self.terms
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_cost_sum() {
        let costs = vec![Cost(0.1), Cost(1.0), Cost(0.01)];
        assert_eq!(costs.iter().cloned().sum::<Cost>(), Cost(1.11));
    }

    #[test]
    fn test_score_sum() {
        let costs = vec![Score(17.0), Score(1.0), Score(0.01)];
        assert_eq!(costs.iter().cloned().sum::<Score>(), Score(18.01));
    }

    #[test]
    fn test_query_deref() {
        let query = Query::new(vec![Term(0), Term(1), Term(0)]);
        assert_eq!(*query, vec![Term(0), Term(1), Term(0)]);
    }
}
