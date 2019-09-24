use crate::{Query, Term};
use failure::ResultExt;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;

/// The underlying type of the mask used by the term bag.
///
/// In this representation, n-th bit represents the n-th term, and
/// is equal to 1 if it's present, and 0 if it's missing.
pub type TermMask = u8;

/// A bitset representation of a subset of terms in a query.
///
/// For example, given a three-term query, `TermBitset(0b001)` represents a single posting list
/// of the first term, while `TermBitset(0b101)` represents the intersection of the first and
/// the last term.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TermBitset(pub TermMask);

impl Into<usize> for TermBitset {
    fn into(self) -> usize {
        self.0 as usize
    }
}

impl TermBitset {
    /// Constructs a term bag from a query and a subset of query terms.
    pub fn from(query: &Query, term_subset: &[Term]) -> Self {
        let mut mask: TermMask = 0;
        let mut mapping = HashMap::<Term, usize>::new();
        for (idx, &term) in query.terms.iter().enumerate() {
            mapping.insert(term, idx);
        }
        for term in term_subset.iter() {
            let idx = mapping.get(term).expect("Term not in query.");
            let one: TermMask = 1;
            mask |= one << idx;
        }
        Self(mask)
    }

    /// Generates all possible result classes for this term bag.
    ///
    /// Essentially, it generates a vector of values from 0 to `2^len`.
    pub fn result_classes(len: u8) -> Vec<Self> {
        let two: TermMask = 2;
        (0..two.pow(u32::from(len))).map(Self).collect()
    }

    /// Checks if this term mag covers a given result class.
    #[inline]
    pub fn covers(self, class: Self) -> bool {
        self.0 & class.0 == self.0
    }

    /// Marks classes that this term bag covers as covered.
    #[inline]
    pub fn cover(self, classes: &[Self], covered: &mut [u8]) {
        for (covered, &class) in covered.iter_mut().zip(classes) {
            *covered |= self.covers(class) as u8;
        }
    }

    /// Number of terms in the set.
    #[inline]
    pub fn degree(self) -> u32 {
        self.0.count_ones()
    }
}

impl FromStr for TermBitset {
    type Err = failure::Error;

    /// Parses a text binary representation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use intersect::TermBitset;
    /// # fn main() -> Result<(), failure::Error> {
    /// let tb: TermBitset = "01010".parse()?;
    /// assert_eq!(tb, TermBitset(0b1010));
    /// assert!("0x21".parse::<TermBitset>().is_err());
    /// # Ok(())
    /// # }
    /// ```
    fn from_str(index: &str) -> Result<Self, Self::Err> {
        Ok(Self(u8::from_str_radix(index, 2).with_context(|_| {
            format!("Invalid bitset string: {}", index)
        })?))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_from_str() {
        let tb: TermBitset = "01010".parse().unwrap();
        assert_eq!(tb, TermBitset(0b1010));
        assert!("0x21".parse::<TermBitset>().is_err());
    }

    #[test]
    #[should_panic]
    fn test_new_term_vec_wrong_id() {
        let query = Query::new(vec![Term(0), Term(7), Term(1)]);
        TermBitset::from(&query, &[Term(2)]);
    }

    #[test]
    fn test_new_term_vec() {
        let query = Query::new(vec![Term(0), Term(7), Term(1)]);
        assert_eq!(TermBitset::from(&query, &[Term(0)]).0, 1);
        assert_eq!(TermBitset::from(&query, &[Term(7)]).0, 2);
        assert_eq!(TermBitset::from(&query, &[Term(1)]).0, 4);
        assert_eq!(TermBitset::from(&query, &[Term(1), Term(7)]).0, 6);
        assert_eq!(TermBitset::from(&query, &[Term(0), Term(1), Term(7)]).0, 7);
    }

    #[test]
    fn test_covers_all() {
        let query = Query::new(vec![Term(0), Term(1), Term(2)]);
        let inter = TermBitset::from(&query, &[Term(1), Term(2)]);
        assert!(inter.covers(TermBitset::from(&query, &[Term(1), Term(2)])));
        assert!(!inter.covers(TermBitset::from(&query, &[Term(0), Term(2)])));
    }

    #[test]
    fn test_covers() {
        assert!(!TermBitset(1).covers(TermBitset(2)));
        assert!(TermBitset(1).covers(TermBitset(3)));
        assert!(TermBitset(2).covers(TermBitset(3)));

        let query = Query::new(vec![Term(0), Term(1), Term(2)]);
        let inter = TermBitset::from(&query, &[Term(1), Term(2)]);
        assert!(inter.covers(TermBitset::from(&query, &[Term(1), Term(2)])));
        assert!(!inter.covers(TermBitset::from(&query, &[Term(0), Term(2)])));
    }

    #[test]
    fn test_into_usize() {
        let n: usize = TermBitset(7).into();
        assert_eq!(n, 7);
    }

    #[test]
    fn test_result_classes() {
        assert_eq!(
            TermBitset::result_classes(3),
            vec![
                TermBitset(0b000),
                TermBitset(0b001),
                TermBitset(0b010),
                TermBitset(0b011),
                TermBitset(0b100),
                TermBitset(0b101),
                TermBitset(0b110),
                TermBitset(0b111),
            ]
        );
    }

    #[test]
    fn test_cover() {
        let classes: Vec<_> = (0..8).map(TermBitset).collect();

        let mut covered = [0_u8; 8];
        TermBitset(0b101).cover(&classes, &mut covered);
        assert_eq!(covered, [0, 0, 0, 0, 0, 1, 0, 1]);

        covered = [0_u8; 8];
        TermBitset(0b10).cover(&classes, &mut covered);
        assert_eq!(covered, [0, 0, 1, 1, 0, 0, 1, 1]);
    }
}
