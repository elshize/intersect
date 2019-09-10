use crate::{Query, Term};
use std::collections::HashMap;

/// The underlying type of the mask used by the term bag.
///
/// In this representation, n-th bit represents the n-th term, and
/// is equal to 1 if it's present, and 0 if it's missing.
pub type TermMask = u8;

/// A bitset representation of a subset of terms in a query.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TermBag(pub TermMask);

impl Into<usize> for TermBag {
    fn into(self) -> usize {
        self.0 as usize
    }
}

impl TermBag {
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
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[should_panic]
    fn test_new_term_vec_wrong_id() {
        let query = Query::new(vec![Term(0), Term(7), Term(1)]);
        TermBag::from(&query, &[Term(2)]);
    }

    #[test]
    fn test_new_term_vec() {
        let query = Query::new(vec![Term(0), Term(7), Term(1)]);
        assert_eq!(TermBag::from(&query, &[Term(0)]).0, 1);
        assert_eq!(TermBag::from(&query, &[Term(7)]).0, 2);
        assert_eq!(TermBag::from(&query, &[Term(1)]).0, 4);
        assert_eq!(TermBag::from(&query, &[Term(1), Term(7)]).0, 6);
        assert_eq!(TermBag::from(&query, &[Term(0), Term(1), Term(7)]).0, 7);
    }

    #[test]
    fn test_covers_all() {
        let query = Query::new(vec![Term(0), Term(1), Term(2)]);
        let inter = TermBag::from(&query, &[Term(1), Term(2)]);
        assert!(inter.covers(TermBag::from(&query, &[Term(1), Term(2)])));
        assert!(!inter.covers(TermBag::from(&query, &[Term(0), Term(2)])));
    }

    #[test]
    fn test_covers() {
        assert!(!TermBag(1).covers(TermBag(2)));
        assert!(TermBag(1).covers(TermBag(3)));
        assert!(TermBag(2).covers(TermBag(3)));

        let query = Query::new(vec![Term(0), Term(1), Term(2)]);
        let inter = TermBag::from(&query, &[Term(1), Term(2)]);
        assert!(inter.covers(TermBag::from(&query, &[Term(1), Term(2)])));
        assert!(!inter.covers(TermBag::from(&query, &[Term(0), Term(2)])));
    }

    #[test]
    fn test_into_usize() {
        let n: usize = TermBag(7).into();
        assert_eq!(n, 7);
    }

    #[test]
    fn test_result_classes() {
        assert_eq!(
            TermBag::result_classes(3),
            vec![
                TermBag(0b000),
                TermBag(0b001),
                TermBag(0b010),
                TermBag(0b011),
                TermBag(0b100),
                TermBag(0b101),
                TermBag(0b110),
                TermBag(0b111),
            ]
        );
    }

    #[test]
    fn test_cover() {
        let classes: Vec<_> = (0..8).map(TermBag).collect();

        let mut covered = [0_u8; 8];
        TermBag(0b101).cover(&classes, &mut covered);
        assert_eq!(covered, [0, 0, 0, 0, 0, 1, 0, 1]);

        covered = [0_u8; 8];
        TermBag(0b10).cover(&classes, &mut covered);
        assert_eq!(covered, [0, 0, 1, 1, 0, 0, 1, 1]);
    }
}
