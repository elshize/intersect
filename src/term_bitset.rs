use crate::{Query, Term};
use failure::{format_err, Error, ResultExt};
use num::ToPrimitive;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

/// The underlying type of the mask used by the term bag.
///
/// In this representation, n-th bit represents the n-th term, and
/// is equal to 1 if it's present, and 0 if it's missing.
pub type TermMask = u8;

fn term_mask_from_query(query: &Query, term_subset: &[Term]) -> Result<TermMask, Error> {
    let mut mask: TermMask = 0;
    let mut mapping = HashMap::<Term, usize>::new();
    for (idx, &term) in query.terms.iter().enumerate() {
        mapping.insert(term, idx);
    }
    for term in term_subset.iter() {
        let idx = mapping
            .get(term)
            .ok_or_else(|| format_err!("Term not in query."))?;
        let one: TermMask = 1;
        mask |= one << idx;
    }
    Ok(mask)
}

fn term_mask_from_str(index: &str) -> Result<TermMask, Error> {
    let num = u8::from_str_radix(index, 2)
        .with_context(|_| format!("Invalid bitset string: {}", index))?;
    Ok(num)
}

/// Represents a class of results having all terms marked as 1 but no terms marked as 0.
///
/// Given a query, all possible resulting documents can be partitioned into
/// a number of disjoint classes defined by the terms they contain (exclusively).
/// For example, in a 3-term query, the class `100` includes all documents containing
/// **only** the first term, while the class `111` includes the documents containing
/// all three terms.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ResultClass(pub TermMask);

impl FromStr for ResultClass {
    type Err = Error;
    fn from_str(index: &str) -> Result<Self, Self::Err> {
        term_mask_from_str(index).map(Self)
    }
}

impl Into<usize> for ResultClass {
    fn into(self) -> usize {
        self.0 as usize
    }
}

impl ResultClass {
    /// Iterates over all possible result classes for this query length.
    ///
    /// Essentially, it generates a vector of values from 0 to `2^len`.
    ///
    /// # Examples
    /// ```
    /// # use intersect::ResultClass;
    /// assert_eq!(
    ///     ResultClass::all(2).map(|ResultClass(c)| c).collect::<Vec<u8>>(),
    ///     vec![0_u8, 1, 2, 3]
    /// );
    /// ```
    pub fn all(len: u8) -> impl Iterator<Item = Self> {
        let two: TermMask = 2;
        (0..two.pow(u32::from(len))).map(Self)
    }

    /// Collects all possible result classes to a vector.
    ///
    /// # Examples
    ///
    /// This is a convenient shorthand to [`all`](#methods.all) that collects elements
    /// instead of returning an iterator.
    /// ```
    /// # use intersect::ResultClass;
    /// assert_eq!(ResultClass::all_to_vec(3), ResultClass::all(3).collect::<Vec<_>>());
    /// ```
    pub fn all_to_vec(len: u8) -> Vec<Self> {
        Self::all(len).collect()
    }

    /// Number of terms in this class.
    #[inline]
    pub fn degree(self) -> u32 {
        self.0.count_ones()
    }

    /// Iterator over result class of each single term.
    pub fn components(self) -> ClassComponents {
        ClassComponents { mask: self.0 }
    }
}

pub struct ClassComponents {
    mask: TermMask,
}

impl Iterator for ClassComponents {
    type Item = ResultClass;

    fn next(&mut self) -> Option<Self::Item> {
        if self.mask == 0 {
            None
        } else {
            let item = 1 << self.mask.trailing_zeros();
            self.mask ^= item;
            Some(ResultClass(item))
        }
    }
}

/// Represents a posting list of a single term, if only one `1` is present,
/// or an intersection of multiple terms.
///
/// Although intersection are in a close relation with [`ResultClass`](#structs.ResultClass)
/// objects, there is an important distinction.
/// A result class is an abstract and implicit object, while intersections are provided
/// explicitly by the user.
/// The relations between result classes and intersections are not symmetric,
/// hence different type to avoid mistakes.
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Intersection(pub TermMask);

impl fmt::Debug for Intersection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Intersection({:08b})", self.0)
    }
}

impl FromStr for Intersection {
    type Err = Error;
    fn from_str(index: &str) -> Result<Self, Self::Err> {
        term_mask_from_str(index).map(Self)
    }
}

impl Into<usize> for Intersection {
    fn into(self) -> usize {
        self.0 as usize
    }
}

impl Intersection {
    /// Constructs an intersection struct from a query and a subset of query terms.
    pub fn from(query: &Query, term_subset: &[Term]) -> Result<Self, Error> {
        term_mask_from_query(query, term_subset).map(Self)
    }

    /// Checks if this intersection covers a given result class.
    #[inline]
    pub fn covers(self, class: ResultClass) -> bool {
        self.0 & class.0 == self.0
    }

    /// Marks classes that this intersection covers.
    #[inline]
    pub fn cover(self, classes: &[ResultClass], covered: &mut [u8]) {
        for (covered, &class) in covered.iter_mut().zip(classes) {
            *covered |= self.covers(class) as u8;
        }
    }

    /// Number of terms in the intersection.
    #[inline]
    pub fn degree(self) -> u32 {
        self.0.count_ones()
    }

    /// Iterator over positions in the query of contained terms.
    ///
    /// # Examples
    ///
    /// Notice that you can freely call it on a temporary object, because `Intersection`
    /// implements [`Copy`](https://doc.rust-lang.org/std/marker/trait.Copy.html) trait.
    /// ```
    /// # use intersect::Intersection;
    /// let mut terms = Intersection(0b10101).terms();
    /// assert_eq!(terms.next(), Some(0));
    /// assert_eq!(terms.next(), Some(2));
    /// assert_eq!(terms.next(), Some(4));
    /// assert_eq!(terms.next(), None);
    /// ```
    pub fn terms(self) -> IntersectionTerms {
        IntersectionTerms { mask: self.0 }
    }
}

pub struct IntersectionTerms {
    mask: TermMask,
}

impl Iterator for IntersectionTerms {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        if self.mask == 0 {
            None
        } else {
            let item = self.mask.trailing_zeros().to_u8().unwrap();
            self.mask ^= 1 << item;
            Some(item)
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_intersection_terms() {
        let inter = Intersection(0b10101);
        assert_eq!(inter.terms().collect::<Vec<_>>(), vec![0, 2, 4]);
    }

    #[test]
    fn test_from_str() {
        let inter: Intersection = "01010".parse().unwrap();
        assert_eq!(inter, Intersection(0b1010));
        assert!("0x21".parse::<Intersection>().is_err());

        let class: ResultClass = "01010".parse().unwrap();
        assert_eq!(class, ResultClass(0b1010));
        assert!("0x21".parse::<ResultClass>().is_err());
    }

    #[test]
    fn test_new_term_vec_wrong_id() {
        let query = Query::new(vec![Term(0), Term(7), Term(1)]);
        assert!(Intersection::from(&query, &[Term(2)]).is_err());
    }

    #[test]
    fn test_new_term_vec() {
        let query = Query::new(vec![Term(0), Term(7), Term(1)]);
        assert_eq!(Intersection::from(&query, &[Term(0)]).unwrap().0, 1);
        assert_eq!(Intersection::from(&query, &[Term(7)]).unwrap().0, 2);
        assert_eq!(Intersection::from(&query, &[Term(1)]).unwrap().0, 4);
        assert_eq!(
            Intersection::from(&query, &[Term(1), Term(7)]).unwrap().0,
            6
        );
        assert_eq!(
            Intersection::from(&query, &[Term(0), Term(1), Term(7)])
                .unwrap()
                .0,
            7
        );
    }

    #[test]
    fn test_covers() {
        assert!(!Intersection(1).covers(ResultClass(0b10)));
        assert!(Intersection(1).covers(ResultClass(0b11)));
        assert!(Intersection(0b10).covers(ResultClass(0b11)));

        let query = Query::new(vec![Term(0), Term(1), Term(2)]);
        let inter = Intersection::from(&query, &[Term(1), Term(2)]).unwrap();
        assert!(inter.covers(ResultClass(0b110)));
        assert!(!inter.covers(ResultClass(0b101)));
    }

    #[test]
    fn test_into_usize() {
        let n: usize = Intersection(7).into();
        assert_eq!(n, 7);
        let m: usize = ResultClass(6).into();
        assert_eq!(m, 6);
    }

    #[test]
    fn test_result_classes() {
        assert_eq!(
            ResultClass::all_to_vec(3),
            vec![
                ResultClass(0b000),
                ResultClass(0b001),
                ResultClass(0b010),
                ResultClass(0b011),
                ResultClass(0b100),
                ResultClass(0b101),
                ResultClass(0b110),
                ResultClass(0b111),
            ]
        );
    }

    #[test]
    fn test_cover() {
        // let classes: Vec<_> = (0..8).map(ResultClass).collect();
        let classes: Vec<_> = ResultClass::all_to_vec(3);

        let mut covered = [0_u8; 8];
        Intersection(0b101).cover(&classes, &mut covered);
        assert_eq!(covered, [0, 0, 0, 0, 0, 1, 0, 1]);

        covered = [0_u8; 8];
        Intersection(0b10).cover(&classes, &mut covered);
        assert_eq!(covered, [0, 0, 1, 1, 0, 0, 1, 1]);
    }
}
