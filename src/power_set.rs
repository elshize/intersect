use num::cast::{FromPrimitive, ToPrimitive};
use num::{One, PrimInt, Unsigned, Zero};

pub struct SubsetIter<'a, T, U> {
    mask: U,
    slice: &'a [T],
}

/// The type alias for the return type of
/// [`power_set_iter`](fn.power_set_iter.html) function.
pub type PowerSetIter<'a, T> = std::iter::Map<
    std::iter::Zip<std::iter::Repeat<&'a [T]>, std::ops::Range<u64>>,
    fn((&'a [T], u64)) -> SubsetIter<'_, T, u64>,
>;

impl<'a, T, U> SubsetIter<'a, T, U> {
    pub fn from(slice: &'a [T], mask: U) -> Self {
        SubsetIter { mask, slice }
    }
    pub fn from_tuple((slice, mask): (&'a [T], U)) -> Self {
        Self::from(slice, mask)
    }
}

impl<'a, T, U> Iterator for SubsetIter<'a, T, U>
where
    U: Unsigned + FromPrimitive + ToPrimitive + PrimInt + Zero + One + std::ops::BitAndAssign,
{
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.mask == U::zero() {
            None
        } else {
            let idx = self.mask.trailing_zeros() as usize;
            if idx >= self.slice.len() {
                self.mask = U::zero();
                None
            } else {
                self.mask &= !(U::one() << idx);
                Some(&self.slice[idx])
            }
        }
    }
}

/// Returns a power set iterator.
///
/// The `Item` type of this iterator is a `SubsetIter` instance that iterates over elements
/// of a particular subset in the power set. Thus, this is an iterator of iterators over
/// elements of a subset.
///
/// # Examples
///
/// ```
/// # extern crate intersect;
/// # use intersect::power_set_iter;
/// let vec = vec![0, 1, 2];
/// let iter: Vec<Vec<_>> = power_set_iter(&vec).map(|s| s.cloned().collect()).collect();
/// assert_eq!(
///     iter,
///     vec![
///         vec![],
///         vec![0],
///         vec![1],
///         vec![0, 1],
///         vec![2],
///         vec![0, 2],
///         vec![1, 2],
///         vec![0, 1, 2],
///     ]
/// );
/// ```
pub fn power_set_iter<T>(slice: &[T]) -> PowerSetIter<T> {
    let len = slice.len().to_u8().expect("Slice too long.");
    std::iter::repeat(slice)
        .zip(0..2_u64.pow(u32::from(len)))
        .map(SubsetIter::from_tuple)
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_subset_iter() {
        let vec = vec![0, 1, 2, 3, 4, 5, 6];
        assert_eq!(
            SubsetIter::from_tuple((&vec, 0_u64)).collect::<Vec<_>>(),
            Vec::<&usize>::new()
        );
        assert_eq!(
            SubsetIter::from_tuple((&vec, 0b0100_1010_u64)).collect::<Vec<_>>(),
            vec![&1_usize, &3, &6]
        );
        assert_eq!(
            SubsetIter::from_tuple((&vec, 0b1001_0010_1001_u64)).collect::<Vec<_>>(),
            vec![&0_usize, &3, &5]
        );
        assert_eq!(
            SubsetIter::from_tuple((&Vec::<usize>::new(), 1_u64)).collect::<Vec<_>>(),
            Vec::<&usize>::new()
        );
    }

    #[test]
    fn test_power_set_iter() {
        let pwset: Vec<Vec<_>> = power_set_iter(&[0, 1, 2, 3])
            .map(|s| s.cloned().collect())
            .collect();
        assert_eq!(
            pwset,
            vec![
                vec![],
                vec![0],
                vec![1],
                vec![0, 1],
                vec![2],
                vec![0, 2],
                vec![1, 2],
                vec![0, 1, 2],
                vec![3],
                vec![0, 3],
                vec![1, 3],
                vec![0, 1, 3],
                vec![2, 3],
                vec![0, 2, 3],
                vec![1, 2, 3],
                vec![0, 1, 2, 3],
            ]
        );
    }
}
