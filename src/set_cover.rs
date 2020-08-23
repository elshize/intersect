//! This module implements a solution to a weighted set cover problem.

use crate::power_set_iter_enum;
use fixedbitset::FixedBitSet;
use itertools::Itertools;
use num::{FromPrimitive, One, PrimInt, ToPrimitive, Unsigned};
use ordered_float::OrderedFloat;

/// Interface for the weighted set cover problem.
///
/// # Note
///
/// Types implementing this trait might return an approximate results.
/// No assumption on accurracy of the results can be made.
/// This trait is meant to test and compare different implementations of potentially
/// approximate solutions.
pub trait WeightedSetCover<U> {
    /// Solve the following weighted set cover problem:
    /// - the elements are denoted as integers of an unsigned type `U`
    ///   in range `0..cardinality`.
    /// - each subset is defined by a triple `(elements, weight)`,
    ///   where `elements` is a vector of elements in the subset, and `weight` is its weight.
    /// - returned is a vector of the indices of the selected subsets.
    fn solve(cardinality: U, subsets: &[(Vec<U>, f32)]) -> Vec<usize>
    where
        U: Unsigned + One + PrimInt + ToPrimitive + FromPrimitive + Copy + std::fmt::Debug;
}

/// Exact, brute-force solution to weighted set cover problem.
///
/// # Warning
///
/// This implementation is extremely inefficient, and should really not be used for
/// anything than testing small instances of the problem. Use with caution.
///
/// # Panics
///
/// Panics if elements cannot be cast to `usize`.
pub struct BruteForceSetCover {}

impl BruteForceSetCover {
    /// Construct a new solver.
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }
}

impl<U> WeightedSetCover<U> for BruteForceSetCover {
    fn solve(cardinality: U, subsets: &[(Vec<U>, f32)]) -> Vec<usize>
    where
        U: Unsigned + One + PrimInt + ToPrimitive + FromPrimitive + Copy + std::fmt::Debug,
    {
        let size: usize = cardinality.to_usize().expect("Failed to cast U to u32");
        for (subset, _) in subsets {
            for &element in subset {
                assert!(element < cardinality);
            }
        }
        let mut min_cost = std::f32::MAX;
        let mut best: Vec<usize> = vec![];
        for candidate_iter in power_set_iter_enum(subsets) {
            let mut covered = FixedBitSet::with_capacity(size);
            let candidate: Vec<_> = candidate_iter.collect();
            'outer: for (_, (subset, _)) in &candidate {
                for elem in subset {
                    if !covered[elem.to_usize().unwrap()] {
                        covered.set(elem.to_usize().unwrap(), true);
                        if covered.count_ones(..) == size {
                            break 'outer;
                        }
                    }
                }
            }
            if covered.count_ones(..) == size {
                let cost: f32 = candidate.iter().map(|(_, (_, cost))| cost).sum();
                if cost < min_cost {
                    min_cost = cost;
                    best = candidate.iter().map(|(idx, (_, _))| idx).copied().collect();
                }
            }
        }
        best
    }
}

/// Iterates over subsets until all elements are covered, and returns the prefix as the result.
/// **Note:** Only for testing.
pub struct FirstSeenSetCover;

impl FirstSeenSetCover {
    /// Constructs a new solver.
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }
}

impl<U> WeightedSetCover<U> for FirstSeenSetCover {
    fn solve(cardinality: U, subsets: &[(Vec<U>, f32)]) -> Vec<usize>
    where
        U: Unsigned + One + PrimInt + ToPrimitive + FromPrimitive + Copy + std::fmt::Debug,
    {
        let size: usize = cardinality.to_usize().expect("Failed to cast U to u32");
        let mut covered = FixedBitSet::with_capacity(size);
        let mut solution: Vec<usize> = Vec::new();
        for (idx, (subset, _)) in subsets.iter().enumerate() {
            for &element in subset {
                assert!(element < cardinality);
                covered.insert(element.to_usize().unwrap());
            }
            solution.push(idx);
            if covered.count_ones(..) == covered.len() {
                return solution;
            }
        }
        vec![]
    }
}

/// Greedy weighted set cover.
pub struct GreedySetCover;

impl GreedySetCover {
    /// Constructs a new solver.
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }
}

impl<U> WeightedSetCover<U> for GreedySetCover {
    fn solve(cardinality: U, subsets: &[(Vec<U>, f32)]) -> Vec<usize>
    where
        U: Unsigned + One + PrimInt + ToPrimitive + FromPrimitive + Copy + std::fmt::Debug,
    {
        let size = cardinality.to_usize().unwrap();
        if subsets.is_empty() {
            return vec![];
        }

        let subsets: Vec<_> = subsets
            .iter()
            .map(|(v, c)| {
                let subset: FixedBitSet = v.iter().map(|elem| elem.to_usize().unwrap()).collect();
                (subset, c)
            })
            .collect();
        let mut selected = FixedBitSet::with_capacity(subsets.len());
        let mut available = FixedBitSet::with_capacity(subsets.len());
        available.set_range(.., true);

        let mut covered = FixedBitSet::with_capacity(size);
        while available.count_ones(..) != 0 && covered.count_ones(..) < size {
            let min = subsets
                .iter()
                .enumerate()
                .map(|s| vec![s])
                //.chain(subsets.iter().enumerate().combinations(2))
                //.chain(subsets.iter().enumerate().combinations(3))
                //.chain(subsets.iter().enumerate().combinations(4))
                //.chain(subsets.iter().enumerate().combinations(5))
                .filter_map(|subs| {
                    let mut bs = FixedBitSet::with_capacity(subsets.len());
                    let mut idxs = Vec::<usize>::new();
                    let mut cost = 0_f32;
                    for (idx, (subset, c)) in subs {
                        bs.union_with(&subset);
                        idxs.push(idx);
                        cost += **c;
                    }
                    let diff: FixedBitSet = bs.difference(&covered).collect();
                    if diff.count_ones(..) == 0 {
                        for idx in idxs {
                            available.set(idx, false);
                        }
                        None
                    } else {
                        Some((
                            idxs,
                            bs,
                            cost,
                            OrderedFloat(cost / diff.count_ones(..).to_f32().unwrap()),
                        ))
                    }
                })
                //.filter_map(|(idx, (subset, cost))| {
                //    if selected.contains(idx) {
                //        None
                //    } else {
                //        let diff: FixedBitSet = subset.difference(&covered).collect();
                //        if diff.count_ones(..) == 0 {
                //            available.set(idx, false);
                //            None
                //        } else {
                //            Some((
                //                idx,
                //                subset,
                //                *cost,
                //                OrderedFloat(*cost / diff.count_ones(..).to_f32().unwrap()),
                //            ))
                //        }
                //    }
                //})
                .min_by(|lhs, rhs| lhs.3.cmp(&rhs.3));
            if let Some((idxs, subset, _cost, _)) = min {
                //println!("SELECTED: {:?}", idxs);
                for idx in idxs {
                    selected.insert(idx);
                    available.set(idx, false);
                }
                covered.union_with(&subset);
            }
        }
        selected.ones().collect()
    }
}

/// Greedy weighted set cover.
pub struct GreedyNSetCover;

impl GreedyNSetCover {
    /// Constructs a new solver.
    pub fn new() -> Self {
        Self {}
    }
}

impl<U> WeightedSetCover<U> for GreedyNSetCover {
    fn solve(cardinality: U, subsets: &[(Vec<U>, f32)]) -> Vec<usize>
    where
        U: Unsigned + One + PrimInt + ToPrimitive + FromPrimitive + Copy + std::fmt::Debug,
    {
        let size = cardinality.to_usize().unwrap();
        if subsets.is_empty() {
            return vec![];
        }

        let subsets: Vec<_> = subsets
            .iter()
            .map(|(v, c)| {
                let subset: FixedBitSet = v.iter().map(|elem| elem.to_usize().unwrap()).collect();
                (subset, c)
            })
            .collect();
        let mut selected = FixedBitSet::with_capacity(subsets.len());
        let mut available = FixedBitSet::with_capacity(subsets.len());
        available.set_range(.., true);

        let calc_gain = |state: FixedBitSet, subsets: &[(Vec<U>, f32)]| {
            //
        };

        let mut covered = FixedBitSet::with_capacity(size);
        while available.count_ones(..) != 0 && covered.count_ones(..) < size {
            let min = subsets
                .iter()
                .enumerate()
                .combinations(2)
                .filter_map(|subs| {
                    let mut bs = FixedBitSet::with_capacity(subsets.len());
                    let mut idxs = Vec::<usize>::new();
                    let mut cost = 0_f32;
                    for (idx, (subset, c)) in subs {
                        bs.union_with(&subset);
                        idxs.push(idx);
                        cost += **c;
                    }
                    let diff: FixedBitSet = bs.difference(&covered).collect();
                    if diff.count_ones(..) == 0 {
                        for idx in idxs {
                            available.set(idx, false);
                        }
                        None
                    } else {
                        Some((
                            idxs,
                            bs,
                            cost,
                            OrderedFloat(cost / diff.count_ones(..).to_f32().unwrap()),
                        ))
                    }
                })
                .min_by(|lhs, rhs| lhs.3.cmp(&rhs.3));
            if let Some((idxs, subset, _cost, _)) = min {
                for idx in idxs {
                    selected.insert(idx);
                    available.set(idx, false);
                }
                covered.union_with(&subset);
            }
        }
        selected.ones().collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use proptest::prelude::*;
    use proptest::strategy::Strategy;
    use std::collections::HashSet;

    #[test]
    fn test_set_cover() {
        assert_eq!(
            BruteForceSetCover::solve(4_u8, &[(Vec::<u8>::new(), 0.1)]),
            Vec::<usize>::new()
        );
        assert_eq!(
            BruteForceSetCover::solve(4_u8, &[(vec![0, 1, 2, 3], 0.1)]),
            &[0]
        );
        assert_eq!(
            BruteForceSetCover::solve(
                4_u8,
                &[
                    (vec![0, 1, 2, 3], 1.0),
                    (vec![0, 1], 0.4),
                    (vec![2, 3], 0.4)
                ]
            ),
            &[1, 2]
        );
    }

    #[test]
    fn test_greedy_set_cover() {
        assert_eq!(
            GreedySetCover::solve(4_u8, &[(vec![], 0.1)]),
            Vec::<usize>::new()
        );
        assert_eq!(
            GreedySetCover::solve(4_u8, &[(vec![0, 1, 2, 3], 0.1)]),
            &[0]
        );
        assert_eq!(
            GreedySetCover::solve(
                4_u8,
                &[
                    (vec![0, 1, 2, 3], 1.0),
                    (vec![0, 1], 0.4),
                    (vec![2, 3], 0.4)
                ]
            )
            .into_iter()
            .collect::<HashSet<_>>(),
            (1..3).collect()
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]
        #[test]
        fn test_brute_force(
            (size, subsets, costs) in (4_usize..10).prop_flat_map(|size| {
                let elements: Vec<_> = (0..size).collect();
                let subsets = proptest::collection::vec(
                    proptest::sample::subsequence(elements.clone(), 1..size),
                    3..16
                );
                let costs = proptest::collection::vec(0.1_f32..10.0, size);
                (Just(size), subsets, costs)
            })
        ) {
            let input: Vec<_> = subsets.into_iter().zip(costs).collect();
            let cost = |solution: Vec<usize>| -> f32 {
                solution.into_iter().map(|idx| input[idx].1).sum()
            };
            assert!(
                cost(BruteForceSetCover::solve(size, &input))
                <= cost(FirstSeenSetCover::solve(size, &input))
            );
        }
    }
}
