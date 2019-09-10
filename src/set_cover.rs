//! This module implements a solution to a weighted set cover problem.

use crate::power_set_iter;
use num::{FromPrimitive, One, PrimInt, ToPrimitive, Unsigned};
use ordered_float::OrderedFloat;
use std::collections::HashSet;

/// Exact set cover solution, which is an NP problem.
///
/// This implementation simply iterates over all posibilities,
/// checks if a given configuration of set covers the universe,
/// and updates the current minimum cost.
pub fn set_cover<U>(universe_cardinality: U, subsets: &[(U, Vec<U>, f32)]) -> Vec<U>
where
    U: Unsigned + One + PrimInt + ToPrimitive + FromPrimitive + Copy + std::fmt::Debug,
{
    let size: u32 = universe_cardinality
        .to_u32()
        .expect("Failed to cast U to u32");
    let mut min_cost = std::f32::MAX;
    let mut best: Vec<U> = vec![];
    for subsets in power_set_iter(subsets) {
        let mut covered = vec![false; size as usize];
        let mut covsum = 0;
        let candidate: Vec<_> = subsets.cloned().collect();
        'outer: for (_, subset, _) in &candidate {
            for elem in subset {
                if !covered[elem.to_usize().unwrap()] {
                    covered[elem.to_usize().unwrap()] = true;
                    covsum += 1;
                    if covsum == size {
                        break 'outer;
                    }
                }
            }
        }
        if covsum == size {
            let cost: f32 = candidate.iter().map(|(_idx, _v, cost)| cost).sum();
            if cost < min_cost {
                min_cost = cost;
                best = candidate.iter().map(|(idx, _v, _c)| idx).cloned().collect();
            }
        }
    }
    best
}

/// Exact set cover solution, which is an NP problem.
///
/// This implementation simply iterates over all posibilities,
/// checks if a given configuration of set covers the universe,
/// and updates the current minimum cost.
pub fn greedy_set_cover(
    universe_cardinality: usize,
    subsets: &[(usize, Vec<usize>, f32)],
) -> Vec<usize> {
    if subsets.is_empty() {
        return vec![];
    }
    let mut available: HashSet<usize> = subsets.iter().map(|(idx, _, _)| *idx).collect();
    let mut chosen: HashSet<usize> = HashSet::new();

    let mut covered = vec![false; universe_cardinality];
    let mut covsum = 0_usize;

    while !available.is_empty() && covsum < universe_cardinality {
        let min = subsets
            .iter()
            .filter_map(|(subset_idx, subset, cost)| {
                if chosen.contains(subset_idx) {
                    None
                } else {
                    let diff_size: usize = subset
                        .iter()
                        .map(|&elem| if covered[elem] { 0 } else { 1 })
                        .sum();
                    if diff_size == 0 {
                        available.remove(subset_idx);
                        None
                    } else {
                        Some((
                            subset_idx,
                            subset,
                            *cost,
                            diff_size,
                            OrderedFloat(*cost / diff_size.to_f32().unwrap()),
                        ))
                    }
                }
            })
            .min_by(|lhs, rhs| lhs.4.cmp(&rhs.4));
        if let Some((subset_idx, subset, _cost, diff_size, _)) = min {
            covsum += diff_size;
            chosen.insert(*subset_idx);
            available.remove(&subset_idx);
            for &elem in subset {
                covered[elem] = true;
            }
        }
    }
    chosen.into_iter().collect()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_set_cover() {
        assert_eq!(set_cover(4_u8, &[(0, vec![], 0.1)]), vec![]);
        assert_eq!(set_cover(4_u8, &[(0, vec![0, 1, 2, 3], 0.1)]), &[0]);
        assert_eq!(
            set_cover(
                4_u8,
                &[
                    (0, vec![0, 1, 2, 3], 1.0),
                    (1, vec![0, 1], 0.4),
                    (2, vec![2, 3], 0.4)
                ]
            ),
            &[1, 2]
        );
    }

    #[test]
    fn test_greedy_set_cover() {
        assert_eq!(greedy_set_cover(4, &[(0, vec![], 0.1)]), vec![]);
        assert_eq!(greedy_set_cover(4, &[(0, vec![0, 1, 2, 3], 0.1)]), &[0]);
        assert_eq!(
            greedy_set_cover(
                4,
                &[
                    (0, vec![0, 1, 2, 3], 1.0),
                    (1, vec![0, 1], 0.4),
                    (2, vec![2, 3], 0.4)
                ]
            ),
            &[1, 2]
        );
    }
}
