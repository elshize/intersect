use crate::graph::Graph;
use crate::power_set::power_set_iter;
use crate::set_cover::{greedy_set_cover, set_cover};
use crate::{Cost, Score, TermBitset, MAX_LIST_COUNT};
use itertools::Itertools;
use std::collections::HashSet;
use std::iter::FromIterator;

#[derive(Debug)]
struct CandidateGraph {
    candidates: Vec<TermBitset>,
    leaves: Vec<TermBitset>,
}

impl CandidateGraph {
    fn cardinality(&self) -> usize {
        self.leaves.len()
    }

    fn subsets(&self, index: &Index) -> Vec<(usize, Vec<usize>, f32)> {
        self.candidates
            .iter()
            .enumerate()
            .filter_map(|(set_idx, &candidate)| {
                let set: Vec<usize> = self
                    .leaves
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, &leaf)| {
                        if candidate.covers(leaf) {
                            Some(idx)
                        } else {
                            None
                        }
                    })
                    .collect();
                let cand_idx: usize = candidate.into();
                let Cost(cost) = index.costs[cand_idx];
                if set.is_empty() {
                    None
                } else {
                    Some((set_idx, set, cost))
                }
            })
            .collect()
    }

    fn solve(&self, index: &Index) -> Vec<usize> {
        set_cover(self.cardinality(), &self.subsets(index))
    }

    fn solve_greedy(&self, index: &Index) -> Vec<usize> {
        greedy_set_cover(self.cardinality(), &self.subsets(index))
    }
}

#[derive(Default, Debug)]
struct Switch {
    remove: Vec<TermBitset>,
    insert: Vec<TermBitset>,
    gain: Cost,
}

/// Represents a full set of available posting lists for a given query.
pub struct Index {
    /// All available posting lists partitioned by number of terms.
    /// Each element contains a vector of posting lists of the same length,
    /// and the shorter lists always come before the longer ones.
    pub degrees: Vec<Vec<TermBitset>>,
    /// An array of posting list costs. The cost of `t` is in `costs[t]`.
    pub costs: [Cost; MAX_LIST_COUNT],
    /// An array of posting list upper bounds. The bound of `t` is in `upper_bound[t]`.
    pub upper_bounds: [Score; MAX_LIST_COUNT],
}

/// Method of selecting optimal set of intersections.
#[derive(Debug, Clone, Copy)]
pub enum OptimizeMethod {
    /// Brute-force method calculating exactly all possible subsets.
    /// Very slow on anything non-trivial.
    BruteForce,
    /// Returns the result equivalent to `BruteForce` but limits the number
    /// of possible candidates significantly.
    /// Faster than `BruteForce` but still very slow on longer queries.
    Exact,
    /// This is essentially a greedy approximation algorithm for weighted set cover.
    /// The result is approximate but this method is very fast.
    Greedy,
    /// Another approximate solution that has potential of being more accurate
    /// than `Greedy` (to be determined) but not as fast.
    /// Still, is reasonably fast if yields better results.
    Graph,
}

impl Index {
    /// Construct a new index containing the given posting lists.
    pub fn new(posting_lists: &[Vec<(TermBitset, Cost, Score)>]) -> Self {
        let mut costs = [Cost::default(); MAX_LIST_COUNT];
        let mut upper_bounds = [Score::default(); MAX_LIST_COUNT];
        let mut degrees: Vec<Vec<TermBitset>> = Vec::new();
        for arity in posting_lists {
            let mut bags: Vec<TermBitset> = Vec::with_capacity(arity.len());
            for &(terms, cost, upper_bound) in arity {
                let idx: usize = terms.into();
                costs[idx] = cost;
                upper_bounds[idx] = upper_bound;
                bags.push(terms);
            }
            degrees.push(bags);
        }
        Self {
            degrees,
            costs,
            upper_bounds,
        }
    }

    fn cost(&self, terms: TermBitset) -> Cost {
        self.costs[terms.0 as usize]
    }

    /// Select optimal set of posting lists for query execution.
    pub fn optimize(
        &self,
        query_len: u8,
        threshold: Score,
        method: OptimizeMethod,
    ) -> Vec<TermBitset> {
        match method {
            OptimizeMethod::BruteForce => self.optimize_brute_force(query_len, threshold),
            OptimizeMethod::Exact => self.optimize_smart(query_len, threshold),
            OptimizeMethod::Greedy => self.optimize_greedy(query_len, threshold),
            OptimizeMethod::Graph => self.optimize_graph(query_len, threshold),
        }
    }

    fn optimize_brute_force(&self, query_len: u8, threshold: Score) -> Vec<TermBitset> {
        let classes = TermBitset::result_classes(query_len);
        let must_cover = std::iter::once(false)
            .chain(
                classes[1..]
                    .iter()
                    .map(|&TermBitset(mask)| self.upper_bounds[mask as usize] >= threshold),
            )
            .collect::<Vec<_>>();
        let candidates = self.candidates(query_len, threshold);
        let mut min_cost = Cost(std::f32::MAX);
        let mut min_subset: Vec<TermBitset> = vec![];
        for candidate_subset in power_set_iter(&candidates) {
            let candidate_subset: Vec<_> = candidate_subset.cloned().collect();
            let mut covered = vec![0_u8; 2_usize.pow(u32::from(query_len))];
            for bag in &candidate_subset {
                bag.cover(&classes, &mut covered);
            }
            if covered
                .iter()
                .zip(&must_cover)
                .any(|(&covered, &must_cover)| must_cover && covered == 0_u8)
            {
                continue;
            }
            let cost = candidate_subset
                .iter()
                .map(|&TermBitset(mask)| self.costs[mask as usize])
                .sum::<Cost>();
            if cost < min_cost {
                min_cost = cost;
                min_subset = candidate_subset;
            }
        }
        min_subset
    }

    fn optimize_smart(&self, query_len: u8, threshold: Score) -> Vec<TermBitset> {
        let candidates = self.candidate_graph(query_len, threshold);
        let solution = candidates.solve(&self);
        solution
            .into_iter()
            .map(|set| candidates.candidates[set])
            .collect()
    }

    fn optimize_greedy(&self, query_len: u8, threshold: Score) -> Vec<TermBitset> {
        let candidates = self.candidate_graph(query_len, threshold);
        let solution = candidates.solve_greedy(&self);
        solution
            .into_iter()
            .map(|set| candidates.candidates[set])
            .collect()
    }

    fn optimize_graph(&self, query_len: u8, threshold: Score) -> Vec<TermBitset> {
        let CandidateGraph { candidates, leaves } = self.candidate_graph(query_len, threshold);
        let graph = Graph::from_iter(candidates.into_iter());
        let mut solution: HashSet<TermBitset> = leaves.into_iter().collect();
        for layer in graph.layers().rev() {
            let switch_candidates = layer
                .filter_map(|node| {
                    if solution.contains(&node) {
                        graph.parents(node)
                    } else {
                        None
                    }
                })
                .flatten()
                .cloned()
                .unique()
                .collect::<Vec<TermBitset>>();
            let mut best_switch = Switch::default();
            for candidate_set in power_set_iter(&switch_candidates) {
                let insert: Vec<_> = candidate_set.cloned().collect();
                let remove: Vec<_> = insert
                    .iter()
                    .flat_map(|&p| graph.children(p).unwrap_or(&[]))
                    .filter(|n| solution.contains(n))
                    .cloned()
                    .unique()
                    .collect();
                let gain = remove.iter().map(|&ch| self.cost(ch)).sum::<Cost>()
                    - insert.iter().map(|&ch| self.cost(ch)).sum::<Cost>();
                if gain > best_switch.gain {
                    best_switch = Switch {
                        remove,
                        insert,
                        gain,
                    };
                }
            }
            if best_switch.gain > Cost(0_f32) {
                for node in best_switch.remove {
                    solution.remove(&node);
                }
                for node in best_switch.insert {
                    solution.insert(node);
                }
            }
        }
        let mut solution: Vec<_> = solution.into_iter().collect();
        solution.sort();
        solution
    }

    /// Returns candidates, i.e., such posting lists that can be in an optimal solution.
    ///
    /// The following lists will be included:
    /// a. Any _minimal_ list with an upper bound above the threshold.
    ///    A minimal list is a list such that no list containing a strict subset
    ///    of its terms has a bound above the threshold.
    ///    In other words, it is a list exceeding the threshold that is not covered
    ///    by any other list.
    /// b. Lists below the threshold that are not covered by any minimal list.
    ///
    /// In other words, we **exclude** lists that are covered by another list from (a).
    /// This is because these will be either covered automatically by a list from (a)
    /// or one of the list from (b) that covers a list from (a).
    #[inline]
    fn candidates(&self, query_len: u8, threshold: Score) -> Vec<TermBitset> {
        self.layered_candidates(query_len, threshold)
            .into_iter()
            .flatten()
            .collect()
    }

    #[inline]
    fn layered_candidates(&self, query_len: u8, threshold: Score) -> Vec<Vec<TermBitset>> {
        let mut cand: Vec<Vec<TermBitset>> = Vec::with_capacity(self.degrees.len());
        let mut covered = vec![0_u8; 2_usize.pow(u32::from(query_len))];
        let classes = TermBitset::result_classes(query_len);
        for arity in &self.degrees {
            let mut arity_cand: Vec<TermBitset> = Vec::with_capacity(arity.len());
            for &posting_list in arity {
                let TermBitset(mask) = posting_list;
                if covered[mask as usize] == 0_u8 {
                    arity_cand.push(posting_list);
                    if self.upper_bounds[mask as usize] >= threshold {
                        posting_list.cover(&classes, &mut covered);
                    }
                }
            }
            cand.push(arity_cand);
        }
        cand
    }

    #[inline]
    fn candidate_graph(&self, query_len: u8, threshold: Score) -> CandidateGraph {
        let mut cand: Vec<Vec<TermBitset>> = Vec::with_capacity(self.degrees.len());
        let mut leaves: Vec<TermBitset> = Vec::with_capacity(query_len.pow(2) as usize);
        let mut covered = vec![0_u8; 2_usize.pow(u32::from(query_len))];
        let classes = TermBitset::result_classes(query_len);
        for arity in &self.degrees {
            let mut arity_cand: Vec<TermBitset> = Vec::with_capacity(arity.len());
            for &posting_list in arity {
                let TermBitset(mask) = posting_list;
                if covered[mask as usize] == 0_u8 {
                    arity_cand.push(posting_list);
                    if self.upper_bounds[mask as usize] >= threshold {
                        leaves.push(posting_list);
                        posting_list.cover(&classes, &mut covered);
                    }
                }
            }
            cand.push(arity_cand);
        }
        CandidateGraph {
            candidates: cand.into_iter().flatten().collect(),
            leaves,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use OptimizeMethod::BruteForce;

    #[test]
    fn test_new_index() {
        let index = Index::new(&[]);
        assert!(index.degrees.is_empty());
        let index = Index::new(&[
            vec![
                (TermBitset(13), Cost(0.1), Score(0.0)),
                (TermBitset(4), Cost(1.1), Score(5.6)),
            ],
            vec![
                (TermBitset(3), Cost(4.1), Score(9.0)),
                (TermBitset(1), Cost(4.1), Score(1.6)),
            ],
        ]);
        assert_eq!(
            index.degrees,
            vec![
                vec![TermBitset(13), TermBitset(4)],
                vec![TermBitset(3), TermBitset(1)],
            ]
        );
        let mut expected_costs = [Cost::default(); MAX_LIST_COUNT];
        expected_costs[13] = Cost(0.1);
        expected_costs[4] = Cost(1.1);
        expected_costs[3] = Cost(4.1);
        expected_costs[1] = Cost(4.1);
        assert_eq!(index.costs.to_vec(), expected_costs.to_vec());
        let mut expected_scores = [Score::default(); MAX_LIST_COUNT];
        expected_scores[13] = Score(0.0);
        expected_scores[4] = Score(5.6);
        expected_scores[3] = Score(9.0);
        expected_scores[1] = Score(1.6);
        assert_eq!(index.upper_bounds.to_vec(), expected_scores.to_vec());
    }

    #[test]
    fn test_layered_candidates() {
        let query_len = 4_u8;
        let unigrams = (0..query_len)
            .map(|term| {
                (
                    TermBitset(1 << term),
                    Cost::default(),
                    Score(f32::from(term)),
                )
            })
            .collect::<Vec<_>>();
        let mut bigrams: Vec<(TermBitset, Cost, Score)> = Vec::new();
        for left in 0..query_len {
            for right in (left + 1)..query_len {
                bigrams.push((
                    TermBitset((1 << left) | (1 << right)),
                    Cost::default(),
                    Score(f32::from(left + right + 1)),
                ));
            }
        }
        let index = Index::new(&[unigrams, bigrams]);

        assert_eq!(
            index.layered_candidates(query_len, Score(0.0)),
            vec![
                vec![
                    TermBitset(0b0001),
                    TermBitset(0b0010),
                    TermBitset(0b0100),
                    TermBitset(0b1000)
                ],
                vec![]
            ]
        );

        assert_eq!(
            index.layered_candidates(query_len, Score(1.0)),
            vec![
                vec![
                    TermBitset(0b0001),
                    TermBitset(0b0010),
                    TermBitset(0b0100),
                    TermBitset(0b1000),
                ],
                vec![]
            ]
        );

        assert_eq!(
            index.layered_candidates(query_len, Score(2.0)),
            vec![
                vec![
                    TermBitset(0b0001),
                    TermBitset(0b0010),
                    TermBitset(0b0100),
                    TermBitset(0b1000),
                ],
                vec![TermBitset(0b0011)],
            ]
        );
    }

    #[test]
    fn test_candidates() {
        let query_len = 4_u8;
        let unigrams = (0..query_len)
            .map(|term| {
                (
                    TermBitset(1 << term),
                    Cost::default(),
                    Score(f32::from(term)),
                )
            })
            .collect::<Vec<_>>();
        let mut bigrams: Vec<(TermBitset, Cost, Score)> = Vec::new();
        for left in 0..query_len {
            for right in (left + 1)..query_len {
                bigrams.push((
                    TermBitset((1 << left) | (1 << right)),
                    Cost::default(),
                    Score(f32::from(left + right + 1)),
                ));
            }
        }
        let index = Index::new(&[unigrams, bigrams]);

        assert_eq!(
            index.candidates(query_len, Score(0.0)),
            vec![
                TermBitset(0b0001),
                TermBitset(0b0010),
                TermBitset(0b0100),
                TermBitset(0b1000)
            ]
        );

        assert_eq!(
            index.candidates(query_len, Score(1.0)),
            vec![
                TermBitset(0b0001),
                TermBitset(0b0010),
                TermBitset(0b0100),
                TermBitset(0b1000),
            ]
        );

        assert_eq!(
            index.candidates(query_len, Score(2.0)),
            vec![
                TermBitset(0b0001),
                TermBitset(0b0010),
                TermBitset(0b0100),
                TermBitset(0b1000),
                TermBitset(0b0011),
            ]
        );
    }

    #[test]
    fn test_optimize_brute_force() {
        let query_len = 4_u8;
        let unigrams = (0..query_len)
            .map(|term| (TermBitset(1 << term), Cost(1.0), Score(f32::from(term))))
            .collect::<Vec<_>>();
        let mut bigrams: Vec<(TermBitset, Cost, Score)> = Vec::new();
        for left in 0..query_len {
            for right in (left + 1)..query_len {
                bigrams.push((
                    TermBitset((1 << left) | (1 << right)),
                    Cost(0.4),
                    Score(f32::from(left + right + 1)),
                ));
            }
        }
        let index = Index::new(&[unigrams, bigrams]);

        assert_eq!(
            index.optimize(query_len, Score(0.0), BruteForce),
            vec![
                TermBitset(0b0001),
                TermBitset(0b0010),
                TermBitset(0b0100),
                TermBitset(0b1000)
            ]
        );

        assert_eq!(
            index.optimize(query_len, Score(1.0), BruteForce),
            vec![TermBitset(0b0010), TermBitset(0b0100), TermBitset(0b1000)]
        );

        assert_eq!(
            index.optimize(query_len, Score(2.0), BruteForce),
            vec![TermBitset(0b0100), TermBitset(0b1000), TermBitset(0b0011)]
        );

        assert_eq!(
            index.optimize(query_len, Score(3.0), BruteForce),
            vec![TermBitset(0b1000), TermBitset(0b0101), TermBitset(0b0110)]
        );

        assert_eq!(
            index.optimize(query_len, Score(4.0), BruteForce),
            vec![TermBitset(0b1000), TermBitset(0b0110)]
        );

        assert_eq!(
            index.optimize(query_len, Score(5.0), BruteForce),
            vec![TermBitset(0b1010), TermBitset(0b1100)]
        );

        assert_eq!(
            index.optimize(query_len, Score(6.0), BruteForce),
            vec![TermBitset(0b1100)]
        );
    }

    #[test]
    fn test_optimize_smart() {
        let query_len = 4_u8;
        let unigrams = (0..query_len)
            .map(|term| (TermBitset(1 << term), Cost(1.0), Score(f32::from(term))))
            .collect::<Vec<_>>();
        let mut bigrams: Vec<(TermBitset, Cost, Score)> = Vec::new();
        for left in 0..query_len {
            for right in (left + 1)..query_len {
                bigrams.push((
                    TermBitset((1 << left) | (1 << right)),
                    Cost(0.4),
                    Score(f32::from(left + right + 1)),
                ));
            }
        }
        let index = Index::new(&[unigrams, bigrams]);

        assert_eq!(
            index.optimize_smart(query_len, Score(0.0)),
            vec![
                TermBitset(0b0001),
                TermBitset(0b0010),
                TermBitset(0b0100),
                TermBitset(0b1000)
            ]
        );

        assert_eq!(
            index.optimize_smart(query_len, Score(1.0)),
            vec![TermBitset(0b0010), TermBitset(0b0100), TermBitset(0b1000)]
        );

        assert_eq!(
            index.optimize_smart(query_len, Score(2.0)),
            vec![TermBitset(0b0100), TermBitset(0b1000), TermBitset(0b0011)]
        );

        assert_eq!(
            index.optimize_smart(query_len, Score(3.0)),
            vec![TermBitset(0b1000), TermBitset(0b0101), TermBitset(0b0110)]
        );

        assert_eq!(
            index.optimize_smart(query_len, Score(4.0)),
            vec![TermBitset(0b1000), TermBitset(0b0110)]
        );

        assert_eq!(
            index.optimize_smart(query_len, Score(5.0)),
            vec![TermBitset(0b1010), TermBitset(0b1100)]
        );

        assert_eq!(
            index.optimize_smart(query_len, Score(6.0)),
            vec![TermBitset(0b1100)]
        );
    }

    #[test]
    fn test_optimize_graph() {
        let query_len = 4_u8;
        let unigrams = (0..query_len)
            .map(|term| (TermBitset(1 << term), Cost(1.0), Score(f32::from(term))))
            .collect::<Vec<_>>();
        let mut bigrams: Vec<(TermBitset, Cost, Score)> = Vec::new();
        for left in 0..query_len {
            for right in (left + 1)..query_len {
                bigrams.push((
                    TermBitset((1 << left) | (1 << right)),
                    Cost(0.4),
                    Score(f32::from(left + right + 1)),
                ));
            }
        }
        let index = Index::new(&[unigrams, bigrams]);

        assert_eq!(
            index.optimize_graph(query_len, Score(0.0)),
            vec![
                TermBitset(0b0001),
                TermBitset(0b0010),
                TermBitset(0b0100),
                TermBitset(0b1000)
            ]
        );

        assert_eq!(
            index.optimize_graph(query_len, Score(1.0)),
            vec![TermBitset(0b0010), TermBitset(0b0100), TermBitset(0b1000)]
        );

        assert_eq!(
            index.optimize_graph(query_len, Score(2.0)),
            vec![TermBitset(0b0011), TermBitset(0b0100), TermBitset(0b1000)]
        );

        assert_eq!(
            index.optimize_graph(query_len, Score(3.0)),
            vec![TermBitset(0b0101), TermBitset(0b0110), TermBitset(0b1000)]
        );

        assert_eq!(
            index.optimize_graph(query_len, Score(4.0)),
            vec![TermBitset(0b0110), TermBitset(0b1000)]
        );

        assert_eq!(
            index.optimize_graph(query_len, Score(5.0)),
            vec![TermBitset(0b1010), TermBitset(0b1100)]
        );

        assert_eq!(
            index.optimize_graph(query_len, Score(6.0)),
            vec![TermBitset(0b1100)]
        );
    }
}
