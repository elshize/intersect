use crate::graph::Graph;
use crate::power_set::power_set_iter;
use crate::set_cover::{greedy_set_cover, set_cover};
use crate::{Cost, Intersection, ResultClass, Score, MAX_LIST_COUNT};
use failure::{format_err, Error};
use itertools::Itertools;
use num::ToPrimitive;
use std::collections::HashSet;
use std::convert::Into;
use std::fmt;
use std::iter::FromIterator;

#[derive(Debug, PartialEq, Eq)]
struct Candidates {
    all: Vec<Intersection>,
    leaves: Vec<Intersection>,
}

impl Candidates {
    fn cardinality(&self) -> usize {
        self.leaves.len()
    }

    fn subsets(&self, index: &Index) -> Vec<(usize, Vec<usize>, f32)> {
        self.all
            .iter()
            .enumerate()
            .filter_map(|(set_idx, &candidate)| {
                let set: Vec<usize> = self
                    .leaves
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, &leaf)| {
                        if candidate.covers(ResultClass(leaf.0)) {
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
    remove: Vec<Intersection>,
    insert: Vec<Intersection>,
    gain: Cost,
}

/// Represents a full set of available posting lists for a given query.
pub struct Index {
    /// All available posting lists partitioned by number of terms.
    /// Each element contains a vector of intersections of the same degree,
    /// and the shorter intersections always come before the longer ones.
    pub degrees: Vec<Vec<Intersection>>,
    /// An array of posting list costs. The cost of `t` is in `costs[t]`.
    pub costs: [Cost; MAX_LIST_COUNT],
    /// An array of posting list upper bounds. The bound of `t` is in `upper_bound[t]`.
    pub upper_bounds: [Score; MAX_LIST_COUNT],
    /// Number of terms in the query.
    pub query_len: u8,
}

pub struct Pretty<'a> {
    index: &'a Index,
}

impl<'a> fmt::Display for Pretty<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let index_fmt = self
            .index
            .degrees
            .iter()
            .format_with("\n", |level, callback| {
                let level_fmt = level.iter().format_with(" ", |inter, callback| {
                    let cost: Cost = self.index.costs[inter.0 as usize];
                    let upper_bound: Score = self.index.upper_bounds[inter.0 as usize];
                    let terms = inter.terms().map(|t| t.to_string()).join("");
                    callback(&format_args!("[{}|{},{}]", terms, cost.0, upper_bound.0))
                });
                callback(&level_fmt)
            });
        write!(f, "{}", index_fmt)
    }
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
    pub fn new(
        query_len: u8,
        posting_lists: &[Vec<(Intersection, Cost, Score)>],
    ) -> Result<Self, Error> {
        let max_class = 2_u16
            .pow(u32::from(query_len))
            .to_u8()
            .ok_or_else(|| format_err!("Query too long"))?;
        let mut costs = [Cost::default(); MAX_LIST_COUNT];
        let mut upper_bounds = [Score::default(); MAX_LIST_COUNT];
        let mut degrees: Vec<Vec<Intersection>> = Vec::new();
        for arity in posting_lists {
            let mut intersections: Vec<Intersection> = Vec::with_capacity(arity.len());
            for &(intersection, cost, upper_bound) in arity {
                let idx: usize = intersection.into();
                costs[idx] = cost;
                upper_bounds[idx] = upper_bound;
                intersections.push(intersection);
            }
            degrees.push(intersections);
        }
        for class in (0..max_class).map(ResultClass) {
            // NOTE: This is clearly not as good as we can get because we always
            // use single terms to infer missing scores. However, using the best
            // of our knowledge might turn out costly here (to be determined).
            if upper_bounds[class.0 as usize] == Score(0.0) {
                let upper_bound = class
                    .components()
                    .map(|ResultClass(c)| upper_bounds[c as usize])
                    .sum::<Score>();
                upper_bounds[class.0 as usize] = upper_bound;
            }
        }
        Ok(Self {
            degrees,
            costs,
            upper_bounds,
            query_len,
        })
    }

    fn cost(&self, terms: Intersection) -> Cost {
        let idx: usize = terms.into();
        self.costs[idx]
    }

    /// Select optimal set of posting lists for query execution.
    pub fn optimize(
        &self,
        query_len: u8,
        threshold: Score,
        method: OptimizeMethod,
    ) -> Vec<Intersection> {
        match method {
            OptimizeMethod::BruteForce => self.optimize_brute_force(query_len, threshold),
            OptimizeMethod::Exact => self.optimize_smart(query_len, threshold),
            OptimizeMethod::Greedy => self.optimize_greedy(query_len, threshold),
            OptimizeMethod::Graph => self.optimize_graph(query_len, threshold),
        }
    }

    fn optimize_brute_force(&self, query_len: u8, threshold: Score) -> Vec<Intersection> {
        let classes = ResultClass::all_to_vec(query_len);
        let must_cover = std::iter::once(false)
            .chain(
                classes[1..]
                    .iter()
                    .map(|&ResultClass(mask)| self.upper_bounds[mask as usize] >= threshold),
            )
            .collect::<Vec<_>>();
        let Candidates {
            all: candidates, ..
        } = self.candidates(query_len, threshold);
        let mut min_cost = Cost(std::f32::MAX);
        let mut min_subset: Vec<Intersection> = vec![];
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
                .map(|&Intersection(mask)| self.costs[mask as usize])
                .sum::<Cost>();
            if cost < min_cost {
                min_cost = cost;
                min_subset = candidate_subset;
            }
        }
        min_subset
    }

    fn optimize_smart(&self, query_len: u8, threshold: Score) -> Vec<Intersection> {
        let candidates = self.candidates(query_len, threshold);
        let solution = candidates.solve(&self);
        solution
            .into_iter()
            .map(|set| candidates.all[set])
            .collect()
    }

    fn optimize_greedy(&self, query_len: u8, threshold: Score) -> Vec<Intersection> {
        let candidates = self.candidates(query_len, threshold);
        let solution = candidates.solve_greedy(&self);
        solution
            .into_iter()
            .map(|set| candidates.all[set])
            .collect()
    }

    fn optimize_graph(&self, query_len: u8, threshold: Score) -> Vec<Intersection> {
        let Candidates {
            all: candidates,
            leaves,
        } = self.candidates(query_len, threshold);
        let graph = Graph::from_iter(candidates.into_iter());
        let mut solution: HashSet<Intersection> = leaves.into_iter().collect();
        for layer in graph.layers().rev() {
            let switch_candidates: Vec<Intersection> = layer
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
                .collect();
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
    fn candidates(&self, query_len: u8, threshold: Score) -> Candidates {
        let mut cand: Vec<Vec<Intersection>> = Vec::with_capacity(self.degrees.len());
        let mut leaves: Vec<Intersection> = Vec::with_capacity(query_len.pow(2) as usize);
        let classes = ResultClass::all_to_vec(query_len);
        let mut covered = vec![0_u8; 2_usize.pow(u32::from(query_len))];
        for arity in &self.degrees {
            let mut arity_cand: Vec<Intersection> = Vec::with_capacity(arity.len());
            for &intersection in arity {
                let Intersection(mask) = intersection;
                if covered[mask as usize] == 0_u8 {
                    arity_cand.push(intersection);
                    if self.upper_bounds[mask as usize] >= threshold {
                        leaves.push(intersection);
                        intersection.cover(&classes, &mut covered);
                    } else {
                        covered[mask as usize] = 1_u8;
                    }
                }
            }
            cand.push(arity_cand);
        }
        Candidates {
            all: cand.into_iter().flatten().collect(),
            leaves,
        }
    }

    /// Returns an object implementing
    /// [`Display`] : https://doc.rust-lang.org/std/fmt/trait.Display.html
    /// trait, pretty-printing the index.
    pub fn pretty(&self) -> Pretty {
        Pretty { index: self }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rstest::{fixture, rstest};
    use OptimizeMethod::{BruteForce, Exact, Graph, Greedy};

    #[test]
    fn test_new_index() {
        let index = Index::new(0, &[]).unwrap();
        assert!(index.degrees.is_empty());
        let index = Index::new(
            3,
            &[
                vec![
                    (Intersection(0b001), Cost(0.1), Score(3.0)),
                    (Intersection(0b010), Cost(4.1), Score(2.0)),
                    (Intersection(0b100), Cost(1.1), Score(5.6)),
                ],
                vec![(Intersection(0b011), Cost(4.1), Score(4.0))],
            ],
        )
        .unwrap();
        assert_eq!(
            index.degrees,
            vec![
                vec![
                    Intersection(0b001),
                    Intersection(0b010),
                    Intersection(0b100),
                ],
                vec![Intersection(0b011)],
            ]
        );
        let mut expected_costs = [Cost::default(); MAX_LIST_COUNT];
        expected_costs[0b001] = Cost(0.1);
        expected_costs[0b010] = Cost(4.1);
        expected_costs[0b100] = Cost(1.1);
        expected_costs[0b011] = Cost(4.1);
        assert_eq!(index.costs.to_vec(), expected_costs.to_vec());
        let mut expected_scores = [Score::default(); MAX_LIST_COUNT];
        expected_scores[0b001] = Score(3.0);
        expected_scores[0b010] = Score(2.0);
        expected_scores[0b100] = Score(5.6);
        expected_scores[0b011] = Score(4.0);
        expected_scores[0b101] = Score(8.6);
        expected_scores[0b110] = Score(7.6);
        expected_scores[0b111] = Score(10.6);
        assert_eq!(index.upper_bounds.to_vec(), expected_scores.to_vec());
    }

    #[fixture]
    fn full_bigram_index() -> Index {
        let query_len = 4_u8;
        let unigrams = (0..query_len)
            .map(|term| {
                (
                    Intersection(1 << term),
                    Cost::default(),
                    Score(f32::from(term)),
                )
            })
            .collect::<Vec<_>>();
        let mut bigrams: Vec<(Intersection, Cost, Score)> = Vec::new();
        for left in 0..query_len {
            for right in (left + 1)..query_len {
                bigrams.push((
                    Intersection((1 << left) | (1 << right)),
                    Cost::default(),
                    Score(f32::from(left + right + 1)),
                ));
            }
        }
        Index::new(4, &[unigrams, bigrams]).unwrap()
    }

    #[rstest]
    #[allow(clippy::needless_pass_by_value)]
    fn test_format_index(full_bigram_index: Index) {
        let formatted = full_bigram_index.pretty().to_string();
        assert_eq!(
            formatted,
            "[0|0,0] [1|0,1] [2|0,2] [3|0,3]
[01|0,2] [02|0,3] [03|0,4] [12|0,4] [13|0,5] [23|0,6]"
        );
    }

    #[rstest]
    fn test_candidates(full_bigram_index: Index) {
        let index = full_bigram_index;
        let query_len = index.query_len;

        println!("{}", index.pretty());

        assert_eq!(
            index.candidates(query_len, Score(0.0)),
            Candidates {
                all: vec![
                    Intersection(0b0001),
                    Intersection(0b0010),
                    Intersection(0b0100),
                    Intersection(0b1000)
                ],
                leaves: vec![
                    Intersection(0b0001),
                    Intersection(0b0010),
                    Intersection(0b0100),
                    Intersection(0b1000)
                ]
            }
        );

        assert_eq!(
            index.candidates(query_len, Score(1.0)),
            Candidates {
                all: vec![
                    Intersection(0b0001),
                    Intersection(0b0010),
                    Intersection(0b0100),
                    Intersection(0b1000),
                ],
                leaves: vec![
                    Intersection(0b0010),
                    Intersection(0b0100),
                    Intersection(0b1000),
                ]
            }
        );

        assert_eq!(
            index.candidates(query_len, Score(2.0)),
            Candidates {
                all: vec![
                    Intersection(0b0001),
                    Intersection(0b0010),
                    Intersection(0b0100),
                    Intersection(0b1000),
                    Intersection(0b0011),
                ],
                leaves: vec![
                    Intersection(0b0100),
                    Intersection(0b1000),
                    Intersection(0b0011),
                ]
            }
        );
    }

    #[test]
    fn test_optimize_brute_force() {
        let query_len = 4_u8;
        let unigrams = (0..query_len)
            .map(|term| (Intersection(1 << term), Cost(1.0), Score(f32::from(term))))
            .collect::<Vec<_>>();
        let mut bigrams: Vec<(Intersection, Cost, Score)> = Vec::new();
        for left in 0..query_len {
            for right in (left + 1)..query_len {
                bigrams.push((
                    Intersection((1 << left) | (1 << right)),
                    Cost(0.4),
                    Score(f32::from(left + right + 1)),
                ));
            }
        }
        let index = Index::new(4, &[unigrams, bigrams]).unwrap();

        assert_eq!(
            index.optimize(query_len, Score(0.0), BruteForce),
            vec![
                Intersection(0b0001),
                Intersection(0b0010),
                Intersection(0b0100),
                Intersection(0b1000)
            ]
        );

        assert_eq!(
            index.optimize(query_len, Score(1.0), BruteForce),
            vec![
                Intersection(0b0010),
                Intersection(0b0100),
                Intersection(0b1000)
            ]
        );

        assert_eq!(
            index.optimize(query_len, Score(2.0), BruteForce),
            vec![
                Intersection(0b0100),
                Intersection(0b1000),
                Intersection(0b0011)
            ]
        );

        assert_eq!(
            index.optimize(query_len, Score(3.0), BruteForce),
            vec![
                Intersection(0b1000),
                Intersection(0b0101),
                Intersection(0b0110)
            ]
        );

        assert_eq!(
            index.optimize(query_len, Score(4.0), BruteForce),
            vec![Intersection(0b1000), Intersection(0b0110)]
        );

        assert_eq!(
            index.optimize(query_len, Score(5.0), BruteForce),
            vec![Intersection(0b1010), Intersection(0b1100)]
        );

        assert_eq!(
            index.optimize(query_len, Score(6.0), BruteForce),
            vec![Intersection(0b1100)]
        );
    }

    #[test]
    fn test_optimize_exact() {
        let query_len = 4_u8;
        let unigrams = (0..query_len)
            .map(|term| (Intersection(1 << term), Cost(1.0), Score(f32::from(term))))
            .collect::<Vec<_>>();
        let mut bigrams: Vec<(Intersection, Cost, Score)> = Vec::new();
        for left in 0..query_len {
            for right in (left + 1)..query_len {
                bigrams.push((
                    Intersection((1 << left) | (1 << right)),
                    Cost(0.4),
                    Score(f32::from(left + right + 1)),
                ));
            }
        }
        let index = Index::new(0, &[unigrams, bigrams]).unwrap();

        assert_eq!(
            index.optimize(query_len, Score(0.0), Exact),
            vec![
                Intersection(0b0001),
                Intersection(0b0010),
                Intersection(0b0100),
                Intersection(0b1000)
            ]
        );

        assert_eq!(
            index.optimize(query_len, Score(1.0), Exact),
            vec![
                Intersection(0b0010),
                Intersection(0b0100),
                Intersection(0b1000)
            ]
        );

        assert_eq!(
            index.optimize(query_len, Score(2.0), Exact),
            vec![
                Intersection(0b0100),
                Intersection(0b1000),
                Intersection(0b0011)
            ]
        );

        assert_eq!(
            index.optimize(query_len, Score(3.0), Exact),
            vec![
                Intersection(0b1000),
                Intersection(0b0101),
                Intersection(0b0110)
            ]
        );

        assert_eq!(
            index.optimize(query_len, Score(4.0), Exact),
            vec![Intersection(0b1000), Intersection(0b0110)]
        );

        assert_eq!(
            index.optimize(query_len, Score(5.0), Exact),
            vec![Intersection(0b1010), Intersection(0b1100)]
        );

        assert_eq!(
            index.optimize(query_len, Score(6.0), Exact),
            vec![Intersection(0b1100)]
        );
    }

    #[test]
    fn test_optimize_greedy() {
        let query_len = 4_u8;
        let unigrams = (0..query_len)
            .map(|term| (Intersection(1 << term), Cost(1.0), Score(f32::from(term))))
            .collect::<Vec<_>>();
        let mut bigrams: Vec<(Intersection, Cost, Score)> = Vec::new();
        for left in 0..query_len {
            for right in (left + 1)..query_len {
                bigrams.push((
                    Intersection((1 << left) | (1 << right)),
                    Cost(0.4),
                    Score(f32::from(left + right + 1)),
                ));
            }
        }
        let index = Index::new(0, &[unigrams, bigrams]).unwrap();

        assert_eq!(
            index
                .optimize(query_len, Score(0.0), Greedy)
                .into_iter()
                .collect::<HashSet<Intersection>>(),
            vec![
                Intersection(0b0001),
                Intersection(0b0010),
                Intersection(0b0100),
                Intersection(0b1000)
            ]
            .into_iter()
            .collect::<HashSet<Intersection>>(),
        );

        assert_eq!(
            index
                .optimize(query_len, Score(1.0), Greedy)
                .into_iter()
                .collect::<HashSet<Intersection>>(),
            vec![
                Intersection(0b0010),
                Intersection(0b0100),
                Intersection(0b1000)
            ]
            .into_iter()
            .collect::<HashSet<Intersection>>(),
        );

        assert_eq!(
            index
                .optimize(query_len, Score(2.0), Greedy)
                .into_iter()
                .collect::<HashSet<Intersection>>(),
            vec![
                Intersection(0b0011),
                Intersection(0b0100),
                Intersection(0b1000)
            ]
            .into_iter()
            .collect::<HashSet<Intersection>>(),
        );

        assert_eq!(
            index
                .optimize(query_len, Score(3.0), Greedy)
                .into_iter()
                .collect::<HashSet<Intersection>>(),
            vec![
                Intersection(0b0101),
                Intersection(0b0110),
                Intersection(0b1000)
            ]
            .into_iter()
            .collect::<HashSet<Intersection>>(),
        );

        assert_eq!(
            index
                .optimize(query_len, Score(4.0), Greedy)
                .into_iter()
                .collect::<HashSet<Intersection>>(),
            vec![Intersection(0b0110), Intersection(0b1000)]
                .into_iter()
                .collect::<HashSet<Intersection>>(),
        );

        assert_eq!(
            index
                .optimize(query_len, Score(5.0), Greedy)
                .into_iter()
                .collect::<HashSet<Intersection>>(),
            vec![Intersection(0b1010), Intersection(0b1100)]
                .into_iter()
                .collect::<HashSet<Intersection>>(),
        );

        assert_eq!(
            index
                .optimize(query_len, Score(6.0), Greedy)
                .into_iter()
                .collect::<HashSet<Intersection>>(),
            vec![Intersection(0b1100)]
                .into_iter()
                .collect::<HashSet<Intersection>>(),
        );
    }

    #[test]
    fn test_optimize_graph() {
        let query_len = 4_u8;
        let unigrams = (0..query_len)
            .map(|term| (Intersection(1 << term), Cost(1.0), Score(f32::from(term))))
            .collect::<Vec<_>>();
        let mut bigrams: Vec<(Intersection, Cost, Score)> = Vec::new();
        for left in 0..query_len {
            for right in (left + 1)..query_len {
                bigrams.push((
                    Intersection((1 << left) | (1 << right)),
                    Cost(0.4),
                    Score(f32::from(left + right + 1)),
                ));
            }
        }
        let index = Index::new(0, &[unigrams, bigrams]).unwrap();

        assert_eq!(
            index.optimize(query_len, Score(0.0), Graph),
            vec![
                Intersection(0b0001),
                Intersection(0b0010),
                Intersection(0b0100),
                Intersection(0b1000)
            ]
        );

        assert_eq!(
            index.optimize(query_len, Score(1.0), Graph),
            vec![
                Intersection(0b0010),
                Intersection(0b0100),
                Intersection(0b1000)
            ]
        );

        assert_eq!(
            index.optimize(query_len, Score(2.0), Graph),
            vec![
                Intersection(0b0011),
                Intersection(0b0100),
                Intersection(0b1000)
            ]
        );

        assert_eq!(
            index.optimize(query_len, Score(3.0), Graph),
            vec![
                Intersection(0b0101),
                Intersection(0b0110),
                Intersection(0b1000)
            ]
        );

        assert_eq!(
            index.optimize(query_len, Score(4.0), Graph),
            vec![Intersection(0b0110), Intersection(0b1000)]
        );

        assert_eq!(
            index.optimize(query_len, Score(5.0), Graph),
            vec![Intersection(0b1010), Intersection(0b1100)]
        );

        assert_eq!(
            index.optimize(query_len, Score(6.0), Graph),
            vec![Intersection(0b1100)]
        );
    }
}
