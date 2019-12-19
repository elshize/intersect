use crate::graph::Graph;
use crate::power_set::power_set_iter;
use crate::set_cover::{greedy_set_cover, set_cover};
use crate::{Cost, Intersection, ResultClass, Score, TermMask, MAX_LIST_COUNT};
use failure::{bail, format_err, Error};
use itertools::Itertools;
use num::ToPrimitive;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::convert::{Into, TryInto};
use std::iter::FromIterator;
use std::{fmt, str::FromStr};

#[derive(Debug, PartialEq, Eq)]
struct Candidates {
    all: Vec<Intersection>,
    leaves: Vec<Intersection>,
    uncovered_classes: Vec<ResultClass>,
}

impl Candidates {
    fn cardinality(&self) -> usize {
        self.leaves.len() + self.uncovered_classes.len()
    }

    fn subsets(&self, index: &Index) -> Vec<(usize, Vec<usize>, f32)> {
        self.all
            .iter()
            .enumerate()
            .filter_map(|(set_idx, &candidate)| {
                let set: Vec<usize> = self
                    .leaves
                    .iter()
                    .map(|&Intersection(leaf)| leaf)
                    .chain(
                        self.uncovered_classes
                            .iter()
                            .map(|&ResultClass(class)| class),
                    )
                    .enumerate()
                    .filter_map(|(idx, elem)| {
                        if candidate.covers(ResultClass(elem)) {
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    /// Similar to graph, but faster by greedily selecting parents with smallest costs.
    /// (Is it equivalent to `Graph`?)
    GraphGreedy,
}

impl FromStr for OptimizeMethod {
    type Err = Error;

    fn from_str(method: &str) -> Result<Self, Self::Err> {
        match method {
            "brute-force" => Ok(Self::BruteForce),
            "exact" => Ok(Self::Exact),
            "greedy" => Ok(Self::Greedy),
            "graph" => Ok(Self::Graph),
            "graph-greedy" => Ok(Self::GraphGreedy),
            method => bail!("Invalid optimize method: {}", method),
        }
    }
}

impl Index {
    /// Construct a new index containing the given posting lists.
    pub fn new(
        query_len: u8,
        posting_lists: &[Vec<(Intersection, Cost, Score)>],
    ) -> Result<Self, Error> {
        let max_class: TermMask = num::cast::NumCast::from(2_u16.pow(u32::from(query_len)))
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

    /// Scale costs depending on arity.
    pub fn scale_costs(&mut self, factor: f32) {
        for (mask, cost) in self.costs.iter_mut().enumerate() {
            let arity = mask.count_ones();
            *cost = Cost(cost.0 * factor.powf(arity as f32));
        }
    }

    /// Cost of a given intersection.
    pub fn cost(&self, intersection: Intersection) -> Cost {
        let idx: usize = intersection.into();
        self.costs[idx]
    }

    /// Select optimal set of posting lists for query execution.
    pub fn optimize(&mut self, threshold: Score, method: OptimizeMethod) -> Vec<Intersection> {
        match method {
            OptimizeMethod::BruteForce => self.optimize_brute_force(threshold),
            OptimizeMethod::Exact => self.optimize_exact(threshold),
            OptimizeMethod::Greedy => self.optimize_greedy(threshold),
            OptimizeMethod::Graph => self.optimize_graph(threshold),
            OptimizeMethod::GraphGreedy => self.optimize_graph_greedy(threshold),
        }
    }

    fn optimize_brute_force(&self, threshold: Score) -> Vec<Intersection> {
        let classes = ResultClass::all_to_vec(self.query_len);
        let must_cover = std::iter::once(false)
            .chain(
                classes[1..]
                    .iter()
                    .map(|&ResultClass(mask)| self.upper_bounds[mask as usize] >= threshold),
            )
            .collect::<Vec<_>>();
        let Candidates {
            all: candidates, ..
        } = self.candidates(threshold);
        let mut min_cost = Cost(std::f32::MAX);
        let mut min_subset: Vec<Intersection> = vec![];
        for candidate_subset in power_set_iter(&candidates) {
            let candidate_subset: Vec<_> = candidate_subset.cloned().collect();
            let mut covered = vec![0_u8; 2_usize.pow(u32::from(self.query_len))];
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

    fn optimize_exact(&self, threshold: Score) -> Vec<Intersection> {
        let candidates = self.candidates(threshold);
        let solution = candidates.solve(&self);
        solution
            .into_iter()
            .map(|set| candidates.all[set])
            .collect()
    }

    fn optimize_greedy(&self, threshold: Score) -> Vec<Intersection> {
        let candidates = self.candidates(threshold);
        let solution = candidates.solve_greedy(&self);
        solution
            .into_iter()
            .map(|set| candidates.all[set])
            .collect()
    }

    fn optimize_graph(&mut self, threshold: Score) -> Vec<Intersection> {
        let Candidates {
            all: candidates,
            leaves,
            uncovered_classes,
        } = self.candidates(threshold);

        // These must be covered but cannot be selected.
        let phony_candidates: HashSet<_> = uncovered_classes
            .into_iter()
            .map(|ResultClass(class)| Intersection(class))
            .collect();

        let graph = Graph::from_iter(
            candidates
                .into_iter()
                .chain(phony_candidates.iter().cloned()),
        );
        let mut solution: HashSet<Intersection> = leaves
            .into_iter()
            .chain(phony_candidates.iter().cloned())
            .collect();
        let &max_cost = self.costs.iter().max().unwrap();
        for Intersection(phony) in phony_candidates {
            self.costs[phony as usize] = Cost(max_cost.0 * phony.count_ones().to_f32().unwrap());
        }
        for degree in graph.degrees().rev() {
            let switch_candidates: Vec<Intersection> = graph
                .layer(degree)
                .filter_map(|node| {
                    if solution.contains(&node) {
                        graph.parents(node)
                    //.map(|some| some.iter().filter(|p| parent_layer.contains(p)))
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

    fn optimize_graph_greedy(&mut self, threshold: Score) -> Vec<Intersection> {
        let Candidates {
            all: candidates,
            leaves,
            uncovered_classes,
        } = self.candidates(threshold);

        // These must be covered but cannot be selected.
        let phony_candidates: HashSet<_> = uncovered_classes
            .into_iter()
            .map(|ResultClass(class)| Intersection(class))
            .collect();

        let graph = Graph::from_iter(
            candidates
                .into_iter()
                .chain(phony_candidates.iter().cloned()),
        );
        let mut solution: HashSet<Intersection> = leaves
            .into_iter()
            .chain(phony_candidates.iter().cloned())
            .collect();
        let &max_cost = self.costs.iter().max().unwrap();
        for Intersection(phony) in phony_candidates {
            self.costs[phony as usize] =
                Cost(max_cost.0 * (1.0 + phony.count_ones().to_f32().unwrap()));
        }
        for degree in graph.degrees().rev() {
            let mut remaining: HashSet<_> = graph
                .layer(degree)
                .filter(|c| solution.contains(c))
                .collect();
            if remaining.is_empty() {
                continue;
            }
            let mut remaining_parents: HashSet<_> = graph.layer(degree - 1).collect();

            let parents: Vec<_> = graph
                .layer(degree - 1)
                .sorted_by(|&Intersection(a), &Intersection(b)| {
                    self.costs[a as usize].cmp(&self.costs[b as usize])
                })
                .collect();
            for parent in parents {
                if remaining_parents.contains(&parent) {
                    if let Some(children) = graph.children(parent) {
                        let cost_removed = children
                            .iter()
                            .filter_map(|c| remaining.get(c).map(|&c| self.cost(c)))
                            .sum::<Cost>();
                        let cost_gain = cost_removed - self.cost(parent);
                        if cost_gain > Cost(0.0) {
                            solution.insert(parent);
                            remaining_parents.remove(&parent);
                            for child in children {
                                solution.remove(child);
                                remaining.remove(&child);
                            }
                            if remaining.is_empty() || remaining_parents.is_empty() {
                                break;
                            }
                        }
                    }
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
    fn candidates(&self, threshold: Score) -> Candidates {
        let mut cand: Vec<Vec<Intersection>> = Vec::with_capacity(self.degrees.len());
        let mut leaves: Vec<Intersection> = Vec::with_capacity(self.query_len.pow(2) as usize);
        let classes = ResultClass::all_to_vec(self.query_len);
        let mut covered = vec![0_u8; 2_usize.pow(u32::from(self.query_len))];
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
            uncovered_classes: covered
                .into_iter()
                .enumerate()
                .skip(1)
                .filter_map(|(mask, is_covered)| {
                    if is_covered == 0_u8 && self.upper_bounds[mask] >= threshold {
                        Some(ResultClass(mask.try_into().unwrap()))
                    } else {
                        None
                    }
                })
                .collect(),
        }
    }

    /// Returns an object implementing
    /// [`Display`](https://doc.rust-lang.org/std/fmt/trait.Display.html)
    /// trait, pretty-printing the index.
    pub fn pretty(&self) -> Pretty {
        Pretty { index: self }
    }
}

impl FromIterator<(Intersection, Cost, Score)> for Index {
    fn from_iter<T: IntoIterator<Item = (Intersection, Cost, Score)>>(iter: T) -> Self {
        let mut query_len = 0_u8;
        let mut posting_lists: Vec<Vec<(Intersection, Cost, Score)>> = Vec::new();
        for (inter, cost, score) in iter {
            let len = inter
                .0
                .trailing_zeros()
                .to_u8()
                .expect("Unable to cast u32 to u8")
                + 1;
            query_len = std::cmp::max(query_len, len);
            let level = inter.0.count_ones() as usize;
            if posting_lists.len() < level {
                posting_lists.resize_with(level, Default::default);
            }
            posting_lists[level.checked_sub(1).expect("Intersection cannot be 0")]
                .push((inter, cost, score));
        }
        Self::new(query_len, &posting_lists).expect("Unable to create index")
    }
}

/// Describes intersection as given at the program input.
#[derive(Serialize, Deserialize)]
pub struct IntersectionInput {
    /// Intersection mask.
    pub intersection: Intersection,
    /// Intersection cost, e.g., number of postings.
    pub cost: Cost,
    /// Maximum score of a document in the intersection.
    pub max_score: Score,
}

impl FromIterator<IntersectionInput> for Index {
    fn from_iter<T: IntoIterator<Item = IntersectionInput>>(iter: T) -> Self {
        let mut query_len = 0_u8;
        let mut posting_lists: Vec<Vec<(Intersection, Cost, Score)>> = Vec::new();
        for IntersectionInput {
            intersection,
            cost,
            max_score,
        } in iter
        {
            let len = intersection
                .0
                .trailing_zeros()
                .to_u8()
                .expect("Unable to cast u32 to u8")
                + 1;
            query_len = std::cmp::max(query_len, len);
            let level = intersection.0.count_ones() as usize;
            if posting_lists.len() < level {
                posting_lists.resize_with(level, Default::default);
            }
            posting_lists[level.checked_sub(1).expect("Intersection cannot be 0")].push((
                intersection,
                cost,
                max_score,
            ));
        }
        Self::new(query_len, &posting_lists).expect("Unable to create index")
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use proptest::prelude::*;
    use rstest::{fixture, rstest};
    use OptimizeMethod::{BruteForce, Exact, Graph, GraphGreedy, Greedy};

    //#[test]
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

    #[derive(Clone, Copy)]
    enum FlracMode {
        Bigrams,
        Trigrams,
    }

    fn french_lick_resort_and_casino(mode: FlracMode) -> Index {
        let i = |n, c, s| (Intersection(n), Cost(c), Score(s));
        let unigrams = vec![
            i(0b00001, 3_207_698.0, 5.09),
            i(0b00010, 217_056.0, 10.3128),
            i(0b00100, 1_402_926.0, 6.72921),
            i(0b01000, 44_575_252.0, 1.89675e-06),
            i(0b10000, 802_140.0, 7.81291),
        ];
        let bigrams = vec![
            i(0b00011, 27_634.0, 15.3662),
            i(0b00101, 165_409.0, 11.655),
            i(0b01001, 3_023_926.0, 5.0903),
            i(0b10001, 77_753.0, 12.7071),
            i(0b00110, 12_303.0, 16.8082),
            i(0b01010, 209_498.0, 10.3128),
            i(0b10010, 6_900.0, 17.8669),
            i(0b01100, 1_351_378.0, 6.72921),
            i(0b10100, 147_880.0, 14.5006),
            i(0b11000, 729_376.0, 7.81292),
        ];
        match mode {
            FlracMode::Bigrams => Index::new(5, &[unigrams, bigrams]).unwrap(),
            FlracMode::Trigrams => Index::new(
                5,
                &[
                    unigrams,
                    bigrams,
                    vec![
                        i(0b00111, 3_503.0, 21.8352),
                        i(0b01011, 27_004.0, 15.3662),
                        i(0b10011, 2348.0, 22.8938),
                        i(0b01101, 164_017.0, 11.655),
                        i(0b10101, 21_485.0, 19.3285),
                        i(0b11001, 75_136.0, 12.7071),
                        i(0b01110, 12_188.0, 16.8082),
                        i(0b10110, 2_367.0, 24.4883),
                        i(0b11010, 6_751.0, 17.8669),
                        i(0b11100, 142_814.0, 14.5006),
                    ],
                ],
            )
            .unwrap(),
        }
    }

    #[test]
    fn test_flrac_bigrams_candidates() {
        let index = french_lick_resort_and_casino(FlracMode::Bigrams);
        let mut candidates = index.candidates(Score(16.4));
        let mut expected: Vec<_> = index.degrees.iter().flatten().cloned().collect();
        expected.sort();
        candidates.all.sort();
        assert_eq!(candidates.all, expected);
        assert_eq!(
            candidates.uncovered_classes,
            // ACE + ACDE are not covered by the leaves.
            vec![ResultClass(21), ResultClass(29)]
        );
    }

    #[test]
    fn test_flrac_trigrams_candidates() {
        let index = french_lick_resort_and_casino(FlracMode::Trigrams);
        let mut candidates = index.candidates(Score(16.4));
        let mut expected: Vec<_> = index.degrees.iter().take(2).flatten().cloned().collect();
        expected.push(Intersection(11));
        expected.push(Intersection(13));
        expected.push(Intersection(21));
        expected.push(Intersection(25));
        expected.push(Intersection(28));
        expected.sort();
        candidates.all.sort();
        assert_eq!(candidates.all, expected);
        assert_eq!(
            candidates.leaves,
            vec![Intersection(6), Intersection(18), Intersection(21)]
        );
        assert!(candidates.uncovered_classes.is_empty());
    }

    #[test]
    fn test_flrac_bigrams_solve() {
        let mut index = french_lick_resort_and_casino(FlracMode::Bigrams);

        let threshold = Score(16.4);
        let mut bf_opt = index.optimize(threshold, BruteForce);
        let mut exact_opt = index.optimize(threshold, Exact);
        let mut greedy_opt = index.optimize(threshold, Greedy);
        let mut graph_opt = index.optimize(threshold, Graph);

        bf_opt.sort();
        exact_opt.sort();
        greedy_opt.sort();
        graph_opt.sort();

        let expected = vec![Intersection(6), Intersection(17), Intersection(18)];
        assert_eq!(bf_opt, expected);
        assert_eq!(exact_opt, expected);
        assert_eq!(greedy_opt, expected);
        assert_eq!(graph_opt, expected);
    }

    #[test]
    fn test_flrac_trigrams_solve() {
        let mut index = french_lick_resort_and_casino(FlracMode::Trigrams);

        let threshold = Score(16.4);
        let mut exact_opt = index.optimize(threshold, Exact);
        let mut greedy_opt = index.optimize(threshold, Greedy);
        let mut graph_opt = index.optimize(threshold, Graph);

        exact_opt.sort();
        greedy_opt.sort();
        graph_opt.sort();

        let expected = vec![Intersection(6), Intersection(18), Intersection(21)];
        assert_eq!(exact_opt, expected);
        assert_eq!(greedy_opt, expected);
        assert_eq!(graph_opt, expected);
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

        assert_eq!(
            index.candidates(Score(0.0)),
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
                ],
                uncovered_classes: vec![]
            }
        );

        assert_eq!(
            index.candidates(Score(1.0)),
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
                ],
                uncovered_classes: vec![]
            }
        );

        assert_eq!(
            index.candidates(Score(2.0)),
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
                ],
                uncovered_classes: vec![]
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
        let mut index = Index::new(4, &[unigrams, bigrams]).unwrap();

        assert_eq!(
            index.optimize(Score(0.0), BruteForce),
            vec![
                Intersection(0b0001),
                Intersection(0b0010),
                Intersection(0b0100),
                Intersection(0b1000)
            ]
        );

        assert_eq!(
            index.optimize(Score(1.0), BruteForce),
            vec![
                Intersection(0b0010),
                Intersection(0b0100),
                Intersection(0b1000)
            ]
        );

        assert_eq!(
            index.optimize(Score(2.0), BruteForce),
            vec![
                Intersection(0b0100),
                Intersection(0b1000),
                Intersection(0b0011)
            ]
        );

        assert_eq!(
            index.optimize(Score(3.0), BruteForce),
            vec![
                Intersection(0b1000),
                Intersection(0b0101),
                Intersection(0b0110)
            ]
        );

        assert_eq!(
            index.optimize(Score(4.0), BruteForce),
            vec![Intersection(0b1000), Intersection(0b0110)]
        );

        assert_eq!(
            index.optimize(Score(5.0), BruteForce),
            vec![Intersection(0b1010), Intersection(0b1100)]
        );

        assert_eq!(
            index.optimize(Score(6.0), BruteForce),
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
        let mut index = Index::new(4, &[unigrams, bigrams]).unwrap();

        assert_eq!(
            index.optimize(Score(0.0), Exact),
            vec![
                Intersection(0b0001),
                Intersection(0b0010),
                Intersection(0b0100),
                Intersection(0b1000)
            ]
        );

        assert_eq!(
            index.optimize(Score(1.0), Exact),
            vec![
                Intersection(0b0010),
                Intersection(0b0100),
                Intersection(0b1000)
            ]
        );

        assert_eq!(
            index.optimize(Score(2.0), Exact),
            vec![
                Intersection(0b0100),
                Intersection(0b1000),
                Intersection(0b0011)
            ]
        );

        assert_eq!(
            index.optimize(Score(3.0), Exact),
            vec![
                Intersection(0b1000),
                Intersection(0b0101),
                Intersection(0b0110)
            ]
        );

        assert_eq!(
            index.optimize(Score(4.0), Exact),
            vec![Intersection(0b1000), Intersection(0b0110)]
        );

        assert_eq!(
            index.optimize(Score(5.0), Exact),
            vec![Intersection(0b1010), Intersection(0b1100)]
        );

        assert_eq!(
            index.optimize(Score(6.0), Exact),
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
        let mut index = Index::new(4, &[unigrams, bigrams]).unwrap();

        assert_eq!(
            index
                .optimize(Score(0.0), Greedy)
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
                .optimize(Score(1.0), Greedy)
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
                .optimize(Score(2.0), Greedy)
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
                .optimize(Score(3.0), Greedy)
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
                .optimize(Score(4.0), Greedy)
                .into_iter()
                .collect::<HashSet<Intersection>>(),
            vec![Intersection(0b0110), Intersection(0b1000)]
                .into_iter()
                .collect::<HashSet<Intersection>>(),
        );

        assert_eq!(
            index
                .optimize(Score(5.0), Greedy)
                .into_iter()
                .collect::<HashSet<Intersection>>(),
            vec![Intersection(0b1010), Intersection(0b1100)]
                .into_iter()
                .collect::<HashSet<Intersection>>(),
        );

        assert_eq!(
            index
                .optimize(Score(6.0), Greedy)
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
        let mut index = Index::new(4, &[unigrams, bigrams]).unwrap();

        assert_eq!(
            index.optimize(Score(0.0), Graph),
            vec![
                Intersection(0b0001),
                Intersection(0b0010),
                Intersection(0b0100),
                Intersection(0b1000)
            ]
        );

        assert_eq!(
            index.optimize(Score(1.0), Graph),
            vec![
                Intersection(0b0010),
                Intersection(0b0100),
                Intersection(0b1000)
            ]
        );

        assert_eq!(
            index.optimize(Score(2.0), Graph),
            vec![
                Intersection(0b0011),
                Intersection(0b0100),
                Intersection(0b1000)
            ]
        );

        assert_eq!(
            index.optimize(Score(3.0), Graph),
            vec![
                Intersection(0b0101),
                Intersection(0b0110),
                Intersection(0b1000)
            ]
        );

        assert_eq!(
            index.optimize(Score(4.0), Graph),
            vec![Intersection(0b0110), Intersection(0b1000)]
        );

        assert_eq!(
            index.optimize(Score(5.0), Graph),
            vec![Intersection(0b1010), Intersection(0b1100)]
        );

        assert_eq!(
            index.optimize(Score(6.0), Graph),
            vec![Intersection(0b1100)]
        );
    }

    #[test]
    fn test_optimize_graph_greedy() {
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
        let mut index = Index::new(4, &[unigrams, bigrams]).unwrap();

        assert_eq!(
            index.optimize(Score(0.0), GraphGreedy),
            vec![
                Intersection(0b0001),
                Intersection(0b0010),
                Intersection(0b0100),
                Intersection(0b1000)
            ]
        );

        assert_eq!(
            index.optimize(Score(1.0), GraphGreedy),
            vec![
                Intersection(0b0010),
                Intersection(0b0100),
                Intersection(0b1000)
            ]
        );

        assert_eq!(
            index.optimize(Score(2.0), GraphGreedy),
            vec![
                Intersection(0b0011),
                Intersection(0b0100),
                Intersection(0b1000)
            ]
        );

        assert_eq!(
            index.optimize(Score(3.0), GraphGreedy),
            vec![
                Intersection(0b0101),
                Intersection(0b0110),
                Intersection(0b1000)
            ]
        );

        assert_eq!(
            index.optimize(Score(4.0), GraphGreedy),
            vec![Intersection(0b0110), Intersection(0b1000)]
        );

        assert_eq!(
            index.optimize(Score(5.0), GraphGreedy),
            vec![Intersection(0b1010), Intersection(0b1100)]
        );

        assert_eq!(
            index.optimize(Score(6.0), GraphGreedy),
            vec![Intersection(0b1100)]
        );
    }

    #[test]
    fn test_optimize_method_from_str() {
        assert_eq!(
            "brute-force".parse::<OptimizeMethod>().ok(),
            Some(BruteForce)
        );
        assert_eq!("exact".parse::<OptimizeMethod>().ok(), Some(Exact));
        assert_eq!("greedy".parse::<OptimizeMethod>().ok(), Some(Greedy));
        assert_eq!("graph".parse::<OptimizeMethod>().ok(), Some(Graph));
        assert_eq!(
            "graph-greedy".parse::<OptimizeMethod>().ok(),
            Some(GraphGreedy)
        );
        assert!("unknown".parse::<OptimizeMethod>().is_err());
    }

    proptest! {

        #[test]
        fn index_from_iter(
            unigrams in prop::sample::subsequence(vec![0b001, 0b010, 0b100], 3)
                .prop_filter("Must have at least one unigram", |v| !v.is_empty()),
            bigrams in prop::sample::subsequence(vec![0b011, 0b110, 0b101], 3),
        ) {
            let transform = |v: Vec<_>| -> Vec<_> {
                v
                    .into_iter()
                    .map(|u| {
                        (Intersection(u), Cost(u.to_f32().unwrap()), Score(u.to_f32().unwrap()))
                    })
                    .collect()
            };
            let unigrams: Vec<_> = transform(unigrams);
            let bigrams: Vec<_> = transform(bigrams);
            let trigrams: Vec<_> = transform(vec![0b111]);
            let levels: Vec<Vec<(Intersection, Cost, Score)>> = vec![
                unigrams,
                bigrams,
                trigrams
            ];
            let index_from_iter = Index::from_iter(levels.iter().flatten().cloned());
            let index_from_levels = Index::new(3, &levels).unwrap();
            assert_eq!(index_from_iter.degrees, index_from_levels.degrees);
            assert_eq!(
                index_from_iter.costs.iter().collect::<Vec<_>>(),
                index_from_levels.costs.iter().collect::<Vec<_>>()
            );
            assert_eq!(
                index_from_iter.upper_bounds.iter().collect::<Vec<_>>(),
                index_from_levels.upper_bounds.iter().collect::<Vec<_>>()
            );
            assert_eq!(index_from_iter.query_len, index_from_levels.query_len);
        }

    }
}
