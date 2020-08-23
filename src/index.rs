use crate::graph::Graph;
use crate::power_set::power_set_iter;
use crate::set_cover::{BruteForceSetCover, GreedyNSetCover, GreedySetCover, WeightedSetCover};
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

    fn subsets(&self, index: &Index) -> Vec<(Vec<usize>, f32)> {
        let elements: Vec<_> = self
            .leaves
            .iter()
            .map(|&Intersection(leaf)| leaf)
            .chain(
                self.uncovered_classes
                    .iter()
                    .map(|&ResultClass(class)| class),
            )
            .collect();
        self.all
            .iter()
            .map(|&candidate| {
                let set: Vec<usize> = elements
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, &elem)| {
                        if candidate.covers(ResultClass(elem)) {
                            Some(idx)
                        } else {
                            None
                        }
                    })
                    .collect();
                let cand_idx: usize = candidate.into();
                let Cost(cost) = index.costs[cand_idx];
                (set, cost)
            })
            .collect()
    }

    fn solve(&self, index: &Index) -> Vec<usize> {
        BruteForceSetCover::solve(self.cardinality(), &self.subsets(index))
    }

    fn solve_greedy(&self, index: &Index) -> Vec<usize> {
        let subsets = &self.subsets(index);
        GreedySetCover::solve(self.cardinality(), subsets)
    }

    fn solve_greedy_2(&self, index: &Index) -> Vec<usize> {
        let subsets = &self.subsets(index);
        GreedyNSetCover::solve(self.cardinality(), subsets)
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
    Greedy2,
    /// Another approximate solution that has potential of being more accurate
    /// than `Greedy` (to be determined) but not as fast.
    /// Still, is reasonably fast if yields better results.
    Graph,
    /// Similar to graph, but faster by greedily selecting parents with smallest costs.
    /// (Is it equivalent to `Graph`?)
    GraphGreedy,
    /// Selects only among single-term lists. This is very fast (linear in number of terms),
    /// and can be easily used to compare with a result of an approximate selection, such as
    /// `Greedy` or `GraphGreedy` to see if the cost is lower by using only unigrams.
    Unigram,
    /// This one assumes we only have unigrams and bigrams available.
    /// Furthermore, it is meant to be optimized for when there are relatively few bigrams
    /// (which usually will be true, at least in certain scenarios).
    Bigram,
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
            "unigram" => Ok(Self::Unigram),
            "bigram" => Ok(Self::Bigram),
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
        let max_class: TermMask = num::cast::NumCast::from(2_u16.pow(u32::from(query_len)) - 1)
            .ok_or_else(|| format_err!("Query too long: {}", query_len))?;
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
        for class in (0..=max_class).map(ResultClass) {
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

    /// Cost of a given intersection.
    pub fn upper_bound(&self, intersection: Intersection) -> Score {
        let idx: usize = intersection.into();
        self.upper_bounds[idx]
    }

    /// Returns all essential classes, i.e., those that are above a given threshold.
    pub fn essential_ngram_classes(&self, threshold: Score, min_degree: u32) -> Vec<ResultClass> {
        ResultClass::all(self.query_len)
            .filter(|&ResultClass(mask)| {
                Intersection(mask).degree() >= min_degree
                    && self.upper_bounds[mask as usize] >= threshold
            })
            .collect()
    }

    /// Select optimal set of posting lists for query execution.
    pub fn optimize(&mut self, threshold: Score, method: OptimizeMethod) -> Vec<Intersection> {
        match method {
            OptimizeMethod::BruteForce => self.optimize_brute_force(threshold),
            OptimizeMethod::Exact => self.optimize_exact(threshold),
            OptimizeMethod::Greedy => self.optimize_greedy(threshold),
            OptimizeMethod::Greedy2 => self.optimize_greedy_2(threshold),
            OptimizeMethod::Graph => self.optimize_graph(threshold),
            OptimizeMethod::GraphGreedy => self.optimize_graph_greedy(threshold),
            OptimizeMethod::Unigram => self.optimize_unigram(threshold),
            OptimizeMethod::Bigram => self.optimize_bigram(threshold),
        }
    }

    fn full_cost<'a, I: IntoIterator<Item = &'a Intersection>>(&'a self, selection: I) -> Cost {
        selection.into_iter().map(|&i| self.cost(i)).sum()
    }

    /// First optimizes using `method`, and then runs `OptimizeMethod::Unigram` and returns
    /// the one with lower cost.
    pub fn optimize_or_unigram(
        &mut self,
        threshold: Score,
        method: OptimizeMethod,
    ) -> Vec<Intersection> {
        match method {
            OptimizeMethod::Unigram | OptimizeMethod::Bigram => self.optimize(threshold, method),
            OptimizeMethod::Greedy
            | OptimizeMethod::Greedy2
            | OptimizeMethod::BruteForce
            | OptimizeMethod::Exact
            | OptimizeMethod::Graph
            | OptimizeMethod::GraphGreedy => {
                let approx = self.optimize(threshold, method);
                let approx_cost = self.full_cost(&approx);
                let unigram = self.optimize_unigram(threshold);
                let unigram_cost = self.full_cost(&unigram);
                if unigram_cost <= approx_cost {
                    unigram
                } else {
                    approx
                }
            }
        }
    }

    fn optimize_unigram(&self, threshold: Score) -> Vec<Intersection> {
        let mut unigrams: Vec<_> = self.degrees[0].iter().copied().collect();
        unigrams.sort_unstable_by_key(|&i| self.upper_bound(i));
        let mut sum = 0.0;
        let mut num_non_essential = 0;
        for bound in unigrams.iter().map(|&i| self.upper_bound(i)) {
            sum += bound.0;
            if sum > threshold.0 {
                break;
            }
            num_non_essential += 1;
        }
        if num_non_essential == unigrams.len() {
            num_non_essential -= 1;
        }
        unigrams[num_non_essential..].iter().copied().collect()
    }

    ///// Checks if a given bigram covers all essential classes of the given unigrams.
    ///// This means that `bigram` can replace `unigrams` safely.
    //fn bigram_covers_unigrams(
    //    essential_classes: &[ResultClass],
    //    removed: &[Intersection],
    //    remaining: &[Intersection],
    //) -> bool {
    //    essential_classes
    //        .iter()
    //        .all(|&c| remaining.iter().any(|u| u.covers(c)) || !removed.iter().any(|u| u.covers(c)))
    //}

    ///// For a given `bigram`, it returns a list of unigrams it can safely replace,
    ///// along with the gain of such replacement.
    ///// If no such replacement is possible, `None` is returned.
    //fn replacement(
    //    &self,
    //    graph: &Graph,
    //    bigram: Intersection,
    //    selected_bigrams: &[Intersection],
    //    available_unigrams: &HashSet<Intersection>,
    //    essential_classes: &[ResultClass],
    //    threshold: Score,
    //) -> Option<(Intersection, Vec<Intersection>, Cost)> {
    //    if let Some(remaining) = graph.parents(bigram) {
    //        // TODO: not enough to look at the parents, have to also include other unigrams
    //        // in the `bigram_covers_unigrams` check.
    //        let (removed, mut remaining): (Vec<_>, Vec<_>) = remaining
    //            .iter()
    //            .copied()
    //            .partition(|&u| available_unigrams.contains(&u) && self.upper_bound(u) < threshold);
    //        eprintln!("(removed, remaining) = ({:?}, {:?})", &removed, &remaining);
    //        let remaining: Vec<_> = remaining
    //            .drain(..)
    //            .chain(selected_bigrams.iter().copied())
    //            .chain(std::iter::once(bigram))
    //            .collect();
    //        if removed.is_empty()
    //            || !Self::bigram_covers_unigrams(&essential_classes, &removed, &remaining)
    //        {
    //            eprintln!(
    //                "bigram {:?} not considered; removed: {:?}; remaining: {:?}; essential classes: {:?}",
    //                &bigram, &removed, &remaining, &essential_classes
    //            );
    //            //eprintln!(
    //            //    "{:?} | {:?}",
    //            //    bigram.covers(essential_classes[0]),
    //            //    Self::bigram_covers_unigrams(&essential_classes, &unigrams, bigram)
    //            //);
    //            None
    //        } else {
    //            let gain = self.full_cost(&removed) - self.cost(bigram);
    //            Some((bigram, removed, gain))
    //        }
    //    } else {
    //        None
    //    }
    //}

    fn solution_after_replacement(
        &self,
        graph: &Graph,
        bigrams: Vec<Intersection>,
        remaining_unigrams: &HashSet<Intersection>,
        threshold: Score,
        mut solution: HashSet<Intersection>,
        essential_classes: &[ResultClass],
    ) -> Option<(Vec<Intersection>, HashSet<Intersection>, Cost)> {
        let is_live = |&u: &Intersection| -> bool {
            remaining_unigrams.contains(&u) && self.upper_bound(u) < threshold
        };
        match bigrams
            .iter()
            .copied()
            .filter_map(|bigram| {
                graph
                    .parents(bigram)
                    .map(|unigrams| unigrams.iter().copied().filter(is_live).collect::<Vec<_>>())
            })
            .flatten()
            .collect::<HashSet<_>>()
        {
            //None => None,
            v if v.is_empty() => None,
            replaced_unigrams => {
                for u in &replaced_unigrams {
                    solution.remove(u);
                }
                for &bigram in &bigrams {
                    solution.insert(bigram);
                }
                if essential_classes
                    .iter()
                    .all(|&c| solution.iter().any(|u| u.covers(c)))
                {
                    let gain = self.full_cost(&replaced_unigrams) - self.full_cost(&bigrams);
                    if gain > Cost(0.0) {
                        Some((bigrams, solution, gain))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        }
    }

    fn optimize_bigram(&self, threshold: Score) -> Vec<Intersection> {
        let essential_unigrams = self.optimize_unigram(threshold);
        if let Some(mut bigrams) = self
            .degrees
            .get(1)
            .map(|b| b.iter().copied().collect::<HashSet<_>>())
        {
            let essential_classes = self.essential_ngram_classes(threshold, 2);
            let graph = Graph::from_iter(essential_unigrams.iter().chain(bigrams.iter()).copied());
            let mut remaining_unigrams: HashSet<_> = essential_unigrams.into_iter().collect();
            let mut solution: HashSet<_> = remaining_unigrams.clone();
            //let max_unigram_cost = remaining_unigrams
            //    .iter()
            //    .map(|&u| self.cost(u))
            //    .max()
            //    .unwrap();
            //let x = power_set_iter(&bigrams.iter().copied().collect::<Vec<_>>())
            //    .filter(|&subset| self.full_cost(subset) < max_unigram_cost)
            //    .count();
            //println!("Subsets to check: {}", x);
            while !remaining_unigrams.is_empty() && !bigrams.is_empty() {
                if let Some((selected, new_solution, _)) = bigrams
                    .iter()
                    .copied()
                    .map(|b| vec![b])
                    .chain(bigrams.iter().copied().combinations(2))
                    //.chain(bigrams.iter().copied().combinations(3))
                    .filter_map(|b| {
                        self.solution_after_replacement(
                            &graph,
                            b,
                            &remaining_unigrams,
                            threshold,
                            solution.clone(),
                            &essential_classes,
                        )
                    })
                    .max_by_key(|&(_, _, gain)| gain)
                {
                    solution = new_solution;
                    remaining_unigrams = solution
                        .iter()
                        .copied()
                        .filter(|i| i.0.count_ones() == 1)
                        .collect();
                    for bigram in selected {
                        bigrams.remove(&bigram);
                    }
                } else {
                    break;
                }
            }
            solution.into_iter().collect()
        } else {
            essential_unigrams
        }
    }

    //fn optimize_bigram_(&self, threshold: Score) -> Vec<Intersection> {
    //    let essential_unigrams = self.optimize_unigram(threshold);
    //    if let Some(bigrams) = self.degrees.get(1) {
    //        let essential_classes = self.essential_ngram_classes(threshold, 2);
    //        let mut remaining_unigrams: HashSet<_> = essential_unigrams.iter().copied().collect();
    //        let graph = Graph::from_iter(essential_unigrams.iter().chain(bigrams.iter()).copied());
    //        let mut bigram_candidates: HashSet<_> = bigrams.iter().copied().collect();
    //        let mut selected_bigrams: Vec<Intersection> = Vec::new();
    //        while !remaining_unigrams.is_empty() && !bigram_candidates.is_empty() {
    //            if let Some((bigram, unigrams, gain)) = bigram_candidates
    //                .iter()
    //                .filter_map(|&bigram| {
    //                    self.replacement(
    //                        &graph,
    //                        bigram,
    //                        &selected_bigrams,
    //                        &remaining_unigrams,
    //                        &essential_classes,
    //                        threshold,
    //                    )
    //                })
    //                .max_by_key(|&(_, _, gain)| gain)
    //            {
    //                if gain <= Cost(0.0) {
    //                    break;
    //                }
    //                for unigram in unigrams {
    //                    remaining_unigrams.remove(&unigram);
    //                }
    //                bigram_candidates.remove(&bigram);
    //                selected_bigrams.push(bigram);
    //            } else {
    //                break;
    //            }
    //        }
    //        remaining_unigrams
    //            .into_iter()
    //            .chain(selected_bigrams.into_iter())
    //            .collect()
    //    } else {
    //        essential_unigrams
    //    }
    //}

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

    fn optimize_greedy_2(&self, threshold: Score) -> Vec<Intersection> {
        let candidates = self.candidates(threshold);
        let solution = candidates.solve_greedy_2(&self);
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

    /// Returns the number of terms.
    pub fn query_length(&self) -> u8 {
        self.query_len
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntersectionInput {
    /// Intersection mask.
    pub mask: Intersection,
    /// Intersection cost, e.g., number of postings.
    pub length: Cost,
    /// Maximum score of a document in the intersection.
    pub max_score: Score,
}

impl FromIterator<IntersectionInput> for Index {
    fn from_iter<T: IntoIterator<Item = IntersectionInput>>(iter: T) -> Self {
        let mut query_len = 0_u8;
        let mut posting_lists: Vec<Vec<(Intersection, Cost, Score)>> = Vec::new();
        for IntersectionInput {
            mask,
            length,
            max_score,
        } in iter
        {
            let len = mask
                .0
                .trailing_zeros()
                .to_u8()
                .expect("Unable to cast u32 to u8")
                + 1;
            query_len = std::cmp::max(query_len, len);
            let level = mask.0.count_ones() as usize;
            if posting_lists.len() < level {
                posting_lists.resize_with(level, Default::default);
            }
            posting_lists[level.checked_sub(1).expect("Intersection cannot be 0")]
                .push((mask, length, max_score));
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

    fn sorted<T: Ord>(mut v: Vec<T>) -> Vec<T> {
        v.sort();
        v
    }

    #[test]
    fn test_school_unified_district_downey() {
        let intersections: Vec<IntersectionInput> = serde_json::from_str(
            r#"[
            {"cost": 2423488, "intersection": 1, "max_score": 5.653356075286865},
            {"cost": 935805, "intersection": 5, "max_score": 8.899179458618164},
            {"cost": 72728, "intersection": 9, "max_score": 14.799269676208496},
            {"cost": 67622, "intersection": 2, "max_score": 12.529928207397461},
            {"cost": 7660019, "intersection": 4, "max_score": 3.252655029296875},
            {"cost": 133514, "intersection": 12, "max_score": 12.412330627441406},
            {"cost": 395014, "intersection": 8, "max_score": 9.169288635253906}
            ]"#,
        )
        .unwrap();
        let index = Index::from_iter(intersections.into_iter());
        assert_eq!(
            &sorted(index.optimize_bigram(Score(16.043301))),
            &[Intersection(2), Intersection(9)]
        );
    }

    #[test]
    fn test_buses_pittsburgh() {
        let intersections: Vec<IntersectionInput> = serde_json::from_str(
            r#"[
            {"cost": 46969, "intersection": 1, "max_score": 13.052966117858888},
            {"cost": 1435, "intersection": 3, "max_score": 19.0411434173584},
            {"cost": 500508, "intersection": 2, "max_score": 8.719024658203125}
            ]"#,
        )
        .unwrap();
        let index = Index::from_iter(intersections.into_iter());
        assert!(&index.optimize_bigram(Score(10.110200)) == &[Intersection(1)]);
    }

    #[test]
    fn test_50_cent() {
        let intersections: Vec<IntersectionInput> = serde_json::from_str(
            r#"[
            {"cost": 7150457, "intersection": 1, "max_score": 3.408390998840332},
            {"cost": 319771, "intersection": 3, "max_score": 11.532732963562012},
            {"cost": 677700, "intersection": 2, "max_score": 8.134215354919434}
            ]"#,
        )
        .unwrap();
        let index = Index::from_iter(intersections.into_iter());
        assert_eq!(&index.optimize_bigram(Score(10.919300)), &[Intersection(3)]);
    }

    #[test]
    fn test_party_slumber_bears() {
        let intersections: Vec<IntersectionInput> = serde_json::from_str(
            r#"[
            {"cost": 1994671, "intersection": 1, "max_score": 6.039915561676025},
            {"cost": 11059, "intersection": 5, "max_score": 18.312904357910156},
            {"cost": 4606787, "intersection": 2, "max_score": 4.34684419631958},
            {"cost": 20953, "intersection": 6, "max_score": 17.45214080810547},
            {"cost": 46983, "intersection": 4, "max_score": 13.127897262573242}
            ]"#,
        )
        .unwrap();
        let index = Index::from_iter(intersections.into_iter());
        assert_eq!(
            &index.optimize_exact(Score(14.999400)),
            &[Intersection(5), Intersection(6)]
        );
        assert_eq!(
            &index.optimize_bigram(Score(14.999400)),
            &[Intersection(5), Intersection(6)]
        );
    }

    #[test]
    fn test_wine_food_pairing() {
        let intersections: Vec<IntersectionInput> = serde_json::from_str(
            r#"[
            {"cost": 5569437, "intersection": 1, "max_score": 3.946219682693481},
            {"cost": 651602,  "intersection": 5, "max_score": 10.726018905639648},
            {"cost": 1118862, "intersection": 2, "max_score": 7.167963981628418},
            {"cost": 1334056, "intersection": 4, "max_score": 6.829254150390625}
            ]"#,
        )
        .unwrap();
        let index = Index::from_iter(intersections.into_iter());
        assert_eq!(&index.optimize_exact(Score(15.075000)), &[Intersection(5)]);
        assert_eq!(&index.optimize_bigram(Score(15.075000)), &[Intersection(5)]);
    }

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
        assert_eq!(candidates.leaves, vec![Intersection(6), Intersection(18)]);
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

    #[test]
    fn test_drivers_education_honolulu_hawaii() {
        let intersections: Vec<IntersectionInput> = serde_json::from_str(
            r#"[{"cost":1468706,"intersection":1,"max_score":6.639634609222412},{"cost":338950,"intersection":3,"max_score":9.69923973083496},{"cost":47357,"intersection":5,"max_score":13.511035919189451},{"cost":10676,"intersection":9,"max_score":15.92648696899414},{"cost":719578,"intersection":17,"max_score":7.662487983703613},{"cost":8267086,"intersection":2,"max_score":3.081190586090088},{"cost":241985,"intersection":6,"max_score":10.240900039672852},{"cost":61706,"intersection":10,"max_score":12.95743179321289},{"cost":3456044,"intersection":18,"max_score":4.114166259765625},{"cost":1062881,"intersection":4,"max_score":7.269174098968506},{"cost":114242,"intersection":12,"max_score":17.30234718322754},{"cost":435092,"intersection":20,"max_score":8.251961708068848},{"cost":238947,"intersection":8,"max_score":10.131380081176758},{"cost":120172,"intersection":24,"max_score":11.09640407562256},{"cost":18089189,"intersection":16,"max_score":1.090119481086731}]"#,
        ).unwrap();
        assert_eq!(
            Index::from_iter(intersections.clone().into_iter())
                .optimize(Score(17.60650062561035), BruteForce),
            Index::from_iter(intersections.into_iter()).optimize(Score(17.60650062561035), Greedy)
        )
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
