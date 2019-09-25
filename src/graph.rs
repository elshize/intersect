//! In this module, we consider graphs formed by posting list intersections.
//!
//! The graph consists of layers of nodes (intersections) of certain
//! [`Degree`s](struct.Degree.html).
//! There is an edge between an intersection `n` of degree `d` and an intersection `m`
//! of degree `d + 1` if `n` covers `m`, or (another way of looking at it)
//! `m` contains all of the terms of `n` (and one additional term).

use crate::{Intersection, TermMask};
use failure::{format_err, Error};
use num::cast::ToPrimitive;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::TryFrom;
use std::convert::TryInto;
use std::iter::FromIterator;

/// Type-safe representation of an n-gram degree.
///
/// E.g., an intersection of 3 terms is of degree 3, while a single posting list is of degree 1.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Degree(pub u8);

impl TryFrom<u32> for Degree {
    type Error = Error;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Ok(Self(
            value
                .to_u8()
                .ok_or_else(|| format_err!("Degree too high"))?,
        ))
    }
}

/// Represents a graph formed by posting lists. See [`graph`](index.html) module.
#[derive(Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Graph {
    nodes: Vec<Vec<Intersection>>,
    parents: Vec<Vec<Intersection>>,
    children: Vec<Vec<Intersection>>,
}

impl FromIterator<Intersection> for Graph {
    fn from_iter<I: IntoIterator<Item = Intersection>>(iter: I) -> Self {
        let mut node_map: HashMap<u8, Vec<Intersection>> = HashMap::new();
        let mut max_degree = 0_u8;
        for terms in iter {
            let degree = terms.0.count_ones().try_into().unwrap();
            node_map.entry(degree).or_default().push(terms);
            max_degree = std::cmp::max(degree, max_degree);
        }
        let mut nodes: Vec<Vec<Intersection>> = vec![vec![]; (max_degree + 1) as usize];
        node_map
            .into_iter()
            .for_each(|(deg, terms)| nodes[deg as usize] = terms);
        Self::from_nodes(nodes)
    }
}

impl Graph {
    /// Constructs a full graph having all n-grams up to a given degree.
    pub fn full(nterms: u8, max_degree: Degree) -> Result<Self, Error> {
        let Degree(max_degree) = max_degree;
        if max_degree > nterms {
            return Err(format_err!("Degree must be at most number of terms"));
        }
        let two: TermMask = 2;
        Ok(Self::from_iter((1..two.pow(u32::from(nterms))).filter_map(
            |mask| {
                if mask.count_ones() > u32::from(max_degree) {
                    None
                } else {
                    Some(Intersection(mask))
                }
            },
        )))
    }

    /// Constructs a graph by connecting neighboring layers in the given vector.
    pub fn from_nodes(mut nodes: Vec<Vec<Intersection>>) -> Self {
        let &Intersection(max_node) = nodes.iter().flatten().max().unwrap_or(&Intersection(0));
        nodes.iter_mut().for_each(|v| v.sort());
        let mut parents: Vec<Vec<Intersection>> = vec![vec![]; max_node as usize + 1];
        let mut children: Vec<Vec<Intersection>> = vec![vec![]; max_node as usize + 1];
        for window in nodes[1..].windows(2).rev() {
            match window {
                [higher, lower] => Self::connect_layers(higher, lower, &mut parents, &mut children),
                _ => unreachable!(),
            }
        }
        parents.iter_mut().for_each(|v| v.sort());
        children.iter_mut().for_each(|v| v.sort());
        Self {
            nodes,
            parents,
            children,
        }
    }

    fn connect_layers(
        higher: &[Intersection],
        lower: &[Intersection],
        parents: &mut Vec<Vec<Intersection>>,
        children: &mut Vec<Vec<Intersection>>,
    ) {
        for &Intersection(node) in lower {
            let mut recipe = node;
            while recipe > 0 {
                let bit_to_flip = 1 << recipe.trailing_zeros();
                let parent = node ^ bit_to_flip;
                if higher.binary_search(&Intersection(parent)).is_ok() {
                    parents[node as usize].push(Intersection(parent));
                    children[parent as usize].push(Intersection(node));
                }
                recipe -= bit_to_flip;
            }
        }
    }

    /// Returns an iterator over all nodes of a certain degree.
    /// For example, `layer(1)` will return unigrams, while `layer(2)` -- bigrams.
    ///
    /// # Examples
    ///
    /// ```
    /// # use intersect::graph::{Degree, Graph};
    /// # use intersect::Intersection;
    /// # fn main() -> Result<(), failure::Error> {
    /// let graph = Graph::full(3, Degree(3))?;
    /// // Layer of degree 0 is always empty.
    /// assert!(graph.layer(Degree(0)).collect::<Vec<_>>().is_empty());
    /// assert_eq!(
    ///     graph.layer(Degree(2)).collect::<Vec<_>>(),
    ///     vec![Intersection(0b011), Intersection(0b101), Intersection(0b110)]
    /// );
    /// // Any layer above the maximum degree will be empty.
    /// assert!(graph.layer(Degree(4)).collect::<Vec<_>>().is_empty());
    /// # Ok(())
    /// # }
    /// ```
    pub fn layer(&self, degree: Degree) -> Layer {
        Layer {
            iter: self.nodes.get(degree.0 as usize).map(|v| v.iter()),
        }
    }

    /// Returns the highest degree in the graph.
    pub fn max_degree(&self) -> Degree {
        Degree(self.nodes.len().to_u8().unwrap() - 1)
    }

    /// Returns the layer of the highest degree and that degree.
    pub fn last_layer(&self) -> Option<(Degree, Layer)> {
        if self.nodes.len() < 2 {
            None
        } else {
            let degree = self.max_degree();
            Some((degree, self.layer(degree)))
        }
    }

    /// Returns the iterator over layers, starting from degree 1.
    pub fn layers(&self) -> Layers {
        Layers {
            iter: self.nodes[1..].iter(),
        }
    }

    /// Parents of the given term subset.
    /// These are nodes of lower degree (fewer terms).
    pub fn parents(&self, terms: Intersection) -> Option<&[Intersection]> {
        self.parents.get(terms.0 as usize).map(|v| &v[..])
    }

    /// Children of the given term subset.
    /// These are nodes of higher degree (more terms).
    pub fn children(&self, terms: Intersection) -> Option<&[Intersection]> {
        self.children.get(terms.0 as usize).map(|v| &v[..])
    }

    /// An iterator over all edges: parents (if any) following by children (if any).
    pub fn edges(&self, term: Intersection) -> Edges {
        Edges {
            iter: self
                .parents(term)
                .unwrap_or(&[])
                .iter()
                .chain(self.children(term).unwrap_or(&[])),
        }
    }
}

/// Iterator over nodes in a single layer of a graph. The return type of
/// [`layer`](struct.Graph.html#method.layer).
pub struct Layer<'a> {
    iter: Option<std::slice::Iter<'a, Intersection>>,
}

impl<'a> Iterator for Layer<'a> {
    type Item = Intersection;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.as_mut()?.next().cloned()
    }
}

/// Iterator over layers of a graph. The return type of
/// [`layers`](struct.Graph.html#method.layers).
pub struct Layers<'a> {
    iter: std::slice::Iter<'a, Vec<Intersection>>,
}

impl<'a> Iterator for Layers<'a> {
    type Item = Layer<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|v| Layer {
            iter: Some(v[..].iter()),
        })
    }
}

impl<'a> DoubleEndedIterator for Layers<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|v| Layer {
            iter: Some(v[..].iter()),
        })
    }
}

/// Iterator over all edges of a node (both parents and children). The return type of
/// [`edges`](struct.Graph.html#method.edges).
pub struct Edges<'a> {
    iter: std::iter::Chain<std::slice::Iter<'a, Intersection>, std::slice::Iter<'a, Intersection>>,
}

impl<'a> Iterator for Edges<'a> {
    type Item = Intersection;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().cloned()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_connect_layers() {
        let nodes = vec![
            vec![],
            vec![
                Intersection(0b001),
                Intersection(0b010),
                Intersection(0b100),
            ],
            vec![
                Intersection(0b011),
                Intersection(0b101),
                Intersection(0b110),
            ],
        ];
        let mut parents: Vec<Vec<Intersection>> = vec![vec![]; 7];
        let mut children: Vec<Vec<Intersection>> = vec![vec![]; 7];
        Graph::connect_layers(&nodes[1], &nodes[2], &mut parents, &mut children);
        assert_eq!(
            parents,
            vec![
                vec![],
                vec![],
                vec![],
                vec![Intersection(0b010), Intersection(0b001)],
                vec![],
                vec![Intersection(0b100), Intersection(0b001)],
                vec![Intersection(0b100), Intersection(0b010)],
            ]
        );
        assert_eq!(
            children,
            vec![
                vec![],
                vec![Intersection(0b011), Intersection(0b101)],
                vec![Intersection(0b011), Intersection(0b110)],
                vec![],
                vec![Intersection(0b101), Intersection(0b110)],
                vec![],
                vec![],
            ]
        );
    }

    #[test]
    fn test_full() {
        assert_eq!(
            Graph::full(3, Degree(0)).unwrap(),
            Graph {
                nodes: vec![vec![]],
                parents: vec![vec![]],
                children: vec![vec![]]
            }
        );
        assert_eq!(
            Graph::full(3, Degree(1)).unwrap(),
            Graph {
                nodes: vec![
                    vec![],
                    vec![
                        Intersection(0b001),
                        Intersection(0b010),
                        Intersection(0b100),
                    ],
                ],
                parents: vec![vec![]; 5],
                children: vec![vec![]; 5],
            }
        );
        assert_eq!(
            Graph::full(3, Degree(2)).unwrap(),
            Graph {
                nodes: vec![
                    vec![],
                    vec![
                        Intersection(0b001),
                        Intersection(0b010),
                        Intersection(0b100),
                    ],
                    vec![
                        Intersection(0b011),
                        Intersection(0b101),
                        Intersection(0b110),
                    ],
                ],
                parents: vec![
                    vec![],
                    vec![],
                    vec![],
                    vec![Intersection(0b001), Intersection(0b010)],
                    vec![],
                    vec![Intersection(0b001), Intersection(0b100)],
                    vec![Intersection(0b010), Intersection(0b100)],
                ],
                children: vec![
                    vec![],
                    vec![Intersection(0b011), Intersection(0b101)],
                    vec![Intersection(0b011), Intersection(0b110)],
                    vec![],
                    vec![Intersection(0b101), Intersection(0b110)],
                    vec![],
                    vec![],
                ],
            }
        );
        assert_eq!(
            Graph::full(3, Degree(3)).unwrap(),
            Graph {
                nodes: vec![
                    vec![],
                    vec![
                        Intersection(0b001),
                        Intersection(0b010),
                        Intersection(0b100),
                    ],
                    vec![
                        Intersection(0b011),
                        Intersection(0b101),
                        Intersection(0b110),
                    ],
                    vec![Intersection(0b111)]
                ],
                parents: vec![
                    vec![],
                    vec![],                                         // 0b001
                    vec![],                                         // 0b010
                    vec![Intersection(0b001), Intersection(0b010)], // 0b011
                    vec![],                                         // 0b100
                    vec![Intersection(0b001), Intersection(0b100)], // 0b101
                    vec![Intersection(0b010), Intersection(0b100)], // 0b110
                    vec![
                        Intersection(0b011),
                        Intersection(0b101),
                        Intersection(0b110)
                    ], // 0b111
                ],
                children: vec![
                    vec![],
                    vec![Intersection(0b011), Intersection(0b101)], // 0b001
                    vec![Intersection(0b011), Intersection(0b110)], // 0b010
                    vec![Intersection(0b111)],                      // 0b011
                    vec![Intersection(0b101), Intersection(0b110)], // 0b100
                    vec![Intersection(0b111)],                      // 0b101
                    vec![Intersection(0b111)],                      // 0b110
                    vec![],                                         // 0b111
                ]
            }
        );
        assert!(Graph::full(3, Degree(4)).is_err());
    }

    #[test]
    fn test_layer() {
        let graph = Graph::full(3, Degree(3)).unwrap();
        assert_eq!(graph.layer(Degree(0)).collect::<Vec<_>>(), vec![]);
        assert_eq!(
            graph.layer(Degree(1)).collect::<Vec<_>>(),
            vec![
                Intersection(0b001),
                Intersection(0b010),
                Intersection(0b100)
            ]
        );
        assert_eq!(
            graph.layer(Degree(2)).collect::<Vec<_>>(),
            vec![
                Intersection(0b011),
                Intersection(0b101),
                Intersection(0b110)
            ]
        );
        assert_eq!(
            graph.layer(Degree(3)).collect::<Vec<_>>(),
            vec![Intersection(0b111)]
        );
    }

    #[test]
    fn test_layers() {
        let graph = Graph::full(3, Degree(3)).unwrap();
        let mut layers = graph.layers();
        assert_eq!(
            layers.next().unwrap().collect::<Vec<_>>(),
            vec![
                Intersection(0b001),
                Intersection(0b010),
                Intersection(0b100)
            ]
        );
        assert_eq!(
            layers.next().unwrap().collect::<Vec<_>>(),
            vec![
                Intersection(0b011),
                Intersection(0b101),
                Intersection(0b110)
            ]
        );
        assert_eq!(
            layers.next().unwrap().collect::<Vec<_>>(),
            vec![Intersection(0b111)]
        );
        assert!(layers.next().is_none());
    }

    #[test]
    fn test_layers_rev() {
        let graph = Graph::full(3, Degree(3)).unwrap();
        let mut layers = graph.layers().rev();
        assert_eq!(
            layers.next().unwrap().collect::<Vec<_>>(),
            vec![Intersection(0b111)]
        );
        assert_eq!(
            layers.next().unwrap().collect::<Vec<_>>(),
            vec![
                Intersection(0b011),
                Intersection(0b101),
                Intersection(0b110)
            ]
        );
        assert_eq!(
            layers.next().unwrap().collect::<Vec<_>>(),
            vec![
                Intersection(0b001),
                Intersection(0b010),
                Intersection(0b100)
            ]
        );
        assert!(layers.next().is_none());
    }

    #[test]
    fn test_last_layer() {
        let graph = Graph::full(3, Degree(3)).unwrap();
        let (Degree(degree), layer) = graph.last_layer().unwrap();
        assert_eq!(degree, 3);
        assert_eq!(layer.collect::<Vec<_>>(), vec![Intersection(0b111)]);

        let graph = Graph::full(3, Degree(0)).unwrap();
        assert!(graph.last_layer().is_none());
    }

    #[test]
    fn test_edges() {
        let graph = Graph::full(3, Degree(3)).unwrap();
        assert_eq!(graph.edges(Intersection(0b000)).collect::<Vec<_>>(), vec![]);
        assert_eq!(
            graph.edges(Intersection(0b001)).collect::<Vec<_>>(),
            vec![Intersection(0b011), Intersection(0b101)]
        );
        assert_eq!(
            graph.edges(Intersection(0b010)).collect::<Vec<_>>(),
            vec![Intersection(0b011), Intersection(0b110)]
        );
        assert_eq!(
            graph.edges(Intersection(0b101)).collect::<Vec<_>>(),
            vec![
                Intersection(0b001),
                Intersection(0b100),
                Intersection(0b111)
            ]
        );
        assert_eq!(
            graph.edges(Intersection(0b111)).collect::<Vec<_>>(),
            vec![
                Intersection(0b011),
                Intersection(0b101),
                Intersection(0b110)
            ]
        );
    }
}
