#[macro_use]
extern crate criterion;
extern crate intersect;

use criterion::{black_box, Criterion};

use intersect::{Cost, Index, OptimizeMethod, Score, TermBitset, TermMask};

pub fn optimize_query(crit: &mut Criterion, threshold: Score, method: OptimizeMethod) {
    let query_len = 7u8;
    let unigrams = (0..query_len)
        .map(|term| {
            (
                TermBitset((1 as TermMask) << term),
                Cost(1.0),
                Score(term as f32),
            )
        })
        .collect::<Vec<_>>();
    let mut bigrams: Vec<(TermBitset, Cost, Score)> = Vec::new();
    for left in 0..query_len {
        for right in (left + 1)..query_len {
            bigrams.push((
                TermBitset(((1 as TermMask) << left) | ((1 as TermMask) << right)),
                Cost(0.4),
                Score((left + right + 1) as f32),
            ));
        }
    }
    let index = Index::new(&vec![unigrams, bigrams]);
    crit.bench_function(
        &format!("optimize query with threshold {}", threshold.0),
        |b| b.iter(|| black_box(index.optimize(query_len, threshold, method))),
    );
}

pub fn opt_graph_0(crit: &mut Criterion) {
    optimize_query(crit, Score(0f32), OptimizeMethod::Graph)
}

pub fn opt_graph_1(crit: &mut Criterion) {
    optimize_query(crit, Score(1f32), OptimizeMethod::Graph)
}

pub fn opt_graph_2(crit: &mut Criterion) {
    optimize_query(crit, Score(2f32), OptimizeMethod::Graph)
}

pub fn opt_graph_3(crit: &mut Criterion) {
    optimize_query(crit, Score(3f32), OptimizeMethod::Graph)
}

pub fn opt_graph_4(crit: &mut Criterion) {
    optimize_query(crit, Score(4f32), OptimizeMethod::Graph)
}

pub fn opt_graph_5(crit: &mut Criterion) {
    optimize_query(crit, Score(5f32), OptimizeMethod::Graph)
}

pub fn opt_graph_6(crit: &mut Criterion) {
    optimize_query(crit, Score(6f32), OptimizeMethod::Graph)
}

criterion_group!(
    benches,
    opt_graph_0,
    opt_graph_1,
    opt_graph_2,
    opt_graph_3,
    opt_graph_4,
    opt_graph_5,
    opt_graph_6,
);
criterion_main!(benches);
