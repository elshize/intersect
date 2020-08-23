#[macro_use]
extern crate criterion;
extern crate intersect;

use criterion::{black_box, Criterion};
use intersect::{Index, IntersectionInput, OptimizeMethod, Score};
use std::iter::FromIterator;

pub fn optimize_query(
    crit: &mut Criterion,
    intersections: Vec<IntersectionInput>,
    threshold: Score,
    method: OptimizeMethod,
) {
    crit.bench_function(
        &format!(
            "optimize query {:?} with threshold {} and method {:?}",
            &intersections, threshold.0, method
        ),
        |b| {
            b.iter(|| {
                let mut index = Index::from_iter(intersections.clone().into_iter());
                black_box(index.optimize(threshold, method))
            })
        },
    );
}

pub fn opt_long(crit: &mut Criterion, method: OptimizeMethod) {
    let intersections: Vec<IntersectionInput> = serde_json::from_str(
        r#"
[
  {
    "cost": 2136636,
    "intersection": 1,
    "max_score": 5.903607368469238
  },
  {
    "cost": 967608,
    "intersection": 3,
    "max_score": 7.781198024749756
  },
  {
    "cost": 1073759,
    "intersection": 5,
    "max_score": 7.198347091674805
  },
  {
    "cost": 1754136,
    "intersection": 9,
    "max_score": 5.903608798980713
  },
  {
    "cost": 75423,
    "intersection": 17,
    "max_score": 14.566667556762695
  },
  {
    "cost": 481067,
    "intersection": 33,
    "max_score": 10.143877029418945
  },
  {
    "cost": 2021017,
    "intersection": 65,
    "max_score": 5.903608798980713
  },
  {
    "cost": 1461356,
    "intersection": 129,
    "max_score": 5.902857780456543
  },
  {
    "cost": 13241415,
    "intersection": 2,
    "max_score": 1.9462953805923462
  },
  {
    "cost": 7634851,
    "intersection": 6,
    "max_score": 3.2859904766082764
  },
  {
    "cost": 11777055,
    "intersection": 10,
    "max_score": 1.946297287940979
  },
  {
    "cost": 276737,
    "intersection": 18,
    "max_score": 10.784050941467285
  },
  {
    "cost": 2871791,
    "intersection": 34,
    "max_score": 6.275369167327881
  },
  {
    "cost": 13117378,
    "intersection": 66,
    "max_score": 1.9462971687316895
  },
  {
    "cost": 9696614,
    "intersection": 130,
    "max_score": 1.9462968111038208
  },
  {
    "cost": 16342542,
    "intersection": 4,
    "max_score": 1.381404995918274
  },
  {
    "cost": 14859369,
    "intersection": 12,
    "max_score": 1.3814067840576172
  },
  {
    "cost": 302813,
    "intersection": 20,
    "max_score": 10.190311431884766
  },
  {
    "cost": 3002838,
    "intersection": 36,
    "max_score": 5.674585819244385
  },
  {
    "cost": 16138537,
    "intersection": 68,
    "max_score": 1.3814067840576172
  },
  {
    "cost": 11751934,
    "intersection": 132,
    "max_score": 1.3814067840576172
  },
  {
    "cost": 36388265,
    "intersection": 8,
    "max_score": 1.8957789507112466e-06
  },
  {
    "cost": 419239,
    "intersection": 24,
    "max_score": 8.923745155334473
  },
  {
    "cost": 4122199,
    "intersection": 40,
    "max_score": 4.367137432098389
  },
  {
    "cost": 35201647,
    "intersection": 72,
    "max_score": 3.7824970604560804e-06
  },
  {
    "cost": 20076941,
    "intersection": 136,
    "max_score": 3.7796316973981448e-06
  },
  {
    "cost": 449638,
    "intersection": 16,
    "max_score": 8.92374324798584
  },
  {
    "cost": 154753,
    "intersection": 48,
    "max_score": 12.98859691619873
  },
  {
    "cost": 443255,
    "intersection": 80,
    "max_score": 8.923745155334473
  },
  {
    "cost": 321482,
    "intersection": 144,
    "max_score": 8.923745155334473
  },
  {
    "cost": 4557431,
    "intersection": 32,
    "max_score": 4.367135524749756
  },
  {
    "cost": 4479586,
    "intersection": 96,
    "max_score": 4.367137432098389
  },
  {
    "cost": 3580538,
    "intersection": 160,
    "max_score": 4.367137432098389
  },
  {
    "cost": 44786909,
    "intersection": 64,
    "max_score": 1.8966732113767648e-06
  },
  {
    "cost": 24170272,
    "intersection": 192,
    "max_score": 3.7819727367605083e-06
  },
  {
    "cost": 25234562,
    "intersection": 128,
    "max_score": 1.8952830487251049e-06
  }
]
        "#,
    )
    .unwrap();
    optimize_query(crit, intersections, Score(36.573299407958984), method);
}

pub fn opt_short(crit: &mut Criterion, method: OptimizeMethod) {
    let intersections: Vec<IntersectionInput> = serde_json::from_str(
        r#"[
            {"cost":2423488,"intersection":1,"max_score":5.653356075286865},
            {"cost":935805,"intersection":9,"max_score":8.899179458618164},
            {"cost":617318,"intersection":2,"max_score":8.318435668945312},
            {"cost":47454,"intersection":4,"max_score":13.09547233581543},
            {"cost":7660019,"intersection":8,"max_score":3.252655029296875}
        ]"#,
    )
    .unwrap();
    optimize_query(crit, intersections, Score(15.080699920654297), method);
}

pub fn opt_bigram_short(crit: &mut Criterion) {
    opt_short(crit, OptimizeMethod::Bigram);
}

pub fn opt_bigram_long(crit: &mut Criterion) {
    opt_long(crit, OptimizeMethod::Bigram);
}

pub fn opt_greedy_short(crit: &mut Criterion) {
    opt_short(crit, OptimizeMethod::Greedy);
}

pub fn opt_greedy_long(crit: &mut Criterion) {
    opt_long(crit, OptimizeMethod::Greedy);
}

criterion_group!(
    benches,
    opt_bigram_short,
    opt_bigram_long,
    opt_greedy_short,
    opt_greedy_long
);
criterion_main!(benches);
