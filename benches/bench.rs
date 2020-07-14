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

pub fn opt_bigram_long(crit: &mut Criterion) {
    let intersections: Vec<IntersectionInput> = serde_json::from_str(
        r#"[{"cost":1104252,"intersection":1,"max_score":7.195741653442383},{"cost":107955,"intersection":3,"max_score":13.53505039215088},{"cost":503015,"intersection":5,"max_score":11.256619453430176},{"cost":1068156,"intersection":9,"max_score":7.195743083953857},{"cost":73385,"intersection":17,"max_score":16.30108070373535},{"cost":13643,"intersection":33,"max_score":20.84843635559082},{"cost":433309,"intersection":65,"max_score":11.471548080444336},{"cost":1088404,"intersection":129,"max_score":7.195743083953857},{"cost":311664,"intersection":257,"max_score":11.696033477783203},{"cost":1489023,"intersection":2,"max_score":6.612133979797363},{"cost":224663,"intersection":6,"max_score":10.733896255493164},{"cost":1441624,"intersection":10,"max_score":6.612135887145996},{"cost":56422,"intersection":18,"max_score":15.703750610351564},{"cost":13776,"intersection":34,"max_score":20.25110626220703},{"cost":282168,"intersection":66,"max_score":10.756011962890623},{"cost":1443263,"intersection":130,"max_score":6.612135887145996},{"cost":410582,"intersection":258,"max_score":11.016401290893556},{"cost":4949306,"intersection":4,"max_score":4.196615219116211},{"cost":4639152,"intersection":12,"max_score":4.196617126464844},{"cost":122787,"intersection":20,"max_score":13.404427528381348},{"cost":15744,"intersection":36,"max_score":17.912965774536133},{"cost":1388716,"intersection":68,"max_score":8.519875526428223},{"cost":4718418,"intersection":132,"max_score":4.196617126464844},{"cost":895009,"intersection":260,"max_score":8.700217247009277},{"cost":45161193,"intersection":8,"max_score":1.8963453385367757e-06},{"cost":313949,"intersection":24,"max_score":9.569229125976564},{"cost":26890,"intersection":40,"max_score":14.128093719482422},{"cost":4371675,"intersection":72,"max_score":4.35020637512207},{"cost":43046072,"intersection":136,"max_score":3.7890906696702582e-06},{"cost":3987475,"intersection":264,"max_score":4.607618808746338},{"cost":320639,"intersection":16,"max_score":9.569228172302246},{"cost":14224,"intersection":48,"max_score":23.58435821533203},{"cost":138359,"intersection":80,"max_score":13.565455436706545},{"cost":316991,"intersection":144,"max_score":9.56923007965088},{"cost":157362,"intersection":272,"max_score":13.811692237854004},{"cost":27582,"intersection":32,"max_score":14.128091812133787},{"cost":22143,"intersection":96,"max_score":18.44595718383789},{"cost":27204,"intersection":160,"max_score":14.128093719482422},{"cost":22673,"intersection":288,"max_score":18.701658248901367},{"cost":4599690,"intersection":64,"max_score":4.350481033325195},{"cost":4407233,"intersection":192,"max_score":4.35048246383667},{"cost":1035086,"intersection":320,"max_score":8.959571838378906},{"cost":45737872,"intersection":128,"max_score":1.8965687331728984e-06},{"cost":4004770,"intersection":384,"max_score":4.6090922355651855},{"cost":4060730,"intersection":256,"max_score":4.609090805053711}]"#,
    )
    .unwrap();
    optimize_query(
        crit,
        intersections,
        Score(36.573299407958984),
        OptimizeMethod::Bigram,
    );
}

pub fn opt_bigram(crit: &mut Criterion) {
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
    optimize_query(
        crit,
        intersections,
        Score(15.080699920654297),
        OptimizeMethod::Bigram,
    );
}

criterion_group!(benches, opt_bigram, opt_bigram_long);
criterion_main!(benches);
