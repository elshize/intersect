use failure::{format_err, Error};
use intersect::{Index, IntersectionInput, OptimizeMethod, Score};
use serde_json::{
    value::{from_value, to_value},
    Deserializer, Value,
};
use std::fs;
use std::io::{self, prelude::*, BufReader};
use std::iter::FromIterator;
use std::path::PathBuf;

#[derive(structopt::StructOpt)]
struct Args {
    /// Input file, stdin if not present
    #[structopt(parse(from_os_str))]
    input: Option<PathBuf>,
    #[structopt(short = "m", long = "method", default_value = "\t")]
    method: OptimizeMethod,
    /// Filters out available intersections to contain only those
    /// having max a given number of terms.
    #[structopt(long = "max")]
    max_terms: Option<u32>,
    /// Times each selection instead of printing the result.
    #[structopt(long = "time", conflicts_with = "terse")]
    time: bool,
    /// Scale costs by this factor.
    #[structopt(long = "scale", conflicts_with = "scale-by-query-len")]
    scale: Option<f32>,
    /// Scale costs by this factor.
    #[structopt(long = "scale-by-query-len", conflicts_with = "scale")]
    scale_by_query_length: bool,
}

fn reader(input: Option<&PathBuf>) -> Box<dyn BufRead> {
    match input {
        None => Box::new(BufReader::new(io::stdin())),
        Some(path) => Box::new(BufReader::new(
            fs::File::open(path).expect("Unable to read input file"),
        )),
    }
}

#[paw::main]
fn main(args: Args) -> Result<(), Error> {
    let input_records = Deserializer::from_reader(reader(args.input.as_ref())).into_iter::<Value>();
    for record in input_records {
        let mut record = record?;
        let query_record = record
            .as_object_mut()
            .ok_or_else(|| format_err!("Failed to parse query object"))?;
        let threshold = Score(
            query_record
                .get("threshold")
                .ok_or_else(|| format_err!("Missing threshold"))?
                .as_f64()
                .ok_or_else(|| format_err!("Failed to parse threshold"))? as f32,
        );
        let intersections: Vec<IntersectionInput> = from_value(
            query_record
                .get("intersections")
                .ok_or_else(|| format_err!("Missing `intersections' field"))?
                .clone(),
        )?;
        let mut index = if let Some(max) = args.max_terms {
            Index::from_iter(intersections.into_iter().filter(
                |IntersectionInput { intersection, .. }| intersection.0.count_ones() <= max,
            ))
        } else {
            Index::from_iter(intersections.into_iter())
        };
        if let Some(factor) = args.scale {
            index.scale_costs(factor);
        } else if args.scale_by_query_length {
            index.scale_costs(index.query_len as f32);
        }
        let now = std::time::Instant::now();
        //let mut optimal = if index.query_length() < 8 {
        //    index.optimize_or_unigram(threshold, OptimizeMethod::Exact)
        //} else {
        let mut optimal = index.optimize_or_unigram(threshold, args.method);
        //};
        let elapsed = now.elapsed().as_micros() as u64;
        optimal.sort();
        if args.time {
            query_record.insert("elapsed".to_string(), to_value(elapsed)?);
        } else {
            query_record.insert("selections".to_string(), to_value(optimal)?);
        }
        println!("{}", record.to_string());
    }
    Ok(())
}
