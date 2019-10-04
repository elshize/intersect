use failure::{bail, format_err, Error, ResultExt};
use intersect::{Cost, Index, Intersection, OptimizeMethod, Score};
use itertools::Itertools;
use std::collections::HashMap;
use std::io::{self, prelude::*, BufReader};
use std::iter::FromIterator;
use std::path::{Path, PathBuf};
use std::{fs, iter};

#[derive(structopt::StructOpt)]
struct Args {
    /// Input file, stdin if not present
    #[structopt(parse(from_os_str))]
    input: Option<PathBuf>,
    /// Input file, stdin if not present
    #[structopt(parse(from_os_str), long = "thresholds", short = "t")]
    thresholds: PathBuf,
    /// Column separator, a tab by default
    #[structopt(long = "sep", default_value = "\t")]
    separator: String,
    #[structopt(short = "m", long = "method", default_value = "\t")]
    method: OptimizeMethod,
}

fn split_columns<'a>(
    line: &'a str,
    sep: &str,
    expected_columns: usize,
) -> Result<Vec<&'a str>, Error> {
    let split: Vec<&str> = line.split(sep).collect();
    if split.len() != expected_columns {
        bail!(
            "Wrong number of columns in line: {} (separator is: {})",
            line,
            sep
        );
    }
    Ok(split)
}

fn parse_line(line: &str, sep: &str) -> Result<(String, (Intersection, Cost, Score)), Error> {
    let split: Vec<&str> = split_columns(line, sep, 4)?;
    Ok((
        String::from(split[0]),
        (
            Intersection(split[1].parse()?),
            Cost(split[2].parse()?),
            Score(split[3].parse()?),
        ),
    ))
}

fn line_parser<'a>(
    sep: &'a str,
) -> impl 'a + FnMut(io::Result<String>) -> (String, (Intersection, Cost, Score)) {
    move |line: io::Result<String>| {
        parse_line(&line.expect("Error while reading lines"), sep)
            .unwrap_or_else(|err| panic!("{}", err))
    }
}

fn line_iterator<'a>(
    input: Option<&'a PathBuf>,
    sep: &'a str,
) -> iter::Peekable<impl 'a + Iterator<Item = (String, (Intersection, Cost, Score))>> {
    reader(input).lines().map(line_parser(sep)).peekable()
}

fn load_thresholds(path: &Path, sep: &str) -> Result<HashMap<String, Score>, Error> {
    let mut thresholds: HashMap<String, Score> = HashMap::new();
    for line in BufReader::new(fs::File::open(path)?).lines() {
        let line = line?;
        let split: Vec<&str> = split_columns(&line, sep, 2)?;
        let threshold: f32 = split[1].parse()?;
        thresholds.insert(split[0].to_string(), Score(threshold));
    }
    Ok(thresholds)
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
    let thresholds =
        load_thresholds(&args.thresholds, &args.separator).context("Unable to parse thresholds")?;
    let mut lines = line_iterator(args.input.as_ref(), &args.separator);
    while let Some((query, inter)) = lines.next() {
        let iter = iter::once(inter).chain(
            lines
                .peeking_take_while(|(q, _)| q == &query)
                .map(|(_, i)| i),
        );
        let mut index = Index::from_iter(iter);
        let &threshold = thresholds
            .get(&query)
            .ok_or_else(|| format_err!("Missing threshold for query: {}", query))?;
        let mut optimal = index.optimize(threshold, args.method);
        optimal.sort();
        for inter in optimal {
            println!(
                "{1}{0}{2}{0}{3}",
                &args.separator,
                query,
                inter.0,
                index.cost(inter).0,
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use tempdir::TempDir;

    #[rstest::fixture]
    fn tmp() -> TempDir {
        TempDir::new("").expect("Unable to create temp dir")
    }

    #[test]
    fn test_split_columns() {
        assert!(split_columns("line", "\t", 2).is_err());
        assert_eq!(&split_columns("line", "\t", 1).unwrap(), &["line"]);
        assert_eq!(&split_columns("line\t2", "\t", 2).unwrap(), &["line", "2"]);
        assert!(&split_columns("line\t2", "\t", 1).is_err());
    }

    #[test]
    fn test_parse_line() {
        assert!(&parse_line("1\t4\t4.0", "\t").is_err());
        assert_eq!(
            parse_line("1\t4\t4.0\t5.1", "\t").unwrap(),
            (String::from("1"), (Intersection(4), Cost(4.0), Score(5.1)))
        );
    }

    #[test]
    fn test_line_parser() {
        let parser = line_parser("\t");
        let expected: Vec<_> =
            iter::repeat((String::from("1"), (Intersection(4), Cost(4.0), Score(5.1))))
                .take(4)
                .collect();
        let lines: Vec<Result<String, io::Error>> = iter::repeat("1\t4\t4.0\t5.1".to_string())
            .take(4)
            .map(Ok)
            .collect();
        let actual: Vec<_> = lines.into_iter().map(parser).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    #[should_panic]
    fn test_line_parser_panic() {
        let parser = line_parser("\t");
        let lines: Vec<Result<String, io::Error>> = iter::repeat("1 4 4.0".to_string())
            .take(4)
            .map(Ok)
            .collect();
        let _: Vec<_> = lines.into_iter().map(parser).collect();
    }

    #[rstest::rstest]
    fn test_line_iterator(tmp: TempDir) {
        let path = tmp.path().join("file");
        std::fs::write(
            &path,
            "1 4 4.0 5.1
1 2 4.1 5.2
2 3 3.3 3.9",
        )
        .expect("Unable to write to temp file");
        let mut iter = line_iterator(Some(&path), " ");
        let expected = vec![
            ("1".to_string(), (Intersection(4), Cost(4.0), Score(5.1))),
            ("1".to_string(), (Intersection(2), Cost(4.1), Score(5.2))),
            ("2".to_string(), (Intersection(3), Cost(3.3), Score(3.9))),
        ];
        assert_eq!(iter.peek(), Some(&expected[0]));
        assert_eq!(iter.collect::<Vec<_>>(), expected);
    }

    #[rstest::rstest]
    fn test_load_thresholds(tmp: TempDir) {
        let path = tmp.path().join("file");
        std::fs::write(
            &path,
            "1 4.0
2 3.0
5 8.8",
        )
        .expect("Unable to write to temp file");
        let thresholds = load_thresholds(&path, " ").unwrap();
        assert_eq!(thresholds.len(), 3);
        assert_eq!(thresholds["1"], Score(4.0));
        assert_eq!(thresholds["2"], Score(3.0));
        assert_eq!(thresholds["5"], Score(8.8));
    }

    #[rstest::rstest]
    fn test_load_thresholds_invalid(tmp: TempDir) {
        let path = tmp.path().join("file");
        std::fs::write(
            &path,
            "1 4.0
2 3.0
5 8.8 0", // Invalid line
        )
        .expect("Unable to write to temp file");
        assert!(load_thresholds(&path, " ").is_err());
    }
}
