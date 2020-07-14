This program is used to select (approximately) optimal selection of available intersections to use for safe disjunctive queries.

# Usage

```bash
intersect 0.1.0

USAGE:
    intersect [FLAGS] [OPTIONS] -k <k> [input]

FLAGS:
    -h, --help                  Prints help information
        --scale-by-query-len    Scale costs by this factor.
        --time                  Times each selection instead of printing the result.
    -V, --version               Prints version information

OPTIONS:
    -k <k>                   k value, at which to compute intersections.
        --max <max-terms>    Filters out available intersections to contain only those having max a given number of
                             terms.
    -m, --method <method>     [default: 	]
        --scale <scale>      Scale costs by this factor.

ARGS:
    <input>    Input file, stdin if not present
```
