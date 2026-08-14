# Replication Paper Selection

## Selected paper

Michael Schneider, Andreas Stenger, and Dominik Goeke (2014), "The Electric
Vehicle-Routing Problem with Time Windows and Recharging Stations,"
*Transportation Science*, 48(4), 500-520.

- DOI: <https://doi.org/10.1287/trsc.2013.0490>
- Publisher page: <https://pubsonline.informs.org/doi/10.1287/trsc.2013.0490>
- Benchmark data: <https://doi.org/10.17632/h3mrm5dhxw.1>
- Citation metadata: [Semantic Scholar](https://www.semanticscholar.org/paper/bcd897026535d3c38d3b143473bcfd73b0c0a85a)
- Bibliographic record: [RWTH Aachen University](https://publications.rwth-aachen.de/record/661591)

## Why this paper was selected

1. **Exact topic match.** The paper introduces the Electric Vehicle-Routing
   Problem with Time Windows and Recharging Stations (E-VRPTW), which directly
   matches this repository's research topic.
2. **Strong research venue.** INFORMS describes *Transportation Science* as the
   flagship journal of its Transportation Science and Logistics Society and a
   foremost journal for transportation analysis.
3. **Strong institutional background.** The first author was affiliated with
   Technical University of Darmstadt when the paper was published. TU Darmstadt
   is a member of TU9, the alliance of Germany's leading universities of
   technology.
4. **High research impact.** Semantic Scholar reported 1,240 citations and
   OpenAlex reported 1,234 citations when checked on 2026-06-14. Citation counts
   vary by database and will change over time.
5. **Reproducibility.** The authors introduced public benchmark instances. The
   paper includes a mixed-integer linear programming formulation and a hybrid
   Variable Neighborhood Search/Tabu Search method, allowing both small-instance
   validation and heuristic experiments on larger instances.

## Shortlisted alternatives

Citation counts below are Semantic Scholar values checked on 2026-06-14. They
are included for comparison, not as permanent metrics.

| Paper | Year | Approx. citations | Reason not selected as the primary paper |
|---|---:|---:|---|
| [The Electric Fleet Size and Mix Vehicle Routing Problem with Time Windows and Recharging Stations](https://doi.org/10.1016/j.ejor.2016.01.038) | 2016 | 569 | Strong and influential, but adds fleet-size and fleet-mix decisions beyond the repository's current core scope. |
| [Partial recharge strategies for the electric vehicle routing problem with time windows](https://doi.org/10.1016/j.trc.2016.01.013) | 2016 | 539 | Highly relevant extension, but it builds on the Schneider benchmark and is better treated as an innovation reference. |
| [The electric vehicle routing problem with nonlinear charging function](https://doi.org/10.1016/j.trb.2017.02.004) | 2017 | 517 | Important realistic charging extension, but nonlinear charging raises implementation complexity for an initial replication. |
| [Exact Algorithms for Electric Vehicle-Routing Problems with Time Windows](https://doi.org/10.1287/opre.2016.1535) | 2016 | 470 | Excellent institutions and journal, but its exact branch-price-and-cut style methods are substantially harder to reproduce in an undergraduate project. |

## Initial replication boundary

The first milestone should reproduce the selected paper's problem definition,
feasibility constraints, hierarchical objectives, benchmark parser, and results
on small instances. The larger-instance VNS/TS baseline can then be implemented
and validated before adding the required innovation.

The formal 2014 journal Version of Record supplied by the user is kept in the
project's local literature folder outside Git. Other shortlisted papers are
recorded by title and DOI only; the user is responsible for obtaining their
formal journal PDFs. Codex does not download papers or retain technical reports,
preprints, accepted manuscripts, or author manuscripts as substitutes. After
the user supplies a PDF, it must be verified as the Version of Record before it
is added to the local collection. The local `document/literature` shortcut is
excluded by `.gitignore`.

## Evidence sources

- INFORMS journal statement: <https://pubsonline.informs.org/page/trsc/editorial-statement>
- TU9 member list: <https://www.tu9.de/en/>
- OpenAlex work record: <https://api.openalex.org/works/https://doi.org/10.1287/trsc.2013.0490>
- Semantic Scholar work record: <https://api.semanticscholar.org/graph/v1/paper/DOI:10.1287/trsc.2013.0490?fields=title,authors,year,citationCount,influentialCitationCount,venue,externalIds,openAccessPdf>
