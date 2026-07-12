# Local Research Environment

This repository uses a native Apple Silicon environment for reproducing
Schneider, Stenger, and Goeke (2014).

## Installed stack

- Python 3.13.13 managed by `uv` in `.venv`
- C++20 compiled by Apple Clang
- CMake and Ninja for native builds
- pybind11/scikit-build-core for the Python-to-C++ extension
- NumPy, SciPy, Pandas, NetworkX, Numba, Matplotlib, Seaborn, and JupyterLab
- OR-Tools and HiGHS as open-source fallback solvers
- IBM CPLEX 22.2 Python runtime and DOcplex
- Gurobi 13.0 Python runtime
- pytest, coverage, Ruff, and mypy development tools for future verification

Python 3.13.14 was requested initially, but no `uv` ARM64 build was available on
2026-06-14. The project therefore pins the latest available 3.13 patch release,
3.13.13, and deliberately does not use the system Python 3.14 installation.

## Bootstrap

```bash
uv sync --all-groups
uv run evrptw-env
```

`uv sync` creates `.venv`, resolves `uv.lock`, builds the C++ extension, and
installs all Python dependencies.

## Solver licenses

The solver Python packages are installed, but meaningful benchmark models
require unrestricted academic licences.

1. Request IBM CPLEX through the
   [IBM Academic Initiative](https://www.ibm.com/products/ilog-cplex-optimization-studio/pricing).
   Install the macOS ARM64 edition and upgrade the Python runtime with:

   ```bash
   uv run docplex config --upgrade /path/to/CPLEX_Studio
   ```

2. Request a free
   [Gurobi Academic License](https://www.gurobi.com/academics). For a single Mac,
   use a Named-User license and run the `grbgetkey` command supplied by the
   Gurobi portal. Academic verification in China may require the
   [manual application route](https://support.gurobi.com/hc/en-us/articles/33982552193937-As-a-student-at-a-university-in-China-how-do-I-request-a-Free-Academic-License).

Do not commit license files, account tokens, or solver credentials.

## Experiment profiles

- `configs/reproduction.toml`: CPLEX, one thread, seed 2014, ten repetitions,
  and the paper's 7,200-second exact-solver limit.
- `configs/performance.toml`: Gurobi primary, CPLEX cross-check, and all locally
  available CPU threads.

The paper comparison must use the reproduction profile. Performance-profile
results should be reported separately because they are not hardware-equivalent
to the original Intel i5/4GB/Windows 7 experiments.

## Benchmark data

Download the public Schneider instances from
<https://doi.org/10.17632/h3mrm5dhxw.1> and place them under
`data/schneider/`. The repository ignores `data/` so the dataset is never
uploaded accidentally.

Parse and validate an instance in Python:

```python
from evrptw import parse_schneider, validate_routes

instance = parse_schneider("data/schneider/c101C5.txt")
report = validate_routes(instance, [["D0", "C1", "S0", "D0"]])
print(report.feasible, report.total_distance)
```

## Reproducibility records

Write machine-readable environment metadata next to an experiment:

```bash
uv run evrptw-env --output results/runs/environment.json
```

The report records the OS, architecture, CPU count, Python executable, package
versions, compiler/build tools, and C++ extension. Generated files under
`results/` are ignored by Git.

## Week 5 advanced benchmark profile

The verified Week 5 ALNS/BPC run used the following fixed profile:

- macOS 27.0 on Apple Silicon (Apple M5), 10 logical CPUs, 16 GiB RAM;
- CPython 3.13.13 and Apple clang 17.0.0;
- SciPy 1.17.1 with HiGHS 1.14.0 for restricted-master LPs;
- OR-Tools 9.15.6755, Gurobi 13.0.2, and CPLEX 22.2.0.0 installed;
- one solver thread, no GPU, 30-second wall-clock limit per method/instance run;
- seeds 2014, 2015, and 2016 for stochastic methods;
- `PYTHONHASHSEED` left unset; algorithm randomness uses explicit local RNG objects.

The exact captured values, including every dependency version and executable,
are written to `results/week05_advanced/environment.json`. Reproduce the run with:

```bash
uv sync --all-groups
uv run python -m evrptw.experiments.week05_advanced_benchmark \
  --benchmark-dir data/schneider \
  --output-dir results/week05_advanced \
  --instances c101C5,r105C5,rc105C5,c104C10,r103C10,rc102C10,c106C15,r105C15,rc103C15,c101_21,r101_21,rc101_21 \
  --stress-instances c101C5,r105C5,rc105C5 \
  --seeds 2014,2015,2016 \
  --alns-iterations 1000 \
  --time-limit-seconds 30 \
  --ga-population-size 60 \
  --ga-generations 80
```

The current BPC implementation uses open-source HiGHS despite the installed
commercial runtimes. This keeps the small exact reference reproducible without
a commercial licence and does not change its enumerated-column optimality proof.
