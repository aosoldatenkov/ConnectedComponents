# conncomp

Computer search for real algebraic varieties with many connected components.

The current target is **Hessian curves**. For a real polynomial f(x, y) of degree d,
the Hessian curve is Hess(f) = f_xx f_yy − f_xy² = 0, which has degree 2d − 4. Its image on the graph
z = f(x, y) is the parabolic curve. We look for f whose Hessian curve has as many
ovals as possible, in connection with Arnold's problem on the topology of parabolic curves
and the work of Ortiz-Rodríguez (and Sottile). For reference, the best known examples so far are
4 ovals (d = 4), 8 ovals (d = 5) and 11 ovals (d = 6) [Ortiz-Rodríguez–Sottile, *Real Hessian Curves*].
The longer-term goal is higher-dimensional varieties (e.g. cubic surfaces).

## Method

1. Forms f(x, y, z) of degree d are sampled from the unit sphere in coefficient space, either uniformly
   or as perturbations of good earlier candidates.
2. The coefficients of Hess(f) are computed with a symbolic map derived once per degree by SymPy.
3. Hess(f) is evaluated in batches by PyTorch (on the GPU if one is available). The evaluation grid is a
   W × W pixel grid in a stereographic chart of S², with antipodal points identified, so it models RP².
4. The sign regions {Hess(f) ≥ 0} and {Hess(f) < 0} are labelled with union-find by a multithreaded
   C++ extension. Components smaller than a pixel threshold are dropped as discretization noise.
   An even-degree curve with k ovals in RP² has k + 1 complementary regions.

## Layout

```
src/conncomp/
  _connected.cpp   C++/pybind11 union-find component labelling (built automatically)
  polynomials.py   monomials, sampling, batched evaluation
  hessian.py       symbolic Hessian map and its batched application
  grid.py          RP^2 pixel grid and 12-neighbour pattern (8 adjacent + 4 antipodal)
  components.py    Python wrapper around the extension
  scan.py          Experiment class, batch/center scans
  search.py        interactive curses search (`conncomp-search`)
  plotting.py      sign-region plots
  io.py            coefficient files
scripts/explore.py           exploratory #%% cells
benchmarks/bench_components.py
tests/                       pytest suite
data/                        coefficient vectors found by searches (one per line)
figures/                     plots of found curves
papers/                      reference papers (local only, not tracked by git)
```

Coefficient vectors are listed in the order of `monomials(d)`, so a file with 21 columns holds
quintics and one with 28 columns holds sextics.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a C++17 compiler (CMake is installed automatically as a build dependency).

```bash
uv sync                 # creates .venv, installs deps (PyTorch with CUDA 12.8), builds the C++ extension
uv run pytest           # run tests
```

The extension is rebuilt automatically by `uv sync` / `uv run` whenever a `.cpp` file,
`CMakeLists.txt` or `pyproject.toml` changes. To force a rebuild, run `uv sync --reinstall-package conncomp`.

PyTorch is installed from the CUDA 12.8 index, which supports Blackwell (RTX 50xx) GPUs. On a machine
without an NVIDIA GPU, change the index in `pyproject.toml` to `https://download.pytorch.org/whl/cpu`.

## Usage

```bash
uv run conncomp-search --deg 6 --width 100 --batch 10000   # interactive search for Hessians of sextics; q to stop
uv run conncomp-search --deg 5 --no-hessian                # components of f itself
uv run python benchmarks/bench_components.py
```

```python
from conncomp.scan import batch_scan
from conncomp.plotting import plot_negative_regions

found = batch_scan(deg=5, width=200, nsamples=10000, niter=10, lo=9)   # {component count: [coefs]}
plot_negative_regions(5, 300, found[9][:4])
```
