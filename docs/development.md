# Development

Install the development and documentation dependencies from a repository clone:

```bash
uv sync --locked --extra dev --extra docs
```

This uses the dependency lock and the validated Python 3.12.14 version in
`.python-version`. Python 3.11 remains supported. Commands below use
`uv run`; include any CUDA extra you need in the sync command.

## Checks

Run the test suite:

```bash
uv run pytest
```

Renaming or removing a module leaves its `__pycache__` behind. A directory that
holds only stale bytecode, or that is left empty once the cache is cleared,
still imports as a namespace package, so a path that no longer exists appears
to import successfully. `git clean -fd` does not remove it, because the cache
is ignored. After switching between branches that move modules, clear both:

```bash
find . -name __pycache__ -type d -not -path "./.venv/*" -exec rm -rf {} +
find prism tests experiments -type d -empty -delete
```

Run formatting, lint, and repository hooks:

```bash
uv run ruff format --check prism tests examples experiments
uv run ruff check prism tests examples experiments
uv run pre-commit run --all-files
```

## Documentation

For prose edits, render saved notebook outputs and build the site:

```bash
uv run python docs/render_examples.py --no-execute
uv run zensical build --clean --strict
```

When notebook code or the APIs it exercises change, execute the notebooks
before building:

```bash
uv run python docs/render_examples.py
uv run zensical build --clean --strict
```

The renderer discovers notebooks recursively under `examples/` and preserves
their subdirectories in `docs/examples/`. Each source produces a Markdown page,
figures, and a downloadable copy. These outputs are ignored by Git; only the
index is maintained there. Source notebooks are left unchanged. The site is
written to `site/`.

The Docs CI workflow executes the symbolic notebooks and dMRI notebooks 01–04,
then checks a strict build without publishing it. Notebook 03 requires the
public pretrained checkpoint, which is cached after its first download.
Notebook 05 is always rendered from saved outputs: its data download and
full-volume prediction commands are run manually, outside the normal CI build.
The symbolic `minimal` notebook creates and restores its own temporary EMA
checkpoint within a single run.

## Examples

Run notebooks from `examples/`. To execute one without overwriting its source:

```bash
uv run jupyter nbconvert \
  --execute --to notebook --stdout examples/symbolic/minimal.ipynb \
  >/dev/null
```

To refresh its saved outputs:

```bash
uv run jupyter nbconvert \
  --execute --to notebook --inplace examples/symbolic/minimal.ipynb
```

The online symbolic prior-conditioning check is also available separately:

```bash
uv run python tests/check_symbolic_prior.py --components 15
```

`--components` accepts `15`, `30`, or `50`. It trains a reduced model for four
steps, restores best EMA weights, and samples at `lambda=0.1` and `0.9`. It
writes checkpoints, `smoke.json`, and `samples.npz` below
`results/symbolic15_smoke/` for K=15; use `--output-dir` for a fresh destination.

## Documentation conventions

- Prefer runnable commands and concrete file layouts.
- Keep workflow explanation in guides and exact option lists in CLI pages.
- Document only Python functions intended for reuse.
- State hardware and data assumptions.
- Record the code revision, checkpoint selection, environment and evaluation inputs.
- Update examples when an API or configuration schema changes.
