"""Render examples notebooks into generated docs/examples pages.

Run before ``zensical build``. Sources remain unchanged; rendered pages,
figures, and downloadable notebook copies are ignored by Git. Notebooks are
executed unless --no-execute is supplied. The real-data notebook is always
rendered from saved outputs.
"""

import argparse
import re
import shutil
from pathlib import Path

import nbformat
from nbconvert import MarkdownExporter
from nbconvert.writers import FilesWriter

#: How long a single notebook may take before the build gives up.
EXECUTE_TIMEOUT_SECONDS = 900
SOURCE_DIR = Path(__file__).resolve().parents[1] / "examples"
OUTPUT_DIR = Path(__file__).resolve().parent / "examples"
NOT_EXECUTED = frozenset({"dmri/05_example_data.ipynb"})
NOTEBOOK_ORDER = {"symbolic/minimal.ipynb": 0}


def _exporter_config():
    """Exporter settings, as a traitlets Config.

    Must be a `Config` rather than a plain dict: a dict reaches the exporter's
    own traits but is silently ignored by nested preprocessors, which is how
    the SVG setting below can look applied while doing nothing.
    """
    from traitlets.config import Config

    config = Config()
    config.MarkdownExporter.exclude_input_prompt = True
    config.MarkdownExporter.exclude_output_prompt = True
    # Extract SVG alongside raster figures so the rendered pages can display it.
    config.ExtractOutputPreprocessor.extract_output_types = {
        "image/png",
        "image/jpeg",
        "image/svg+xml",
        "application/pdf",
    }
    return config


def execute(notebook, notebook_path: Path):
    """Run every cell, leaving the outputs on the in-memory notebook."""
    from nbclient import NotebookClient

    client = NotebookClient(
        notebook,
        timeout=EXECUTE_TIMEOUT_SECONDS,
        kernel_name="python3",
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    client.execute()
    return notebook


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="prism._docs_render", description=__doc__)
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help="Render committed outputs without running any notebook.",
    )
    args = parser.parse_args(argv)

    notebook_paths = sorted(
        (
            path
            for path in SOURCE_DIR.rglob("*.ipynb")
            if not any(
                part.startswith(".") for part in path.relative_to(SOURCE_DIR).parts
            )
        ),
        key=lambda path: (
            path.relative_to(SOURCE_DIR).parent.as_posix(),
            NOTEBOOK_ORDER.get(path.relative_to(SOURCE_DIR).as_posix(), 0),
            path.name,
        ),
    )
    if not notebook_paths:
        raise FileNotFoundError(f"No example notebooks found in {SOURCE_DIR}")
    for notebook_path in notebook_paths:
        relative_path = notebook_path.relative_to(SOURCE_DIR)
        output_dir = OUTPUT_DIR / relative_path.parent
        downloads = output_dir / "downloads"
        downloads.mkdir(parents=True, exist_ok=True)
        output_files_dir = f"{notebook_path.stem}_files"
        shutil.rmtree(output_dir / output_files_dir, ignore_errors=True)

        notebook = nbformat.read(notebook_path, as_version=4)
        nbformat.validate(notebook)

        skipped = args.no_execute or relative_path.as_posix() in NOT_EXECUTED
        print(
            f"{'rendering saved outputs for' if skipped else 'executing'} "
            f"{relative_path.as_posix()}",
            flush=True,
        )
        if not skipped:
            notebook = execute(notebook, notebook_path)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs = [
                    output
                    for output in cell.outputs
                    if not (output.output_type == "stream" and output.name == "stderr")
                ]
        exporter = MarkdownExporter(config=_exporter_config())
        body, resources = exporter.from_notebook_node(
            notebook,
            resources={"output_files_dir": output_files_dir},
        )
        # Notebook sources link from examples/ to docs/; pages live inside docs/.
        parent_prefix = "../" * (len(relative_path.parent.parts) + 1)
        body = body.replace(f"]({parent_prefix}docs/", f"]({parent_prefix}")
        body = re.sub(r"\]\(([^):#]+)\.ipynb(#[^)]*)?\)", r"](\1.md\2)", body)
        body = add_figure_alt_text(body)
        title, separator, content = body.partition("\n")
        body = (
            f"<!-- Generated from examples/{relative_path.as_posix()}. -->\n\n"
            f"{title}{separator}\n"
            f"[Download notebook](downloads/{notebook_path.name}){{ .md-button }}\n\n"
            f"{content.lstrip()}"
        )
        FilesWriter(build_directory=str(output_dir)).write(
            body,
            resources,
            notebook_name=notebook_path.stem,
        )
        nbformat.write(notebook, downloads / notebook_path.name)


def add_figure_alt_text(markdown: str) -> str:
    """Describe exported figures using their nearest section heading."""
    heading = "Notebook output"
    lines = []
    for line in markdown.splitlines():
        if line.startswith("## "):
            heading = re.sub(r"[`*_]", "", line.removeprefix("## "))
        line = re.sub(r"!\[(?:png|svg)\]", f"![{heading}]", line)
        lines.append(line)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
