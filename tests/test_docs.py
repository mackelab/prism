"""Keep the documented experiment commands and links consistent with the code."""

import importlib.util
import re
import unicodedata
from pathlib import Path

import pytest

from prism.utils.slurm import partitions

ROOT = Path(__file__).resolve().parents[1]
GITHUB = re.compile(
    r"https://github\.com/mackelab/prism/(?:tree|blob)/main/([^)#\s]*)(#[^)\s]*)?"
)
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")
FENCE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)


def markdown_files():
    files = [ROOT / "README.md", *(ROOT / "experiments").rglob("*.md")]
    files += [
        path
        for path in (ROOT / "docs").rglob("*.md")
        if "examples" not in path.relative_to(ROOT / "docs").parts
    ]
    return sorted(files)


def recipe_files():
    return sorted((ROOT / "experiments").rglob("README.md"))


def commands(path):
    """``uv run`` lines inside fenced code blocks."""
    lines = []
    for block in FENCE.findall(path.read_text()):
        lines += [line for line in block.splitlines() if line.startswith("uv run ")]
    return lines


def slugs(heading):
    """Anchor spellings used by GitHub and by the documentation site."""
    text = unicodedata.normalize("NFKD", heading).replace("`", "").strip().lower()
    github = re.sub(r"[^\w\- ]", "", text).replace(" ", "-")
    site = re.sub(r"[-\s]+", "-", re.sub(r"[^\w\s-]", "", text)).strip("-")
    return {github, site}


def anchors(path):
    text = FENCE.sub("", path.read_text())
    found = set()
    for heading in re.findall(r"^#{1,6} (.+)$", text, re.MULTILINE):
        found |= slugs(heading)
    return found


DOCUMENTED = [path for path in markdown_files() if commands(path)]


@pytest.mark.parametrize("path", DOCUMENTED, ids=lambda p: str(p.relative_to(ROOT)))
def test_documented_commands_name_existing_presets_modules_and_partitions(path):
    for line in commands(path):
        for preset in re.findall(r"\+experiment/train=(\S+)", line):
            assert (ROOT / "conf/experiment/train" / f"{preset}.yaml").is_file(), line
        for module in re.findall(r"python -m (experiments\.\S+)", line):
            assert importlib.util.find_spec(module) is not None, line
        for partition in re.findall(r"--slurm (\S+)", line):
            assert partition in partitions(), line


def test_readme_commands_match_a_recipe():
    """The root README repeats one command per step; the recipes own them."""
    text = (ROOT / "README.md").read_text()
    section = text[text.index("## Run the experiments") :]
    section = section[: section.index("\n## ", 1)]
    readme = [
        line
        for block in FENCE.findall(section)
        for line in block.splitlines()
        if line.startswith("uv run ")
    ]
    recipes = {line for path in recipe_files() for line in commands(path)}
    assert readme
    assert [line for line in readme if line not in recipes] == [
        "uv run prism train +experiment/train=symbolic"
    ]


def link_targets():
    for path in markdown_files():
        text = FENCE.sub("", path.read_text())
        for target in LINK.findall(text):
            match = GITHUB.match(target)
            if match:
                file, anchor = match.group(1), match.group(2)
                yield path, target, ROOT / file, anchor
            elif not re.match(r"[a-z]+:", target):
                file, _, anchor = target.partition("#")
                resolved = (path.parent / file).resolve() if file else path
                yield (
                    path,
                    target,
                    rendered_source(resolved),
                    ("#" + anchor if anchor else None),
                )


def rendered_source(resolved):
    """docs/examples pages are generated from examples/ notebooks.

    In a fresh checkout (as in CI) they have not been rendered yet, so a link
    to one resolves to its source notebook.
    """
    generated = ROOT / "docs" / "examples"
    if resolved.exists() or not resolved.is_relative_to(generated):
        return resolved
    if resolved.suffix != ".md":
        return resolved
    relative = resolved.relative_to(generated).with_suffix(".ipynb")
    return ROOT / "examples" / relative


@pytest.mark.parametrize(
    "path,target,resolved,anchor",
    list(link_targets()),
    ids=lambda value: str(value) if isinstance(value, str) else None,
)
def test_repository_links_and_anchors_resolve(path, target, resolved, anchor):
    assert resolved.exists(), f"{path.relative_to(ROOT)} links to missing {target}"
    if anchor:
        page = resolved / "README.md" if resolved.is_dir() else resolved
        if page.suffix == ".md":
            assert anchor[1:] in anchors(page), (
                f"{path.relative_to(ROOT)} links to missing anchor {target}"
            )
