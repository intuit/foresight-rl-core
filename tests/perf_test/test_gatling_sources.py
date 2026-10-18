"""Structural checks on the vendored Gatling Java sources.

There is no JDK in CI, so nothing here compiles Java. What it does catch is the
class of breakage a package rename or a file move actually causes: a package
declaration that no longer matches its directory, a public type whose name
drifted from its filename, an import pointing at a type that moved, and -- the
reason the sources were split in the first place -- Intuit-specific identifiers
reappearing in the OSS tree.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pytest

from foresightrlcore.config.perf_schema import GatlingConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OSS_JAVA_DIR = (
    _REPO_ROOT
    / "foresightrlcore"
    / "perf_test"
    / "gatling"
    / "src"
    / "gatling"
    / "java"
)
_OSS_PACKAGE_ROOT = "org.foresightrl.perf"


def _extension_java_dir() -> Optional[Path]:
    """Resolve the extension's Gatling sources the way the runner does.

    Going through the import system rather than a hardcoded ``intuit/`` path
    keeps this honest for an installed wheel as well as a monorepo checkout,
    and yields ``None`` -- rather than a path that isn't there -- when the
    extension ships no Java or isn't installed.
    """
    spec = importlib.util.find_spec("foresightrl_intuit")
    if spec is None or not spec.submodule_search_locations:
        return None
    candidate = (
        Path(list(spec.submodule_search_locations)[0]) / "perf_test" / "gatling_java"
    )
    return candidate if candidate.is_dir() else None


_PACKAGE_RE = re.compile(r"^package\s+([\w.]+);", re.MULTILINE)
_TYPE_RE = re.compile(
    r"^public\s+(?:final\s+|abstract\s+)?(?:class|interface|enum)\s+(\w+)",
    re.MULTILINE,
)
_IMPORT_RE = re.compile(r"^import\s+(?:static\s+)?([\w.]+);", re.MULTILINE)

#: Substrings that must never appear in the OSS Gatling sources. These are the
#: things that made the original tree unpublishable: a vendor namespace, an
#: internal project/ticket name, and a served-model name from one deployment.
_FORBIDDEN_IN_OSS = ("intuit", "film", "delphiai", "chat-film")


def _java_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.java"))


def _declarations(root: Path) -> Iterator[Tuple[Path, str, str]]:
    """Yield ``(path, package, public_type_name)`` for each source file."""
    for path in _java_files(root):
        text = path.read_text(encoding="utf-8")
        package_match = _PACKAGE_RE.search(text)
        type_match = _TYPE_RE.search(text)
        assert package_match, f"{path} has no package declaration"
        assert type_match, f"{path} declares no public type"
        yield path, package_match.group(1), type_match.group(1)


def _declared_types(root: Path) -> Dict[str, Path]:
    return {f"{pkg}.{name}": path for path, pkg, name in _declarations(root)}


class TestOssGatlingSources:
    def test_sources_are_present(self):
        """A missing tree would make every other check here vacuously pass."""
        assert _java_files(_OSS_JAVA_DIR), f"no Java sources under {_OSS_JAVA_DIR}"

    def test_package_matches_directory(self):
        for path, package, _ in _declarations(_OSS_JAVA_DIR):
            expected = _OSS_JAVA_DIR.joinpath(*package.split("."))
            assert path.parent == expected, (
                f"{path} declares package {package} but sits in {path.parent}; "
                f"javac requires {expected}"
            )

    def test_public_type_matches_filename(self):
        for path, _, type_name in _declarations(_OSS_JAVA_DIR):
            assert type_name == path.stem, (
                f"{path} declares public type {type_name}; javac requires the "
                f"file to be named {type_name}.java"
            )

    def test_everything_lives_under_the_neutral_package(self):
        for _, package, _ in _declarations(_OSS_JAVA_DIR):
            assert package.startswith(_OSS_PACKAGE_ROOT), (
                f"{package} is outside {_OSS_PACKAGE_ROOT}; OSS Gatling sources "
                "must not carry a vendor namespace"
            )

    def test_no_vendor_identifiers(self):
        offenders = []
        for path in _java_files(_OSS_JAVA_DIR):
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                lowered = line.lower()
                for token in _FORBIDDEN_IN_OSS:
                    if token in lowered:
                        rel = path.relative_to(_REPO_ROOT)
                        offenders.append(f"{rel}:{lineno}: {token!r} in {line.strip()}")
        assert (
            not offenders
        ), "vendor identifiers in OSS Gatling sources:\n" + "\n".join(offenders)

    def test_internal_imports_resolve(self):
        declared = _declared_types(_OSS_JAVA_DIR)
        for path in _java_files(_OSS_JAVA_DIR):
            text = path.read_text(encoding="utf-8")
            for imported in _IMPORT_RE.findall(text):
                if not imported.startswith(_OSS_PACKAGE_ROOT):
                    continue
                assert (
                    imported in declared
                ), f"{path} imports {imported}, which no source file declares"

    def test_default_simulation_class_exists(self):
        """The shipped default must be runnable with no extension installed."""
        default = GatlingConfig(data_s3_path="s3://bucket/data/").simulation_class
        assert default in _declared_types(_OSS_JAVA_DIR), (
            f"default simulation_class {default} is not declared in the OSS "
            "Gatling project, so a stock perf run would fail to start"
        )


@pytest.fixture
def extension_java_dir() -> Path:
    java_dir = _extension_java_dir()
    if java_dir is None:
        pytest.skip("the foresightrl-intuit extension ships no Gatling sources here")
    return java_dir


@pytest.mark.requires_intuit
class TestExtensionGatlingSources:
    """The extension's sources must still compile against the OSS ones."""

    def test_sources_are_present(self, extension_java_dir):
        assert _java_files(extension_java_dir)

    def test_package_matches_directory(self, extension_java_dir):
        for path, package, _ in _declarations(extension_java_dir):
            expected = extension_java_dir.joinpath(*package.split("."))
            assert (
                path.parent == expected
            ), f"{path} declares package {package} but sits in {path.parent}"

    def test_public_type_matches_filename(self, extension_java_dir):
        for path, _, type_name in _declarations(extension_java_dir):
            assert type_name == path.stem

    def test_imports_of_oss_types_resolve(self, extension_java_dir):
        """Catches an OSS rename that silently orphans the extension."""
        oss_types = _declared_types(_OSS_JAVA_DIR)
        checked = 0
        for path in _java_files(extension_java_dir):
            for imported in _IMPORT_RE.findall(path.read_text(encoding="utf-8")):
                if not imported.startswith(_OSS_PACKAGE_ROOT):
                    continue
                assert imported in oss_types, (
                    f"{path} imports {imported}, which the OSS Gatling project "
                    "no longer declares"
                )
                checked += 1
        assert checked, "expected the extension to import at least one OSS type"
