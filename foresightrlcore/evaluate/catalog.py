"""The benchmark catalog: benchmarks as data, not code.

A catalog is a YAML file of :class:`BenchmarkSpec` entries. Running one becomes
a single line in a config::

    benchmarks:
      - name: demo_mcq

...because everything else -- type, dataset, metrics, prompts -- is already in
the catalog. Overriding a field is a second line; the entry is merged over the
catalog record with the entry winning.

Three sources are merged, later winning over earlier:

1. the built-in OSS catalog (``catalog/benchmarks.yaml``), which contains only
   self-contained demo benchmarks;
2. catalogs registered by extension packages via :func:`register_catalog`;
3. ``evaluate.catalog_paths`` from the config.

**Placement rule:** a catalog entry naming an internal host, bucket, or account
belongs in an extension package, never in the OSS tree. ``scripts/
check_intuit_fingerprints.sh`` enforces this in CI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

import yaml

if TYPE_CHECKING:
    from foresightrlcore.config.evaluate_schema import BenchmarkSpec

logger = logging.getLogger(__name__)

BUILTIN_CATALOG = Path(__file__).parent / "catalog" / "benchmarks.yaml"

# Catalog files contributed by extension packages, in registration order.
_REGISTERED_CATALOGS: List[Path] = []


def register_catalog(path: str | Path) -> None:
    """Register an extension-provided catalog file.

    Call this from an extension's ``_register_evaluate_plugins()`` hook. Later
    registrations win over earlier ones for the same benchmark name.
    """
    resolved = Path(path)
    if resolved not in _REGISTERED_CATALOGS:
        _REGISTERED_CATALOGS.append(resolved)


def registered_catalogs() -> List[Path]:
    """Return the extension-registered catalog paths, in registration order."""
    return list(_REGISTERED_CATALOGS)


def read_catalog_file(path: str | Path) -> Dict[str, "BenchmarkSpec"]:
    """Parse one catalog YAML file into ``{name: BenchmarkSpec}``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file is malformed or an entry fails validation.
    """
    from foresightrlcore.config.evaluate_schema import BenchmarkSpec

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"benchmark catalog not found: {file_path}")

    raw = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{file_path}: catalog must be a mapping at the top level")

    entries = raw.get("benchmarks", [])
    if not isinstance(entries, list):
        raise ValueError(f"{file_path}: 'benchmarks' must be a list")

    catalog: Dict[str, BenchmarkSpec] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(
                f"{file_path}: benchmark #{index} must be a mapping, "
                f"got {type(entry).__name__}"
            )
        try:
            spec = BenchmarkSpec.model_validate(entry)
        except Exception as exc:  # noqa: BLE001 - re-raised with file context
            raise ValueError(
                f"{file_path}: benchmark #{index} is invalid: {exc}"
            ) from exc
        if spec.name in catalog:
            raise ValueError(f"{file_path}: duplicate benchmark name {spec.name!r}")
        catalog[spec.name] = spec
    return catalog


def load_catalog(
    extra_paths: Optional[Iterable[str | Path]] = None,
) -> Dict[str, "BenchmarkSpec"]:
    """Load the merged catalog: built-in, then extensions, then ``extra_paths``."""
    catalog: Dict[str, "BenchmarkSpec"] = {}

    for path in [BUILTIN_CATALOG, *_REGISTERED_CATALOGS, *(extra_paths or [])]:
        entries = read_catalog_file(path)
        overridden = sorted(set(entries) & set(catalog))
        if overridden:
            logger.info("%s overrides catalog entries: %s", path, overridden)
        catalog.update(entries)

    return catalog


def resolve_benchmark(
    entry: "BenchmarkSpec", catalog: Dict[str, "BenchmarkSpec"]
) -> "BenchmarkSpec":
    """Merge a config entry over its catalog record, if it has one.

    Raises:
        KeyError: If the entry names no catalog benchmark and defines no type.
    """
    if entry.name in catalog:
        return entry.merged_over(catalog[entry.name])
    if entry.type is None:
        raise KeyError(
            f"benchmark {entry.name!r} is not in the catalog and defines no "
            f"'type'. Known catalog benchmarks: {sorted(catalog)}"
        )
    return entry


__all__ = [
    "BUILTIN_CATALOG",
    "load_catalog",
    "read_catalog_file",
    "register_catalog",
    "registered_catalogs",
    "resolve_benchmark",
]
