import ast
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[2] / "app"


def _app_modules() -> dict[str, Path]:
    modules: dict[str, Path] = {}
    for path in APP_DIR.rglob("*.py"):
        parts = list(path.relative_to(APP_DIR).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        module = "app" + (f".{'.'.join(parts)}" if parts else "")
        modules[module] = path
    return modules


def _import_targets(
    source: str,
    package: str,
    modules: dict[str, Path],
    *,
    filename: str = "<source>",
) -> set[str]:
    tree = ast.parse(source, filename=filename)
    targets: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package_parts = package.split(".")
                parent_parts = package_parts[: len(package_parts) - node.level + 1]
                base = ".".join(parent_parts)
                target = f"{base}.{node.module}" if node.module and base else (
                    node.module or base
                )
            else:
                target = node.module or ""

            if target:
                targets.add(target)
                targets.update(
                    candidate
                    for alias in node.names
                    if (candidate := f"{target}.{alias.name}") in modules
                )

    return targets


def _dependencies(module: str, path: Path, modules: dict[str, Path]) -> set[str]:
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    return {
        dependency
        for dependency in _import_targets(
            path.read_text(encoding="utf-8"),
            package,
            modules,
            filename=str(path),
        )
        if dependency in modules and dependency != module
    }


def _prohibited_imports(
    source: str,
    package: str,
    modules: dict[str, Path],
) -> set[str]:
    forbidden = (
        "app.itinerary_planning.infrastructure",
        "app.infrastructure",
        "app.models",
        "sqlalchemy",
        "httpx",
        "app.interfaces",
        "app.agents",
    )
    return {
        target
        for target in _import_targets(source, package, modules)
        if any(target == prefix or target.startswith(f"{prefix}.") for prefix in forbidden)
    }


def test_app_module_import_graph_is_acyclic() -> None:
    modules = _app_modules()
    graph = {
        module: _dependencies(module, path, modules)
        for module, path in modules.items()
    }
    visited: set[str] = set()
    active: list[str] = []

    def visit(module: str) -> None:
        if module in active:
            start = active.index(module)
            cycle = " -> ".join(active[start:] + [module])
            raise AssertionError(f"Application import cycle detected: {cycle}")
        if module in visited:
            return

        active.append(module)
        for dependency in graph[module]:
            visit(dependency)
        active.pop()
        visited.add(module)

    for module in graph:
        visit(module)


def test_context_applications_do_not_depend_on_infrastructure_or_global_schemas() -> None:
    modules = _app_modules()
    forbidden = (
        "sqlalchemy",
        "app.infrastructure.persistence.postgres.database",
        "app.infrastructure",
        "app.models",
        "app.infrastructure.embeddings.text",
        "app.infrastructure.embeddings.image",
        "app.schemas",
    )
    context_applications = (
        "app.restaurant_discovery.application.",
        "app.knowledge.application.",
    )
    violations = []
    for module, path in modules.items():
        if not module.startswith(context_applications):
            continue
        for dependency in _dependencies(module, path, modules):
            if any(
                dependency == prefix or dependency.startswith(f"{prefix}.")
                for prefix in forbidden
            ):
                violations.append(f"{path}: {dependency}")

    assert not violations, (
        "A context application depends on infrastructure or global schemas:\n"
        + "\n".join(sorted(violations))
    )


def test_itinerary_planning_application_does_not_depend_on_transport_or_agents() -> None:
    modules = _app_modules()
    violations = []
    for module, path in modules.items():
        if (
            module != "app.itinerary_planning.application"
            and not module.startswith("app.itinerary_planning.application.")
        ):
            continue
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        violations.extend(
            f"{path}: {dependency}"
            for dependency in _prohibited_imports(
                path.read_text(encoding="utf-8"),
                package,
                modules,
            )
        )

    assert not violations, (
        "Itinerary Planning application depends on transport or Agent schemas:\n"
        + "\n".join(sorted(violations))
    )


def test_planning_import_rule_detects_absolute_and_relative_prohibited_imports() -> None:
    modules = _app_modules()
    source = """
import httpx
from sqlalchemy.orm import Session
from app.infrastructure.persistence.postgres.database import get_db
from app.models.osm_place import OsmPlace
from app.interfaces.a2a import itinerary_planner_server
from app.agents.schemas import RestaurantSearchAgentResponse
from ..infrastructure.a2a import restaurant_discovery_client
from ...infrastructure.persistence.postgres.database import get_db as relative_db
from ...models.osm_place import OsmPlace as RelativeOsmPlace
from ...interfaces.a2a import itinerary_planner_server as relative_server
from ...agents.schemas import RestaurantSearchAgentResponse as RelativeResponse
"""

    detected = _prohibited_imports(
        source,
        "app.itinerary_planning.application",
        modules,
    )

    assert {
        "httpx",
        "sqlalchemy.orm",
        "app.infrastructure.persistence.postgres.database",
        "app.itinerary_planning.infrastructure.a2a",
        "app.models.osm_place",
        "app.interfaces.a2a",
        "app.agents.schemas",
    } <= detected


def test_postgres_ingestion_adapter_does_not_import_the_external_osm_provider() -> None:
    modules = _app_modules()
    module = "app.infrastructure.persistence.postgres.ingestion"
    path = modules[module]
    targets = _import_targets(
        path.read_text(encoding="utf-8"),
        module.rpartition(".")[0],
        modules,
        filename=str(path),
    )

    assert not any(
        target == "app.infrastructure.external_data.open_data_sources"
        or target.startswith("app.infrastructure.external_data.open_data_sources.")
        for target in targets
    )
