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


def _dependencies(module: str, path: Path, modules: dict[str, Path]) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    dependencies: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            dependencies.update(
                alias.name
                for alias in node.names
                if alias.name in modules and alias.name != module
            )
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

            if target in modules and target != module:
                dependencies.add(target)
            elif node.module is None and target:
                dependencies.update(
                    candidate
                    for alias in node.names
                    if (candidate := f"{target}.{alias.name}") in modules
                    and candidate != module
                )

    return dependencies


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
    forbidden = (
        "app.itinerary_planning.infrastructure",
        "app.interfaces",
        "app.agents",
    )
    violations = []
    for module, path in modules.items():
        if not module.startswith("app.itinerary_planning.application."):
            continue
        for dependency in _dependencies(module, path, modules):
            if any(
                dependency == prefix or dependency.startswith(f"{prefix}.")
                for prefix in forbidden
            ):
                violations.append(f"{path}: {dependency}")

        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported_modules = []
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)
            violations.extend(
                f"{path}: {imported_module}"
                for imported_module in imported_modules
                if imported_module == "httpx"
                or imported_module.startswith("httpx.")
            )

    assert not violations, (
        "Itinerary Planning application depends on transport or Agent schemas:\n"
        + "\n".join(sorted(violations))
    )
