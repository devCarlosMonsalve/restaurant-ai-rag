import ast
from pathlib import Path
import sys


APP_DIR = Path(__file__).resolve().parents[2] / "app"
DOMAIN_DIRS = [
    APP_DIR / "domain",
    APP_DIR / "restaurant_discovery" / "domain",
    APP_DIR / "knowledge" / "domain",
]
DOMAIN_IMPORT_ROOTS = (
    "app.domain",
    "app.knowledge.domain",
    "app.restaurant_discovery.domain",
)


def _domain_modules() -> dict[str, Path]:
    modules: dict[str, Path] = {}
    for domain_dir in DOMAIN_DIRS:
        for path in domain_dir.rglob("*.py"):
            relative = path.relative_to(APP_DIR).with_suffix("")
            parts = list(relative.parts)
            if parts[-1] == "__init__":
                parts.pop()
            modules["app." + ".".join(parts)] = path
    return modules


def _import_targets(path: Path, module_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    targets: set[str] = set()
    package = (
        module_name
        if path.name == "__init__.py"
        else module_name.rpartition(".")[0]
    )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package_parts = package.split(".")
                parent_parts = package_parts[: len(package_parts) - node.level + 1]
                target = ".".join(parent_parts)
                if node.module:
                    target = f"{target}.{node.module}" if target else node.module
            else:
                target = node.module or ""
            if target:
                targets.add(target)

    return targets


def test_domain_imports_respect_layer_boundaries() -> None:
    violations: list[str] = []

    for module_name, path in _domain_modules().items():
        for target in _import_targets(path, module_name):
            root = target.split(".", maxsplit=1)[0]
            if root == "app":
                if not any(
                    target == domain_root
                    or target.startswith(f"{domain_root}.")
                    for domain_root in DOMAIN_IMPORT_ROOTS
                ):
                    violations.append(f"{path}: {target}")
            elif root not in sys.stdlib_module_names:
                violations.append(f"{path}: {target}")

    assert not violations, "Domain modules crossed architectural boundaries:\n" + "\n".join(
        sorted(violations)
    )


def test_domain_internal_imports_are_acyclic() -> None:
    modules = _domain_modules()
    graph: dict[str, set[str]] = {module: set() for module in modules}

    for module_name, path in modules.items():
        for target in _import_targets(path, module_name):
            if target in modules:
                graph[module_name].add(target)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(module: str, trail: list[str]) -> None:
        if module in visiting:
            cycle_start = trail.index(module)
            cycle = " -> ".join(trail[cycle_start:] + [module])
            raise AssertionError(f"Domain import cycle detected: {cycle}")
        if module in visited:
            return

        visiting.add(module)
        trail.append(module)
        for dependency in graph[module]:
            visit(dependency, trail)
        trail.pop()
        visiting.remove(module)
        visited.add(module)

    for module in graph:
        visit(module, [])
