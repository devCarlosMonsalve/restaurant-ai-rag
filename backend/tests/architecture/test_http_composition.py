import ast
from pathlib import Path


MAIN_PATH = Path(__file__).resolve().parents[2] / "app" / "main.py"


def test_http_entrypoint_does_not_import_orm_models_or_query_builder() -> None:
    tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"), filename=str(MAIN_PATH))
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "app.models" or module.startswith("app.models."):
                violations.append(f"ORM model import: {module}")
            if module == "sqlalchemy" and any(
                alias.name in {"select", "insert", "update", "delete"}
                for alias in node.names
            ):
                violations.append(f"SQL query builder import: {module}")

    assert not violations, "HTTP entrypoint contains persistence queries:\n" + "\n".join(
        violations
    )
