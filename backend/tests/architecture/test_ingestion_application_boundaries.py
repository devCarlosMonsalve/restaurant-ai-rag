import ast
from pathlib import Path


def _imported_modules(source: str) -> set[str]:
    tree = ast.parse(source)
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def test_ingestion_use_cases_depend_only_on_standard_library_and_ports() -> None:
    backend_root = Path(__file__).resolve().parents[2]
    use_case_directory = (
        backend_root / "app" / "ingestion" / "application"
    )
    use_case_files = (
        "documents_usecase.py",
        "images_usecase.py",
        "ports.py",
    )
    forbidden_prefixes = (
        "sqlalchemy",
        "app.infrastructure",
        "app.models",
        "app.embeddings",
        "app.image_embeddings",
        "app.open_data_sources",
    )

    for filename in use_case_files:
        modules = _imported_modules((use_case_directory / filename).read_text())
        assert not any(
            module == prefix or module.startswith(f"{prefix}.")
            for module in modules
            for prefix in forbidden_prefixes
        ), filename
