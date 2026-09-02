import ast
from pathlib import Path


def test_molt_stream_imports_follow_inward_boundaries():
    root = Path("src/molt_stream")
    ranks = {
        "core": 0,
        "data": 1,
        "kernels": 1,
        "measurement": 1,
        "experiments": 1,
        "streaming": 2,
        "training": 3,
        "cli": 4,
    }
    violations = []
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        if len(relative.parts) == 1:
            source_area = "cli" if relative.name == "cli.py" else "core"
        else:
            source_area = relative.parts[0]
        source_rank = ranks.get(source_area, 0)
        tree = ast.parse(path.read_text("utf-8"))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                if not module.startswith("molt_stream."):
                    continue
                target_area = module.split(".")[1]
                target_rank = ranks.get(target_area.removesuffix(".py"), 0)
                if target_area != source_area and target_rank >= source_rank:
                    violations.append(f"{relative}: {source_area} -> {target_area}")
    assert not violations, violations
