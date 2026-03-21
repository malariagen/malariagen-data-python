import ast
from pathlib import Path


def _module_ast(path: str):
    src = Path(path).read_text(encoding="utf-8")
    return ast.parse(src)


def _module_string_constants(mod: ast.Module):
    out = {}
    for node in mod.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            out[node.targets[0].id] = node.value.value
    return out


def _class_assignments(mod: ast.Module, class_name: str):
    class_node = next(
        node
        for node in mod.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    out = {}
    for node in class_node.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            out[node.targets[0].id] = node.value
    return out


def _resolve_value(value_node, constants):
    if isinstance(value_node, ast.Constant) and isinstance(value_node.value, str):
        return value_node.value
    if isinstance(value_node, ast.Name):
        return constants[value_node.id]
    raise TypeError(f"Unsupported value node type: {type(value_node)}")


def _assert_cache_name_attrs(*, path: str, class_name: str, prefix: str):
    required_attrs = (
        "_xpehh_gwss_cache_name",
        "_ihs_gwss_cache_name",
        "_roh_hmm_cache_name",
    )
    mod = _module_ast(path)
    constants = _module_string_constants(mod)
    class_values = _class_assignments(mod, class_name)

    for attr in required_attrs:
        assert attr in class_values
        value = _resolve_value(class_values[attr], constants)
        assert value.startswith(f"{prefix}_")
        assert value.endswith("_v1")


def test_ag3_cache_names():
    _assert_cache_name_attrs(
        path="malariagen_data/ag3.py",
        class_name="Ag3",
        prefix="ag3",
    )


def test_af1_cache_names():
    _assert_cache_name_attrs(
        path="malariagen_data/af1.py",
        class_name="Af1",
        prefix="af1",
    )


def test_amin1_cache_names():
    _assert_cache_name_attrs(
        path="malariagen_data/amin1.py",
        class_name="Amin1",
        prefix="amin1",
    )


def test_adir1_cache_names():
    _assert_cache_name_attrs(
        path="malariagen_data/adir1.py",
        class_name="Adir1",
        prefix="adir1",
    )
