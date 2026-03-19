from pathlib import Path

from malariagen_data.util import _init_filesystem


def test_init_filesystem_decodes_local_file_uri_path(tmp_path: Path):
    root = tmp_path / "dir with spaces and apostrophe's"
    root.mkdir()
    config_path = root / "v3-config.json"
    config_path.write_text("{}", encoding="utf-8")

    fs, path = _init_filesystem(root.as_uri())

    with fs.open(f"{path}/v3-config.json") as f:
        assert f.read() == b"{}"
