import os
from malariagen_data.util import _init_filesystem


def test_init_filesystem_decodes_file_uri_escaped_path(tmp_path):
    dir_with_space = tmp_path / "dir with space's"
    dir_with_space.mkdir()
    file_path = dir_with_space / "v3-config.json"
    file_path.write_text('{"foo": "bar"}')

    uri = dir_with_space.as_uri()

    fs, path = _init_filesystem(uri)

    assert "%20" in uri
    assert "%20" not in path
    assert "%27" not in path

    # Using local path with os.path.join should now succeed
    with fs.open(os.path.join(path, "v3-config.json"), "r") as f:
        assert f.read() == '{"foo": "bar"}'


def test_init_filesystem_plain_local_path_unchanged(tmp_path):
    dir_with_space = tmp_path / "dir with space's"
    dir_with_space.mkdir()
    file_path = dir_with_space / "v3-config.json"
    file_path.write_text('{"foo": "bar"}')

    uri = str(dir_with_space)

    fs, path = _init_filesystem(uri)

    with fs.open(os.path.join(path, "v3-config.json"), "r") as f:
        assert f.read() == '{"foo": "bar"}'
