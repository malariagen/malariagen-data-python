from packaging.version import Version, parse

import malariagen_data

def test_version():
    assert hasattr(malariagen_data, "__version__")
    version_str = malariagen_data.__version__

    assert isinstance(version_str, str)
    assert version_str != ""

    version = parse(version_str)

    assert isinstance(version, Version)
    assert str(version) == version_str  # normalized format