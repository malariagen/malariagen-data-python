from packaging.version import Version, parse

import malariagen_data


def test_version():
    assert hasattr(malariagen_data, "__version__")
    assert isinstance(malariagen_data.__version__, str)
    version = parse(malariagen_data.__version__)
    assert isinstance(version, Version)
