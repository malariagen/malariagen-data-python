import pytest
from malariagen_data.util import SafeStore


def test_safestore_getitem_success():
    store = {"foo": "bar"}
    safe_store = SafeStore(store)
    assert safe_store["foo"] == "bar"


def test_safestore_getitem_keyerror_raises_filenotfounderror():
    store = {}
    safe_store = SafeStore(store)
    with pytest.raises(FileNotFoundError) as excinfo:
        _ = safe_store["missing_key"]
    # Ensure the missing key is represented in the error message
    assert "missing_key" in str(excinfo.value)
    # Ensure the original KeyError is preserved as the cause
    assert isinstance(excinfo.value.__cause__, KeyError)


def test_safestore_len():
    store = {"a": 1, "b": 2}
    safe_store = SafeStore(store)
    assert len(safe_store) == 2


def test_safestore_iter():
    store = {"a": 1, "b": 2}
    safe_store = SafeStore(store)
    assert set(iter(safe_store)) == {"a", "b"}


def test_safestore_getattr_passthrough():
    class MockStore:
        def __init__(self):
            self.attr = "value"

    store = MockStore()
    safe_store = SafeStore(store)
    assert safe_store.attr == "value"


def test_safestore_getattr_setstate_raises_attributeerror():
    store = {}
    safe_store = SafeStore(store)
    with pytest.raises(AttributeError):
        _ = safe_store.__setstate__


def test_safestore_setitem_raises_notimplemented():
    store = {}
    safe_store = SafeStore(store)
    with pytest.raises(NotImplementedError):
        safe_store["foo"] = "bar"


def test_safestore_delitem_raises_notimplemented():
    store = {"foo": "bar"}
    safe_store = SafeStore(store)
    with pytest.raises(NotImplementedError):
        del safe_store["foo"]
