"""Carga de datos de radio: tope de tamano antes de leer (radio_page._load_array)."""
import io

import numpy as np
import pytest

import radio_page


class _Up:
    def __init__(self, name, data, size=None):
        self.name, self._data = name, data
        self.size = len(data) if size is None else size

    def getvalue(self):
        return self._data


def _npy(arr):
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def test_small_npy_loads():
    arr = np.arange(1000, dtype=np.float32)
    np.testing.assert_array_equal(radio_page._load_array(_Up("s.npy", _npy(arr))), arr)


def test_too_big_is_refused_before_reading():
    up = _Up("big.npy", b"", size=(radio_page.MAX_RADIO_MB + 1) * 1048576)
    up.getvalue = lambda: pytest.fail("no debe leer un archivo por encima del tope")
    with pytest.raises(ValueError, match="máximo"):
        radio_page._load_array(up)
