#cython: language_level=3
import warnings
from typing import Optional, Sequence, Union

from libcpp.memory cimport unique_ptr

from extensions.climiter cimport CLimiter
from extensions.climiter cimport ReverbRIR as CReverbRIR
import numpy as np


cdef class Limiter:
    cdef unique_ptr[CLimiter] _ptr

    def __init__(self, attack: float = 0.9, release: float = 0.9995, delay: int = 40, threshold: float = 0.95):
        assert 0 < attack < 1, "Permitted attack value range is (0 - 1)."
        assert 0 < release < 1, "Permitted release value range is (0 - 1)."
        assert 0 < threshold, "threshold has to be a float greater than zero."
        assert isinstance(delay, int) and delay > 0, "Delay has to be an integer greater than zero."
        self._ptr.reset(new CLimiter(attack, release, delay, threshold))

    def __setstate__(self, state: bytes) -> None:
        self.__init__()
        self._ptr.get().read_from_string(state)

    def __getstate__(self) -> bytes:
        return self.write_to_string()

    def write_to_string(self) -> bytes:
        return self._ptr.get().write_to_string()

    def apply_inplace(self, audio: np.ndarray) -> None:
        _check_array(audio)
        cdef float[:] audio_memview = audio
        self._ptr.get().apply_inplace(&audio_memview[0], audio_memview.shape[0])

    def apply(self, audio) -> np.ndarray:
        audio_arr = np.copy(np.ascontiguousarray(audio, dtype=np.float32))
        self.apply_inplace(audio_arr)
        return audio_arr

    def limit_inplace(self, audio: np.ndarray) -> None:
        self.apply_inplace(audio)

    def limit(self, audio) -> np.ndarray:
        return self.apply(audio)

    def reset(self) -> None:
        return self._ptr.get().reset()


cdef class ReverbRIR:
    cdef unique_ptr[CReverbRIR] _ptr

    def __init__(self, rir: Optional[Union[Sequence[float], np.ndarray]] = None, mix: float = 1.0):
        assert 0.0 <= mix <= 1.0, "Argument 'mix' must be between 0 and 1."
        if rir is None:
            self._ptr.reset(new CReverbRIR(mix))
        else:
            self._init_from_rir(rir, mix)

    def _init_from_rir(self, rir, mix: float):
        # There will be two copies total, but it shouldn't be too bad...
        rir = np.copy(np.ascontiguousarray(rir, dtype=np.float32))
        cdef float[:] rir_memview = rir
        self._ptr.reset(new CReverbRIR(&rir_memview[0], rir_memview.shape[0], mix))

    def __setstate__(self, state: bytes) -> None:
        self.__init__()
        self._ptr.get().read_from_string(state)

    def __getstate__(self) -> bytes:
        return self.write_to_string()

    def write_to_string(self) -> bytes:
        return self._ptr.get().write_to_string()

    def apply_inplace(self, audio: np.ndarray) -> None:
        _check_array(audio)
        cdef float[:] audio_memview = audio
        self._ptr.get().apply_inplace(&audio_memview[0], audio_memview.shape[0])

    def apply(self, audio) -> np.ndarray:
        audio_arr = np.copy(np.ascontiguousarray(audio, dtype=np.float32))
        self.apply_inplace(audio_arr)
        return audio_arr

    def reset(self) -> None:
        return self._ptr.get().reset()


def _check_array(arr) -> None:
    assert isinstance(arr, np.ndarray), "For in-place operations, we only support Numpy arrays. Either convert your input to a 1D Numpy array, or use the non-in-place operation ('effect.apply(arr)')."
    assert arr.dtype == np.float32, "We only support np.float32 dtype for in-place operations."
    assert arr.ndim == 1, "The input array has to be single-dimensional (only mono audio is supported)."
    assert arr.flags['C_CONTIGUOUS'], "The input array has to be contiguous (you can use np.ascontiguousarray)."
