#cython: language_level=3
import warnings

from libcpp.memory cimport unique_ptr

from extensions.climiter cimport CLimiter
import numpy as np


cdef class Limiter:
    cdef unique_ptr[CLimiter] _limiter

    def __init__(self, attack: float = 0.9, release: float = 0.9995, delay: int = 40, threshold: float = 0.95):
        assert 0 < attack < 1, "Permitted attack value range is (0 - 1)."
        assert 0 < release < 1, "Permitted release value range is (0 - 1)."
        assert 0 < threshold, "threshold has to be a float greater than zero."
        assert isinstance(delay, int) and delay > 0, "Delay has to be an integer greater than zero."
        self._limiter.reset(new CLimiter(attack, release, delay, threshold))

    def __setstate__(self, state: bytes) -> None:
        self.__init__()
        self._limiter.get().read_from_string(state)

    def __getstate__(self) -> bytes:
        return self.write_to_string()

    def write_to_string(self) -> bytes:
        return self._limiter.get().write_to_string()

    def limit_inplace(self, audio: np.ndarray) -> np.ndarray:
        assert isinstance(audio, np.ndarray), "For in-place limiter, we only support Numpy arrays. Either convert your input to a 1D Numpy array, or use the non-in-place operation ('limiter.limit(arr)')."
        assert audio.dtype == np.float32, "We only support np.float32 dtype for in-place operations."
        assert audio.ndim == 1, "The input audio array has to be single-dimensional (only mono audio is supported)."
        assert audio.flags['C_CONTIGUOUS'], "The input array has to be contiguous (you can use np.ascontiguousarray)."

        cdef float[:] audio_memview = audio
        self._limiter.get().limit_inplace(&audio_memview[0], audio_memview.shape[0])

    def limit(self, audio) -> np.ndarray:
        audio_arr = np.copy(np.ascontiguousarray(audio, dtype=np.float32))
        self.limit_inplace(audio_arr)
        return audio_arr

    def reset(self) -> None:
        return self._limiter.get().reset()
