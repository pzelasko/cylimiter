#cython: language_level=3
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

from extensions.climiter cimport CLimiter, CLimiterState


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

    def _validate_input(self, audio):
        if hasattr(audio, "ndim"):
            assert audio.ndim == 1, "The input audio array has to be single-dimensional (only mono audio is supported)."

    def limit_inplace(self, audio):
        self._validate_input(audio)
        self._limiter.get().limit_inplace(audio)

    def limit(self, audio):
        self._validate_input(audio)
        return self._limiter.get().limit(audio)

    def reset(self):
        return self._limiter.get().reset()


def _create_default() -> Limiter:
    return Limiter()
