#cython: language_level=3
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

from extensions.climiter cimport CLimiter


cdef class Limiter:
    cdef unique_ptr[CLimiter] _limiter

    def __init__(self, attack: float = 0.9, release: float = 0.9995, delay: int = 40, threshold: float = 0.95):
        self._limiter.reset(new CLimiter(attack, release, delay, threshold))

    def limit_inplace(self, audio):
        self._limiter.get().limit_inplace(audio)

    def limit(self, audio):
        return self._limiter.get().limit(audio)

    def reset(self):
        return self._limiter.get().reset()
