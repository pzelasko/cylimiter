#cython: language_level=3
from libcpp.vector cimport vector

cdef extern from "limiter.h" nogil:

    cdef cppclass CLimiter:

        CLimiter(float, float, int, float)

        void limit_inplace(vector[float] &)

        vector[float] limit(const vector[float] &)

        void reset()
