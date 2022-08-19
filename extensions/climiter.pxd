#cython: language_level=3
from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "limiter.h" nogil:

    cdef cppclass CLimiter:

        CLimiter(float, float, int, float)

        CLimiter read_from_string(string data)

        string write_to_string() const

        void limit_inplace(float * const, size_t)

        vector[float] limit(const float * const, size_t)

        void reset()
