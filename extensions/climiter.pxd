#cython: language_level=3
from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "limiter.h" nogil:

    cdef cppclass CLimiterState

    cdef cppclass CLimiter:

        CLimiter(float, float, int, float)

        @staticmethod
        CLimiter read_from_string(string data)

        string write_to_string() const

        void limit_inplace(vector[float] &)

        vector[float] limit(const vector[float] &)

        void reset()
