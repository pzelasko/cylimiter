#cython: language_level=3
from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "limiter.h" nogil:

    cdef cppclass CLimiter:

        CLimiter(float, float, int, float)

        CLimiter read_from_string(string data)

        string write_to_string() const

        void apply_inplace(float * const, size_t)

        vector[float] apply(const float * const, size_t)

        void reset()


cdef extern from "reverb_rir.h" nogil:

    cdef cppclass ReverbRIR:

        ReverbRIR(const float)

        ReverbRIR(const float * const, const size_t, const float)

        ReverbRIR read_from_string(string data)

        string write_to_string() const

        void apply_inplace(float * const, const size_t)

        vector[float] apply(const float * const, const size_t)

        void reset()
