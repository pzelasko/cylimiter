#include <stdio.h>
#include <vector>
#include <deque>
#include <string>


class ReverbRIR {
    // We are maintaining a C-style API with pair of pointer + size
    // to be able to avoid memory copies from python/numpy/cython to C++
    // and provide an inplace modification API for numpy arrays.
    public:
        ReverbRIR(const float mix = 1.0f);

        ReverbRIR(const float * const rir, const std::size_t num_samples, const float mix = 1.0f);

        void apply_inplace(float * const audio, const std::size_t num_samples);
        std::vector<float> apply(float const * const audio, const std::size_t num_samples);
        void reset();

        void read_from_string(const std::string &data);
        std::string write_to_string() const;

    // Mutable state
    private:
        std::vector<float> buffer_;

    // Settings
    private:
        std::vector<float> rir_;
        float mix_;
};