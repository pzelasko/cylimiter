#include <stdio.h>
#include <vector>
#include <string>


class CLimiter {
    // We are maintaining a C-style API with pair of pointer + size
    // to be able to avoid memory copies from python/numpy/cython to C++
    // and provide an inplace modification API for numpy arrays.
    public:
        CLimiter(float attack, float release, int delay, float threshold);

        void limit_inplace(float * const audio, std::size_t num_samples);
        std::vector<float> limit(float const * const audio, std::size_t num_samples);
        void reset();

        void read_from_string(const std::string &data);
        std::string write_to_string() const;

    // Mutable state
    private:
        std::vector<float> delay_line_;
        int delay_index_;
        float envelope_;
        float gain_;

    // Settings
    private:
        float attack_;
        float release_;
        int delay_;
        float threshold_;
};