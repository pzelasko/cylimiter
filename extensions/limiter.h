#include <vector>


class CLimiter {
    public:
        CLimiter(float attack, float release, int delay, float threshold);

        void limit_inplace(std::vector<float> &audio);
        std::vector<float> limit(const std::vector<float> &audio);
        void reset();

    // Mutable state
    private:
        std::vector<float> delay_line_;
        int delay_index_;
        float envelope_;
        float gain_;

    // Settings
    private:
        const float attack_;
        const float release_;
        const int delay_;
        const float threshold_;
};
