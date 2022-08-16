#include <vector>
#include <string>


class CLimiter {
    public:
        CLimiter(float attack, float release, int delay, float threshold);

        void limit_inplace(std::vector<float> &audio);
        std::vector<float> limit(const std::vector<float> &audio);
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