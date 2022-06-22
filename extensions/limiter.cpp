#include <cmath>
#include <algorithm>
#include <iterator>

#include "limiter.h"

using namespace std;

CLimiter::CLimiter(float attack, float release, int delay, float threshold)
    : attack_{attack},
      release_{release},
      delay_{delay},
      threshold_{threshold} {
    reset();
}

void CLimiter::limit_inplace(vector<float> &audio) {
    for (unsigned long idx = 0; idx < audio.size(); ++idx) {
        const auto sample = audio[idx];
        delay_line_[delay_index_] = sample;
        delay_index_ = (delay_index_ + 1) % delay_;

        // calculate an envelope of the signal
        envelope_ = max(abs(sample), envelope_ * release_);

        float target_gain = 1.0f;
        if (envelope_ > threshold_) {
            target_gain = threshold_ / envelope_;
        }

        // have gain_ go towards a desired limiter gain
        gain_ = gain_* attack_ + target_gain * (1.0f - attack_);

        // limit the delayed signal
        audio[idx] = delay_line_[delay_index_] * gain_;
    }
}

std::vector<float> CLimiter::limit(const vector<float> &audio) {
    vector<float> out;
    copy(begin(audio), end(audio), back_inserter(out));
    limit_inplace(out);
    return out;
}

void CLimiter::reset() {
    delay_index_ = 0;
    gain_ = 1.0f;
    envelope_ = 0.0f;
    delay_line_.resize(delay_);
    for (unsigned long i = 0; i < delay_line_.size(); ++i) {
        delay_line_[i] = 0.0f;
    }
}
