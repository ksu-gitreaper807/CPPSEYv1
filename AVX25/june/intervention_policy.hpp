#pragma once

#include <cmath>
#include <limits>

enum class PSEInterventionLevel { NONE = 0, DIM = 1, BLACKOUT = 2 };

// Pure decision policy; no display, clock, threads, or detector dependencies.
// confirmed_flash must be the detector's spatial + opposing-transition +
// onset-rate + frequency decision, never a score/stripe/static-red threshold.
//
// Use one policy for the display, with serialized observations from BOTH
// detector paths. Supply seconds from ONE monotonic clock (steady_clock for
// live use, media timestamps for offline replay). Call level_at() on the
// display worker's timer too, so capture stalls/silent windows still release.
// Reset when starting a new stream/clock epoch. This class is not thread-safe.
class PSEInterventionPolicy {
public:
    explicit PSEInterventionPolicy(double hold_seconds = 1.0) noexcept
        : hold_seconds_(std::isfinite(hold_seconds) && hold_seconds > 0.0
                            ? hold_seconds : 1.0) {}

    [[nodiscard]] PSEInterventionLevel update(double timestamp_s,
                                             bool confirmed_flash,
                                             float probability) noexcept {
        // Stale/invalid observations cannot extend a hold, release protection,
        // or resurrect an expired event. Equal timestamps allow both detectors
        // to contribute to the same frame; the strongest decision wins.
        if (!std::isfinite(timestamp_s) || timestamp_s < last_timestamp_s_)
            return level_;
        last_timestamp_s_ = timestamp_s;

        if (level_ != PSEInterventionLevel::NONE &&
            timestamp_s - last_confirmed_s_ >= hold_seconds_)
            level_ = PSEInterventionLevel::NONE;

        if (confirmed_flash) {
            // A confirmed flash always gets at least DIM, even if the score
            // underestimates a slow flash. Score alone cannot intervene.
            const auto requested = std::isfinite(probability) && probability >= 0.70f
                ? PSEInterventionLevel::BLACKOUT : PSEInterventionLevel::DIM;
            if (requested > level_) level_ = requested;
            last_confirmed_s_ = timestamp_s;
        }
        // Keep the strongest level through the hold. Only confirmed evidence
        // refreshes it; unconfirmed high scores neither escalate nor prolong it.
        return level_;
    }

    [[nodiscard]] PSEInterventionLevel level_at(double timestamp_s) noexcept {
        return update(timestamp_s, false, 0.0f);
    }

    void reset() noexcept {
        level_ = PSEInterventionLevel::NONE;
        last_timestamp_s_ = -std::numeric_limits<double>::infinity();
        last_confirmed_s_ = 0.0;
    }

private:
    double hold_seconds_;
    double last_timestamp_s_ = -std::numeric_limits<double>::infinity();
    double last_confirmed_s_ = 0.0;
    PSEInterventionLevel level_ = PSEInterventionLevel::NONE;
};
