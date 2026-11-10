// Standalone regression proof (no display/FFmpeg dependencies):
// g++ -std=c++17 -O2 -Wall -Wextra test/sensitivity_tests.cpp -o /tmp/pse_sensitivity_tests
#include "../june/pse_probability.hpp"
#include "../june/intervention_policy.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>

namespace {
int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        if (failures < 20) std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

bool near(float a, float b) {
    return std::fabs(a - b) < 2e-6f;
}

void check_bounds(const PSEProbResult& p) {
    const std::array<float, 11> components = {
        p.P, p.P_sp, p.P_osc, p.P_rt, p.P_sat, p.P_str,
        p.P_core, p.P_red, p.P_pat, p.W_rt, p.P_pct / 100.0f
    };
    for (float v : components)
        check(std::isfinite(v) && v >= 0.0f && v <= 1.0f,
              "every probability component and weight must be finite and bounded");
    check(std::isfinite(p.thr_c_pct) && p.thr_c_pct >= 25.0f && p.thr_c_pct <= 60.0f,
          "drift-corrected threshold stays within the documented input range");
    check(near(p.P_pct, p.P * 100.0f), "percentage matches probability");
}

void probability_tests() {
    const auto strong = pse_flash_probability(100, 0, 0, 7, 25, 0, 50, 0);
    check_bounds(strong);
    check(near(strong.P_osc, 0.60f + 0.40f * pse_sigmoid(14.0f)),
          "oscillation amplifier follows the bounded documented formula");
    check(near(strong.W_rt, 0.40f + 0.60f * strong.P_rt),
          "rate weight follows the bounded documented formula");
    const auto both = pse_flash_probability(100, 0, 100, 7, 25, 0, 50, 0);
    check_bounds(both);
    check(both.P >= strong.P, "adding saturated-red evidence cannot reduce risk");

    const auto quiet = pse_flash_probability(0, 0, 0, 0, 25, 0, 0, 0);
    check(quiet.P < 0.01f, "quiet scene has low diagnostic risk");
    check(strong.P > 0.99f, "strong repeated spatial evidence retains high sensitivity");
    const auto slow = pse_flash_probability(100, 0, 0, 1, 60, 0, 3, 0);
    check(slow.P > 0.40f, "slow confirmed flashes retain the oscillation floor");
    check(near(slow.thr_c_pct, 25.0f), "large concurrent area restores base threshold");

    float previous_area = 0, previous_red = 0, previous_rate = 0, previous_stripes = 0;
    for (int i = 0; i <= 100; ++i) {
        const auto area = pse_flash_probability(i, 0, 0, 7, 25, 0, 8, 0);
        const auto red = pse_flash_probability(100, 0, i, 7, 25, 0, 8, 0);
        const auto rate = pse_flash_probability(70, 20, 50, 4, 40, 0, i, 0);
        const auto stripes = pse_flash_probability(0, 0, 0, 0, 25, i / 10.0f, 0, 0);
        check(area.P >= previous_area, "risk is monotone in concurrent area");
        check(red.P >= previous_red, "risk is monotone in saturated-red area");
        check(rate.P >= previous_rate, "risk is monotone in onset count");
        check(stripes.P >= previous_stripes, "risk is monotone in stripe evidence");
        previous_area = area.P;
        previous_red = red.P;
        previous_rate = rate.P;
        previous_stripes = stripes.P;
    }

    std::mt19937 rng(102);
    std::uniform_real_distribution<float> percent(0, 100), threshold(25, 60), stripes(0, 40);
    std::uniform_int_distribution<int> oscillations(0, 7), onsets(0, 64);
    for (int i = 0; i < 10000; ++i)
        check_bounds(pse_flash_probability(percent(rng), percent(rng), percent(rng),
                     oscillations(rng), threshold(rng), stripes(rng), onsets(rng), onsets(rng)));

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const float max = std::numeric_limits<float>::max();
    for (float bad : {-100.0f, 0.0f, 200.0f, nan, inf, -inf, max}) {
        check_bounds(pse_flash_probability(bad, bad, bad, -7, bad, bad, -4, -3));
        check_bounds(pse_flash_probability(bad, 30, 40, std::numeric_limits<int>::max(),
                     60, 10, std::numeric_limits<int>::max(), 0));
    }
    const auto invalid = pse_flash_probability(nan, inf, -inf, -1, nan, nan, -1, -1);
    check(near(invalid.P, quiet.P), "invalid evidence does not manufacture risk");
    const auto valid_red = pse_flash_probability(0, 80, 20, 3, 25, 0, 0, 4);
    const auto nan_luma = pse_flash_probability(nan, 80, 20, 3, 25, 0, 0, 4);
    check(near(valid_red.P, nan_luma.P), "invalid luma cannot hide valid red evidence");
    const auto valid_luma = pse_flash_probability(80, 0, 20, 3, 25, 0, 4, 0);
    const auto nan_red = pse_flash_probability(80, nan, 20, 3, 25, 0, 4, 0);
    check(near(valid_luma.P, nan_red.P), "invalid red cannot hide valid luma evidence");
    const auto excess = pse_flash_probability(200, 200, 200, 99, 200, 10, -1, -1);
    const auto clamped = pse_flash_probability(100, 100, 100, 7, 60, 10, 0, 0);
    check(near(excess.P, clamped.P), "out-of-range inputs use documented bounds");
}

void policy_tests() {
    using Level = PSEInterventionLevel;
    PSEInterventionPolicy policy;
    const std::array<PSEProbResult, 4> unconfirmed = {
        pse_flash_probability(100, 0, 0, 0, 25, 0, 0, 0), // single scene cut
        pse_flash_probability(0, 0, 100, 0, 25, 0, 0, 0), // static red
        pse_flash_probability(0, 0, 0, 0, 25, 40, 0, 0), // stripe evidence
        pse_flash_probability(100, 100, 100, 7, 25, 40, 64, 64)
    };
    for (const auto& p : unconfirmed) {
        check(policy.update(0, false, p.P) == Level::NONE,
              "scene cut, static red, stripes, and high score require flash confirmation");
    }
    for (int i = 0; i <= 100; ++i)
        check(policy.update(i, false, i / 100.0f) == Level::NONE,
              "score changes alone never initiate intervention");

    policy.reset();
    const auto slow = pse_flash_probability(100, 0, 0, 1, 60, 0, 3, 0);
    check(slow.P < 0.45f, "slow-flash fixture demonstrates score-only false negative");
    check(policy.update(10, true, slow.P) == Level::DIM,
          "a confirmed slow flash intervenes even below the old score threshold");
    check(policy.update(10.25, false, 1.0f) == Level::DIM,
          "unconfirmed high score cannot escalate existing protection");
    check(policy.level_at(10.999) == Level::DIM, "protection lasts through the hold");
    check(policy.level_at(11) == Level::NONE,
          "timer tick releases at the exact deadline without another captured frame");

    check(policy.update(20, true, 0.70f) == Level::BLACKOUT,
          "blackout threshold is inclusive and requires confirmation");
    check(policy.update(20.25, true, 0) == Level::BLACKOUT,
          "a lower confirmed score preserves strongest protection and refreshes hold");
    check(policy.update(20.5, false, 1) == Level::BLACKOUT,
          "unconfirmed input holds without refreshing the deadline");
    check(policy.level_at(21) == Level::BLACKOUT, "hold starts at the last confirmed timestamp");
    check(policy.update(21.25, false, 1) == Level::NONE,
          "even a maximal unconfirmed score releases at the last-confirmation deadline");

    check(policy.update(30, true, 0.699f) == Level::DIM, "confirmed score below 70% dims");
    check(policy.update(30, true, 0.90f) == Level::BLACKOUT,
          "second detector can escalate immediately at the same timestamp");
    check(policy.update(30, true, 0.1f) == Level::BLACKOUT,
          "second detector cannot downgrade protection at the same timestamp");
    check(policy.update(30, false, 0) == Level::BLACKOUT,
          "unconfirmed fast path cannot clear a confirmed slow path");
    check(policy.level_at(31) == Level::NONE, "duplicate observations do not lengthen hold");

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    const double nan_time = std::numeric_limits<double>::quiet_NaN();
    policy.reset();
    check(policy.update(nan_time, true, 1) == Level::NONE, "invalid timestamp cannot activate");
    check(policy.update(40, true, nan) == Level::DIM,
          "invalid score does not suppress independently confirmed flash evidence");
    for (double bad : {39.0, nan_time, inf, -inf}) {
        check(policy.update(bad, true, 1) == Level::DIM,
              "stale/invalid timestamps cannot escalate or extend protection");
        check(policy.level_at(bad) == Level::DIM,
              "stale/invalid timestamps cannot release protection");
    }
    check(policy.level_at(41) == Level::NONE, "invalid observations did not extend the hold");
    check(policy.update(40.5, true, 1) == Level::NONE,
          "out-of-order confirmation cannot resurrect an expired event");
    check(policy.update(1000, true, 0) == Level::DIM, "fresh evidence after a long gap intervenes");
    check(policy.level_at(2000) == Level::NONE, "long capture gap expires protection");
    policy.reset();
    check(policy.update(0, true, 1) == Level::BLACKOUT, "reset permits a new timestamp epoch");

    PSEInterventionPolicy short_hold(0.25);
    check(short_hold.update(0, true, 1) == Level::BLACKOUT, "custom hold accepts confirmation");
    check(short_hold.level_at(0.249) == Level::BLACKOUT, "custom hold persists until its deadline");
    check(short_hold.level_at(0.25) == Level::NONE, "custom hold releases on timestamps");
    for (double invalid_hold : {0.0, -1.0, inf, nan_time}) {
        PSEInterventionPolicy fallback(invalid_hold);
        check(fallback.update(0, true, 1) == Level::BLACKOUT, "invalid hold uses default");
        check(fallback.level_at(0.5) == Level::BLACKOUT, "default hold is one second");
        check(fallback.level_at(1) == Level::NONE, "default hold expires after one second");
    }

    for (int fps : {24, 30, 60, 144}) {
        PSEInterventionPolicy timed;
        for (int frame = 0; frame <= 4 * fps; ++frame) {
            const bool confirmed = frame == 0 || frame == fps || frame == 2 * fps;
            const auto result = timed.update(static_cast<double>(frame) / fps, confirmed, 1);
            check(result == (frame < 3 * fps ? Level::BLACKOUT : Level::NONE),
                  "hold release is independent of analysis frame rate");
        }
    }
}
} // namespace

int main() {
    probability_tests();
    policy_tests();
    if (failures) {
        std::cerr << failures << " sensitivity checks failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "Probability and intervention policy checks passed (10,000 score samples; 24/30/60/144 fps holds)\n";
}
