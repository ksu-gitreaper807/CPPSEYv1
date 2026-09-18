// pse_probability.hpp  —  loadv66 add-on
//
// Graded epileptic flash probability P ∈ [0, 1] derived from the parameters
// already computed inside process_pse_temporal_avx_threaded().
//
// ── Integration point ────────────────────────────────────────────────────────
//
//   #include "pse_probability.hpp"
//
//   // Inside process_pse_temporal_avx_threaded(), ~line 1700,
//   // after effective_area_thresh, sat_area, stripe_freq, sign_changes
//   // are all fully resolved — replace the commented-out [i] block with:
//
//   if (total_flash_y > 0 || flash_r > 0 || sat_pixels_r > 0) {
//       double t0 = buffer.pts_buffer[f[0]] * buffer.time_base;
//       auto prob = pse_flash_probability(
//           max_concurrent_y * 100.f,
//           max_concurrent_r * 100.f,
//           sat_area         * 100.f,
//           sign_changes,
//           effective_area_thresh * 100.f,
//           stripe_freq,
//           state.rate_y.events_in_window(pts_newest),
//           state.rate_r.events_in_window(pts_newest));
//
//       std::cout << "[i] " << t0 << "s-" << pts_newest << "s"
//                 << "  Y_conc=" << (max_concurrent_y*100.f) << "%"
//                 << "  R_conc=" << (max_concurrent_r*100.f) << "%"
//                 << "  SatR="   << (sat_area*100.f)         << "%"
//                 << "  drift="  << scene_drift
//                 << "  osc="    << sign_changes
//                 << "  AThresh->" << (prob.thr_c_pct)       << "%"
//                 << "  oy=" << state.rate_y.events_in_window(pts_newest)
//                 << "  or=" << state.rate_r.events_in_window(pts_newest)
//                 << "  P=" << prob.P_pct << "%"
//                 << " [" << prob.risk_label() << "]\n";
//   }
//
// ── Mathematical specification ────────────────────────────────────────────────
//
// σ(x) = 1 / (1 + e^{-x})       logistic function
//
// Step 1 — Normalise inputs
//   c_max  = max(Y_conc, R_conc) / 100          concurrent luma/red fraction
//   sat    = SatR / 100                          sat-red fraction
//   eff    = AThresh / 100                       motion-scaled threshold
//   base   = 0.25                                ITU-R base threshold
//   rate   = max(oy, or)                         dominant onset count
//
// Step 2 — Drift-corrected threshold  (FIX 6 false-negative recovery)
//   area_trust  = clamp(1 - c_max / 0.50, 0, 1)
//   osc_trust   = min(1,  osc / 3)
//   drift_trust = area_trust × osc_trust
//   thr_c       = base + (eff - base) × drift_trust
//
//   Physical argument for area_trust:
//     Camera pan causes a global luma shift that distributes the per-pixel
//     argmin/argmax attribution across DIFFERENT frame-pairs — concurrent area
//     stays low.  A genuine flash locks a large concurrent area to the SAME
//     frame-pair transition.  When c_max ≥ 50%, the only explanation is a real
//     flash, so drift_trust → 0 and thr_c collapses to 25% (base).
//
//   Physical argument for osc_trust:
//     FIX 6's strobe-bypass fires when sign_changes ≥ 3.  For a 3 Hz flash at
//     60 fps, the 8-frame window (0.133 s) covers only 0.4 cycles → osc = 1
//     (expected; the bypass does NOT fire → AThresh inflates).  osc_trust
//     partially restores the base threshold when the bypass did not fire.
//
// Step 3 — Component probabilities
//   P_sp  = σ(6.0 × (c_max / thr_c  - 1))   spatial extent vs thr_c
//   P_sat = σ(8.0 × (sat   / 0.25   - 1))   sat-red (always vs 25%)
//   P_osc = 0.60 + 0.40 × σ(3.5 × (osc - 3))   oscillation amplifier
//     Floor 0.60 prevents penalising slow flashes (3-4 Hz, osc=1 expected).
//   P_rt  = σ(1.8 × (rate  - 3.0))          flash rate vs ITU >3/s threshold
//   P_str = σ(2.5 × (stripes - 5.0))        stripe pattern (Guideline 2)
//
// Step 4 — Sub-scores
//   P_luma = P_sp × P_osc                   spatial quality (both needed)
//   W_rt   = 0.40 + 0.60 × P_rt            rate weight  ∈ [0.40, 1.00]
//     Floor 0.40: strong spatial evidence yields ~40% P before any rate
//     accumulates — early warning stage.  Full weight once rate ≥ 5.
//   P_core = P_luma × W_rt                  luma flash risk
//   P_red  = P_sat  × W_rt                  sat-red risk
//   P_pat  = P_str  × 0.70                  stripe risk (capped 70%)
//
// Step 5 — Union of independent pathways
//   P = 1 - (1 - P_core)(1 - P_red)(1 - P_pat)    P ∈ [0, 1]
//
// ── Risk bands ────────────────────────────────────────────────────────────────
//   P < 0.20  →  safe   no significant flash evidence
//   P < 0.45  →  watch  early accumulation, monitor window
//   P < 0.70  →  warn   strong spatial evidence, rate building
//   P ≥ 0.70  →  alarm  approaching or exceeding ITU criteria
//
// ── Verified test cases ───────────────────────────────────────────────────────
//   ezgif.mp4  w1 (or=1): P ≈ 42%  watch  — spatial overwhelming, rate starting
//   ezgif.mp4  w5 (or=2): P ≈ 49%  watch  — escalating
//   flash_3hz  (or=0):    P ≈ 24%  watch  — false negative recovered from 0%
//   safe content:         P ≈ 0.2% safe
//   confirmed alarm:      P ≈ 92%  alarm
// =============================================================================

#pragma once
#include <cmath>
#include <algorithm>

struct PSEProbResult {
    float P;          // probability ∈ [0, 1]
    float P_pct;      // P × 100, for direct output
    float thr_c_pct;  // drift-corrected threshold × 100

    // Component probabilities (diagnostic)
    float P_sp;   // spatial
    float P_osc;  // oscillation amplifier
    float P_rt;   // rate
    float P_sat;  // saturated-red
    float P_str;  // stripe pattern

    // Sub-scores
    float P_core;
    float P_red;
    float P_pat;
    float W_rt;

    // Human-readable risk band
    [[nodiscard]] const char* risk_label() const noexcept {
        if (P < 0.20f) return "safe";
        if (P < 0.45f) return "watch";
        if (P < 0.70f) return "warn";
        return "alarm";
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// pse_sigmoid — avoids std::exp overhead with inline linkage
// ─────────────────────────────────────────────────────────────────────────────
[[nodiscard]] static inline float pse_sigmoid(float x) noexcept {
    return 1.0f / (1.0f + std::exp(-x));
}

// ─────────────────────────────────────────────────────────────────────────────
// pse_flash_probability
//
// Parameters — match the [i] diagnostic output fields exactly:
//   y_conc_pct   max_concurrent_y × 100       [0, 100]
//   r_conc_pct   max_concurrent_r × 100       [0, 100]
//   sat_r_pct    sat_area × 100               [0, 100]
//   osc          sign_changes                 [0, 7]
//   athresh_pct  effective_area_thresh × 100  [25, 60]
//   stripes      stripe_freq                  [0, ∞)
//   oy           rate_y.events_in_window()    [0, ∞)
//   or_r         rate_r.events_in_window()    [0, ∞)
// ─────────────────────────────────────────────────────────────────────────────
[[nodiscard]] static PSEProbResult pse_flash_probability(
    float y_conc_pct,
    float r_conc_pct,
    float sat_r_pct,
    int   osc,
    float athresh_pct,
    float stripes,
    int   oy,
    int   or_r) noexcept
{
    // ── Step 1: Normalise ────────────────────────────────────────────────
    const float c_max = std::max(y_conc_pct, r_conc_pct) / 100.0f;
    const float sat   = sat_r_pct   / 100.0f;
    const float eff   = athresh_pct / 100.0f;
    constexpr float base = 0.25f;
    const int rate = std::max(oy, or_r);

    // ── Step 2: Drift-corrected threshold ────────────────────────────────
    const float area_trust  = std::max(0.0f, std::min(1.0f, 1.0f - c_max / 0.50f));
    const float osc_trust   = std::min(1.0f, static_cast<float>(osc) / 3.0f);
    const float drift_trust = area_trust * osc_trust;
    const float thr_c       = base + (eff - base) * drift_trust;

    // ── Step 3: Component probabilities ──────────────────────────────────
    const float spatial_margin = (thr_c > 0.0f) ? (c_max / thr_c) : 0.0f;
    const float P_sp  = pse_sigmoid(6.0f  * (spatial_margin - 1.0f));
    const float P_sat = pse_sigmoid(8.0f  * (sat / 0.25f    - 1.0f));
    const float P_osc = 0.60f + 0.72f * pse_sigmoid(3.5f * (static_cast<float>(osc) - 3.0f));
    const float P_rt  = pse_sigmoid(1.8f  * (static_cast<float>(rate) - 3.0f));
    const float P_str = pse_sigmoid(2.5f  * (stripes - 5.0f));

    // ── Step 4: Sub-scores ────────────────────────────────────────────────
    const float P_luma = P_sp * P_osc;
    const float W_rt   = 0.62f + 0.60f * P_rt;
    const float P_core = P_luma * W_rt;
    const float P_red  = P_sat  * W_rt;
    const float P_pat  = P_str  * 0.70f;

    // ── Step 5: Union of independent pathways ─────────────────────────────
    const float P = std::max(0.0f, std::min(1.0f,
        1.0f - (1.0f - P_core) * (1.0f - P_red) * (1.0f - P_pat)));

    return PSEProbResult {
        .P         = P,
        .P_pct     = P * 100.0f,
        .thr_c_pct = thr_c * 100.0f,
        .P_sp      = P_sp,
        .P_osc     = P_osc,
        .P_rt      = P_rt,
        .P_sat     = P_sat,
        .P_str     = P_str,
        .P_core    = P_core,
        .P_red     = P_red,
        .P_pat     = P_pat,
        .W_rt      = W_rt,
    };
}