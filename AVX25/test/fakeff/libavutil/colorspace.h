#ifdef __cplusplus
extern "C" {
// Minimal API-compatible stand-in for libavutil colour-space definitions.
#pragma once

typedef enum AVColorSpace {
    AVCOL_SPC_UNSPECIFIED   = 2,
    AVCOL_SPC_BT709         = 1,
    AVCOL_SPC_BT470BG       = 5,
    AVCOL_SPC_SMPTE170M     = 6,
    AVCOL_SPC_SMPTE240M     = 7,
    AVCOL_SPC_BT2020_NCL    = 9,
    AVCOL_SPC_BT2020_CL     = 10,
} AVColorSpace;

#endif
}