#ifdef __cplusplus
extern "C" {
// Minimal API-compatible stand-in for libavdevice.
#pragma once

typedef struct AVInputFormat {
    const char *name;
} AVInputFormat;

int             avdevice_register_all(void);
const AVInputFormat *av_find_input_format(const char *name);

#endif
}