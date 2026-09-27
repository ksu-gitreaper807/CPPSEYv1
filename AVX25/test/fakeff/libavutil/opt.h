#ifdef __cplusplus
extern "C" {
// Minimal API-compatible stand-in for libavutil AVDictionary.
#pragma once
#include <stddef.h>

typedef struct AVDictionaryEntry {
    char *key;
    char *value;
} AVDictionaryEntry;

typedef struct AVDictionary AVDictionary;

int  av_dict_set(AVDictionary **pm, const char *key, const char *value, int flags);
void av_dict_free(AVDictionary **m);

#endif
}