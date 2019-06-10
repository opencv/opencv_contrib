/* Copyright (C) 2013-2016, The Regents of The University of Michigan.
All rights reserved.
This software was developed in the APRIL Robotics Lab under the
direction of Edwin Olson, ebolson@umich.edu. This software may be
available under alternative licensing terms; contact the address above.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the Regents of The University of Michigan.
*/

#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <string.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>

#include "string_util.h"
#include "zarray.h"

struct string_buffer
{
    char *s;
    int alloc;
    size_t size; // as if strlen() was called; not counting terminating \0
};

#define MIN_PRINTF_ALLOC 16

char *sprintf_alloc(const char *fmt, ...)
{
    assert(fmt != NULL);

    va_list args;

    va_start(args,fmt);
    char *buf = vsprintf_alloc(fmt, args);
    va_end(args);

    return buf;
}

char *vsprintf_alloc(const char *fmt, va_list orig_args)
{
    assert(fmt != NULL);

    int size = MIN_PRINTF_ALLOC;
    char *buf = malloc(size * sizeof(char));

    int returnsize;
    va_list args;

    va_copy(args, orig_args);
    returnsize = vsnprintf(buf, size, fmt, args);
    va_end(args);

    // it was successful
    if (returnsize < size) {
        return buf;
    }

    // otherwise, we should try again
    free(buf);
    size = returnsize + 1;
    buf = malloc(size * sizeof(char));

    va_copy(args, orig_args);
    returnsize = vsnprintf(buf, size, fmt, args);
    va_end(args);

    assert(returnsize <= size);
    return buf;
}

char *_str_concat_private(const char *first, ...)
{
    size_t len = 0;

    // get the total length (for the allocation)
    {
        va_list args;
        va_start(args, first);
        const char *arg = first;
        while(arg != NULL) {
            len += strlen(arg);
            arg = va_arg(args, const char *);
        }
        va_end(args);
    }

    // write the string
    char *str = malloc(len*sizeof(char) + 1);
    char *ptr = str;
    {
        va_list args;
        va_start(args, first);
        const char *arg = first;
        while(arg != NULL) {
            while(*arg)
                *ptr++ = *arg++;
            arg = va_arg(args, const char *);
        }
        *ptr = '\0';
        va_end(args);
    }

    return str;
}

// Returns the index of the first character that differs:
int str_diff_idx(const char * a, const char * b)
{
    assert(a != NULL);
    assert(b != NULL);

    int i = 0;

    size_t lena = strlen(a);
    size_t lenb = strlen(b);

    size_t minlen = lena < lenb ? lena : lenb;

    for (; i < minlen; i++)
        if (a[i] != b[i])
            break;

    return i;
}


zarray_t *str_split(const char *str, const char *delim)
{
    assert(str != NULL);
    assert(delim != NULL);

    zarray_t *parts = zarray_create(sizeof(char*));
    string_buffer_t *sb = string_buffer_create();

    size_t delim_len = strlen(delim);
    size_t len = strlen(str);
    size_t pos = 0;

    while (pos < len) {
        if (str_starts_with(&str[pos], delim) && delim_len > 0) {
            pos += delim_len;
            // never add empty strings (repeated tokens)
            if (string_buffer_size(sb) > 0) {
                char *part = string_buffer_to_string(sb);
                zarray_add(parts, &part);
            }
            string_buffer_reset(sb);
        } else {
            string_buffer_append(sb, str[pos]);
            pos++;
        }
    }

    if (string_buffer_size(sb) > 0) {
        char *part = string_buffer_to_string(sb);
        zarray_add(parts, &part);
    }

    string_buffer_destroy(sb);
    return parts;
}

// split on one or more spaces.
zarray_t *str_split_spaces(const char *str)
{
  zarray_t *parts = zarray_create(sizeof(char*));
  size_t len = strlen(str);
  size_t pos = 0;

  while (pos < len) {

    while (pos < len && str[pos] == ' ')
      pos++;

    // produce a token?
    if (pos < len) {
      // yes!
      size_t off0 = pos;
      while (pos < len && str[pos] != ' ')
	pos++;
      size_t off1 = pos;

      size_t len_off = off1 - off0;
      char *tok = malloc(len_off + 1);
      memcpy(tok, &str[off0], len_off);
      tok[len_off] = 0;
      zarray_add(parts, &tok);
    }
  }

  return parts;
}

void str_split_destroy(zarray_t *za)
{
    if (!za)
        return;

    zarray_vmap(za, free);
    zarray_destroy(za);
}

char *str_trim(char *str)
{
    assert(str != NULL);

    return str_lstrip(str_rstrip(str));
}

char *str_lstrip(char *str)
{
    assert(str != NULL);

    char *ptr = str;
    char *end = str + strlen(str);
    for(; ptr != end && isspace(*ptr); ptr++);
    // shift the string to the left so the original pointer still works
    memmove(str, ptr, strlen(ptr)+1);
    return str;
}

char *str_rstrip(char *str)
{
    assert(str != NULL);

    char *ptr = str + strlen(str) - 1;
    for(; ptr+1 != str && isspace(*ptr); ptr--);
    *(ptr+1) = '\0';
    return str;
}

int str_indexof(const char *haystack, const char *needle)
{
	assert(haystack != NULL);
	assert(needle != NULL);

    // use signed types for hlen/nlen because hlen - nlen can be negative.
    int hlen = (int) strlen(haystack);
    int nlen = (int) strlen(needle);

    if (nlen > hlen) return -1;

    for (int i = 0; i <= hlen - nlen; i++) {
        if (!strncmp(&haystack[i], needle, nlen))
            return i;
    }

    return -1;
}

int str_last_indexof(const char *haystack, const char *needle)
{
	assert(haystack != NULL);
	assert(needle != NULL);

    // use signed types for hlen/nlen because hlen - nlen can be negative.
    int hlen = (int) strlen(haystack);
    int nlen = (int) strlen(needle);

    int last_index = -1;
    for (int i = 0; i <= hlen - nlen; i++) {
        if (!strncmp(&haystack[i], needle, nlen))
            last_index = i;
    }

    return last_index;
}

// in-place modification.
char *str_tolowercase(char *s)
{
	assert(s != NULL);

    size_t slen = strlen(s);
    for (int i = 0; i < slen; i++) {
        if (s[i] >= 'A' && s[i] <= 'Z')
            s[i] = s[i] + 'a' - 'A';
    }

    return s;
}

char *str_touppercase(char *s)
{
    assert(s != NULL);

    size_t slen = strlen(s);
    for (int i = 0; i < slen; i++) {
        if (s[i] >= 'a' && s[i] <= 'z')
            s[i] = s[i] - ('a' - 'A');
    }

    return s;
}

string_buffer_t* string_buffer_create()
{
    string_buffer_t *sb = (string_buffer_t*) calloc(1, sizeof(string_buffer_t));
    assert(sb != NULL);
    sb->alloc = 32;
    sb->s = calloc(sb->alloc, 1);
    return sb;
}

void string_buffer_destroy(string_buffer_t *sb)
{
    if (sb == NULL)
        return;

    if (sb->s)
        free(sb->s);

    memset(sb, 0, sizeof(string_buffer_t));
    free(sb);
}

void string_buffer_append(string_buffer_t *sb, char c)
{
    assert(sb != NULL);

    if (sb->size+2 >= sb->alloc) {
        sb->alloc *= 2;
        sb->s = realloc(sb->s, sb->alloc);
    }

    sb->s[sb->size++] = c;
    sb->s[sb->size] = 0;
}

char string_buffer_pop_back(string_buffer_t *sb) {
    assert(sb != NULL);
    if (sb->size == 0)
        return 0;

    char back = sb->s[--sb->size];
    sb->s[sb->size] = 0;
    return back;
}

void string_buffer_appendf(string_buffer_t *sb, const char *fmt, ...)
{
    assert(sb != NULL);
    assert(fmt != NULL);

    int size = MIN_PRINTF_ALLOC;
    char *buf = malloc(size * sizeof(char));

    int returnsize;
    va_list args;

    va_start(args,fmt);
    returnsize = vsnprintf(buf, size, fmt, args);
    va_end(args);

    if (returnsize >= size) {
        // otherwise, we should try again
        free(buf);
        size = returnsize + 1;
        buf = malloc(size * sizeof(char));

        va_start(args, fmt);
        returnsize = vsnprintf(buf, size, fmt, args);
        va_end(args);

        assert(returnsize <= size);
    }

    string_buffer_append_string(sb, buf);
    free(buf);
}

void string_buffer_append_string(string_buffer_t *sb, const char *str)
{
    assert(sb != NULL);
    assert(str != NULL);

    size_t len = strlen(str);

    while (sb->size+len + 1 >= sb->alloc) {
        sb->alloc *= 2;
        sb->s = realloc(sb->s, sb->alloc);
    }

    memcpy(&sb->s[sb->size], str, len);
    sb->size += len;
    sb->s[sb->size] = 0;
}

bool string_buffer_ends_with(string_buffer_t *sb, const char *str)
{
    assert(sb != NULL);
    assert(str != NULL);

    return str_ends_with(sb->s, str);
}

char *string_buffer_to_string(string_buffer_t *sb)
{
    assert(sb != NULL);

    return strdup(sb->s);
}

// returns length of string (not counting \0)
size_t string_buffer_size(string_buffer_t *sb)
{
    assert(sb != NULL);

    return sb->size;
}

void string_buffer_reset(string_buffer_t *sb)
{
    assert(sb != NULL);

    sb->s[0] = 0;
    sb->size = 0;
}

string_feeder_t *string_feeder_create(const char *str)
{
    assert(str != NULL);

    string_feeder_t *sf = (string_feeder_t*) calloc(1, sizeof(string_feeder_t));
    sf->s = strdup(str);
    sf->len = strlen(sf->s);
    sf->line = 1;
    sf->col = 0;
    sf->pos = 0;
    return sf;
}

int string_feeder_get_line(string_feeder_t *sf)
{
    assert(sf != NULL);
    return sf->line;
}

int string_feeder_get_column(string_feeder_t *sf)
{
    assert(sf != NULL);
    return sf->col;
}

void string_feeder_destroy(string_feeder_t *sf)
{
    if (sf == NULL)
        return;

    free(sf->s);
    memset(sf, 0, sizeof(string_feeder_t));
    free(sf);
}

bool string_feeder_has_next(string_feeder_t *sf)
{
    assert(sf != NULL);

    return sf->s[sf->pos] != 0 && sf->pos <= sf->len;
}

char string_feeder_next(string_feeder_t *sf)
{
    assert(sf != NULL);
    assert(sf->pos <= sf->len);

    char c = sf->s[sf->pos++];
    if (c == '\n') {
        sf->line++;
        sf->col = 0;
    } else {
        sf->col++;
    }

    return c;
}

char *string_feeder_next_length(string_feeder_t *sf, size_t length)
{
    assert(sf != NULL);
    assert(length >= 0);
    assert(sf->pos <= sf->len);

    if (sf->pos + length > sf->len)
        length = sf->len - sf->pos;

    char *substr = calloc(length+1, sizeof(char));
    for (int i = 0 ; i < length ; i++)
        substr[i] = string_feeder_next(sf);
    return substr;
}

char string_feeder_peek(string_feeder_t *sf)
{
    assert(sf != NULL);
    assert(sf->pos <= sf->len);

    return sf->s[sf->pos];
}

char *string_feeder_peek_length(string_feeder_t *sf, size_t length)
{
    assert(sf != NULL);
    assert(length >= 0);
    assert(sf->pos <= sf->len);

    if (sf->pos + length > sf->len)
        length = sf->len - sf->pos;

    char *substr = calloc(length+1, sizeof(char));
    memcpy(substr, &sf->s[sf->pos], length*sizeof(char));
    return substr;
}

bool string_feeder_starts_with(string_feeder_t *sf, const char *str)
{
    assert(sf != NULL);
    assert(str != NULL);
    assert(sf->pos <= sf->len);

    return str_starts_with(&sf->s[sf->pos], str);
}

void string_feeder_require(string_feeder_t *sf, const char *str)
{
    assert(sf != NULL);
    assert(str != NULL);
    assert(sf->pos <= sf->len);

    size_t len = strlen(str);

    for (int i = 0; i < len; i++) {
        char c = string_feeder_next(sf);
        assert(c == str[i]);
    }
}

////////////////////////////////////////////
bool str_ends_with(const char *haystack, const char *needle)
{
    assert(haystack != NULL);
    assert(needle != NULL);

    size_t lens = strlen(haystack);
    size_t lenneedle = strlen(needle);

    if (lenneedle > lens)
        return false;

    return !strncmp(&haystack[lens - lenneedle], needle, lenneedle);
}

#ifndef _MSC_VER
inline
#endif
bool str_starts_with(const char *haystack, const char *needle)
{
    assert(haystack != NULL);
    assert(needle != NULL);

    // haystack[pos] doesn't have to be compared to zero; if it were
    // zero, it either doesn't match needle (in which case the loop
    // terminates) or it matches needle[pos] (in which case the loop
    // terminates).
    int pos = 0;
    while (haystack[pos] == needle[pos] && needle[pos] != 0)
        pos++;

    return (needle[pos] == 0);
}

bool str_starts_with_any(const char *haystack, const char **needles, int num_needles)
{
    assert(haystack != NULL);
    assert(needles != NULL);
    assert(num_needles >= 0);

    for (int i = 0; i < num_needles; i++) {
        assert(needles[i] != NULL);
        if (str_starts_with(haystack, needles[i]))
            return true;
    }

    return false;
}

bool str_matches_any(const char *haystack, const char **needles, int num_needles)
{
    assert(haystack != NULL);
    assert(needles != NULL);
    assert(num_needles >= 0);

    for (int i = 0; i < num_needles; i++) {
        assert(needles[i] != NULL);
        if (!strcmp(haystack, needles[i]))
            return true;
    }

    return false;
}

char *str_substring(const char *str, size_t startidx, long endidx)
{
    assert(str != NULL);
    assert(startidx >= 0 && startidx <= strlen(str)+1);
    assert(endidx < 0 || endidx >= startidx);
    assert(endidx < 0 || endidx <= strlen(str)+1);

    if (endidx < 0)
        endidx = (long) strlen(str);

    size_t blen = endidx - startidx; // not counting \0
    char *b = malloc(blen + 1);
    memcpy(b, &str[startidx], blen);
    b[blen] = 0;
    return b;
}

char *str_replace(const char *haystack, const char *needle, const char *replacement)
{
    assert(haystack != NULL);
    assert(needle != NULL);
    assert(replacement != NULL);

    string_buffer_t *sb = string_buffer_create();
    size_t haystack_len = strlen(haystack);
    size_t needle_len = strlen(needle);

    int pos = 0;
    while (pos < haystack_len) {
        if (needle_len > 0 && str_starts_with(&haystack[pos], needle)) {
            string_buffer_append_string(sb, replacement);
            pos += needle_len;
        } else {
            string_buffer_append(sb, haystack[pos]);
            pos++;
        }
    }
    if (needle_len == 0 && haystack_len == 0)
        string_buffer_append_string(sb, replacement);

    char *res = string_buffer_to_string(sb);
    string_buffer_destroy(sb);
    return res;
}

char *str_replace_many(const char *_haystack, ...)
{
    va_list ap;
    va_start(ap, _haystack);

    char *haystack = strdup(_haystack);

    while (true) {
        char *needle = va_arg(ap, char*);
        if (!needle)
            break;

        char *replacement = va_arg(ap, char*);
        char *tmp = str_replace(haystack, needle, replacement);
        free(haystack);
        haystack = tmp;
    }

    va_end(ap);

    return haystack;
}

static void buffer_appendf(char **_buf, int *bufpos, void *fmt, ...)
{
    char *buf = *_buf;
    va_list ap;

    int salloc = 128;
    char *s = malloc(salloc);

    va_start(ap, fmt);
    int slen = vsnprintf(s, salloc, fmt, ap);
    va_end(ap);

    if (slen >= salloc) {
        s = realloc(s, slen + 1);
        va_start(ap, fmt);
        vsprintf((char*) s, fmt, ap);
        va_end(ap);
    }

    buf = realloc(buf, *bufpos + slen + 1);
    *_buf = buf;

    memcpy(&buf[*bufpos], s, slen + 1); // get trailing \0
    (*bufpos) += slen;

    free(s);
}

static int is_variable_character(char c)
{
    if (c >= 'a' && c <= 'z')
        return 1;

    if (c >= 'A' && c <= 'Z')
        return 1;

    if (c >= '0' && c <= '9')
        return 1;

    if (c == '_')
        return 1;

    return 0;
}

char *str_expand_envs(const char *in)
{
    size_t inlen = strlen(in);
    size_t inpos = 0;

    char *out = NULL;
    int  outpos = 0;

    while (inpos < inlen) {

        if (in[inpos] != '$') {
            buffer_appendf(&out, &outpos, "%c", in[inpos]);
            inpos++;
            continue;

        } else {
            inpos++; // consume '$'

            char *varname = NULL;
            int  varnamepos = 0;

            while (inpos < inlen && is_variable_character(in[inpos])) {
                buffer_appendf(&varname, &varnamepos, "%c", in[inpos]);
                inpos++;
            }

            char *env = getenv(varname);
            if (env)
                buffer_appendf(&out, &outpos, "%s", env);

            free(varname);
        }
    }

    return out;
}
