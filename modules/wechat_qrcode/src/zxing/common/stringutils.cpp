// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").
#include "../../precomp.hpp"
#include "../common/stringutils.hpp"
#include "../decodehints.hpp"

using namespace zxing::common;

// N.B.: these are the iconv strings for at least some versions of iconv

char const* const StringUtils::PLATFORM_DEFAULT_ENCODING = "ANY";
char const* const StringUtils::ASCII = "ASCII";
char const* const StringUtils::SHIFT_JIS = "SHIFT-JIS";
char const* const StringUtils::GBK = "GBK";
char const* const StringUtils::EUC_JP = "EUC-JP";
char const* const StringUtils::UTF8 = "UTF-8";
char const* const StringUtils::ISO88591 = "ISO8859-1";
char const* const StringUtils::GB2312 = "GB2312";
char const* const StringUtils::BIG5 = "BIG5";
char const* const StringUtils::GB18030 = "GB18030";

const bool StringUtils::ASSUME_SHIFT_JIS = false;

#ifdef USE_UCHARDET
#include "uchardet/uchardet.h"
#endif

// Added convertString
#ifndef NO_ICONV
#include <iconv.h>

// Required for compatibility. TODO: test on Symbian
//#ifdef ZXING_ICONV_CONST
#undef ICONV_CONST
#define ICONV_CONST const
//#endif

#ifndef ICONV_CONST
#define ICONV_CONST /**/
#endif

// Add this to fix both Mac and Windows compilers
// by Skylook
template <class T>
class sloppy {};

// convert between T** and const T**
template <class T>
class sloppy<T**> {
    T** t;

public:
    sloppy(T** mt) : t(mt) {}
    sloppy(const T** mt) : t(const_cast<T**>(mt)) {}

    operator T* *() const { return t; }
    operator const T* *() const { return const_cast<const T**>(t); }
};
#endif

string StringUtils::convertString(const char* rawData, int length, const char* fromCharset,
                                  const char* toCharset) {
    string result;
    const char* bufIn = rawData;
    int nIn = length;

    // If from and to charset are the same, return
    int ret = strcmp(fromCharset, toCharset);
    if (ret == 0) {
        result.append((const char*)bufIn, nIn);
        return result;
    }

#ifndef NO_ICONV
    if (nIn == 0) {
        return "";
    }
    iconv_t cd;
    // cout<<src<<endl;
    cd = iconv_open(toCharset, fromCharset);

    // iconv_t cd = iconv_open(StringUtils::GBK, src);
    if (cd == (iconv_t)-1) {
        // result.append((const char *)bufIn, nIn);
        result = "";
        return result;
    }

    const int maxOut = 4 * nIn + 1;
    char* bufOut = new char[maxOut];

    ICONV_CONST char* fromPtr = (ICONV_CONST char*)bufIn;
    size_t nFrom = nIn;
    char* toPtr = (char*)bufOut;
    size_t nTo = maxOut;

    size_t oneway = -1;

    if (nFrom > 0) {
        // size_t oneway = iconv(cd, &fromPtr, &nFrom, &toPtr, &nTo);
        oneway = iconv(cd, sloppy<char**>(&fromPtr), &nFrom, sloppy<char**>(&toPtr), &nTo);
    }
    iconv_close(cd);

    int nResult = maxOut - nTo;
    bufOut[nResult] = '\0';
    result.append((const char*)bufOut);
    delete[] bufOut;

    // Cannot convert string
    if (oneway == (size_t)(-1)) {
        // result.append((const char *)bufIn, nIn);
        result = "";
    }
#else
    result.append((const char*)bufIn, nIn);
#endif

    return result;
}

string StringUtils::guessEncoding(char* bytes, int length) {
#ifdef USE_UCHARDET
    if (length < 10) {
        return guessEncodingZXing(bytes, length);
    } else {
        return guessEncodingUCharDet(bytes, length);
    }
#else
    return guessEncodingZXing(bytes, length);
#endif
}

#ifdef USE_UCHARDET

string StringUtils::guessEncodingUCharDet(char* bytes, int length) {
    uchardet_t handle = uchardet_new();

    int retval = uchardet_handle_data(handle, bytes, length);

    if (retval != 0) {
        fprintf(stderr, "Handle data error.\n");
        exit(0);
    }

    uchardet_data_end(handle);

    const char* charset = uchardet_get_charset(handle);

    string charsetStr(charset);

    uchardet_delete(handle);

    if (charsetStr.size() != 0) {
        return charsetStr;
    } else {
        return guessEncodingZXing(bytes, length);
    }

    // Otherwise, we take a wild guess with platform encoding
    // return PLATFORM_DEFAULT_ENCODING;
}
#endif

string StringUtils::guessEncodingZXing(char* bytes, int length) {
    //
    // typedef bool boolean;
    // For now, merely tries to distinguish ISO-8859-1, UTF-8 and Shift_JIS,
    // which should be by far the most common encodings.
    bool canBeISO88591 = true;
    bool canBeShiftJIS = true;
    bool canBeUTF8 = true;
    bool canBeGB2312 = true;
    bool canBeGBK = true;
    bool canBeBIG5 = true;
    bool canBeASCII = true;

    int utf8BytesLeft = 0;
    int utf2BytesChars = 0;
    int utf3BytesChars = 0;
    int utf4BytesChars = 0;
    int sjisBytesLeft = 0;
    int sjisKatakanaChars = 0;
    int sjisCurKatakanaWordLength = 0;
    int sjisCurDoubleBytesWordLength = 0;
    int sjisMaxKatakanaWordLength = 0;
    int sjisMaxDoubleBytesWordLength = 0;
    int isoHighOther = 0;

    int gb2312SCByteChars = 0;
    int big5TWBytesChars = 0;

    bool utf8bom = length > 3 && (unsigned char)bytes[0] == 0xEF &&
                   (unsigned char)bytes[1] == 0xBB && (unsigned char)bytes[2] == 0xBF;

    for (int i = 0; i < length && (canBeISO88591 || canBeShiftJIS || canBeUTF8 || canBeGBK); i++) {
        int value = bytes[i] & 0xFF;

        // UTF-8 stuff
        if (canBeUTF8) {
            if (utf8BytesLeft > 0) {
                if ((value & 0x80) == 0) {
                    canBeUTF8 = false;
                } else {
                    utf8BytesLeft--;
                }
            } else if ((value & 0x80) != 0) {
                if ((value & 0x40) == 0) {
                    canBeUTF8 = false;
                } else {
                    utf8BytesLeft++;
                    if ((value & 0x20) == 0) {
                        utf2BytesChars++;
                    } else {
                        utf8BytesLeft++;
                        if ((value & 0x10) == 0) {
                            utf3BytesChars++;
                        } else {
                            utf8BytesLeft++;
                            if ((value & 0x08) == 0) {
                                utf4BytesChars++;
                            } else {
                                canBeUTF8 = false;
                            }
                        }
                    }
                }
            }
        }
        // Shift_JIS stuff
        if (canBeShiftJIS) {
            if (sjisBytesLeft > 0) {
                if (value < 0x40 || value == 0x7F || value > 0xFC) {
                    canBeShiftJIS = false;
                } else {
                    sjisBytesLeft--;
                }
            } else if (value == 0x80 || value == 0xA0 || value > 0xEF) {
                canBeShiftJIS = false;
            } else if (value > 0xA0 && value < 0xE0) {
                sjisKatakanaChars++;
                sjisCurDoubleBytesWordLength = 0;
                sjisCurKatakanaWordLength++;
                if (sjisCurKatakanaWordLength > sjisMaxKatakanaWordLength) {
                    sjisMaxKatakanaWordLength = sjisCurKatakanaWordLength;
                }
            } else if (value > 0x7F) {
                sjisBytesLeft++;
                // sjisDoubleBytesChars++;
                sjisCurKatakanaWordLength = 0;
                sjisCurDoubleBytesWordLength++;
                if (sjisCurDoubleBytesWordLength > sjisMaxDoubleBytesWordLength) {
                    sjisMaxDoubleBytesWordLength = sjisCurDoubleBytesWordLength;
                }
            } else {
                // sjisLowChars++;
                sjisCurKatakanaWordLength = 0;
                sjisCurDoubleBytesWordLength = 0;
            }
        }

        // ISO-8859-1 stuff
        if (canBeISO88591) {
            if (value > 0x7F && value < 0xA0) {
                canBeISO88591 = false;
            } else if (value > 0x9F) {
                if (value < 0xC0 || value == 0xD7 || value == 0xF7) {
                    isoHighOther++;
                }
            }
        }
    }

    // Get how many chinese sc & tw words
    gb2312SCByteChars = is_gb2312_code(bytes, length);
    big5TWBytesChars = is_big5_code(bytes, length);

    if (gb2312SCByteChars <= 0) {
        canBeGB2312 = false;
    }

    if (big5TWBytesChars <= 0) {
        canBeBIG5 = false;
    }

    if (!is_gbk_code(bytes, length)) {
        canBeGBK = false;
    }

    if (canBeUTF8 && utf8BytesLeft > 0) {
        canBeUTF8 = false;
    }
    if (canBeShiftJIS && sjisBytesLeft > 0) {
        canBeShiftJIS = false;
    }

    if (is_ascii_code(bytes, length) <= 0) {
        canBeASCII = false;
    }

    // Easy -- if there is BOM or at least 1 valid not-single byte character
    // (and no evidence it can't be UTF-8), done
    if (canBeUTF8 && (utf8bom || utf2BytesChars + utf3BytesChars + utf4BytesChars > 0)) {
        return UTF8;
    }

    // if ( canBeBIG5 == false && canBeGB2312 == false )
    int chineseWordLen =
        gb2312SCByteChars > big5TWBytesChars ? gb2312SCByteChars : big5TWBytesChars;
    int chineseByteLen = chineseWordLen * 2;
    int japaneseByteLen = sjisMaxKatakanaWordLength + sjisMaxDoubleBytesWordLength * 2;

    // if ( chineseByteLen < japaneseByteLen || (japaneseByteLen == 0 &&
    // chineseByteLen == 0) ) if ( (gb2312SCByteChars < sjisKatakanaChars) &&
    // (big5TWBytesChars < sjisKatakanaChars) )
    //{
    // Easy -- if assuming Shift_JIS or at least 3 valid consecutive not-ascii
    // characters (and no evidence it can't be), done
    if (canBeShiftJIS &&
        (ASSUME_SHIFT_JIS || sjisMaxKatakanaWordLength >= 3 || sjisMaxDoubleBytesWordLength >= 3)) {
        // return SHIFT_JIS;
        if (chineseByteLen <= japaneseByteLen) {
            if (chineseByteLen == japaneseByteLen) {
                if (chineseWordLen < sjisKatakanaChars) {
                    return SHIFT_JIS;
                }
            } else {
                return SHIFT_JIS;
            }
        }
    }

    // Distinguishing Shift_JIS and ISO-8859-1 can be a little tough for short
    // words. The crude heuristic is:
    // - If we saw
    //   - only two consecutive katakana chars in the whole text, or
    //   - at least 10% of bytes that could be "upper" not-alphanumeric Latin1,
    // - then we conclude Shift_JIS, else ISO-8859-1
    if (canBeISO88591 && canBeShiftJIS) {
        if ((sjisMaxKatakanaWordLength == 2 && sjisKatakanaChars == 2) ||
            isoHighOther * 10 >= length) {
            /*
            if ( chineseByteLen < japaneseByteLen )
            {
                return SHIFT_JIS;
            }
            */
            if (chineseByteLen <= japaneseByteLen) {
                if (chineseByteLen == japaneseByteLen) {
                    if (chineseWordLen < sjisKatakanaChars) {
                        return SHIFT_JIS;
                    }
                } else {
                    return SHIFT_JIS;
                }
            }
        } else {
            if (chineseByteLen <= 0 && !canBeGB2312 && !canBeBIG5) {
                return ISO88591;
            }
        }
    }
    //}

    // Otherwise, try in order ISO-8859-1, Shift JIS, UTF-8 and fall back to
    // default platform encoding
    if (canBeGB2312) {
        return GB2312;
    }

    if (canBeBIG5) {
        return BIG5;
    }

    if (canBeShiftJIS) {
        return SHIFT_JIS;
    }

    if (canBeGBK) {
        return GBK;
    }

    if (canBeISO88591) {
        return ISO88591;
    }

    if (canBeUTF8) {
        return UTF8;
    }

    if (canBeASCII) {
        return ASCII;
    }

    // Otherwise, we take a wild guess with platform encoding
    return PLATFORM_DEFAULT_ENCODING;
}

// judge the byte whether begin with binary 10
int StringUtils::is_utf8_special_byte(unsigned char c) {
    unsigned char special_byte = 0X02;  // binary 00000010
    if (c >> 6 == special_byte) {
        return 1;
    } else {
        return 0;
    }
}

int StringUtils::is_utf8_code(char* str, int length) {
    unsigned char one_byte = 0X00;    // binary 00000000
    unsigned char two_byte = 0X06;    // binary 00000110
    unsigned char three_byte = 0X0E;  // binary 00001110
    unsigned char four_byte = 0X1E;   // binary 00011110
    unsigned char five_byte = 0X3E;   // binary 00111110
    unsigned char six_byte = 0X7E;    // binary 01111110

    int utf8_yes = 0;
    int utf8_no = 0;

    unsigned char k = 0;
    unsigned char m = 0;
    unsigned char n = 0;
    unsigned char p = 0;
    unsigned char q = 0;

    unsigned char c = 0;
    for (int i = 0; i < length;) {
        c = (unsigned char)str[i];
        if (c >> 7 == one_byte) {
            i++;
            continue;
        } else if (c >> 5 == two_byte) {
            k = (unsigned char)str[i + 1];
            if (is_utf8_special_byte(k)) {
                utf8_yes++;
                i += 2;
                continue;
            }
        } else if (c >> 4 == three_byte) {
            m = (unsigned char)str[i + 1];
            n = (unsigned char)str[i + 2];
            if (is_utf8_special_byte(m) && is_utf8_special_byte(n)) {
                utf8_yes++;
                i += 3;
                continue;
            }
        } else if (c >> 3 == four_byte) {
            k = (unsigned char)str[i + 1];
            m = (unsigned char)str[i + 2];
            n = (unsigned char)str[i + 3];
            if (is_utf8_special_byte(k) && is_utf8_special_byte(m) && is_utf8_special_byte(n)) {
                utf8_yes++;
                i += 4;
                continue;
            }
        } else if (c >> 2 == five_byte) {
            k = (unsigned char)str[i + 1];
            m = (unsigned char)str[i + 2];
            n = (unsigned char)str[i + 3];
            p = (unsigned char)str[i + 4];
            if (is_utf8_special_byte(k) && is_utf8_special_byte(m) && is_utf8_special_byte(n) &&
                is_utf8_special_byte(p)) {
                utf8_yes++;
                i += 5;
                continue;
            }
        } else if (c >> 1 == six_byte) {
            k = (unsigned char)str[i + 1];
            m = (unsigned char)str[i + 2];
            n = (unsigned char)str[i + 3];
            p = (unsigned char)str[i + 4];
            q = (unsigned char)str[i + 5];
            if (is_utf8_special_byte(k) && is_utf8_special_byte(m) && is_utf8_special_byte(n) &&
                is_utf8_special_byte(p) && is_utf8_special_byte(q)) {
                utf8_yes++;
                i += 6;
                continue;
            }
        }

        utf8_no++;
        i++;
    }

    // printf("uft8_yes: %d utf8_no:%d\n", utf8_yes, utf8_no);
    if ((utf8_yes + utf8_no) != 0) {
        int ret = (100 * utf8_yes) / (utf8_yes + utf8_no);
        if (ret > 90) {
            return 1;
        } else {
            return 0;
        }
    }
    return 0;
}
int StringUtils::is_gb2312_code(char* str, int length) {
    unsigned char one_byte = 0X00;  // binary 00000000

    int gb2312_yes = 0;
    int gb2312_no = 0;

    unsigned char k = 0;

    unsigned char c = 0;
    for (int i = 0; i < length;) {
        c = (unsigned char)str[i];
        if (c >> 7 == one_byte) {
            i++;
            continue;
        } else if (c >= 0XA1 && c <= 0XF7) {
            k = (unsigned char)str[i + 1];
            if (k >= 0XA1 && k <= 0XFE) {
                gb2312_yes++;
                i += 2;
                continue;
            }
        }

        gb2312_no++;
        i += 2;
    }

    // printf("gb2312_yes: %d gb2312_no:%d\n", gb2312_yes, gb2312_no);
    if ((gb2312_yes + gb2312_no) > 0) {
        int ret = (100 * gb2312_yes) / (gb2312_yes + gb2312_no);
        if (ret == 100) {
            // if (ret > 90) {
            // gb2312SCByteChars = gb2312_yes;
            return gb2312_yes;
        } else {
            return 0;
        }
    }
    return 0;
}

int StringUtils::is_big5_code(char* str, int length) {
    unsigned char one_byte = 0X00;  // binary 00000000

    int big5_yes = 0;
    int big5_no = 0;

    unsigned char k = 0;

    unsigned char c = 0;
    for (int i = 0; i < length;) {
        c = (unsigned char)str[i];
        if (c >> 7 == one_byte) {
            i++;
            continue;
        } else if (c >= 0XA1 && c <= 0XF9) {
            k = (unsigned char)str[i + 1];
            if ((k >= 0X40 && k <= 0X7E) || (k >= 0XA1 && k <= 0XFE)) {
                big5_yes++;
                i += 2;
                continue;
            }
        }

        big5_no++;
        i += 2;
    }

    // printf("%d %d\n", big5_yes, big5_no);
    if ((big5_yes + big5_no) > 0) {
        int ret = (100 * big5_yes) / (big5_yes + big5_no);
        if (ret == 100) {
            // if (ret > 90) {
            // big5TWBytesChars = big5_yes;
            return big5_yes;
        } else {
            return 0;
        }
    }
    return 0;
}

int StringUtils::is_gbk_code(char* str, int length) {
    unsigned char one_byte = 0X00;  // binary 00000000

    int gbk_yes = 0;
    int gbk_no = 0;

    unsigned char k = 0;

    unsigned char c = 0;
    for (int i = 0; i < length;) {
        c = (unsigned char)str[i];
        if (c >> 7 == one_byte) {
            i++;
            continue;
        } else if (c >= 0X81 && c <= 0XFE) {
            k = (unsigned char)str[i + 1];
            if (k >= 0X40 && k <= 0XFE) {
                gbk_yes++;
                i += 2;
                continue;
            }
        }

        gbk_no++;
        i += 2;
    }

    // printf("gbk_yes: %d gbk_no:%d\n", gbk_yes, gbk_no);
    if ((gbk_yes + gbk_no) > 0) {
        int ret = (100 * gbk_yes) / (gbk_yes + gbk_no);
        if (ret == 100) {
            // if (ret > 90) {
            return 1;
        } else {
            return 0;
        }
    }
    return 0;
}

int StringUtils::is_ascii_code(char* str, int length) {
    unsigned char c = 0;

    bool isASCII = true;

    for (int i = 0; i < length; i++) {
        c = (unsigned char)str[i];

        if ((c > 127)) {
            isASCII = false;
        }
    }
    return (isASCII ? 1 : -1);
}

//#define DEBUG

int StringUtils::shift_jis_to_jis(const unsigned char* may_be_shift_jis, int* jis_first_ptr,
                                  int* jis_second_ptr) {
    int status = 0;
    unsigned char first = may_be_shift_jis[0];
    unsigned char second = may_be_shift_jis[1];
    int jis_first = 0;
    int jis_second = 0;
    /* Check first byte is valid shift JIS. */
    if ((first >= 0x81 && first <= 0x84) || (first >= 0x87 && first <= 0x9f)) {
        jis_first = 2 * (first - 0x70) - 1;
        if (second >= 0x40 && second <= 0x9e) {
            jis_second = second - 31;
            if (jis_second > 95) {
                jis_second -= 1;
            }

            status = 1;
        } else if (second >= 0x9f && second <= 0xfc) {
            jis_second = second - 126;
            jis_first += 1;
            status = 1;
        } else {
        }
    } else if (first >= 0xe0 && first <= 0xef) {
        jis_first = 2 * (first - 0xb0) - 1;
        if (second >= 0x40 && second <= 0x9e) {
            jis_second = second - 31;
            if (jis_second > 95) {
                jis_second -= 1;
            }
            status = 1;
        } else if (second >= 0x9f && second <= 0xfc) {
            jis_second = second - 126;
            jis_first += 1;
            status = 1;
        }
    } else {
    }
    *jis_first_ptr = jis_first;
    *jis_second_ptr = jis_second;
    return status;
}
