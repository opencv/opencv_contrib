FreeType2 Wrapper Module
========================

This FreeType2 wrapper module allows to draw strings with outlines and bitmaps.

Requested external libraries
----------------------------

harfbuzz is requested to convert UTF8 to gid(GlyphID).

freetype library is requested to rasterize given gid.

- harfbuzz https://www.freedesktop.org/wiki/Software/HarfBuzz/
- freetype https://www.freetype.org/

Usage
-----

```
cv::Ptr<cv::freetype::FreeType2> ft2;
ft2 = cv::freetype::createFreeType2();
ft2->loadFontData(ttf_pathname, 0);
ft2->putText(mat, "hello world", cv::Point(20, 200),
             30, CV_RGB(0, 0, 0), cv::FILLED, cv::LINE_AA, true);
```

Option
------
- 2nd argument of loadFontData is used if font file has many font data.
- 3 drawing mode is available.
    - outline mode is used if lineWidth is larger than 0. (like original putText)
    - bitmap  mode is used if lineWidth is less than 0.
        - 1bit bitmap mode is used if lineStyle is 4 or 8.
        - gray bitmap mode is used if lineStyle is 16.
