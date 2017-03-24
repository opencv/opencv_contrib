FreeType Module
===========

This FreeType module allows you to draw strings with outlines and bitmaps.

Installation
-----------
harfbuzz is requested to convert UTF8 to gid(GlyphID).
freetype library is requested to rasterize given gid.

harfbuzz https://www.freedesktop.org/wiki/Software/HarfBuzz/
freetype https://www.freetype.org/

Usage
-----------
cv::freetype::FreeType2 ft2;
ft2.loadFontData("your-font.ttf", 0);
ft2.setSplitNumber( 4 ); // Bezier-line is splited by 4 segment.
ft2.putText(src, .... )

Option
------------
- 2nd argument of loadFontData is used if font file has many font data.
- 3 drawing mode is available.
-- outline mode is used if lineWidth is larger than 0. (like original putText)
-- bitmap  mode is used if lineWidth is less than 0.
--- 1bit bitmap mode is used if lineStyle is 4 or 8.
--- gray bitmap mode is used if lineStyle is 16.

Future work
------------
- test
-- CJK and ...
- RTL,LTR,TTB,BTT...
