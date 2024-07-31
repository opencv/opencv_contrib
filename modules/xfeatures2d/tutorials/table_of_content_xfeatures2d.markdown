Feature Detection and Description (xfeatures module) {#tutorial_table_of_content_xfeatures2d}
=============================================================================================

-   @subpage tutorial_py_surf_intro

    SIFT is really good,
    but not fast enough, so people came up with a speeded-up version called SURF.

-   @subpage tutorial_py_brief

    SIFT uses a feature
    descriptor with 128 floating point numbers. Consider thousands of such features. It takes lots of
    memory and more time for matching. We can compress it to make it faster. But still we have to
    calculate it first. There comes BRIEF which gives the shortcut to find binary descriptors with
    less memory, faster matching, still higher recognition rate.