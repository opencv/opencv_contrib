#pragma once
class CmShow
{
public:
    static Mat HistBins(CMat& color3f, CMat& val, CStr& title, bool descendShow = false, CMat &with = Mat());
    static void showTinyMat(CStr &title, CMat &m);
    static inline void SaveShow(CMat& img, CStr& title);
};

