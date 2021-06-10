// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.

#ifndef __ZXING_COMMON_UNICOMBLOCK_HPP__
#define __ZXING_COMMON_UNICOMBLOCK_HPP__
#include "bitmatrix.hpp"
#include "counted.hpp"

namespace zxing {
class UnicomBlock : public Counted {
public:
    UnicomBlock(int iMaxHeight, int iMaxWidth);
    ~UnicomBlock();

    void Init();
    void Reset(Ref<BitMatrix> poImage);

    unsigned short GetUnicomBlockIndex(int y, int x);

    int GetUnicomBlockSize(int y, int x);

    int GetMinPoint(int y, int x, int &iMinY, int &iMinX);
    int GetMaxPoint(int y, int x, int &iMaxY, int &iMaxX);

private:
    void Bfs(int y, int x);

    int m_iHeight;
    int m_iWidth;

    unsigned int m_iNowIdx;
    bool m_bInit;
    std::vector<unsigned int> m_vcIndex;
    std::vector<unsigned int> m_vcCount;
    std::vector<int> m_vcMinPnt;
    std::vector<int> m_vcMaxPnt;
    std::vector<int> m_vcQueue;
    static short SEARCH_POS[4][2];

    Ref<BitMatrix> m_poImage;
};
}  // namespace zxing
#endif  // __ZXING_COMMON_UNICOMBLOCK_HPP__
