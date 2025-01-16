// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
#include "../../precomp.hpp"
#include "unicomblock.hpp"

namespace zxing {
short UnicomBlock::SEARCH_POS[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
UnicomBlock::UnicomBlock(int iMaxHeight, int iMaxWidth)
    : m_iHeight(iMaxHeight), m_iWidth(iMaxWidth), m_bInit(false) {}

UnicomBlock::~UnicomBlock() {}

void UnicomBlock::Init() {
    if (m_bInit) return;
    m_vcIndex = std::vector<unsigned int>(m_iHeight * m_iWidth, 0);
    m_vcCount = std::vector<unsigned int>(m_iHeight * m_iWidth, 0);
    m_vcMinPnt = std::vector<int>(m_iHeight * m_iWidth, 0);
    m_vcMaxPnt = std::vector<int>(m_iHeight * m_iWidth, 0);
    m_vcQueue = std::vector<int>(m_iHeight * m_iWidth, 0);
    m_bInit = true;
}

void UnicomBlock::Reset(Ref<BitMatrix> poImage) {
    m_poImage = poImage;
    memset(&m_vcIndex[0], 0, m_vcIndex.size() * sizeof(m_vcIndex[0]));
    m_iNowIdx = 0;
}

unsigned short UnicomBlock::GetUnicomBlockIndex(int y, int x) {
    if (y >= m_iHeight || x >= m_iWidth) return 0;
    if (m_vcIndex[y * m_iWidth + x]) return m_vcIndex[y * m_iWidth + x];
    Bfs(y, x);
    return m_vcIndex[y * m_iWidth + x];
}

int UnicomBlock::GetUnicomBlockSize(int y, int x) {
    if (y >= m_iHeight || x >= m_iWidth) return 0;
    if (m_vcIndex[y * m_iWidth + x]) return m_vcCount[y * m_iWidth + x];
    Bfs(y, x);
    return m_vcCount[y * m_iWidth + x];
}

int UnicomBlock::GetMinPoint(int y, int x, int &iMinY, int &iMinX) {
    if (y >= m_iHeight || x >= m_iWidth) return -1;
    if (m_vcIndex[y * m_iWidth + x]) {
        iMinY = m_vcMinPnt[y * m_iWidth + x] >> 16;
        iMinX = m_vcMinPnt[y * m_iWidth + x] & (0xFFFF);
        return 0;
    }
    Bfs(y, x);
    iMinY = m_vcMinPnt[y * m_iWidth + x] >> 16;
    iMinX = m_vcMinPnt[y * m_iWidth + x] & (0xFFFF);
    return 0;
}

int UnicomBlock::GetMaxPoint(int y, int x, int &iMaxY, int &iMaxX) {
    if (y >= m_iHeight || x >= m_iWidth) return -1;
    if (m_vcIndex[y * m_iWidth + x]) {
        iMaxY = m_vcMaxPnt[y * m_iWidth + x] >> 16;
        iMaxX = m_vcMaxPnt[y * m_iWidth + x] & (0xFFFF);
        return 0;
    }
    Bfs(y, x);
    iMaxY = m_vcMaxPnt[y * m_iWidth + x] >> 16;
    iMaxX = m_vcMaxPnt[y * m_iWidth + x] & (0xFFFF);
    return 0;
}

void UnicomBlock::Bfs(int y, int x) {
    m_iNowIdx++;

    int iFront = 0;
    int iTail = 0;
    int iCount = 1;

    int iMaxX = x, iMaxY = y;
    int iMinX = x, iMinY = y;

    const bool bValue = (m_poImage->get(x, y) != (unsigned char)0);

    m_vcIndex[y * m_iWidth + x] = m_iNowIdx;
    m_vcQueue[iTail++] = y << 16 | x;

    while (iFront < iTail) {
        int iNode = m_vcQueue[iFront++];
        int iX = iNode & (0xFFFF);
        int iY = iNode >> 16;
        iMaxX = max(iX, iMaxX);
        iMaxY = max(iY, iMaxY);
        iMinX = min(iX, iMinX);
        iMinY = min(iY, iMinY);

        iCount++;

        for (int i = 0; i < 4; ++i) {
            const int iNextX = iX + SEARCH_POS[i][0], iNextY = iY + SEARCH_POS[i][1];
            const int iPosition = iNextY * m_iWidth + iNextX;

            if (iPosition >= 0 && iPosition < int(m_vcIndex.size()) && 0 == m_vcIndex[iPosition]) {
                if (iNextX < 0 || iNextX >= m_poImage->getWidth() || iNextY < 0 ||
                    iNextY >= m_poImage->getHeight() ||
                    bValue != (m_poImage->get(iNextX, iNextY) != (unsigned char)0))
                    continue;

                m_vcIndex[iPosition] = m_iNowIdx;
                m_vcQueue[iTail++] = iNextY << 16 | iNextX;
            }
        }
    }

    if (iCount >= (1 << 16) - 1) iCount = 0xFFFF;

    const int iMinCombine = iMinY << 16 | iMinX;
    const int iMaxCombine = iMaxY << 16 | iMaxX;
    for (int i = 0; i < iTail; ++i) {
        const int iPosition = (m_vcQueue[i] >> 16) * m_iWidth + (m_vcQueue[i] & (0xFFFF));

        m_vcCount[iPosition] = iCount;
        m_vcMinPnt[iPosition] = iMinCombine;
        m_vcMaxPnt[iPosition] = iMaxCombine;
    }
}
}  // namespace zxing
