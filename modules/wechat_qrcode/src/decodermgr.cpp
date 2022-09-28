// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
#include "precomp.hpp"
#include "decodermgr.hpp"


using zxing::ArrayRef;
using zxing::BinaryBitmap;
using zxing::DecodeHints;
using zxing::ErrorHandler;
using zxing::LuminanceSource;
using zxing::Ref;
using zxing::Result;
using zxing::UnicomBlock;
namespace cv {
namespace wechat_qrcode {
int DecoderMgr::decodeImage(cv::Mat src, bool use_nn_detector, vector<string>& results, vector<vector<Point2f>>& zxing_points) {
    int width = src.cols;
    int height = src.rows;
    if (width <= 20 || height <= 20)
        return -1;  // image data is not enough for providing reliable results

    std::vector<uint8_t> scaled_img_data(src.data, src.data + width * height);
    zxing::ArrayRef<uint8_t> scaled_img_zx =
        zxing::ArrayRef<uint8_t>(new zxing::Array<uint8_t>(scaled_img_data));

    vector<zxing::Ref<zxing::Result>> zx_results;

    decode_hints_.setUseNNDetector(use_nn_detector);

    Ref<ImgSource> source;
    qbarUicomBlock_ = new UnicomBlock(width, height);

    // Four Binarizers
    int tryBinarizeTime = 4;
    for (int tb = 0; tb < tryBinarizeTime; tb++) {
        if (source == NULL || height * width > source->getMaxSize()) {
            source = ImgSource::create(scaled_img_zx.data(), width, height);
        } else {
            source->reset(scaled_img_zx.data(), width, height);
        }
        int ret = TryDecode(source, zx_results);
        if (!ret) {
            for(size_t k = 0; k < zx_results.size(); k++) {
                results.emplace_back(zx_results[k]->getText()->getText());
                vector<Point2f> tmp_qr_points;
                auto tmp_zx_points = zx_results[k]->getResultPoints();
                for (int i = 0; i < tmp_zx_points->size() / 4; i++) {
                    const int ind = i * 4;
                    for (int j = 1; j < 4; j++){
                        tmp_qr_points.emplace_back(tmp_zx_points[ind + j]->getX(), tmp_zx_points[ind + j]->getY());
                    }
                    tmp_qr_points.emplace_back(tmp_zx_points[ind]->getX(), tmp_zx_points[ind]->getY());
                }
                zxing_points.push_back(tmp_qr_points);
            }
            return ret;
        }
        // try different binarizers
        binarizer_mgr_.SwitchBinarizer();
    }
    return -1;
}

int DecoderMgr::TryDecode(Ref<LuminanceSource> source, vector<Ref<Result>>& results) {
    int res = -1;
    string cell_result;

    // get binarizer
    zxing::Ref<zxing::Binarizer> binarizer = binarizer_mgr_.Binarize(source);
    zxing::Ref<zxing::BinaryBitmap> binary_bitmap(new BinaryBitmap(binarizer));
    binary_bitmap->m_poUnicomBlock = qbarUicomBlock_;

    results = Decode(binary_bitmap, decode_hints_);
    res = (results.size() == 0) ? 1 : 0;

    if (res == 0) {
        results[0]->setBinaryMethod(int(binarizer_mgr_.GetCurBinarizer()));
    }

    return res;
}

vector<Ref<Result>> DecoderMgr::Decode(Ref<BinaryBitmap> image, DecodeHints hints) {
    return reader_->decode(image, hints);
}
}  // namespace wechat_qrcode
}  // namespace cv