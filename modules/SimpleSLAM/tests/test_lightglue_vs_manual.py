import pytest
import cv2
import numpy as np
import torch
from types import SimpleNamespace

# adjust this import to match your project layout
from slam.core.features_utils import (
    init_feature_pipeline,
    _lightglue_detect_and_match,
    feature_extractor,
    feature_matcher,
)

@pytest.fixture(scope="module")
def synthetic_pair():
    """
    Create two simple 200×200 BGR images, each with four white dots.
    The second is a small translation of the first.
    """
    img1 = np.zeros((200,200,3), dtype=np.uint8)
    img2 = np.zeros((200,200,3), dtype=np.uint8)
    pts = [(50,50), (150,50), (50,150), (150,150)]
    for x,y in pts:
        cv2.circle(img1, (x,y),  5, (255,255,255), -1)
        cv2.circle(img2, (x+5,y+3), 5, (255,255,255), -1)
    return img1, img2

def test_lightglue_pipeline_matches_manual(synthetic_pair):
    img1, img2 = synthetic_pair

    # build a fake args object
    args = SimpleNamespace(use_lightglue=True,
                           detector=None,  # not used for LG
                           matcher=None,
                           min_conf=0.0)

    # 1) initialize LightGlue extractor & matcher
    extractor, matcher = init_feature_pipeline(args)

    # 2) run the “direct” LG detect+match
    kp1_dir, kp2_dir, desc1_dir, desc2_dir, matches_dir = (
        _lightglue_detect_and_match(img1, img2, extractor, matcher)
    )

    # 3) run our new split API
    kp1_man, desc1_man = feature_extractor(args, img1, extractor)
    kp2_man, desc2_man = feature_extractor(args, img2, extractor)
    matches_man = feature_matcher(args,
                                  kp1_man, kp2_man,
                                  desc1_man, desc2_man,
                                  matcher)

    # --- Assertions ---

    # a) same number of keypoints & identical positions
    assert len(kp1_dir) == len(kp1_man)
    for kd, km in zip(kp1_dir, kp1_man):
        assert kd.pt == pytest.approx(km.pt)

    assert len(kp2_dir) == len(kp2_man)
    for kd, km in zip(kp2_dir, kp2_man):
        assert kd.pt == pytest.approx(km.pt)

    # b) descriptors are identical tensors
    #    (direct returns torch.Tensor, manual too)
    assert torch.allclose(desc1_dir, desc1_man)
    assert torch.allclose(desc2_dir, desc2_man)

    # c) same matches (queryIdx/trainIdx)
    assert len(matches_dir) == len(matches_man)
    for mdir, mman in zip(matches_dir, matches_man):
        assert mdir.queryIdx == mman.queryIdx
        assert mdir.trainIdx == mman.trainIdx
