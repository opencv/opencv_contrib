#!/usr/bin/env bash
set -e

# ==============================================
# 1) Run with OpenCV SIFT detector + BF matcher:
# ==============================================
# python3 -m slam.monocular.main  \
#   --dataset tum-rgbd \
#   --base_dir ./Dataset \
#   --detector akaze \
#   --matcher bf \
#   --fps 10 \
#   --ransac_thresh 1.0 \
#   --no_viz3d

# ==============================================
# 2) (Alternative) Run with ALIKED + LightGlue:
# ==============================================
# python3 -m slam.monocular.main4 \
#   --dataset kitti \
#   --base_dir ./Dataset \
#   --use_lightglue \
#   --ransac_thresh 1.0
#   # --fps 10 \
#   # --no_viz3d


# # TEST
# python3 -m slam.monocular.main_revamped  \
#   --dataset kitti \
#   --base_dir ./Dataset \
#   --detector orb \
#   --matcher bf \
#   --fps 10 \
#   --ransac_thresh 1.0 \
#   --no_viz3d


python3 -m slam.monocular.main_revamped \
  --dataset kitti \
  --base_dir ./Dataset \
  --use_lightglue \
  # --fps 10 \
  # --no_viz3d