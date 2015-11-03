#! /bin/bash

echo "just for reference"
exit 0

for idx in 0 1 2 3 4; do ./find_fooling_image.py --push_idx $idx --N 1500 --decay .03 --lr .001 --prefix 'result_idx3/idx_%(push_idx)03d_decay_%(decay).03f_lr_%(lr).03f_'; done
for idx in 0 1 2 3 4; do ./find_fooling_image.py --push_idx $idx --N 1500 --decay .00 --lr .001 --prefix 'result_idx3/idx_%(push_idx)03d_decay_%(decay).03f_lr_%(lr).03f_'; done
