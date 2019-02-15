Improved Background-Foreground Segmentation Methods
===================================================

This algorithm combines statistical background image estimation and per-pixel Bayesian segmentation. It[1] was introduced by Andrew B. Godbehere, Akihiro Matsukawa, Ken Goldberg in 2012. As per the paper, the system ran a successful interactive audio art installation called "Are We There Yet?" from March 31 - July 31 2011 at the Contemporary Jewish Museum in San Francisco, California.

It uses first few (120 by default) frames for background modelling. It employs probabilistic foreground segmentation algorithm that identifies possible foreground objects using Bayesian inference. The estimates are adaptive; newer observations are more heavily weighted than old observations to accommodate variable illumination. Several morphological filtering operations like closing and opening are done to remove unwanted noise. You will get a black window during first few frames.

References
----------
[1]: A.B. Godbehere, A. Matsukawa, K. Goldberg. Visual tracking of human visitors under variable-lighting conditions for a responsive audio art installation. American Control Conference. (2012), pp. 4305–4312
