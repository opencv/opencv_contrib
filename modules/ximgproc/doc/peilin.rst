PeiLinNormalization
-------------------
Calculates an affine transformation normalizing given image.

.. ocv:function:: Matx23d PeiLinNormalization ( InputArray I )
.. ocv:pyfunction:: cv2.PeiLinNormalization ( InputArray I ) -> Matx23d

Assume given image :math:`I=T(\bar{I})` where :math:`\bar{I}` is a normalized image and :math:`T` is is an affine transformation distorting this image by translation, rotation, scaling and skew.
The function returns an affine transformation matrix corresponding to the transformation :math:`T^{-1}` described in [PeiLin95].

.. [PeiLin95] Soo-Chang Pei and Chao-Nan Lin. Image normalization for pattern recognition. Image and Vision Computing, Vol. 13, N.10, pp. 711-723, 1995.
