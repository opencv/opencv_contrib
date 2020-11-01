# Computer Vision based Alpha Matting

This project was part of the Google Summer of Code 2019.

####Student: Muskaan Kularia
####Mentor: Sunita Nayak
***
Alphamatting is the problem of extracting the foreground from an image. Given the input of an image and its corresponding trimap, we try to extract the foreground from the background.

This project is implementation of "[Designing Effective Inter-Pixel Information Flow for Natural Image Matting](https://www.researchgate.net/publication/318489370_Designing_Effective_Inter-Pixel_Information_Flow_for_Natural_Image_Matting)" by Yağız Aksoy, Tunç Ozan Aydın and Marc Pollefeys[1]. It required implementation of parts of other papers [2,3,4].


## References

[1] Yagiz Aksoy, Tunc Ozan Aydin, Marc Pollefeys, "[Designing Effective Inter-Pixel Information Flow for Natural Image Matting](https://www.researchgate.net/publication/318489370_Designing_Effective_Inter-Pixel_Information_Flow_for_Natural_Image_Matting)", CVPR, 2017.

[2] Roweis, Sam T., and Lawrence K. Saul. "[Nonlinear dimensionality reduction by locally linear embedding](https://science.sciencemag.org/content/290/5500/2323)" Science 290.5500 (2000): 2323-2326.

[3] Anat Levin, Dani Lischinski, Yair Weiss, "[A Closed Form Solution to Natural Image Matting](https://www.researchgate.net/publication/5764820_A_Closed-Form_Solution_to_Natural_Image_Matting)", IEEE TPAMI, 2008.

[4] Qifeng Chen, Dingzeyu Li, Chi-Keung Tang, "[KNN Matting](http://dingzeyu.li/files/knn-matting-tpami.pdf)", IEEE TPAMI, 2013.

[5] Yagiz Aksoy, "[Affinity Based Matting Toolbox](https://github.com/yaksoy/AffinityBasedMattingToolbox)".
