# Textons - Texture Based Image Segmentation
Textons using LM filters

The code here is an implementation of textons using LM filters.

The image is mapped into a 48 dimensions vector space using the features obtained from the LM filters.

Three more features are added to the end of each vector. These features are row, col and centroids allocated for each pixel 

K-Means using Eculidean distance is used to segment image.

inputs:
1. image ==> numpy array (grayscaled image)
2. number of cluster centers ==> integer
3. number of iterations ==> integers
4. type of colors assignment in the final image ==> integer 0 for 'RANDOM' or 1 for'DEFINED'
            
output:
* numpy array of image after k means wiht LM filters
            
Assignment type 'RANDOM' is suited more for higher number of clusters,
assignment type 'DEFINED' is better suited when number of clusters is less than 15,
however the assignemnt types do not impact the performance of the code in any significant way


To run the script 

```
from textons_utils import Textons

im = cv2.imread('path_to_image', 0)
textons = Textons(im, number of clusters, number of iterations, assignemnet type)
tex = textons.textons()
cv2.imshow("window name", tex)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Example 
```
from textons_utils import Textons

im = cv2.imread('image.jpg', 0)
textons = Textons(im, 5, 25, 1)
tex = textons.textons()
cv2.imshow("check", tex)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Results with cluster values as 7 and iterations as 10

![original image](https://github.com/BATspock/Textons/blob/master/image.jpg) ![resulting image](https://github.com/BATspock/Textons/blob/master/result.png)


*  Advisable to keep number of iterations high (recommended>=100) for reproducibility of result
