//! @addtogroup ptcloud
//! @{

Point Cloud Module, Object Fitting API
=======================================

Try to segment geometric primitives like planes, spheres and cylinders from a 3d point cloud

2 alternative ransac strategies are implemented here:

- plain ransac loop:
  + generate random minimal hypothesis
  + find inliers for the model,
    - bail out if at 1/4 of the data it does not have more than 1/8 inliers of the current best model
    - if sprt is used, bail out if the probability of finding inliers goes low
  + if this model is the current best one
    - best model = current
    - update stopping criterion (optional SPRT)
    - apply local optimization (generate a non-minimal model from the inliers)
      + if it improved, take that instead

- preemptive ransac loop:
  + generate M minimal random models in advance
  + slice the data into blocks(of size B), for each block:
    - evaluate all M models in parallel
    - sort descending (by inliers or accumulated distance)
    - prune model list, M' = M * (2 ^ -(i/B))
    - stop if there is only one model left, or the last data block reached
  + polish/optimize the last remaining model

To Do
-----------------------------------------
- Integrate (better) with Maksym's work

//! @}