/*
 * Copyright 2014 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include "cudamat_conv_util.cuh"

cudaTextureObject_t getTextureObject(cudamat* mat) {
  if (mat->tex_obj == 0) {
    int size = mat->size[0] * mat->size[1] * sizeof(float);
    if (size <= TEXTURE_SIZE_MAX) {
      struct cudaResourceDesc resDesc;
      memset(&resDesc, 0, sizeof(resDesc));
      resDesc.resType = cudaResourceTypeLinear;
      resDesc.res.linear.devPtr = mat->data_device;
      resDesc.res.linear.sizeInBytes = size;
      resDesc.res.linear.desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
      struct cudaTextureDesc texDesc;
      memset(&texDesc, 0, sizeof(texDesc));
      cudaError_t err = cudaCreateTextureObject(&(mat->tex_obj), &resDesc, &texDesc, NULL);
      if (cudaSuccess != err) {
        fprintf(stderr, "Error creating texture object for matrix of shape %d %d.", mat->size[0], mat->size[1]);
        exit(EXIT_FAILURE);
      }
    }
    assert(mat->tex_obj != 0);  // If this assert is false, then we need to call a kernel which doesn't use textures.
  }
  return mat->tex_obj;
}

bool FitsAsTexture(cudamat* mat) {
  return ((mat->size[0] * mat->size[1] * sizeof(float)) <= TEXTURE_SIZE_MAX);
}
