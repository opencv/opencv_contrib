function modelConvertToBin(model, outname)
% usage:
%  1. Download the pretrained model file "modelFinal.mat"
%  from https://github.com/csukuangfj/epicflow/blob/master/EpicFlow_v1.00/modelFinal.mat
%  or train the model by yourself.
%
%  2. execute the following statements:
%   load('modelFinal.mat');
%   modelConvertToBin(model, 'model')
%
%  3. A model file named "model.bin" is generated.
%
%  4. Pass the model file to the C++ code
%   cv::Ptr<cv::ximgproc::StructuredEdgeDetection> pDollar = cv::ximgproc::StructuredEdgeDetection::create("model.bin");
%
%  5. Refer to the tutorial <http://docs.opencv.org/trunk/d0/da5/tutorial_ximgproc_prediction.html> and pass
%     "model.bin" as the model file name. Notice that the extension should be ".bin"
%
% author: KUANG, Fangjun
% email address: csukuangfj@gmail.com
%
  len_ = length(outname);
  if (len_ < 4) || (outname(end-3) ~= '.') || (outname(end-2) ~= 'b') || ...
     (outname(end-1) ~= 'i') || (outname(end) ~= 'n')
    outname = [outname, '.bin'];
  end

  f = fopen(outname, 'w');
  numberOfTrees = 8;
  numberOfTreesToEvaluate = 4;
  selfsimilarityGridSize = 5;
  stride = 2;
  shrinkNumber = 2;
  patchSize = 32;
  patchInnerSize = 16;
  numberOfGradientOrientations = 4;
  gradientSmoothingRadius = 0;
  regFeatureSmoothingRadius = 2;
  ssFeatureSmoothingRadius = 8;
  gradientNormalizationRadius = 4;

  fwrite(f, stride, 'int32');
  fwrite(f, shrinkNumber, 'int32');
  fwrite(f, patchSize, 'int32');
  fwrite(f, patchInnerSize, 'int32');
  fwrite(f, numberOfGradientOrientations, 'int32');
  fwrite(f, gradientSmoothingRadius, 'int32');
  fwrite(f, regFeatureSmoothingRadius, 'int32');
  fwrite(f, ssFeatureSmoothingRadius, 'int32');
  fwrite(f, gradientNormalizationRadius, 'int32');
  fwrite(f, selfsimilarityGridSize, 'int32');
  fwrite(f, numberOfTrees, 'int32');
  fwrite(f, numberOfTreesToEvaluate, 'int32');

  fwrite(f, prod(size(model.child)), 'uint64');
  fwrite(f, model.child(:), 'int32');

  fwrite(f, prod(size(model.fids)), 'uint64');
  fwrite(f, model.fids(:), 'int32');

  fwrite(f, prod(size(model.thrs)), 'uint64');
  fwrite(f, model.thrs(:), 'float');

  fwrite(f, prod(size(model.eBnds)), 'uint64');
  fwrite(f, model.eBnds(:), 'int32');

  fwrite(f, prod(size(model.eBins)), 'uint64');
  fwrite(f, model.eBins(:), 'int32');

  fclose(f);
end
