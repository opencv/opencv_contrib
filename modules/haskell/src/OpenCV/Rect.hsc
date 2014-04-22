{-# LANGUAGE ForeignFunctionInterface #-}
#include <bindings.dsl.h>
#include <opencv2/opencv.h>

module OpenCV.Rect where
#strict_import

import OpenCV.Types

#ccall cv_create_Rect     , IO (Ptr <Rect>)
#ccall cv_create_Rect4    , CInt -> CInt -> CInt -> CInt -> IO (Ptr <Rect>)
#ccall cv_Rect_assignTo   , Ptr <Rect> -> Ptr <Rect> -> IO (Ptr <Rect>)
#ccall cv_Rect_clone      , Ptr <Rect> -> IO (Ptr <Rect>)

#ccall cv_Rect_tl         , Ptr <Rect> -> IO (Ptr <Point>)
#ccall cv_Rect_br         , Ptr <Rect> -> IO (Ptr <Point>)
#ccall cv_Rect_getX       , Ptr <Rect> -> IO CInt
#ccall cv_Rect_getY       , Ptr <Rect> -> IO CInt
#ccall cv_Rect_getWidth   , Ptr <Rect> -> IO CInt
#ccall cv_Rect_getHeight  , Ptr <Rect> -> IO CInt
#ccall cv_Rect_size       , Ptr <Rect> -> IO (Ptr <Size>)
#ccall cv_Rect_contains   , Ptr <Rect> -> Ptr <Point> -> IO CInt
