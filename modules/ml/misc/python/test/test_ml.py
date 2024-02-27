#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests

class Bindings(NewOpenCVTests):

    def test_inheritance(self):

        boost = cv.ml.Boost_create()
        boost.getBoostType()  # from ml::Boost
        boost.getMaxDepth()  # from ml::DTrees
        boost.isClassifier()  # from ml::StatModel

class Arguments(NewOpenCVTests):

    def test_class_from_submodule_has_global_alias(self):
        self.assertTrue(hasattr(cv.ml, "Boost"),
                        msg="Class is not registered in the submodule")
        self.assertTrue(hasattr(cv, "ml_Boost"),
                        msg="Class from submodule doesn't have alias in the "
                        "global module")
        self.assertEqual(cv.ml.Boost, cv.ml_Boost,
                         msg="Classes from submodules and global module don't refer "
                         "to the same type")

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()