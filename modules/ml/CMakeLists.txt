set(the_description "Machine Learning")

ocv_add_module(ml opencv_core WRAP java objc python)
ocv_glob_module_sources()
ocv_module_include_directories()
ocv_create_module()

ocv_add_accuracy_tests()
ocv_add_perf_tests()
ocv_add_samples(opencv_imgproc opencv_objdetect opencv_video)