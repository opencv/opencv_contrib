# test/CMakeLists.txt - cv_slam 测试程序

include_directories(/root/opencv_contrib_slam/test/cv_slam/tools)

# Phase 3: 消融实验测试
add_executable(test_ablation_study /root/opencv_contrib_slam/test/cv_slam/phase3_ablation_study/test_ablation_study.cpp)
target_link_libraries(test_ablation_study cv_slam opencv_imgcodecs pthread)

# Phase 4.1: 地图保存/加载测试
add_executable(test_map_save_load /root/opencv_contrib_slam/test/cv_slam/phase4_map_save_load/test_map_save_load.cpp)
target_link_libraries(test_map_save_load cv_slam opencv_imgcodecs pthread)

# Phase 4.3: Localization Mode 测试
add_executable(test_localization_mode /root/opencv_contrib_slam/test/cv_slam/phase4_map_save_load/test_localization_mode.cpp)
target_link_libraries(test_localization_mode cv_slam opencv_imgcodecs pthread)
