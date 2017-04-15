set(the_description "Separate world module containing all contrib modules")

set(OPENCV_MODULE_IS_PART_OF_WORLD FALSE)
set(BUILD_opencv_contrib_world_INIT OFF) # disabled by default

# add new submodules to this list
set(OPENCV_MODULE_CHILDREN
  aruco
  bgsegm
  bioinspired
  ccalib
  cvv
  datasets
  dnn
  dpm
  face
  fuzzy
  hdf
  line_descriptor
  optflow
  plot
  reg
  rgbd
  saliency
  sfm
  stereo
  structured_light
  surface_matching
  text
  tracking
  xfeatures2d
  ximgproc
  xobjdetect
  xphoto
)
ocv_list_add_prefix(OPENCV_MODULE_CHILDREN "opencv_")

ocv_define_module(contrib_world)

# ocv_add_module(contrib_world)

# set(link_deps "")
# foreach(m ${OPENCV_MODULE_opencv_contrib_world_CHILDREN})
#   list(APPEND link_deps ${OPENCV_MODULE_${m}_LINK_DEPS})
# endforeach()

#
# ocv_glob_module_sources()
# ocv_module_include_directories()
#
# ocv_create_module(${link_deps})
