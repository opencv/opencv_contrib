SFM module installation {#tutorial_sfm_installation}
=======================

Dependencies
------------

The Structure from Motion module depends on some open source libraries.

  - [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) 3.2.2 or later. \b Required
  - [ng-log](https://github.com/ng-log/ng-log) 0.9.0 or later. \b Required, use instead of Google glog
  - [Google Logging Library(glog)](https://github.com/google/glog) 0.3.1 or later. \b Not-recommended
  - [Google Flags](https://github.com/gflags/gflags). \b Required
  - [Ceres Solver](http://ceres-solver.org). Needed by the reconstruction API in order to solve part of the Bundle Adjustment plus the points Intersect. If Ceres Solver is not installed on your system, the reconstruction funcionality will be disabled. \b Recommended

@warning
- Google glog is deprecated. Use ng-log 0.9.0 or later.
- If ng-log 0.8.x or earlier is used, many configuration and build warnings will be shown. Update ng-log 0.9.0 or later.
- If both ng-log and Google glog are installed, sfm uses Google glog. If you require ng-log, uninstall Google glog.


Installation
------------

__Required Dependencies__

In case you are on [Ubuntu](http://www.ubuntu.com) you can simply install the required dependencies by typing the following command:

@code{.bash}
  sudo apt-get install libeigen3-dev libgflags-dev
@endcode

__ng-log__

See https://ng-log.github.io/ng-log/stable/build/#cmake

Start by installing all the dependencies.

@code{.bash}
    # CMake
    sudo apt-get install cmake
    # gflags
    sudo apt-get install libgflags-dev
@endcode

We are now ready to build, test, and install ng-log.

@code{.bash}
    git clone https://github.com/ng-log/ng-log.git
    cd ng-log
    mkdir build && cd build
    cmake ..
    make -j4
    make test
    sudo make install
@endcode

__Ceres Solver__

Start by installing all the dependencies:

@code{.bash}
  # CMake
  sudo apt-get install cmake
  # google-glog + gflags
  sudo apt-get install libgoogle-glog-dev
  # BLAS & LAPACK
  sudo apt-get install libatlas-base-dev
  # Eigen3
  sudo apt-get install libeigen3-dev
  # SuiteSparse and CXSparse (optional)
  # - If you want to build Ceres as a *static* library (the default)
  #   you can use the SuiteSparse package in the main Ubuntu package
  #   repository:
  sudo apt-get install libsuitesparse-dev
  # - However, if you want to build Ceres as a *shared* library, you must
  #   add the following PPA:
  sudo add-apt-repository ppa:bzindovic/suitesparse-bugfix-1319687
  sudo apt-get update
  sudo apt-get install libsuitesparse-dev
@endcode

We are now ready to build, test, and install Ceres:

@code{.bash}
  git clone https://ceres-solver.googlesource.com/ceres-solver
  cd ceres-solver
  mkdir build && cd build
  cmake ..
  make -j4
  make test
  sudo make install
@endcode
