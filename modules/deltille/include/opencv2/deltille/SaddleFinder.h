
#include <opencv2/deltille/SaddlePoint.h>


#include <vector>

static int findSaddles(std::vector<SaddlePoint> &refclust) {
    std::vector<cv::Point> locations;

    getInitialSaddleLocations(input_lowres, locations);
    std::vector<SaddlePoint> refined;
    lowresFitting.saddleSubpixelRefinement(lowres, locations, refined,
                                           num_iterations, true);

    if (!SaddlePointType::isTriangular) {
      // early prefilter outliers
      double smax = -DBL_MAX;
      for (size_t i = 0; i < refined.size(); ++i)
        if (smax < refined[i].s)
          smax = refined[i].s;

      const double min_angle_width = 15.0 * M_PI / 180;
      const double max_angle_width = 75.0 * M_PI / 180;
      smax *= detector_params.rectangular_saddle_threshold;
      for (size_t i = 0; i < refined.size(); ++i) {
        if (refined[i].s < smax || std::abs(refined[i].a2) < min_angle_width ||
            std::abs(refined[i].a2) > max_angle_width)
          refined[i].x = refined[i].y = std::numeric_limits<double>::infinity();
      }
    }
    // squeeze out diverged points
    refined.resize(std::remove_if(refined.begin(), refined.end(),
                                  PointIsInf<SaddlePointType>) -
                   refined.begin());

#ifdef DEBUG_TIMING
    auto t2 = high_resolution_clock::now();
    printf("Refinement took: %.3f ms\n",
           duration_cast<microseconds>(t2 - t1).count() / 1000.0);
#endif
    int num_clusters = 0;
    std::vector<int> cluster_ids;

    std::vector<typename SaddlePointType::ClusterDescType> cluster_stats;
    clusterPoints2(refined, lowres.size(), cluster_ids, cluster_stats,
                   num_clusters, 1.0);
#ifdef DEBUG_TIMING
    auto t3 = high_resolution_clock::now();
    printf("Clustering took: %.3f ms\n",
           duration_cast<microseconds>(t3 - t2).count() / 1000.0);
#endif
    refclust.resize(cluster_stats.size());
    for (size_t i = 0; i < cluster_stats.size(); i++)
      refclust[i] = cluster_stats[i];

#ifdef DEBUG_INDEXING
    cv::cvtColor(input_lowres, DEBUG, CV_GRAY2RGB);
    for (size_t i = 0; i < refclust.size(); i++) {
      cv::Point2f pt;
      pt.x = refclust[i].x;
      pt.y = refclust[i].y;
      cv::rectangle(DEBUG, cv::Point((pt.x - 3) * 65536, (pt.y - 3) * 65536),
                    cv::Point((pt.x + 3) * 65536, (pt.y + 3) * 65536),
                    cv::Scalar(255, 0, 0), 1, CV_AA, 16);
      if (!SaddlePointType::isTriangular)
        refclust[i].plotPolarities(DEBUG, 3.0 / scaling);
    }
#endif

#ifdef DEBUG_TIMING
    auto t4 = high_resolution_clock::now();
#endif
    if (SaddlePointType::isTriangular) {
      // a bit hackier monkey saddle second filter, that checks if points would
      // converge to the same location with larger scale...
      PolynomialFit<SaddlePointType> tempFitting;
      tempFitting.initSaddleFitting(
          lowresFitting.getHalfKernelSize() +
          detector_params.deltille_stability_kernel_size_increase);

      std::vector<SaddlePointType> tempclust;
      cv::Mat temp;
      cv::filter2D(input_lowres, temp, cv::DataType<FloatImageType>::depth,
                   tempFitting.getSmoothingKernel());
      tempFitting.saddleSubpixelRefinement(temp, refclust, tempclust);
      for (size_t i = 0; i < refclust.size(); i++) {
        double dx = refclust[i].x - tempclust[i].x,
               dy = refclust[i].y - tempclust[i].y;
        if (dx * dx + dy * dy > detector_params.deltille_stability_threshold) {
          refclust[i].x = std::numeric_limits<double>::infinity();
          refclust[i].y = std::numeric_limits<double>::infinity();
        }
      }
    }
    // squeeze out any remaining unwanted points...
    refclust.resize(std::remove_if(refclust.begin(), refclust.end(),
                                   PointIsInf<SaddlePointType>) -
                    refclust.begin());
#ifdef DEBUG_INDEXING
    // show final detections
    for (size_t i = 0; i < refclust.size(); i++) {
      SaddlePointType &s1 = refclust[i];
      int sz = scaling * 3;
      cv::rectangle(DEBUG, cv::Point((s1.x - sz) * 65536, (s1.y - sz) * 65536),
                    cv::Point((s1.x + sz) * 65536, (s1.y + sz) * 65536),
                    cv::Scalar(0, 255, 0), 3.0 / scaling, CV_AA, 16);
      s1.plotPolarities(DEBUG, 3.0 / scaling);
      cv::putText(DEBUG, std::to_string(i), cv::Point((s1.x + sz), (s1.y + sz)),
                  cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(255, 0, 255));
    }
    imshow("detections", DEBUG);
#endif
#ifdef DEBUG_TIMING
    auto t5 = high_resolution_clock::now();
#endif
    return int(refclust.size());
  }
