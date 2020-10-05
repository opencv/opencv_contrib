//
// Created by YIMIN TANG on 9/18/20.
//


//
// Author: Stanislaw Adaszewski, 2019
//

#include "concaveman.h"

extern "C" {
void pyconcaveman2d(double *points_c, size_t num_points,
                    int *hull_points_c, size_t num_hull_points,
                    double concavity, double lengthThreshold,
                    double **concave_points_c, size_t *num_concave_points,
                    void(**p_free)(void*));
}

void pyconcaveman2d(double *points_c, size_t num_points,
                    int *hull_points_c, size_t num_hull_points,
                    double concavity, double lengthThreshold,
                    double **p_concave_points_c,
                    size_t *p_num_concave_points,
                    void(**p_free)(void*)) {

    std::cout << "pyconcaveman2d(), concavity: " << concavity <<
              " lengthThreshold: " << lengthThreshold << std::endl;

    typedef double T;
    typedef std::array<T, 2> point_type;

    std::vector<point_type> points(num_points);
    for (auto i = 0; i < num_points; i++) {
        points[i] = { points_c[i << 1], points_c[(i << 1) + 1] };
    }

    std::cout << "points:" << std::endl;
    for (auto &p : points)
        std::cout << p[0] << " " << p[1] << std::endl;

    std::vector<int> hull(num_hull_points);
    for (auto i = 0; i < num_hull_points; i++) {
        hull[i] = hull_points_c[i];
    }

    std::cout << "hull:" << std::endl;
    for (auto &i : hull)
        std::cout << i << std::endl;

    auto concave_points = concaveman<T, 16>(points, hull, concavity, lengthThreshold);

    std::cout << "concave_points:" << std::endl;
    for (auto &p : concave_points)
        std::cout << p[0] << " " << p[1] << std::endl;

    double *concave_points_c = *p_concave_points_c = (double*) malloc(sizeof(double) * 2 * concave_points.size());
    for (auto i = 0; i < concave_points.size(); i++) {
        concave_points_c[i << 1] = concave_points[i][0];
        concave_points_c[(i << 1) + 1] = concave_points[i][1];
    }

    *p_num_concave_points = concave_points.size();
    *p_free = free;
}
