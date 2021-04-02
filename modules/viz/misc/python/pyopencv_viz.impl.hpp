#ifndef OPENCV_PYTHON_VIZ_IMPL_HPP
#define OPENCV_PYTHON_VIZ_IMPL_HPP

#include "python_viz.hpp"

namespace cv { namespace viz {

PyAffine3d makeTransformToGlobalPy(const Vec3d& axis_x, const Vec3d& axis_y, const Vec3d& axis_z, const Vec3d& origin)
{
    return makeTransformToGlobal(axis_x, axis_y, axis_z, origin);
}

PyAffine3d makeCameraPosePy(const Vec3d& position, const Vec3d& focal_point, const Vec3d& y_dir)
{
    return makeCameraPose(position, focal_point, y_dir);
}

#define WRAP_SHOW_WIDGET_1  return Viz3d::showWidget(id, *widget.widget.get())
#define WRAP_SHOW_WIDGET_2  return Viz3d::showWidget(id, *widget.widget.get(), pose)
inline void PyViz3d::showWidget(const String &id, PyWLine &widget) { WRAP_SHOW_WIDGET_1; }
inline void PyViz3d::showWidget(const String &id, PyWLine &widget, PyAffine3d &pose) { WRAP_SHOW_WIDGET_2; }
inline void PyViz3d::showWidget(const String &id, PyWSphere &widget) { WRAP_SHOW_WIDGET_1; }
inline void PyViz3d::showWidget(const String &id, PyWSphere &widget, PyAffine3d &pose) { WRAP_SHOW_WIDGET_2; }
inline void PyViz3d::showWidget(const String &id, PyWCameraPosition &widget) { WRAP_SHOW_WIDGET_1; }
inline void PyViz3d::showWidget(const String &id, PyWCameraPosition &widget, PyAffine3d &pose) { WRAP_SHOW_WIDGET_2; }
inline void PyViz3d::showWidget(const String &id, PyWArrow &widget) { WRAP_SHOW_WIDGET_1; }
inline void PyViz3d::showWidget(const String &id, PyWArrow &widget, PyAffine3d &pose) { WRAP_SHOW_WIDGET_2; }
inline void PyViz3d::showWidget(const String &id, PyWCircle &widget) { WRAP_SHOW_WIDGET_1; }
inline void PyViz3d::showWidget(const String &id, PyWCircle &widget, PyAffine3d &pose) { WRAP_SHOW_WIDGET_2; }
inline void PyViz3d::showWidget(const String &id, PyWPlane &widget) { WRAP_SHOW_WIDGET_1; }
inline void PyViz3d::showWidget(const String &id, PyWPlane &widget, PyAffine3d &pose) { WRAP_SHOW_WIDGET_2; }
inline void PyViz3d::showWidget(const String &id, PyWCone &widget) { WRAP_SHOW_WIDGET_1; }
inline void PyViz3d::showWidget(const String &id, PyWCone &widget, PyAffine3d &pose) { WRAP_SHOW_WIDGET_2; }
inline void PyViz3d::showWidget(const String &id, PyWCube &widget) { WRAP_SHOW_WIDGET_1; }
inline void PyViz3d::showWidget(const String &id, PyWCube &widget, PyAffine3d &pose) { WRAP_SHOW_WIDGET_2; }
inline void PyViz3d::showWidget(const String &id, PyWCylinder &widget) { WRAP_SHOW_WIDGET_1; }
inline void PyViz3d::showWidget(const String &id, PyWCylinder &widget, PyAffine3d &pose) { WRAP_SHOW_WIDGET_2; }
inline void PyViz3d::showWidget(const String &id, PyWCoordinateSystem &widget) { WRAP_SHOW_WIDGET_1; }
inline void PyViz3d::showWidget(const String &id, PyWPaintedCloud &widget) { WRAP_SHOW_WIDGET_1; }
inline void PyViz3d::showWidget(const String &id, PyWPaintedCloud &widget, PyAffine3d &pose) { WRAP_SHOW_WIDGET_2; }
inline void PyViz3d::showWidget(const String &id, PyWCloudCollection &widget) { WRAP_SHOW_WIDGET_1; }
inline void PyViz3d::showWidget(const String &id, PyWCloudCollection &widget, PyAffine3d &pose) { WRAP_SHOW_WIDGET_2; }
inline void PyViz3d::showWidget(const String &id, PyWGrid &widget) { WRAP_SHOW_WIDGET_1; }
inline void PyViz3d::showWidget(const String &id, PyWGrid &widget, PyAffine3d &pose) { WRAP_SHOW_WIDGET_2; }
inline void PyViz3d::showWidget(const String &id, const cv::Ptr<WMesh> &widget) { return Viz3d::showWidget(id, *widget.get()); }
inline void PyViz3d::showWidget(const String &id, const cv::Ptr<WMesh> &widget, PyAffine3d &pose) { return Viz3d::showWidget(id, *widget.get(), pose); }
inline void PyViz3d::showWidget(const String &id, PyWPolyLine &widget) { WRAP_SHOW_WIDGET_1; }
inline void PyViz3d::showWidget(const String &id, PyWPolyLine &widget, PyAffine3d &pose) { WRAP_SHOW_WIDGET_2; }
inline void PyViz3d::showWidget(const String &id, PyWCloud &widget) { WRAP_SHOW_WIDGET_1; }
inline void PyViz3d::showWidget(const String &id, PyWCloud &widget, PyAffine3d &pose) { WRAP_SHOW_WIDGET_2; }
inline void PyViz3d::showWidget(const String &id, PyWImage3D &widget, PyAffine3d &pose) { WRAP_SHOW_WIDGET_2; }
inline void PyViz3d::showWidget(const String &id, PyWImage3D &widget) { WRAP_SHOW_WIDGET_1; }
inline void PyViz3d::showWidget(const String &id, PyWImageOverlay &widget, PyAffine3d &pose) { WRAP_SHOW_WIDGET_2; }
inline void PyViz3d::showWidget(const String &id, PyWImageOverlay &widget) { WRAP_SHOW_WIDGET_1; }
inline void PyViz3d::showWidget(const String &id, PyWText &widget) { WRAP_SHOW_WIDGET_1; }
inline void PyViz3d::showWidget(const String &id, PyWText &widget, PyAffine3d &pose) { WRAP_SHOW_WIDGET_2; }
inline void PyViz3d::showWidget(const String &id, PyWText3D &widget) { WRAP_SHOW_WIDGET_1; }
inline void PyViz3d::showWidget(const String &id, PyWText3D &widget, PyAffine3d &pose) { WRAP_SHOW_WIDGET_2; }
inline void PyViz3d::showWidget(const String &id, PyWCloudNormals &widget, PyAffine3d &pose) { WRAP_SHOW_WIDGET_2; }
inline void PyViz3d::showWidget(const String &id, PyWCloudNormals &widget) { WRAP_SHOW_WIDGET_1; }
inline void PyViz3d::showWidget(const String &id, PyWTrajectory &widget, PyAffine3d &pose) { WRAP_SHOW_WIDGET_2; }
inline void PyViz3d::showWidget(const String &id, PyWTrajectory &widget) { WRAP_SHOW_WIDGET_1; }
inline void PyViz3d::showWidget(const String &id, PyWTrajectorySpheres &widget, PyAffine3d &pose) { WRAP_SHOW_WIDGET_2; }
inline void PyViz3d::showWidget(const String &id, PyWTrajectorySpheres &widget) { WRAP_SHOW_WIDGET_1; }
inline void PyViz3d::showWidget(const String &id, PyWTrajectoryFrustums &widget, PyAffine3d &pose) { WRAP_SHOW_WIDGET_2; }
inline void PyViz3d::showWidget(const String &id, PyWTrajectoryFrustums &widget) { WRAP_SHOW_WIDGET_1; }
#undef WRAP_SHOW_WIDGET_1
#undef WRAP_SHOW_WIDGET_2


PyWGrid::PyWGrid(InputArray cells, InputArray cells_spacing, const PyColor& color)
{
    if (cells.kind() == _InputArray::MAT && cells_spacing.kind() == _InputArray::MAT)
    {
        Mat k = cells.getMat();
        Mat l = cells_spacing.getMat();
        if (k.total() == 2  && l.total() == 2 )
        {
            CV_Assert(k.type() == CV_64FC1 && k.type() == CV_64FC1);
            Vec2i c1((int)k.at<double>(0), (int)k.at<double>(1));
            Vec2d c2(l.at<double>(0), l.at<double>(1));
            widget = cv::makePtr<WGrid>(c1, c2, color);
        }
        else
            CV_Error(cv::Error::StsVecLengthErr, "unknown size");
    }
    else
        CV_Error(cv::Error::StsUnsupportedFormat, "unknown type");
}


PyWTrajectoryFrustums::PyWTrajectoryFrustums(InputArray path, InputArray K, double scale, const PyColor& color)
{
    if (K.kind() == _InputArray::MAT)
    {
        Mat k = K.getMat();
        if (k.rows == 3 && k.cols == 3)
        {
            Matx33d x = k;
            widget = cv::makePtr<WTrajectoryFrustums>(path, x, scale, color);

        }
        else if (k.total() == 2)
            widget = cv::makePtr<WTrajectoryFrustums>(path, Vec2d(k.at<double>(0), k.at<double>(1)), 1.0, color);
        else
            CV_Error(cv::Error::StsVecLengthErr, "unknown size");
    }
    else
        CV_Error(cv::Error::StsVecLengthErr, "unknown size");
}


}}  // namespace
#endif  // OPENCV_PYTHON_VIZ_IMPL_HPP
