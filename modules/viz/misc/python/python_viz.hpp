#ifndef OPENCV_PYTHON_VIZ_HPP
#define OPENCV_PYTHON_VIZ_HPP

#include "opencv2/viz.hpp"

namespace cv { namespace viz {

struct CV_EXPORTS_W_SIMPLE CV_WRAP_AS(Color) PyColor
{
    CV_WRAP PyColor() {}
    CV_WRAP PyColor(double gray) : c(gray) {}
    CV_WRAP PyColor(double blue, double green, double red) : c(blue, green, red) {}
    PyColor(const Color& v) : c(v) { }

    operator Color() const { return c; }

    CV_WRAP static PyColor black() { return PyColor(Color::black()); }
    CV_WRAP static PyColor white() { return PyColor(Color::white()); }
    CV_WRAP static PyColor blue() { return PyColor(Color::blue()); }
    CV_WRAP static PyColor green() { return PyColor(Color::green()); }
    CV_WRAP static PyColor red() { return PyColor(Color::red()); }
    CV_WRAP static PyColor cyan() { return PyColor(Color::cyan()); }
    CV_WRAP static PyColor yellow() { return PyColor(Color::yellow()); }
    CV_WRAP static PyColor magenta() { return PyColor(Color::magenta()); }

    CV_WRAP static PyColor gray() { return PyColor(Color::gray()); }
    CV_WRAP static PyColor silver() { return PyColor(Color::silver()); }

    CV_WRAP static PyColor mlab() { return PyColor(Color::mlab()); }

    CV_WRAP static PyColor navy() { return PyColor(Color::navy()); }
    CV_WRAP static PyColor maroon() { return PyColor(Color::maroon()); }
    CV_WRAP static PyColor teal() { return PyColor(Color::teal()); }
    CV_WRAP static PyColor olive() { return PyColor(Color::olive()); }
    CV_WRAP static PyColor purple() { return PyColor(Color::olive()); }
    CV_WRAP static PyColor azure() { return PyColor(Color::olive()); }
    CV_WRAP static PyColor chartreuse() { return PyColor(Color::olive()); }
    CV_WRAP static PyColor rose() { return PyColor(Color::olive()); }

    CV_WRAP static PyColor lime() { return PyColor(Color::olive()); }
    CV_WRAP static PyColor gold() { return PyColor(Color::olive()); }
    CV_WRAP static PyColor orange() { return PyColor(Color::olive()); }
    CV_WRAP static PyColor orange_red() { return PyColor(Color::olive()); }
    CV_WRAP static PyColor indigo() { return PyColor(Color::olive()); }

    CV_WRAP static PyColor brown() { return PyColor(Color::olive()); }
    CV_WRAP static PyColor apricot() { return PyColor(Color::olive()); }
    CV_WRAP static PyColor pink() { return PyColor(Color::olive()); }
    CV_WRAP static PyColor raspberry() { return PyColor(Color::olive()); }
    CV_WRAP static PyColor cherry() { return PyColor(Color::olive()); }
    CV_WRAP static PyColor violet() { return PyColor(Color::olive()); }
    CV_WRAP static PyColor amethyst() { return PyColor(Color::amethyst()); }
    CV_WRAP static PyColor bluberry() { return PyColor(Color::bluberry()); }
    CV_WRAP static PyColor celestial_blue() { return PyColor(Color::celestial_blue()); }
    CV_WRAP static PyColor turquoise() { return PyColor(Color::turquoise()); }

    static PyColor not_set() { return PyColor(Color::not_set()); }
    CV_WRAP double get_blue() { return c[0]; }
    CV_WRAP double get_green() { return c[1]; }
    CV_WRAP double get_red() { return c[2]; }

    Color c;
};


struct CV_EXPORTS_W_SIMPLE CV_WRAP_AS(Affine3d) PyAffine3d
#ifndef OPENCV_BINDING_PARSER
    : public Affine3d
#endif
{
    CV_WRAP PyAffine3d()
    {
        // nothing
    }
    inline PyAffine3d(const Affine3d& base) : Affine3d(base)
    {
        // nothing
    }
    CV_WRAP PyAffine3d(const Vec3d &rvec, const Vec3d &t = Vec3d::all(0))
        : Affine3d(rvec, t)
    {
        // nothing
    }
    CV_WRAP PyAffine3d(const Mat &rot, const Vec3f &t = Vec3d::all(0))
        : Affine3d(rot, t)
    {
        // nothing
    }
    CV_WRAP PyAffine3d translate(const Vec3d &t)
    {
        return Affine3d::translate(t);
    }
    CV_WRAP PyAffine3d rotate(const Vec3d &t)
    {
        return Affine3d::rotate(t);
    }
    CV_WRAP PyAffine3d product(const PyAffine3d &t)
    {
        return ((const Affine3d&)(*this)) * (const Affine3d&)t;
    }
    CV_WRAP static PyAffine3d Identity()
    {
        return Affine3d::Identity();
    }
    CV_WRAP PyAffine3d inv()
    {
        return Affine3d::inv();
    }
    CV_WRAP Mat mat()
    {
        return Mat(matrix);
    }
};


struct CV_EXPORTS_W_SIMPLE CV_WRAP_AS(WLine) PyWLine
{
    CV_WRAP PyWLine()
    {
    }
    /** @brief Constructs a WLine.

    @param pt1 Start point of the line.
    @param pt2 End point of the line.
    @param color Color of the line.
     */
    CV_WRAP PyWLine(const Point3d &pt1, const Point3d &pt2, const PyColor& color)
    {
        widget = cv::makePtr<cv::viz::WLine>(pt1, pt2, color);
    }

    CV_WRAP void setRenderingProperty(int property, double value)
    {
        CV_Assert(widget);
        widget->setRenderingProperty(property, value);
    }

    Ptr<cv::viz::WLine> widget;
};

struct CV_EXPORTS_W_SIMPLE CV_WRAP_AS(WPlane) PyWPlane
{
public:
    /** @brief Constructs a default plane with center point at origin and normal oriented along z-axis.

    @param size Size of the plane
    @param color Color of the plane.
    */
    CV_WRAP PyWPlane(const Point2d& size = Point2d(1.0, 1.0), const PyColor& color = Color(255, 255,255))
    {
        widget = cv::makePtr<cv::viz::WPlane>(size, color);
    }

    /** @brief Constructs a repositioned plane

    @param center Center of the plane
    @param normal Plane normal orientation
    @param new_yaxis Up-vector. New orientation of plane y-axis.
    @param size
    @param color Color of the plane.
     */
    CV_WRAP PyWPlane(const Point3d& center, const Vec3d& normal, const Vec3d& new_yaxis,
        const Point2d& size = Point2d(1.0, 1.0), const PyColor& color = Color(255, 255, 255))
    {
        widget = cv::makePtr<cv::viz::WPlane>(center, normal, new_yaxis, size, color);
    }

    CV_WRAP void setRenderingProperty(int property, double value)
    {
        CV_Assert(widget);
        widget->setRenderingProperty(property, value);
    }

    Ptr<cv::viz::WPlane> widget;
};

/** @brief This 3D Widget defines a sphere. :
*/
struct CV_EXPORTS_W_SIMPLE CV_WRAP_AS(WSphere) PyWSphere
{
public:
    CV_WRAP PyWSphere()
    {

    }
    /** @brief Constructs a WSphere.

    @param center Center of the sphere.
    @param radius Radius of the sphere.
    @param sphere_resolution Resolution of the sphere.
    @param color Color of the sphere.
     */
    CV_WRAP PyWSphere(const cv::Point3d &center, double radius, int sphere_resolution = 10, const PyColor& color = Color(255, 255,255))
    {
        widget = cv::makePtr<cv::viz::WSphere>(center, radius, sphere_resolution,  color);
    }

    CV_WRAP void setRenderingProperty(int property, double value)
    {
        CV_Assert(widget);
        widget->setRenderingProperty(property, value);
    }

    Ptr<cv::viz::WSphere> widget;
};

/** @brief This 3D Widget defines an arrow.
*/
struct CV_EXPORTS_W_SIMPLE CV_WRAP_AS(WArrow) PyWArrow
{
public:
    CV_WRAP PyWArrow()
    {
    }
    /** @brief Constructs an WArrow.

    @param pt1 Start point of the arrow.
    @param pt2 End point of the arrow.
    @param thickness Thickness of the arrow. Thickness of arrow head is also adjusted
    accordingly.
    @param color Color of the arrow.

    Arrow head is located at the end point of the arrow.
     */
    CV_WRAP PyWArrow(const Point3d& pt1, const Point3d& pt2, double thickness = 0.03, const PyColor& color = Color(255, 255, 255))
    {
        widget = cv::makePtr<cv::viz::WArrow>(pt1, pt2, thickness, color);
    }

    CV_WRAP void setRenderingProperty(int property, double value)
    {
        CV_Assert(widget);
        widget->setRenderingProperty(property, value);
    }

    Ptr<cv::viz::WArrow> widget;
};

/** @brief This 3D Widget defines a cube.
*/
struct CV_EXPORTS_W_SIMPLE CV_WRAP_AS(WCube) PyWCube
{
public:
    /** @brief Constructs a WCube.

    @param min_point Specifies minimum (or maximum) point of the bounding box.
    @param max_point Specifies maximum (or minimum) point of the bounding box, opposite to the first parameter.
    @param wire_frame If true, cube is represented as wireframe.
    @param color Color of the cube.

    ![Cube Widget](images/cube_widget.png)
     */
    CV_WRAP PyWCube(const Point3d& min_point = Vec3d::all(-0.5), const Point3d& max_point = Vec3d::all(0.5),
        bool wire_frame = true, const PyColor& color = Color(255, 255, 255))
    {
        widget = cv::makePtr<cv::viz::WCube>(min_point, max_point, wire_frame, color);
    }

    CV_WRAP void setRenderingProperty(int property, double value)
    {
        CV_Assert(widget);
        widget->setRenderingProperty(property, value);
    }

    Ptr<cv::viz::WCube> widget;
};

/** @brief This 3D Widget defines a PyWCircle.
*/
struct CV_EXPORTS_W_SIMPLE CV_WRAP_AS(WCircle) PyWCircle
{
public:
    PyWCircle() {}
    /** @brief Constructs default planar circle centered at origin with plane normal along z-axis

    @param radius Radius of the circle.
    @param thickness Thickness of the circle.
    @param color Color of the circle.
     */
    CV_WRAP PyWCircle(double radius, double thickness = 0.01, const PyColor& color = Color::white())
    {
        widget = cv::makePtr<WCircle>(radius, thickness, color);
    }

    /** @brief Constructs repositioned planar circle.

    @param radius Radius of the circle.
    @param center Center of the circle.
    @param normal Normal of the plane in which the circle lies.
    @param thickness Thickness of the circle.
    @param color Color of the circle.
     */
    CV_WRAP PyWCircle(double radius, const Point3d& center, const Vec3d& normal, double thickness = 0.01, const PyColor& color = Color::white())
    {
        widget = cv::makePtr<WCircle>(radius, center, normal, thickness, color);
    }

    CV_WRAP void setRenderingProperty(int property, double value)
    {
        CV_Assert(widget);
        widget->setRenderingProperty(property, value);
    }

    Ptr<cv::viz::WCircle> widget;
};

/** @brief This 3D Widget defines a cone. :
*/
struct CV_EXPORTS_W_SIMPLE CV_WRAP_AS(WCone) PyWCone
{
public:
    PyWCone() {}
    /** @brief Constructs default cone oriented along x-axis with center of its base located at origin

    @param length Length of the cone.
    @param radius Radius of the cone.
    @param resolution Resolution of the cone.
    @param color Color of the cone.
     */
    CV_WRAP PyWCone(double length, double radius, int resolution = 6, const PyColor& color = Color::white())
    {
        widget = cv::makePtr<WCone>(length, radius, resolution, color);
    }

    /** @brief Constructs repositioned planar cone.

    @param radius Radius of the cone.
    @param center Center of the cone base.
    @param tip Tip of the cone.
    @param resolution Resolution of the cone.
    @param color Color of the cone.

     */
    CV_WRAP PyWCone(double radius, const Point3d& center, const Point3d& tip, int resolution = 6, const PyColor& color = Color::white())
    {
        widget = cv::makePtr<WCone>(radius, center, tip, resolution, color);
    }

    CV_WRAP void setRenderingProperty(int property, double value)
    {
        CV_Assert(widget);
        widget->setRenderingProperty(property, value);
    }

    Ptr<cv::viz::WCone> widget;
};

/** @brief This 3D Widget defines a PyWCylinder. :
*/
struct CV_EXPORTS_W_SIMPLE CV_WRAP_AS(WCylinder) PyWCylinder
{
public:
    CV_WRAP PyWCylinder() {}
    /** @brief Constructs a WCylinder.

    @param axis_point1 A point1 on the axis of the cylinder.
    @param axis_point2 A point2 on the axis of the cylinder.
    @param radius Radius of the cylinder.
    @param numsides Resolution of the cylinder.
    @param color Color of the cylinder.
     */
    CV_WRAP PyWCylinder(const Point3d& axis_point1, const Point3d& axis_point2, double radius, int numsides = 30, const PyColor& color = Color::white())
    {
        widget = cv::makePtr<WCylinder>(axis_point1, axis_point2, radius, numsides, color);
    }
    Ptr<cv::viz::WCylinder> widget;
};

/** @brief This 3D Widget represents camera position in a scene by its axes or viewing frustum. :
*/
struct CV_EXPORTS_W_SIMPLE CV_WRAP_AS(WCameraPosition) PyWCameraPosition
{
public:
    /** @brief Creates camera coordinate frame at the origin.

    ![Camera coordinate frame](images/cpw1.png)
     */
    CV_WRAP PyWCameraPosition(double scale = 1.0)
    {
        widget = cv::makePtr<cv::viz::WCameraPosition>(scale);
    }
    /** @brief Display the viewing frustum
    @param K Intrinsic matrix of the camera or fov Field of view of the camera (horizontal, vertical).
    @param scale Scale of the frustum.
    @param color Color of the frustum.

    Creates viewing frustum of the camera based on its intrinsic matrix K.

    ![Camera viewing frustum](images/cpw2.png)
    */
    CV_WRAP PyWCameraPosition(InputArray  K, double scale = 1.0, const PyColor& color = Color(255, 255, 255))
    {
        if (K.kind() == _InputArray::MAT)
        {
            Mat k = K.getMat();
            if (k.rows == 3 && k.cols == 3)
            {
                Matx33d x = k;
                widget = cv::makePtr<cv::viz::WCameraPosition>(x, scale, color);

            }
            else if (k.total() == 2)
                widget = cv::makePtr<cv::viz::WCameraPosition>(Vec2d(k.at<double>(0), k.at<double>(1)), scale, color);
            else
                CV_Error(cv::Error::StsVecLengthErr, "unknown size");
        }
        else
            CV_Error(cv::Error::StsUnsupportedFormat, "unknown type");
    }

    /** @brief Display image on the far plane of the viewing frustum

    @param K Intrinsic matrix of the camera.
    @param image BGR or Gray-Scale image that is going to be displayed on the far plane of the frustum.
    @param scale Scale of the frustum and image.
    @param color Color of the frustum.

    Creates viewing frustum of the camera based on its intrinsic matrix K, and displays image on
    the far end plane.

    ![Camera viewing frustum with image](images/cpw3.png)
     */
    CV_WRAP PyWCameraPosition(InputArray K, InputArray image, double scale = 1.0, const PyColor& color = Color(255, 255, 255))
    {
        if (K.kind() == _InputArray::MAT)
        {
            Mat k = K.getMat();
            if (k.rows == 3 && k.cols == 3)
            {
                Matx33d x = k;
                widget = cv::makePtr<cv::viz::WCameraPosition>(x, image, scale, color);

            }
            else if (k.total() == 2)
                widget = cv::makePtr<cv::viz::WCameraPosition>(Vec2d(k.at<double>(0), k.at<double>(1)), image, scale, color);
            else
                CV_Error(cv::Error::StsVecLengthErr, "unknown size");
        }
        else
            CV_Error(cv::Error::StsUnsupportedFormat, "unknown type");

    }
    /** @brief  Display image on the far plane of the viewing frustum

    @param fov Field of view of the camera (horizontal, vertical).
    @param image BGR or Gray-Scale image that is going to be displayed on the far plane of the frustum.
    @param scale Scale of the frustum and image.
    @param color Color of the frustum.

    Creates viewing frustum of the camera based on its intrinsic matrix K, and displays image on
    the far end plane.

    ![Camera viewing frustum with image](images/cpw3.png)
     */
    CV_WRAP PyWCameraPosition(const Point2d &fov, InputArray image, double scale = 1.0, const PyColor& color = Color(255, 255, 255))
    {
        widget = cv::makePtr<cv::viz::WCameraPosition>(fov, image, scale, color);
    }

    Ptr<cv::viz::WCameraPosition> widget;
};
/////////////////////////////////////////////////////////////////////////////
/// Compound widgets

/** @brief This 3D Widget represents a coordinate system. :
*/
struct CV_EXPORTS_W_SIMPLE CV_WRAP_AS(WCoordinateSystem) PyWCoordinateSystem
{
public:
    /** @brief Constructs a WCoordinateSystem.

    @param scale Determines the size of the axes.
     */
    CV_WRAP PyWCoordinateSystem(double scale = 1.0)
    {
        widget = cv::makePtr<cv::viz::WCoordinateSystem>(scale);

    }
    Ptr<cv::viz::WCoordinateSystem> widget;
};

/////////////////////////////////////////////////////////////////////////////
/// Clouds

/** @brief This 3D Widget defines a point cloud. :

@note In case there are four channels in the cloud, fourth channel is ignored.
*/
struct CV_EXPORTS_W_SIMPLE CV_WRAP_AS(WCloud) PyWCloud
{
public:
    CV_WRAP PyWCloud()
    {
        // nothing
    }

    /** @brief Constructs a WCloud.
    @param cloud Set of points which can be of type: CV_32FC3, CV_32FC4, CV_64FC3, CV_64FC4.
    @param color A single Color for the whole cloud.

    Points in the cloud belong to mask when they are set to (NaN, NaN, NaN).
     */
    CV_WRAP PyWCloud(InputArray cloud, const PyColor& color = Color::white())
    {
        widget = cv::makePtr<cv::viz::WCloud>(cloud, color);
    }

    CV_WRAP PyWCloud(InputArray cloud, InputArray colors)
    {
        widget = cv::makePtr<cv::viz::WCloud>(cloud, colors);
    }

    CV_WRAP PyWCloud(InputArray cloud, InputArray colors, InputArray normals)
    {
        widget = cv::makePtr<cv::viz::WCloud>(cloud, colors, normals);
    }

    /** @brief Constructs a WCloud.
    @param cloud Set of points which can be of type: CV_32FC3, CV_32FC4, CV_64FC3, CV_64FC4.
    @param color A single Color for the whole cloud.
    @param normals Normals for each point in cloud.

    Size and type should match with the cloud parameter.
    Points in the cloud belong to mask when they are set to (NaN, NaN, NaN).
     */
    CV_WRAP PyWCloud(InputArray cloud, const PyColor& color, InputArray normals)
    {
        widget = cv::makePtr<cv::viz::WCloud>(cloud, color, normals);
    }

    CV_WRAP void setRenderingProperty(int property, double value)
    {
        CV_Assert(widget);
        widget->setRenderingProperty(property, value);
    }

    Ptr<cv::viz::WCloud> widget;
};
/** @brief This 3D Widget defines a poly line. :
*/
struct CV_EXPORTS_W_SIMPLE CV_WRAP_AS(WPolyLine) PyWPolyLine
{
public:
    CV_WRAP PyWPolyLine()
    {
    }
    CV_WRAP PyWPolyLine(InputArray points, InputArray colors)
    {
        widget = cv::makePtr<cv::viz::WPolyLine>(points, colors);
    }
    /** @brief Constructs a WPolyLine.

    @param points Point set.
    @param color Color of the poly line.
     */
    CV_WRAP PyWPolyLine(InputArray points, const PyColor& color = Color::white())
    {
        widget = cv::makePtr<cv::viz::WPolyLine>(points, color);
    }

    CV_WRAP void setRenderingProperty(int property, double value)
    {
        CV_Assert(widget);
        widget->setRenderingProperty(property, value);
    }

    Ptr<cv::viz::WPolyLine> widget;
};

/////////////////////////////////////////////////////////////////////////////
/// Text and image widgets

/** @brief This 2D Widget represents text overlay.
*/
struct CV_EXPORTS_W_SIMPLE CV_WRAP_AS(WText) PyWText
{
public:
    CV_WRAP PyWText()
    {
    }
    /** @brief Constructs a WText.

    @param text Text content of the widget.
    @param pos Position of the text.
    @param font_size Font size.
    @param color Color of the text.
     */
    CV_WRAP PyWText(const String &text, const Point &pos, int font_size = 20, const PyColor& color = Color::white())
    {
        widget = cv::makePtr<cv::viz::WText>(text, pos, font_size, color);
    }

    /** @brief Sets the text content of the widget.

    @param text Text content of the widget.
     */
    CV_WRAP void setText(const String &text)
    {
        widget->setText(text);
    }
    /** @brief Returns the current text content of the widget.
    */
    CV_WRAP String getText() const
    {
        return widget->getText();
    }
    Ptr<cv::viz::WText> widget;
};

/** @brief This 3D Widget represents 3D text. The text always faces the camera.
*/
struct CV_EXPORTS_W_SIMPLE CV_WRAP_AS(WText3D) PyWText3D
{
public:
    CV_WRAP PyWText3D()
    {}
    /** @brief Constructs a WText3D.

    @param text Text content of the widget.
    @param position Position of the text.
    @param text_scale Size of the text.
    @param face_camera If true, text always faces the camera.
    @param color Color of the text.
     */
    CV_WRAP PyWText3D(const String &text, const Point3d &position, double text_scale = 1., bool face_camera = true, const PyColor& color = Color::white())
    {
        widget = cv::makePtr<WText3D>(text, position, text_scale, face_camera, color);
    }
    /** @brief Sets the text content of the widget.

    @param text Text content of the widget.

     */
    CV_WRAP void setText(const String &text)
    {
        widget->setText(text);
    }
    /** @brief Returns the current text content of the widget.
    */
    CV_WRAP String getText() const
    {
        return widget->getText();
    }
    Ptr<cv::viz::WText3D> widget;
};

/** @brief This 2D Widget represents an image overlay. :
*/
struct CV_EXPORTS_W_SIMPLE CV_WRAP_AS(WImageOverlay) PyWImageOverlay
{
public:
    CV_WRAP PyWImageOverlay()
    {
    }
    /** @brief Constructs an WImageOverlay.

    @param image BGR or Gray-Scale image.
    @param rect Image is scaled and positioned based on rect.
     */
    CV_WRAP PyWImageOverlay(InputArray image, const Rect &rect)
    {
        widget = cv::makePtr<WImageOverlay>(image, rect);
    }
    /** @brief Sets the image content of the widget.

    @param image BGR or Gray-Scale image.
     */
    CV_WRAP void setImage(InputArray image)
    {
        widget->setImage(image);
    }

    CV_WRAP void setRenderingProperty(int property, double value)
    {
        CV_Assert(widget);
        widget->setRenderingProperty(property, value);
    }

    Ptr<cv::viz::WImageOverlay> widget;
};

/** @brief This 3D Widget represents an image in 3D space. :
*/
struct CV_EXPORTS_W_SIMPLE CV_WRAP_AS(WImage3D) PyWImage3D
{
    CV_WRAP PyWImage3D()
    {
    }
public:
    /** @brief Constructs an WImage3D.

    @param image BGR or Gray-Scale image.
    @param size Size of the image.
     */
    CV_WRAP  PyWImage3D(InputArray image, const Point2d &size)
    {
        widget = cv::makePtr<WImage3D>(image, size);
    }

    /** @brief Constructs an WImage3D.

    @param image BGR or Gray-Scale image.
    @param size Size of the image.
    @param center Position of the image.
    @param normal Normal of the plane that represents the image.
    @param up_vector Determines orientation of the image.
     */
    CV_WRAP PyWImage3D(InputArray image, const Point2d &size, const Vec3d &center, const Vec3d &normal, const Vec3d &up_vector)
    {
        widget = cv::makePtr<WImage3D>(image, size, center, normal, up_vector);
    }


    /** @brief Sets the image content of the widget.

    @param image BGR or Gray-Scale image.
     */
    CV_WRAP void setImage(InputArray image)
    {
        widget->setImage(image);
    }
    /** @brief Sets the image size of the widget.

    @param size the new size of the image.
     */
    CV_WRAP void setSize(const Size& size)
    {
        widget->setSize(size);
    }
    Ptr<cv::viz::WImage3D> widget;
};


/** @brief This 3D Widget defines a grid. :
 */
struct CV_EXPORTS_W_SIMPLE CV_WRAP_AS(WGrid) PyWGrid
{
public:
    PyWGrid() {}
    /** @brief Constructs a WGrid.

    @param cells Number of cell columns and rows, respectively.
    @param cells_spacing Size of each cell, respectively.
    @param color Color of the grid.
     */
    CV_WRAP PyWGrid(InputArray cells, InputArray cells_spacing, const PyColor& color = Color::white());

    //! Creates repositioned grid
    CV_WRAP PyWGrid(const Point3d& center, const Vec3d& normal, const Vec3d& new_yaxis,
        const Vec2i &cells = Vec2i::all(10), const Vec2d &cells_spacing = Vec2d::all(1.0), const PyColor& color = Color::white())
    {
        widget = cv::makePtr<WGrid>(center, normal, new_yaxis, cells, cells_spacing, color);
    }

    CV_WRAP void setRenderingProperty(int property, double value)
    {
        CV_Assert(widget);
        widget->setRenderingProperty(property, value);
    }

    Ptr<cv::viz::WGrid> widget;
};


/////////////////////////////////////////////////////////////////////////////
/// Trajectories

/** @brief This 3D Widget represents a trajectory. :
*/
struct CV_EXPORTS_W_SIMPLE CV_WRAP_AS(WTrajectory) PyWTrajectory
{
public:
    enum { FRAMES = 1, PATH = 2, BOTH = FRAMES + PATH };
    PyWTrajectory() {}
    /** @brief Constructs a WTrajectory.

    @param path List of poses on a trajectory. Takes std::vector\<Affine\<T\>\> with T == [float | double]
    @param display_mode Display mode. This can be PATH, FRAMES, and BOTH.
    @param scale Scale of the frames. Polyline is not affected.
    @param color Color of the polyline that represents path.

    Frames are not affected.
    Displays trajectory of the given path as follows:
    -   PATH : Displays a poly line that represents the path.
    -   FRAMES : Displays coordinate frames at each pose.
    -   PATH & FRAMES : Displays both poly line and coordinate frames.
     */
    CV_WRAP PyWTrajectory(InputArray path, int display_mode = WTrajectory::PATH, double scale = 1.0, const PyColor& color = Color::white())
    {
        widget = cv::makePtr<cv::viz::WTrajectory>(path, display_mode, scale, color);
    }

    CV_WRAP void setRenderingProperty(int property, double value)
    {
        CV_Assert(widget);
        widget->setRenderingProperty(property, value);
    }

    Ptr<cv::viz::WTrajectory> widget;
};

/** @brief This 3D Widget represents a trajectory. :
*/
struct CV_EXPORTS_W_SIMPLE CV_WRAP_AS(WTrajectoryFrustums) PyWTrajectoryFrustums
{
public:
    PyWTrajectoryFrustums() {}
    /** @brief Constructs a WTrajectoryFrustums.

    @param path List of poses on a trajectory. Takes std::vector\<Affine\<T\>\> with T == [float | double]
    @param K Intrinsic matrix of the camera or fov Field of view of the camera (horizontal, vertical).
    @param scale Scale of the frustums.
    @param color Color of the frustums.

    Displays frustums at each pose of the trajectory.
     */
    CV_WRAP PyWTrajectoryFrustums(InputArray path, InputArray K, double scale = 1.0, const PyColor& color = Color::white());

    CV_WRAP void setRenderingProperty(int property, double value)
    {
        CV_Assert(widget);
        widget->setRenderingProperty(property, value);
    }

    Ptr<cv::viz::WTrajectoryFrustums> widget;
};

/** @brief This 3D Widget represents a trajectory using spheres and lines

where spheres represent the positions of the camera, and lines represent the direction from
previous position to the current. :
 */
struct CV_EXPORTS_W_SIMPLE CV_WRAP_AS(WTrajectorySpheres) PyWTrajectorySpheres
{
public:
    PyWTrajectorySpheres() {}
    /** @brief Constructs a WTrajectorySpheres.

    @param path List of poses on a trajectory. Takes std::vector\<Affine\<T\>\> with T == [float | double]
    @param line_length Max length of the lines which point to previous position
    @param radius Radius of the spheres.
    @param from Color for first sphere.
    @param to Color for last sphere. Intermediate spheres will have interpolated color.
     */
    CV_WRAP PyWTrajectorySpheres(InputArray path, double line_length = 0.05, double radius = 0.007,
        const PyColor& from = Color::red(), const PyColor& to = Color::white())
    {
        widget = cv::makePtr<WTrajectorySpheres>(path, line_length, radius, from, to);
    }

    CV_WRAP void setRenderingProperty(int property, double value)
    {
        CV_Assert(widget);
        widget->setRenderingProperty(property, value);
    }

    Ptr<cv::viz::WTrajectorySpheres> widget;
};


struct CV_EXPORTS_W_SIMPLE CV_WRAP_AS(WPaintedCloud) PyWPaintedCloud
{
public:
    //! Paint cloud with default gradient between cloud bounds points
    PyWPaintedCloud()
    {
    }

    CV_WRAP PyWPaintedCloud(InputArray cloud)
    {
        widget = cv::makePtr<WPaintedCloud>(cloud);
    }

    //! Paint cloud with default gradient between given points
    CV_WRAP PyWPaintedCloud(InputArray cloud, const Point3d& p1, const Point3d& p2)
    {
        widget = cv::makePtr<WPaintedCloud>(cloud, p1, p2);
    }

    //! Paint cloud with gradient specified by given colors between given points
    CV_WRAP PyWPaintedCloud(InputArray cloud, const Point3d& p1, const Point3d& p2, const PyColor& c1, const PyColor& c2)
    {
        widget = cv::makePtr<WPaintedCloud>(cloud, p1, p2, c1, c2);
    }

    CV_WRAP void setRenderingProperty(int property, double value)
    {
        CV_Assert(widget);
        widget->setRenderingProperty(property, value);
    }

    Ptr<cv::viz::WPaintedCloud> widget;
};

/** @brief This 3D Widget defines a collection of clouds. :
@note In case there are four channels in the cloud, fourth channel is ignored.
*/
struct CV_EXPORTS_W_SIMPLE CV_WRAP_AS(WCloudCollection) PyWCloudCollection
{
public:
    CV_WRAP PyWCloudCollection()
    {
        widget = cv::makePtr<WCloudCollection>();
    }

    /** @brief Adds a cloud to the collection.

    @param cloud Point set which can be of type: CV_32FC3, CV_32FC4, CV_64FC3, CV_64FC4.
    @param colors Set of colors. It has to be of the same size with cloud.
    @param pose Pose of the cloud. Points in the cloud belong to mask when they are set to (NaN, NaN, NaN).
     */
    CV_WRAP void addCloud(InputArray cloud, InputArray colors, const PyAffine3d &pose = PyAffine3d::Identity())
    {
        widget->addCloud(cloud, colors, pose);
    }
    /** @brief Adds a cloud to the collection.

    @param cloud Point set which can be of type: CV_32FC3, CV_32FC4, CV_64FC3, CV_64FC4.
    @param color A single Color for the whole cloud.
    @param pose Pose of the cloud. Points in the cloud belong to mask when they are set to (NaN, NaN, NaN).
     */
    CV_WRAP void addCloud(InputArray cloud, const PyColor& color = Color::white(), const PyAffine3d& pose = PyAffine3d::Identity())
    {
        widget->addCloud(cloud, color, pose);
    }
    /** @brief Finalizes cloud data by repacking to single cloud.

    Useful for large cloud collections to reduce memory usage
    */
    CV_WRAP void finalize()
    {
        widget->finalize();
    }

    CV_WRAP void setRenderingProperty(int property, double value)
    {
        CV_Assert(widget);
        widget->setRenderingProperty(property, value);
    }

    Ptr<cv::viz::WCloudCollection> widget;
};

/** @brief This 3D Widget represents normals of a point cloud. :
*/
struct CV_EXPORTS_W_SIMPLE CV_WRAP_AS(WCloudNormals) PyWCloudNormals
{
public:
    PyWCloudNormals()
    {
    }
    /** @brief Constructs a WCloudNormals.

    @param cloud Point set which can be of type: CV_32FC3, CV_32FC4, CV_64FC3, CV_64FC4.
    @param normals A set of normals that has to be of same type with cloud.
    @param level Display only every level th normal.
    @param scale Scale of the arrows that represent normals.
    @param color Color of the arrows that represent normals.

    @note In case there are four channels in the cloud, fourth channel is ignored.
     */
    CV_WRAP PyWCloudNormals(InputArray cloud, InputArray normals, int level = 64, double scale = 0.1, const PyColor& color = Color::white())
    {
        widget = cv::makePtr<WCloudNormals>(cloud, normals, level, scale, color);
    }

    CV_WRAP void setRenderingProperty(int property, double value)
    {
        CV_Assert(widget);
        widget->setRenderingProperty(property, value);
    }

    Ptr<cv::viz::WCloudNormals> widget;
};


CV_WRAP_AS(makeTransformToGlobal) static inline
PyAffine3d makeTransformToGlobalPy(const Vec3d& axis_x, const Vec3d& axis_y, const Vec3d& axis_z, const Vec3d& origin = Vec3d::all(0));

CV_WRAP_AS(makeCameraPose) static inline
PyAffine3d makeCameraPosePy(const Vec3d& position, const Vec3d& focal_point, const Vec3d& y_dir);


/** @brief The Viz3d class represents a 3D visualizer window. This class is implicitly shared.
*/
class CV_EXPORTS_AS(Viz3d) PyViz3d
#ifndef OPENCV_BINDING_PARSER
    : public Viz3d
#endif
{
public:
    CV_WRAP static cv::Ptr<PyViz3d> create(const std::string& window_name = std::string())
    {
        return makePtr<PyViz3d>(window_name);
    }

    /** @brief The constructors.

    @param window_name Name of the window.
     */
    CV_WRAP PyViz3d(const std::string& window_name = std::string()) : Viz3d(window_name) {}

    CV_WRAP void showWidget(const String &id, PyWLine &widget);
    CV_WRAP void showWidget(const String &id, PyWLine &widget, PyAffine3d &pose);
    CV_WRAP void showWidget(const String &id, PyWSphere &widget);
    CV_WRAP void showWidget(const String &id, PyWSphere &widget, PyAffine3d &pose);
    CV_WRAP void showWidget(const String &id, PyWCameraPosition &widget);
    CV_WRAP void showWidget(const String &id, PyWCameraPosition &widget, PyAffine3d &pose);
    CV_WRAP void showWidget(const String &id, PyWArrow &widget);
    CV_WRAP void showWidget(const String &id, PyWArrow &widget, PyAffine3d &pose);
    CV_WRAP void showWidget(const String &id, PyWCircle &widget);
    CV_WRAP void showWidget(const String &id, PyWCircle &widget, PyAffine3d &pose);
    CV_WRAP void showWidget(const String &id, PyWPlane &widget);
    CV_WRAP void showWidget(const String &id, PyWPlane &widget, PyAffine3d &pose);
    CV_WRAP void showWidget(const String &id, PyWCone &widget);
    CV_WRAP void showWidget(const String &id, PyWCone &widget, PyAffine3d &pose);
    CV_WRAP void showWidget(const String &id, PyWCube &widget);
    CV_WRAP void showWidget(const String &id, PyWCube &widget, PyAffine3d &pose);
    CV_WRAP void showWidget(const String &id, PyWCylinder &widget);
    CV_WRAP void showWidget(const String &id, PyWCylinder &widget, PyAffine3d &pose);
    CV_WRAP void showWidget(const String &id, PyWCoordinateSystem &widget);
    CV_WRAP void showWidget(const String &id, PyWPaintedCloud &widget);
    CV_WRAP void showWidget(const String &id, PyWPaintedCloud &widget, PyAffine3d &pose);
    CV_WRAP void showWidget(const String &id, PyWCloudCollection &widget);
    CV_WRAP void showWidget(const String &id, PyWCloudCollection &widget, PyAffine3d &pose);
    CV_WRAP void showWidget(const String &id, PyWGrid &widget);
    CV_WRAP void showWidget(const String &id, PyWGrid &widget, PyAffine3d &pose);
    CV_WRAP void showWidget(const String &id, const cv::Ptr<WMesh> &widget);
    CV_WRAP void showWidget(const String &id, const cv::Ptr<WMesh> &widget, PyAffine3d &pose);
    CV_WRAP void showWidget(const String &id, PyWPolyLine &widget);
    CV_WRAP void showWidget(const String &id, PyWPolyLine &widget, PyAffine3d &pose);
    CV_WRAP void showWidget(const String &id, PyWCloud &widget);
    CV_WRAP void showWidget(const String &id, PyWCloud &widget, PyAffine3d &pose);
    CV_WRAP void showWidget(const String &id, PyWImage3D &widget, PyAffine3d &pose);
    CV_WRAP void showWidget(const String &id, PyWImage3D &widget);
    CV_WRAP void showWidget(const String &id, PyWImageOverlay &widget, PyAffine3d &pose);
    CV_WRAP void showWidget(const String &id, PyWImageOverlay &widget);
    CV_WRAP void showWidget(const String &id, PyWText &widget);
    CV_WRAP void showWidget(const String &id, PyWText &widget, PyAffine3d &pose);
    CV_WRAP void showWidget(const String &id, PyWText3D &widget);
    CV_WRAP void showWidget(const String &id, PyWText3D &widget, PyAffine3d &pose);
    CV_WRAP void showWidget(const String &id, PyWCloudNormals &widget, PyAffine3d &pose);
    CV_WRAP void showWidget(const String &id, PyWCloudNormals &widget);
    CV_WRAP void showWidget(const String &id, PyWTrajectory &widget, PyAffine3d &pose);
    CV_WRAP void showWidget(const String &id, PyWTrajectory &widget);
    CV_WRAP void showWidget(const String &id, PyWTrajectorySpheres &widget, PyAffine3d &pose);
    CV_WRAP void showWidget(const String &id, PyWTrajectorySpheres &widget);
    CV_WRAP void showWidget(const String &id, PyWTrajectoryFrustums &widget, PyAffine3d &pose);
    CV_WRAP void showWidget(const String &id, PyWTrajectoryFrustums &widget);


    /** @brief Removes a widget from the window.

    @param id The id of the widget that will be removed.
     */
    CV_WRAP
    void removeWidget(const String &id)
    { return Viz3d::removeWidget(id); }

    /** @brief Removes all widgets from the window.
    */
    CV_WRAP
    void removeAllWidgets()
    { return Viz3d::removeAllWidgets(); }

    /** @brief Removed all widgets and displays image scaled to whole window area.

    @param image Image to be displayed.
    @param window_size Size of Viz3d window. Default value means no change.
     */
    CV_WRAP
    void showImage(InputArray image, const Size& window_size = Size(-1, -1))
    { return Viz3d::showImage(image, window_size); }

    /** @brief Sets pose of a widget in the window.

    @param id The id of the widget whose pose will be set. @param pose The new pose of the widget.
     */
    CV_WRAP
    void setWidgetPose(const String &id, const PyAffine3d &pose)
    { return Viz3d::setWidgetPose(id, pose); }

    /** @brief Updates pose of a widget in the window by pre-multiplying its current pose.

    @param id The id of the widget whose pose will be updated. @param pose The pose that the current
    pose of the widget will be pre-multiplied by.
     */
    CV_WRAP
    void updateWidgetPose(const String &id, const PyAffine3d &pose)
    { return Viz3d::updateWidgetPose(id, pose); }

    /** @brief Returns the current pose of a widget in the window.

    @param id The id of the widget whose pose will be returned.
     */
    CV_WRAP
    PyAffine3d getWidgetPose(const String &id) const
    { return (PyAffine3d)Viz3d::getWidgetPose(id); }

#if 0
    /** @brief Sets the intrinsic parameters of the viewer using Camera.

    @param camera Camera object wrapping intrinsic parameters.
     */
    void setCamera(const Camera &camera);

    /** @brief Returns a camera object that contains intrinsic parameters of the current viewer.
    */
    Camera getCamera() const;
#endif

    /** @brief Returns the current pose of the viewer.
    */
    CV_WRAP PyAffine3d getViewerPose() const
    { return (PyAffine3d)Viz3d::getViewerPose(); }

    /** @brief Sets pose of the viewer.

    @param pose The new pose of the viewer.
     */
    CV_WRAP
    void setViewerPose(const PyAffine3d &pose)
    { return Viz3d::setViewerPose(pose); }

    /** @brief Resets camera viewpoint to a 3D widget in the scene.

    @param id Id of a 3D widget.
     */
    CV_WRAP
    void resetCameraViewpoint(const String &id)
    { return Viz3d::resetCameraViewpoint(id); }

    /** @brief Resets camera.
    */
    CV_WRAP
    void resetCamera()
    { return Viz3d::resetCamera(); }

    /** @brief Transforms a point in world coordinate system to window coordinate system.

    @param pt Point in world coordinate system.
    @param window_coord Output point in window coordinate system.
     */
    CV_WRAP void convertToWindowCoordinates(const Point3d &pt, CV_OUT Point3d &window_coord)
    { return Viz3d::convertToWindowCoordinates(pt, window_coord); }

#if 0
    /** @brief Transforms a point in window coordinate system to a 3D ray in world coordinate system.

    @param window_coord Point in window coordinate system. @param origin Output origin of the ray.
    @param direction Output direction of the ray.
     */
    void converTo3DRay(const Point3d &window_coord, Point3d &origin, Vec3d &direction);
#endif

    /** @brief Returns the current size of the window.
    */
    CV_WRAP
    Size getWindowSize() const
    { return Viz3d::getWindowSize(); }

    /** @brief Sets the size of the window.

    @param window_size New size of the window.
     */
    CV_WRAP
    void setWindowSize(const Size& window_size)
    { return Viz3d::setWindowSize(window_size); }

    /** @brief Returns the name of the window which has been set in the constructor.
     *  `Viz - ` is prepended to the name if necessary.
     */
    CV_WRAP
    String getWindowName() const
    { return Viz3d::getWindowName(); }

    /** @brief Returns the Mat screenshot of the current scene.
    */
    CV_WRAP
    cv::Mat getScreenshot() const
    { return Viz3d::getScreenshot(); }

    /** @brief Saves screenshot of the current scene.

    @param file Name of the file.
     */
    CV_WRAP
    void saveScreenshot(const String &file)
    { return Viz3d::saveScreenshot(file); }

    /** @brief Sets the position of the window in the screen.

    @param window_position coordinates of the window
     */
    CV_WRAP
    void setWindowPosition(const Point& window_position)
    { return Viz3d::setWindowPosition(window_position); }

    /** @brief Sets or unsets full-screen rendering mode.

    @param mode If true, window will use full-screen mode.
     */
    CV_WRAP
    void setFullScreen(bool mode = true)
    { return Viz3d::setFullScreen(mode); }

    /** @brief Sets background color.
    */
    CV_WRAP
    void setBackgroundColor(const PyColor& color, const PyColor& color2 = Color::not_set())
    { return Viz3d::setBackgroundColor(color, color2); }

    CV_WRAP
    void setBackgroundTexture(InputArray image = noArray())
    { return Viz3d::setBackgroundTexture(image); }

    CV_WRAP
    void setBackgroundMeshLab()
    { return Viz3d::setBackgroundMeshLab(); }

    /** @brief The window renders and starts the event loop.
    */
    CV_WRAP
    void spin()
    { return Viz3d::spin(); }

    /** @brief Starts the event loop for a given time.

    @param time Amount of time in milliseconds for the event loop to keep running.
    @param force_redraw If true, window renders.
     */
    CV_WRAP
    void spinOnce(int time = 1, bool force_redraw = false)
    { return Viz3d::spinOnce(time, force_redraw); }

    /** @brief Create a window in memory instead of on the screen.
     */
    CV_WRAP
    void setOffScreenRendering()
    { return Viz3d::setOffScreenRendering(); }

    /** @brief Remove all lights from the current scene.
    */
    CV_WRAP
    void removeAllLights()
    { return Viz3d::removeAllLights(); }

#if 0
    /** @brief Add a light in the scene.

    @param position The position of the light.
    @param focalPoint The point at which the light is shining
    @param color The color of the light
    @param diffuseColor The diffuse color of the light
    @param ambientColor The ambient color of the light
    @param specularColor The specular color of the light
     */
    void addLight(const Vec3d &position, const Vec3d &focalPoint = Vec3d(0, 0, 0), const Color &color = Color::white(),
                  const Color &diffuseColor = Color::white(), const Color &ambientColor = Color::black(),
                  const Color &specularColor = Color::white());
#endif

    /** @brief Returns whether the event loop has been stopped.
    */
    CV_WRAP
    bool wasStopped() const
    { return Viz3d::wasStopped(); }

    CV_WRAP
    void close()
    { return Viz3d::close(); }

    /** @brief Sets rendering property of a widget.

    @param id Id of the widget.
    @param property Property that will be modified.
    @param value The new value of the property.

    Rendering property can be one of the following:
    -   **POINT_SIZE**
    -   **OPACITY**
    -   **LINE_WIDTH**
    -   **FONT_SIZE**

    REPRESENTATION: Expected values are
    -   **REPRESENTATION_POINTS**
    -   **REPRESENTATION_WIREFRAME**
    -   **REPRESENTATION_SURFACE**

    IMMEDIATE_RENDERING:
    -   Turn on immediate rendering by setting the value to 1.
    -   Turn off immediate rendering by setting the value to 0.

    SHADING: Expected values are
    -   **SHADING_FLAT**
    -   **SHADING_GOURAUD**
    -   **SHADING_PHONG**
     */
    CV_WRAP
    void setRenderingProperty(const String &id, int property, double value)
    { return Viz3d::setRenderingProperty(id, property, value); }

    /** @brief Returns rendering property of a widget.

    @param id Id of the widget.
    @param property Property.

    Rendering property can be one of the following:
    -   **POINT_SIZE**
    -   **OPACITY**
    -   **LINE_WIDTH**
    -   **FONT_SIZE**

    REPRESENTATION: Expected values are
    -   **REPRESENTATION_POINTS**
    -   **REPRESENTATION_WIREFRAME**
    -   **REPRESENTATION_SURFACE**

    IMMEDIATE_RENDERING:
    -   Turn on immediate rendering by setting the value to 1.
    -   Turn off immediate rendering by setting the value to 0.

    SHADING: Expected values are
    -   **SHADING_FLAT**
    -   **SHADING_GOURAUD**
    -   **SHADING_PHONG**
     */
    CV_WRAP
    double getRenderingProperty(const String &id, int property)
    { return Viz3d::getRenderingProperty(id, property); }

    /** @brief Sets geometry representation of the widgets to surface, wireframe or points.

    @param representation Geometry representation which can be one of the following:
    -   **REPRESENTATION_POINTS**
    -   **REPRESENTATION_WIREFRAME**
    -   **REPRESENTATION_SURFACE**
     */
    CV_WRAP
    void setRepresentation(int representation)
    { return Viz3d::setRepresentation(representation); }

    CV_WRAP
    void setGlobalWarnings(bool enabled = false)
    { return Viz3d::setGlobalWarnings(enabled); }
};


}}  // namespace
#endif  // OPENCV_PYTHON_VIZ_HPP
