// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _OPENCV_OVIS_H_
#define _OPENCV_OVIS_H_

#include <opencv2/core.hpp>

/**
@defgroup ovis OGRE 3D Visualiser
*/

namespace cv {
namespace ovis {
//! @addtogroup ovis
//! @{

enum SceneSettings
{
    /// the window will use a seperate scene. The scene will be shared otherwise.
    SCENE_SEPERATE = 1,
    /// allow the user to control the camera.
    SCENE_INTERACTIVE = 2,
    /// draw coordinate system crosses for debugging
    SCENE_SHOW_CS_CROSS = 4
};

enum MaterialProperty
{
    MATERIAL_POINT_SIZE,
    MATERIAL_OPACITY,
    MATERIAL_EMISSIVE,
    MATERIAL_TEXTURE
};

/**
 * A 3D viewport and the associated scene
 */
class CV_EXPORTS_W WindowScene {
public:
    virtual ~WindowScene();

    /**
     * set window background to custom image
     *
     * creates a texture named "<title>_Background"
     * @param image
     */
    CV_WRAP virtual void setBackground(InputArray image) = 0;

    /**
     * place an entity of an mesh in the scene
     * @param name entity name
     * @param meshname mesh name
     * @param rot Rodrigues vector or 3x3 rotation matrix
     * @param tvec translation
     */
    CV_WRAP virtual void createEntity(const String& name, const String& meshname,
                                      InputArray tvec = noArray(), InputArray rot = noArray()) = 0;

    /**
     * convenience method to visualize a camera position
     *
     * the entity uses a material with the same name that can be used to change the line color.
     * @param name entity name
     * @param K intrinsic matrix
     * @param imsize image size
     * @param zFar far plane in camera coordinates
     * @param rot Rodrigues vector or 3x3 rotation matrix
     * @param tvec translation
     * @return the extents of the Frustum at far plane, where the top left corner denotes the principal
     * point offset
     */
    CV_WRAP virtual Rect2d createCameraEntity(const String& name, InputArray K, const Size& imsize,
                                              float zFar, InputArray tvec = noArray(),
                                              InputArray rot = noArray()) = 0;

    /**
     * creates a point light in the scene
     * @param name entity name
     * @param rot Rodrigues vector or 3x3 rotation matrix
     * @param tvec translation
     * @param diffuseColor
     * @param specularColor
     */
    CV_WRAP virtual void createLightEntity(const String& name, InputArray tvec = noArray(),
                                           InputArray rot = noArray(),
                                           const Scalar& diffuseColor = Scalar::all(1),
                                           const Scalar& specularColor = Scalar::all(1)) = 0;

    /**
     * update entity pose by transformation in the parent coordinate space. (pre-rotation)
     * @param name entity name
     * @param rot Rodrigues vector or 3x3 rotation matrix
     * @param tvec translation
     */
    CV_WRAP virtual void updateEntityPose(const String& name, InputArray tvec = noArray(),
                                          InputArray rot = noArray()) = 0;

    /**
     * set entity pose in the world coordinate space.
     * @param name enitity name
     * @param rot Rodrigues vector or 3x3 rotation matrix
     * @param tvec translation
     * @param invert use the inverse of the given pose
     */
    CV_WRAP virtual void setEntityPose(const String& name, InputArray tvec = noArray(),
                                       InputArray rot = noArray(), bool invert = false) = 0;

    /**
     * read back image of last call to @ref renderOneFrame
     */
    CV_WRAP virtual void getScreenshot(OutputArray frame) = 0;

    /**
     * convenience method to force the "up" axis to stay fixed
     *
     * works with both programmatic changes and SCENE_INTERACTIVE
     * @param useFixed whether to enforce the fixed yaw axis
     * @param up the axis to be fixed
     */
    CV_WRAP virtual void fixCameraYawAxis(bool useFixed, InputArray up = noArray()) = 0;

    /**
     * Sets the current camera pose
     * @param rot Rodrigues vector or 3x3 rotation matrix
     * @param tvec translation
     * @param invert use the inverse of the given pose
     */
    CV_WRAP virtual void setCameraPose(InputArray tvec = noArray(), InputArray rot = noArray(),
                                       bool invert = false) = 0;

    /**
     * convenience method to orient the camera to a specific entity
     * @param target entity name
     * @param offset offset from entity centre
     */
    CV_WRAP virtual void setCameraLookAt(const String& target, InputArray offset = noArray()) = 0;

    /**
     * Retrieves the current camera pose
     * @param R 3x3 rotation matrix
     * @param tvec translation vector
     * @param invert return the inverted pose
     */
    CV_WRAP virtual void getCameraPose(OutputArray R = noArray(), OutputArray tvec = noArray(),
                                       bool invert = false) = 0;

    /**
     * set intrinsics of the camera
     * @param K intrinsic matrix
     * @param imsize image size
     */
    CV_WRAP virtual void setCameraIntrinsics(InputArray K, const Size& imsize) = 0;
};

/**
 * Add an additional resource location that is search for meshes, textures and materials
 *
 * must be called before the first createWindow. If give path does not exist, retries inside
 * Ogre Media Directory.
 * @param path folder or Zip archive.
 */
CV_EXPORTS_W void addResourceLocation(const String& path);

/**
 * create a new rendering window/ viewport
 * @param title window title
 * @param size size of the window
 * @param flags @see SceneSettings
 */
CV_EXPORTS_W Ptr<WindowScene> createWindow(const String& title, const Size& size,
                                           int flags = SCENE_INTERACTIVE);

/**
 * update all windows
 * @return true if this functian can be called again (i.e. continue rendering). false otherwise.
 */
CV_EXPORTS_W bool renderOneFrame();

/**
 * set the property of a material to the given value
 * @param name material name
 * @param prop property @ref MaterialProperty
 * @param value the value
 */
CV_EXPORTS_W void setMaterialProperty(const String& name, int prop, const Scalar& value);

/// @overload
CV_EXPORTS_W void setMaterialProperty(const String& name, int prop, const String& value);

/**
 * create a 2D plane, X right, Y down, Z up
 *
 * creates a material and a texture with the same name
 * @param name name of the mesh
 * @param size size in world units
 * @param image optional texture
 */
CV_EXPORTS_W void createPlaneMesh(const String& name, const Size2f& size, InputArray image = noArray());

/**
 * creates a point cloud mesh
 *
 * creates a material with the same name
 * @param name name of the mesh
 * @param vertices float vector of positions
 * @param colors uchar vector of colors
 */
CV_EXPORTS_W void createPointCloudMesh(const String& name, InputArray vertices, InputArray colors = noArray());

/**
 * creates a grid
 *
 * creates a material with the same name
 * @param name name of the mesh
 * @param size extents of the grid
 * @param segments number of segments per side
 */
CV_EXPORTS_W void createGridMesh(const String& name, const Size2f& size, const Size& segments = Size(1, 1));
//! @}
}
}

#endif
