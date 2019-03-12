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
    SCENE_SHOW_CS_CROSS = 4,
    /// Apply anti-aliasing. The first window determines the setting for all windows.
    SCENE_AA = 8
};

enum MaterialProperty
{
    MATERIAL_POINT_SIZE,
    MATERIAL_LINE_WIDTH,
    MATERIAL_OPACITY,
    MATERIAL_EMISSIVE,
    MATERIAL_TEXTURE0,
    MATERIAL_TEXTURE = MATERIAL_TEXTURE0,
    MATERIAL_TEXTURE1,
    MATERIAL_TEXTURE2,
    MATERIAL_TEXTURE3,
};

enum EntityProperty
{
    ENTITY_MATERIAL,
    ENTITY_SCALE,
    ENTITY_AABB_WORLD
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

    /// @overload
    CV_WRAP_AS(setBackgroundColor) virtual void setBackground(const Scalar& color) = 0;

    /**
     * enable an ordered chain of full-screen post processing effects
     *
     * this way you can add distortion or SSAO effects.
     * The effects themselves must be defined inside Ogre .compositor scripts.
     * @see addResourceLocation
     * @param names compositor names that will be applied in order of appearance
     */
    CV_WRAP virtual void setCompositors(const std::vector<String>& names) = 0;

    /**
     * place an entity of an mesh in the scene
     *
     * the mesh needs to be created beforehand. Either programmatically
     * by e.g. @ref createPointCloudMesh or by placing an Ogre .mesh file in a resource location.
     * @see addResourceLocation
     * @param name entity name
     * @param meshname mesh name
     * @param rot @ref Rodrigues vector or 3x3 rotation matrix
     * @param tvec translation
     */
    CV_WRAP virtual void createEntity(const String& name, const String& meshname,
                                      InputArray tvec = noArray(), InputArray rot = noArray()) = 0;

    /**
     * remove an entity from the scene
     * @param name entity name
     */
    CV_WRAP virtual void removeEntity(const String& name) = 0;

    /**
     * set the property of an entity to the given value
     * @param name entity name
     * @param prop @ref EntityProperty
     * @param value the value
     */
    CV_WRAP virtual void setEntityProperty(const String& name, int prop, const Scalar& value) = 0;

    /// @overload
    CV_WRAP virtual void setEntityProperty(const String& name, int prop, const String& value) = 0;

    /**
     * get the property of an entity
     * @param name entity name
     * @param prop @ref EntityProperty
     * @param value the value
     */
    CV_WRAP virtual void getEntityProperty(const String& name, int prop, OutputArray value) = 0;

    /**
     * convenience method to visualize a camera position
     *
     * the entity uses a material with the same name that can be used to change the line color.
     * @param name entity name
     * @param K intrinsic matrix
     * @param imsize image size
     * @param zFar far plane in camera coordinates
     * @param rot @ref Rodrigues vector or 3x3 rotation matrix
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
     * @param rot @ref Rodrigues vector or 3x3 rotation matrix
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
     * @param rot @ref Rodrigues vector or 3x3 rotation matrix
     * @param tvec translation
     */
    CV_WRAP virtual void updateEntityPose(const String& name, InputArray tvec = noArray(),
                                          InputArray rot = noArray()) = 0;

    /**
     * set entity pose in the world coordinate space.
     * @param name enitity name
     * @param rot @ref Rodrigues vector or 3x3 rotation matrix
     * @param tvec translation
     * @param invert use the inverse of the given pose
     */
    CV_WRAP virtual void setEntityPose(const String& name, InputArray tvec = noArray(),
                                       InputArray rot = noArray(), bool invert = false) = 0;

    /**
     * read back the image generated by the last call to @ref waitKey
     */
    CV_WRAP virtual void getScreenshot(OutputArray frame) = 0;

    /**
     * read back the texture of an active compositor
     * @param compname name of the compositor
     * @param texname name of the texture inside the compositor
     * @param mrtIndex if texture is a MRT, specifies the attachment
     * @param out the texture contents
     */
    CV_WRAP virtual void getCompositorTexture(const String& compname, const String& texname,
                                              OutputArray out, int mrtIndex = 0) = 0;

    /**
     * get the depth for the current frame.
     *
     * return the per pixel distance to the camera in world units
     */
    CV_WRAP virtual void getDepth(OutputArray depth) = 0;

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
     * @param rot @ref Rodrigues vector or 3x3 rotation matrix
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
     *
     * @param K intrinsic matrix or noArray(). If noArray() is specified, imsize
     * is ignored and zNear/ zFar can be set separately.
     * @param imsize image size
     * @param zNear near clip distance or -1 to keep the current
     * @param zFar  far clip distance or -1 to keep the current
     */
    CV_WRAP virtual void setCameraIntrinsics(InputArray K, const Size& imsize,
                                             float zNear = -1,
                                             float zFar = -1) = 0;
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
 * @param flags a combination of @ref SceneSettings
 */
CV_EXPORTS_W Ptr<WindowScene> createWindow(const String& title, const Size& size,
                                           int flags = SCENE_INTERACTIVE | SCENE_AA);

/**
 * update all windows and wait for keyboard event
 *
 * @param delay 0 is the special value that means "forever".
 *        Any positive number returns after sync to blank (typically 16ms).
 * @return the code of the pressed key or -1 if no key was pressed
 */
CV_EXPORTS_W int waitKey(int delay = 0);

/**
 * set the property of a material to the given value
 * @param name material name
 * @param prop @ref MaterialProperty
 * @param value the value
 */
CV_EXPORTS_W void setMaterialProperty(const String& name, int prop, const Scalar& value);

/// @overload
CV_EXPORTS_W void setMaterialProperty(const String& name, int prop, const String& value);

/**
 * set the shader property of a material to the given value
 * @param name material name
 * @param prop property name
 * @param value the value
 */
CV_EXPORTS_W void setMaterialProperty(const String& name, const String& prop, const Scalar& value);

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

/**
 * updates an existing texture
 *
 * A new texture can be created with @ref createPlaneMesh
 * @param name name of the texture
 * @param image the image data
 */
CV_EXPORTS_W void updateTexture(const String& name, InputArray image);
//! @}
}
}

#endif
