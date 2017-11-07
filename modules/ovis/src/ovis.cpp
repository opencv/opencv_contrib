// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include <OgreApplicationContext.h>
#include <OgreCameraMan.h>
#include <OgreRectangle2D.h>

#include <opencv2/calib3d.hpp>

namespace cv
{
namespace ovis
{
using namespace Ogre;

const char* RESOURCEGROUP_NAME = "OVIS";
Ptr<Application> _app;

static const char* RENDERSYSTEM_NAME = "OpenGL 3+ Rendering Subsystem";
static std::vector<String> _extraResourceLocations;

// convert from OpenCV to Ogre coordinates:
// rotation by 180° around x axis
static Matrix3 toOGRE = Matrix3(1, 0, 0, 0, -1, 0, 0, 0, -1);
static Vector2 toOGRE_SS = Vector2(1, -1);

WindowScene::~WindowScene() {}

void _createTexture(const String& name, Mat image)
{
    TextureManager& texMgr = TextureManager::getSingleton();
    TexturePtr tex = texMgr.getByName(name, RESOURCEGROUP_NAME);

    Image im;
    im.loadDynamicImage(image.ptr(), image.cols, image.rows, 1, PF_BYTE_BGR);

    if (tex)
    {
        // update
        PixelBox box = im.getPixelBox();
        tex->getBuffer()->blitFromMemory(box, box);
        return;
    }

    texMgr.loadImage(name, RESOURCEGROUP_NAME, im);
}

static void _convertRT(InputArray rot, InputArray tvec, Quaternion& q, Vector3& t,
                       bool invert = false)
{
    CV_Assert(rot.empty() || rot.rows() == 3 || rot.size() == Size(3, 3),
              tvec.empty() || tvec.rows() == 3);

    q = Quaternion::IDENTITY;
    t = Vector3::ZERO;

    if (!rot.empty())
    {
        Mat _R;

        if (rot.size() == Size(3, 3))
        {
            _R = rot.getMat();
        }
        else
        {
            Rodrigues(rot, _R);
        }

        Matrix3 R;
        _R.copyTo(Mat_<Real>(3, 3, R[0]));
        q = Quaternion(toOGRE * R);

        if (invert)
        {
            q = q.Inverse();
        }
    }

    if (!tvec.empty())
    {
        tvec.copyTo(Mat_<Real>(3, 1, t.ptr()));
        t = toOGRE * t;

        if(invert)
        {
            t = q * -t;
        }
    }
}

static void _setCameraIntrinsics(Camera* cam, InputArray _K, const Size& imsize)
{
    CV_Assert(_K.size() == Size(3, 3));

    cam->setAspectRatio(float(imsize.width) / imsize.height);

    Matx33f K = _K.getMat();

    float fovy = atan2(K(1, 2), K(1, 1)) + atan2(imsize.height - K(1, 2), K(1, 1));
    cam->setFOVy(Radian(fovy));

    Vec2f pp_offset = Vec2f(0.5, 0.5) - Vec2f(K(0, 2) / imsize.width, K(1, 2) / imsize.height);
    cam->setFrustumOffset(toOGRE_SS * Vector2(pp_offset.val));
}

static SceneNode* _getSceneNode(SceneManager* sceneMgr, const String& name)
{
    MovableObject* mo = NULL;

    try
    {
        mo = sceneMgr->getCamera(name);
    }
    catch (ItemIdentityException&)
    {
        // ignore
    }

    try
    {
        if (!mo)
            mo = sceneMgr->getLight(name);
    }
    catch (ItemIdentityException&)
    {
        // ignore
    }

    if (!mo)
        mo = sceneMgr->getEntity(name);

    return mo->getParentSceneNode();
}

struct Application : public OgreBites::ApplicationContext
{
    Ogre::SceneManager* sceneMgr;
    Ogre::String title;
    uint32_t w;
    uint32_t h;

    Application(const Ogre::String& _title, const Size& sz)
        : OgreBites::ApplicationContext("ovis", false), sceneMgr(NULL), title(_title), w(sz.width),
          h(sz.height)
    {
    }

    void setupInput(bool /*grab*/)
    {
        // empty impl to show cursor
    }

    bool oneTimeConfig()
    {
        Ogre::RenderSystem* rs = getRoot()->getRenderSystemByName(RENDERSYSTEM_NAME);
        CV_Assert(rs);
        getRoot()->setRenderSystem(rs);
        return true;
    }

    OgreBites::NativeWindowPair createWindow(const Ogre::String& name, uint32_t _w, uint32_t _h,
                                             NameValuePairList miscParams = NameValuePairList())
    {
        Ogre::String _name = name;
        if (!sceneMgr)
        {
            _w = w;
            _h = h;
            _name = title;
        }
        miscParams["FSAA"] = "4";
        miscParams["vsync"] = "true";

        return OgreBites::ApplicationContext::createWindow(_name, _w, _h, miscParams);
    }

    void locateResources()
    {
        OgreBites::ApplicationContext::locateResources();
        ResourceGroupManager& rgm = ResourceGroupManager::getSingleton();
        rgm.createResourceGroup(RESOURCEGROUP_NAME);

        for (size_t i = 0; i < _extraResourceLocations.size(); i++)
        {
            String loc = _extraResourceLocations[i];
            String type = StringUtil::endsWith(loc, ".zip") ? "Zip" : "FileSystem";

            if (!FileSystemLayer::fileExists(loc))
            {
                loc = FileSystemLayer::resolveBundlePath(getDefaultMediaDir() + "/" + loc);
            }

            rgm.addResourceLocation(loc, type, RESOURCEGROUP_NAME);
        }
    }

    void setup()
    {
        OgreBites::ApplicationContext::setup();

        MaterialManager& matMgr = MaterialManager::getSingleton();
        matMgr.setDefaultTextureFiltering(TFO_ANISOTROPIC);
        matMgr.setDefaultAnisotropy(16);
    }
};

class WindowSceneImpl : public WindowScene, public OgreBites::InputListener
{
    String title;
    Root* root;
    SceneManager* sceneMgr;
    SceneNode* camNode;
    RenderWindow* rWin;
    Ptr<OgreBites::CameraMan> camman;
    Ptr<Rectangle2D> bgplane;

public:
    WindowSceneImpl(Ptr<Application> app, const String& _title, const Size& sz, int flags)
        : title(_title), root(app->getRoot())
    {
        if (!app->sceneMgr)
        {
            flags |= SCENE_SEPERATE;
        }

        if (flags & SCENE_SEPERATE)
        {
            sceneMgr = root->createSceneManager("DefaultSceneManager", title);
            RTShader::ShaderGenerator& shadergen = RTShader::ShaderGenerator::getSingleton();
            shadergen.addSceneManager(sceneMgr); // must be done before we do anything with the scene

            sceneMgr->setAmbientLight(ColourValue(.1, .1, .1));
            _createBackground();
        }
        else
        {
            sceneMgr = app->sceneMgr;
        }

        if(flags & SCENE_SHOW_CS_CROSS)
        {
            sceneMgr->setDisplaySceneNodes(true);
        }

        Camera* cam = sceneMgr->createCamera(title);
        cam->setNearClipDistance(0.5);
        cam->setAutoAspectRatio(true);
        camNode = sceneMgr->getRootSceneNode()->createChildSceneNode();
        camNode->attachObject(cam);

        if (flags & SCENE_INTERACTIVE)
        {
            camman.reset(new OgreBites::CameraMan(camNode));
            camman->setStyle(OgreBites::CS_ORBIT);
        }

        if (!app->sceneMgr)
        {
            app->sceneMgr = sceneMgr;
            rWin = app->getRenderWindow();
            app->addInputListener(this);

            if (camman)
                app->addInputListener(camman.get());
        }
        else
        {
            OgreBites::NativeWindowPair nwin = app->createWindow(title, sz.width, sz.height);
            rWin = nwin.render;
            if (camman)
                app->addInputListener(nwin.native, camman.get());

            app->addInputListener(nwin.native, this);
        }

        rWin->addViewport(cam);
    }

    void setBackground(InputArray image)
    {
        CV_Assert(image.type() == CV_8UC3, bgplane);

        String name = sceneMgr->getName() + "_Background";

        _createTexture(name, image.getMat());

        // correct for pixel centers
        Vector2 pc(0.5 / image.cols(), 0.5 / image.rows());
        bgplane->setUVs(pc, Vector2(pc[0], 1 - pc[1]), Vector2(1 - pc[0], pc[1]), Vector2(1, 1) - pc);

        Pass* rpass = bgplane->getMaterial()->getBestTechnique()->getPasses()[0];
        rpass->getTextureUnitStates()[0]->setTextureName(name);
    }

    void createEntity(const String& name, const String& meshname, InputArray tvec, InputArray rot)
    {
        Entity* ent = sceneMgr->createEntity(name, meshname, RESOURCEGROUP_NAME);

        Quaternion q;
        Vector3 t;
        _convertRT(rot, tvec, q, t);
        SceneNode* node = sceneMgr->getRootSceneNode()->createChildSceneNode(t, q);
        node->attachObject(ent);
    }

    Rect2d createCameraEntity(const String& name, InputArray K, const Size& imsize, float zFar,
                              InputArray tvec, InputArray rot)
    {
        MaterialPtr mat = MaterialManager::getSingleton().create(name, RESOURCEGROUP_NAME);
        Pass* rpass = mat->getTechniques()[0]->getPasses()[0];
        rpass->setEmissive(ColourValue::White);

        Camera* cam = sceneMgr->createCamera(name);
        cam->setMaterial(mat);

        cam->setVisible(true);
        cam->setDebugDisplayEnabled(true);
        cam->setNearClipDistance(1e-9);
        cam->setFarClipDistance(zFar);

        _setCameraIntrinsics(cam, K, imsize);

        Quaternion q;
        Vector3 t;
        _convertRT(rot, tvec, q, t);
        SceneNode* node = sceneMgr->getRootSceneNode()->createChildSceneNode(t, q);
        node->attachObject(cam);

        RealRect ext = cam->getFrustumExtents();
        float scale = zFar / cam->getNearClipDistance(); // convert to ext at zFar

        return Rect2d(toOGRE_SS[0] * (ext.right - ext.width() / 2) * scale,
                      toOGRE_SS[1] * (ext.bottom - ext.height() / 2) * scale, ext.width() * scale,
                      ext.height() * scale);
    }

    void createLightEntity(const String& name, InputArray tvec, InputArray rot, const Scalar& diffuseColour,
                           const Scalar& specularColour)
    {
        Light* light = sceneMgr->createLight(name);
        light->setDirection(Vector3::NEGATIVE_UNIT_Z);
        // convert to BGR
        light->setDiffuseColour(ColourValue(diffuseColour[2], diffuseColour[1], diffuseColour[0]));
        light->setSpecularColour(ColourValue(specularColour[2], specularColour[1], specularColour[0]));

        Quaternion q;
        Vector3 t;
        _convertRT(rot, tvec, q, t);
        SceneNode* node = sceneMgr->getRootSceneNode()->createChildSceneNode(t, q);
        node->attachObject(light);
    }

    void updateEntityPose(const String& name, InputArray tvec, InputArray rot)
    {
        SceneNode* node = _getSceneNode(sceneMgr, name);
        Quaternion q;
        Vector3 t;
        _convertRT(rot, tvec, q, t);
        node->rotate(q, Ogre::Node::TS_LOCAL);
        node->translate(t, Ogre::Node::TS_LOCAL);
    }

    void setEntityPose(const String& name, InputArray tvec, InputArray rot, bool invert)
    {
        SceneNode* node = _getSceneNode(sceneMgr, name);
        Quaternion q;
        Vector3 t;
        _convertRT(rot, tvec, q, t, invert);
        node->setOrientation(q);
        node->setPosition(t);
    }

    void _createBackground()
    {
        String name = "_" + sceneMgr->getName() + "_DefaultBackground";

        Mat_<Vec3b> img = (Mat_<Vec3b>(2, 1) << Vec3b(2, 1, 1), Vec3b(240, 120, 120));
        _createTexture(name, img);

        MaterialPtr mat = MaterialManager::getSingleton().create(name, RESOURCEGROUP_NAME);
        Pass* rpass = mat->getTechniques()[0]->getPasses()[0];
        rpass->setLightingEnabled(false);
        rpass->setDepthCheckEnabled(false);
        rpass->setDepthWriteEnabled(false);
        rpass->createTextureUnitState(name);

        bgplane.reset(new Rectangle2D(true));
        bgplane->setCorners(-1.0, 1.0, 1.0, -1.0);

        // correct for pixel centers
        Vector2 pc(0.5 / img.cols, 0.5 / img.rows);
        bgplane->setUVs(pc, Vector2(pc[0], 1 - pc[1]), Vector2(1 - pc[0], pc[1]), Vector2(1, 1) - pc);

        bgplane->setMaterial(mat);
        bgplane->setRenderQueueGroup(RENDER_QUEUE_BACKGROUND);
        bgplane->setBoundingBox(AxisAlignedBox(AxisAlignedBox::BOX_INFINITE));

        sceneMgr->getRootSceneNode()->createChildSceneNode()->attachObject(bgplane.get());
    }

    void getScreenshot(OutputArray frame)
    {
        frame.create(rWin->getHeight(), rWin->getWidth(), CV_8UC3);

        Mat out = frame.getMat();
        PixelBox pb(rWin->getWidth(), rWin->getHeight(), 1, PF_BYTE_BGR, out.ptr());
        rWin->copyContentsToMemory(pb, pb);
    }

    bool keyPressed(const OgreBites::KeyboardEvent& evt)
    {
        if (evt.keysym.sym == SDLK_ESCAPE)
            root->queueEndRendering();

        return true;
    }

    void fixCameraYawAxis(bool useFixed, InputArray _up)
    {
        Vector3 up = Vector3::UNIT_Y;
        if (!_up.empty())
        {
            _up.copyTo(Mat_<Real>(3, 1, up.ptr()));
            up = toOGRE * up;
        }

        Camera* cam = sceneMgr->getCamera(title);
        cam->getParentSceneNode()->setFixedYawAxis(useFixed, up);
    }

    void setCameraPose(InputArray tvec, InputArray rot, bool invert)
    {
        Camera* cam = sceneMgr->getCamera(title);

        SceneNode* node = cam->getParentSceneNode();
        Quaternion q;
        Vector3 t;
        _convertRT(rot, tvec, q, t, invert);

        if (!rot.empty())
            node->setOrientation(q);

        if (!tvec.empty())
            node->setPosition(t);
    }

    void getCameraPose(OutputArray R, OutputArray tvec, bool invert)
    {
        Camera* cam = sceneMgr->getCamera(title);
        SceneNode* node = cam->getParentSceneNode();

        Matrix3 _R;
        node->getOrientation().ToRotationMatrix(_R);

        if (invert)
        {
            _R = _R.Transpose();
        }

        if (tvec.needed())
        {
            Vector3 _tvec = node->getPosition();

            if (invert)
            {
                _tvec = _R * -_tvec;
            }

            _tvec = toOGRE.Transpose() * _tvec;
            Mat_<Real>(3, 1, _tvec.ptr()).copyTo(tvec);
        }

        if (R.needed())
        {
            _R = toOGRE.Transpose() * _R;
            Mat_<Real>(3, 3, _R[0]).copyTo(R);
        }
    }

    void setCameraIntrinsics(InputArray K, const Size& imsize)
    {
        Camera* cam = sceneMgr->getCamera(title);
        _setCameraIntrinsics(cam, K, imsize);
    }

    void setCameraLookAt(const String& target, InputArray offset)
    {
        SceneNode* cam = sceneMgr->getCamera(title)->getParentSceneNode();
        SceneNode* tgt = sceneMgr->getEntity(target)->getParentSceneNode();

        Vector3 _offset = Vector3::ZERO;

        if (!offset.empty())
        {
            offset.copyTo(Mat_<Real>(3, 1, _offset.ptr()));
            _offset = toOGRE * _offset;
        }

        cam->lookAt(tgt->_getDerivedPosition() + _offset, Ogre::Node::TS_WORLD);
    }
};

CV_EXPORTS_W void addResourceLocation(const String& path) { _extraResourceLocations.push_back(path); }

Ptr<WindowScene> createWindow(const String& title, const Size& size, int flags)
{
    if (!_app)
    {
        _app = makePtr<Application>(title.c_str(), size);
        _app->initApp();
    }

    return makePtr<WindowSceneImpl>(_app, title, size, flags);
}

CV_EXPORTS_W bool renderOneFrame()
{
    CV_Assert(_app);

    _app->getRoot()->renderOneFrame();
    return not _app->getRoot()->endRenderingQueued();
}

void setMaterialProperty(const String& name, int prop, const Scalar& val)
{
    CV_Assert(_app);

    MaterialPtr mat = MaterialManager::getSingleton().getByName(name, RESOURCEGROUP_NAME);

    CV_Assert(mat);

    Pass* rpass = mat->getTechniques()[0]->getPasses()[0];
    ColourValue col;

    switch (prop)
    {
    case MATERIAL_POINT_SIZE:
        rpass->setPointSize(val[0]);
        break;
    case MATERIAL_OPACITY:
        col = rpass->getDiffuse();
        col.a = val[0];
        rpass->setDiffuse(col);
        rpass->setSceneBlending(SBT_TRANSPARENT_ALPHA);
        rpass->setDepthWriteEnabled(false);
        break;
    case MATERIAL_EMISSIVE:
        col = ColourValue(val[2], val[1], val[0]) / 255; // BGR as uchar
        col.saturate();
        rpass->setEmissive(col);
        break;
    default:
        CV_Error(Error::StsBadArg, "invalid or non Scalar property");
        break;
    }
}

void setMaterialProperty(const String& name, int prop, const String& value)
{
    CV_Assert(prop == MATERIAL_TEXTURE, _app);

    MaterialPtr mat = MaterialManager::getSingleton().getByName(name, RESOURCEGROUP_NAME);
    CV_Assert(mat);

    Pass* rpass = mat->getTechniques()[0]->getPasses()[0];

    if (rpass->getTextureUnitStates().empty())
    {
        rpass->createTextureUnitState(value);
        return;
    }

    rpass->getTextureUnitStates()[0]->setTextureName(value);
}
}
}
