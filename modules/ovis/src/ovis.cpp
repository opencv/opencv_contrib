// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include <OgreApplicationContext.h>
#include <OgreCameraMan.h>
#include <OgreRectangle2D.h>
#include <OgreCompositorManager.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/utils/configuration.private.hpp>

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
static Quaternion toOGRE(Degree(180), Vector3::UNIT_X);
static Vector2 toOGRE_SS = Vector2(1, -1);

WindowScene::~WindowScene() {}

void _createTexture(const String& name, Mat image)
{
    PixelFormat format;
    switch(image.type())
    {
    case CV_8UC4:
        format = PF_BYTE_BGRA;
        break;
    case CV_8UC3:
        format = PF_BYTE_BGR;
        break;
    case CV_8UC1:
        format = PF_BYTE_L;
        break;
    case CV_16UC1:
        format = PF_L16;
        break;
    case CV_32FC1:
        format = PF_FLOAT32_R;
        break;
    default:
        CV_Error(Error::StsBadArg, "currently supported formats are only CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_32FC1");
        break;
    }

    TextureManager& texMgr = TextureManager::getSingleton();
    TexturePtr tex = texMgr.getByName(name, RESOURCEGROUP_NAME);

    Image im;
    im.loadDynamicImage(image.ptr(), image.cols, image.rows, 1, format);

    if (tex)
    {
        // update
        PixelBox box = im.getPixelBox();
        tex->getBuffer()->blitFromMemory(box, box);
        return;
    }

    texMgr.loadImage(name, RESOURCEGROUP_NAME, im);
}

static void _convertRT(InputArray rot, InputArray tvec, Quaternion& q, Vector3& t, bool invert = false)
{
    CV_Assert_N(rot.empty() || rot.rows() == 3 || rot.size() == Size(3, 3),
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
        q = Quaternion(R);

        if (invert)
        {
            q = q.Inverse();
        }
    }

    if (!tvec.empty())
    {
        tvec.copyTo(Mat_<Real>(3, 1, t.ptr()));

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

    float zNear = cam->getNearClipDistance();
    float top = zNear * K(1, 2) / K(1, 1);
    float left = -zNear * K(0, 2) / K(0, 0);
    float right = zNear * (imsize.width - K(0, 2)) / K(0, 0);
    float bottom = -zNear * (imsize.height - K(1, 2)) / K(1, 1);

    // use frustum extents instead of setFrustumOffset as the latter
    // assumes centered FOV, which is not the case
    cam->setFrustumExtents(left, right, top, bottom);

    // top and bottom parts of the FOV
    float fovy = atan2(K(1, 2), K(1, 1)) + atan2(imsize.height - K(1, 2), K(1, 1));
    cam->setFOVy(Radian(fovy));
}

static SceneNode& _getSceneNode(SceneManager* sceneMgr, const String& name)
{
    MovableObject* mo = NULL;

    try
    {
        mo = sceneMgr->getMovableObject(name, "Camera");

        // with cameras we have an extra CS flip node
        if(mo)
            return *mo->getParentSceneNode()->getParentSceneNode();
    }
    catch (const ItemIdentityException&)
    {
        // ignore
    }

    try
    {
        if (!mo)
            mo = sceneMgr->getMovableObject(name, "Light");
    }
    catch (const ItemIdentityException&)
    {
        // ignore
    }

    if (!mo)
        mo = sceneMgr->getMovableObject(name, "Entity"); // throws if not found

    return *mo->getParentSceneNode();
}

struct Application : public OgreBites::ApplicationContext, public OgreBites::InputListener
{
    Ptr<LogManager> logMgr;
    Ogre::SceneManager* sceneMgr;
    Ogre::String title;
    uint32_t w;
    uint32_t h;
    int key_pressed;
    int flags;

    Application(const Ogre::String& _title, const Size& sz, int _flags)
        : OgreBites::ApplicationContext("ovis", false), sceneMgr(NULL), title(_title), w(sz.width),
          h(sz.height), key_pressed(-1), flags(_flags)
    {
        if(utils::getConfigurationParameterBool("OPENCV_OVIS_VERBOSE_LOG", false))
            return;

        // set default log with low log level
        logMgr.reset(new LogManager());
        logMgr->createLog("ovis.log", true, true, true);
        logMgr->setLogDetail(LL_LOW);
    }

    void setupInput(bool /*grab*/)
    {
        // empty impl to show cursor
    }

    bool keyPressed(const OgreBites::KeyboardEvent& evt) CV_OVERRIDE
    {
        key_pressed = evt.keysym.sym;
        return true;
    }

    bool oneTimeConfig() CV_OVERRIDE
    {
        Ogre::String rsname = utils::getConfigurationParameterString("OPENCV_OVIS_RENDERSYSTEM", RENDERSYSTEM_NAME);
        Ogre::RenderSystem* rs = getRoot()->getRenderSystemByName(rsname);
        CV_Assert(rs && "Could not find rendersystem");
        getRoot()->setRenderSystem(rs);
        return true;
    }

    OgreBites::NativeWindowPair createWindow(const Ogre::String& name, uint32_t _w, uint32_t _h,
                                             NameValuePairList miscParams = NameValuePairList()) CV_OVERRIDE
    {
        Ogre::String _name = name;
        if (!sceneMgr)
        {
            _w = w;
            _h = h;
            _name = title;
        }

        if (flags & SCENE_AA)
            miscParams["FSAA"] = "4";

        miscParams["vsync"] = "true";

        OgreBites::NativeWindowPair ret =
            OgreBites::ApplicationContext::createWindow(_name, _w, _h, miscParams);
        addInputListener(ret.native, this); // handle input for all windows

        return ret;
    }

    size_t numWindows() const { return mWindows.size(); }

    void locateResources() CV_OVERRIDE
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

    void setup() CV_OVERRIDE
    {
        OgreBites::ApplicationContext::setup();

        MaterialManager& matMgr = MaterialManager::getSingleton();
        matMgr.setDefaultTextureFiltering(TFO_ANISOTROPIC);
        matMgr.setDefaultAnisotropy(16);
    }
};

class WindowSceneImpl : public WindowScene
{
    String title;
    Root* root;
    SceneManager* sceneMgr;
    SceneNode* camNode;
    RenderWindow* rWin;
    Ptr<OgreBites::CameraMan> camman;
    Ptr<Rectangle2D> bgplane;

    Ogre::RenderTarget* depthRTT;
    int flags;
public:
    WindowSceneImpl(Ptr<Application> app, const String& _title, const Size& sz, int _flags)
        : title(_title), root(app->getRoot()), depthRTT(NULL), flags(_flags)
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
        camNode->setOrientation(toOGRE);
        camNode->attachObject(cam);

        if (flags & SCENE_INTERACTIVE)
        {
            camman.reset(new OgreBites::CameraMan(camNode));
            camman->setStyle(OgreBites::CS_ORBIT);
            camNode->setFixedYawAxis(true, Vector3::NEGATIVE_UNIT_Y);
        }

        if (!app->sceneMgr)
        {
            app->sceneMgr = sceneMgr;
            rWin = app->getRenderWindow();
            if (camman)
                app->addInputListener(camman.get());
        }
        else
        {
            OgreBites::NativeWindowPair nwin = app->createWindow(title, sz.width, sz.height);
            rWin = nwin.render;
            if (camman)
                app->addInputListener(nwin.native, camman.get());
        }

        rWin->addViewport(cam);
    }

    ~WindowSceneImpl()
    {
        if (flags & SCENE_SEPERATE)
        {
            MaterialManager::getSingleton().remove(bgplane->getMaterial());
            bgplane.release();
            String texName = sceneMgr->getName() + "_Background";
            TextureManager::getSingleton().remove(texName, RESOURCEGROUP_NAME);
        }

        if(_app->sceneMgr == sceneMgr && (flags & SCENE_SEPERATE))
        {
            // this is the root window owning the context
            CV_Assert(_app->numWindows() == 1 && "the first OVIS window must be deleted last");
            _app->closeApp();
            _app.release();
        }
    }

    void setBackground(InputArray image) CV_OVERRIDE
    {
        CV_Assert(bgplane);

        String name = sceneMgr->getName() + "_Background";

        _createTexture(name, image.getMat());

        // correct for pixel centers
        Vector2 pc(0.5 / image.cols(), 0.5 / image.rows());
        bgplane->setUVs(pc, Vector2(pc[0], 1 - pc[1]), Vector2(1 - pc[0], pc[1]), Vector2(1, 1) - pc);

        Pass* rpass = bgplane->getMaterial()->getBestTechnique()->getPasses()[0];
        rpass->getTextureUnitStates()[0]->setTextureName(name);

        // ensure bgplane is visible
        bgplane->setVisible(true);
    }

    void setCompositors(const std::vector<String>& names) CV_OVERRIDE
    {
        CompositorManager& cm = CompositorManager::getSingleton();
        // this should be applied to all owned render targets
        Ogre::RenderTarget* targets[] = {rWin, depthRTT};

        for(int j = 0; j < 2; j++)
        {
            Ogre::RenderTarget* tgt = targets[j];
            if(!tgt) continue;

            Viewport* vp = tgt->getViewport(0);
            cm.removeCompositorChain(vp); // remove previous configuration

            for(size_t i = 0; i < names.size(); i++)
            {
                if (!cm.addCompositor(vp, names[i])) {
                    LogManager::getSingleton().logError("Failed to add compositor: " + names[i]);
                    continue;
                }
                cm.setCompositorEnabled(vp, names[i], true);
            }
        }
    }

    void getCompositorTexture(const String& compname, const String& texname, OutputArray out,
                              int mrtIndex) CV_OVERRIDE
    {
        CompositorManager& cm = CompositorManager::getSingleton();
        CompositorChain* chain = cm.getCompositorChain(rWin->getViewport(0));
        CV_Assert(chain && "no active compositors");

        CompositorInstance* inst = chain->getCompositor(compname);
        if(!inst)
            CV_Error_(Error::StsBadArg, ("no active compositor named: %s", compname.c_str()));

        TexturePtr tex = inst->getTextureInstance(texname, mrtIndex);
        if(!tex)
            CV_Error_(Error::StsBadArg, ("no texture named: %s", texname.c_str()));

        PixelFormat src_type = tex->getFormat();
        int dst_type;
        switch(src_type)
        {
        case PF_R8:
        case PF_L8:
            dst_type = CV_8U;
            break;
        case PF_BYTE_RGB:
            dst_type = CV_8UC3;
            break;
        case PF_BYTE_RGBA:
            dst_type = CV_8UC4;
            break;
        case PF_FLOAT32_R:
            dst_type = CV_32F;
            break;
        case PF_FLOAT32_RGB:
            dst_type = CV_32FC3;
            break;
        case PF_FLOAT32_RGBA:
            dst_type = CV_32FC4;
            break;
        case PF_L16:
        case PF_DEPTH16:
            dst_type = CV_16U;
            break;
        default:
            CV_Error(Error::StsNotImplemented, "unsupported texture format");
        }

        out.create(tex->getHeight(), tex->getWidth(), dst_type);

        Mat mat = out.getMat();
        PixelBox pb(tex->getWidth(), tex->getHeight(), 1, src_type, mat.ptr());
        tex->getBuffer()->blitToMemory(pb, pb);

        if(CV_MAT_CN(dst_type) < 3)
            return;

        // convert to OpenCV channel order
        cvtColor(mat, mat, CV_MAT_CN(dst_type) == 3 ? COLOR_RGB2BGR : COLOR_RGBA2BGRA);
    }

    void setBackground(const Scalar& color) CV_OVERRIDE
    {
        // hide background plane
        bgplane->setVisible(false);

        // BGRA as uchar
        ColourValue _color = ColourValue(color[2], color[1], color[0], color[3]) / 255;
        rWin->getViewport(0)->setBackgroundColour(_color);
    }

    void createEntity(const String& name, const String& meshname, InputArray tvec, InputArray rot) CV_OVERRIDE
    {
        Entity* ent = sceneMgr->createEntity(name, meshname, RESOURCEGROUP_NAME);

        Quaternion q;
        Vector3 t;
        _convertRT(rot, tvec, q, t);
        SceneNode* node = sceneMgr->getRootSceneNode()->createChildSceneNode(t, q);
        node->attachObject(ent);
    }

    void removeEntity(const String& name) CV_OVERRIDE {
        SceneNode& node = _getSceneNode(sceneMgr, name);
        node.getAttachedObject(name)->detachFromParent();

        // only one of the following will do something
        sceneMgr->destroyLight(name);
        sceneMgr->destroyEntity(name);
        sceneMgr->destroyCamera(name);

        sceneMgr->destroySceneNode(&node);
    }

    Rect2d createCameraEntity(const String& name, InputArray K, const Size& imsize, float zFar,
                              InputArray tvec, InputArray rot) CV_OVERRIDE
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
        node = node->createChildSceneNode();
        node->setOrientation(toOGRE); // camera mesh is oriented by OGRE conventions by default
        node->attachObject(cam);

        RealRect ext = cam->getFrustumExtents();
        float scale = zFar / cam->getNearClipDistance(); // convert to ext at zFar

        return Rect2d(toOGRE_SS[0] * (ext.right - ext.width() / 2) * scale,
                      toOGRE_SS[1] * (ext.bottom - ext.height() / 2) * scale, ext.width() * scale,
                      ext.height() * scale);
    }

    void createLightEntity(const String& name, InputArray tvec, InputArray rot, const Scalar& diffuseColour,
                           const Scalar& specularColour) CV_OVERRIDE
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

    void updateEntityPose(const String& name, InputArray tvec, InputArray rot) CV_OVERRIDE
    {
        SceneNode& node = _getSceneNode(sceneMgr, name);
        Quaternion q;
        Vector3 t;
        _convertRT(rot, tvec, q, t);
        node.rotate(q, Ogre::Node::TS_LOCAL);
        node.translate(t, Ogre::Node::TS_LOCAL);
    }

    void setEntityPose(const String& name, InputArray tvec, InputArray rot, bool invert) CV_OVERRIDE
    {
        SceneNode& node = _getSceneNode(sceneMgr, name);
        Quaternion q;
        Vector3 t;
        _convertRT(rot, tvec, q, t, invert);
        node.setOrientation(q);
        node.setPosition(t);
    }

    void setEntityProperty(const String& name, int prop, const String& value) CV_OVERRIDE
    {
        CV_Assert(prop == ENTITY_MATERIAL);
        SceneNode& node = _getSceneNode(sceneMgr, name);

        MaterialPtr mat = MaterialManager::getSingleton().getByName(value, RESOURCEGROUP_NAME);
        CV_Assert(mat && "material not found");

        Camera* cam = dynamic_cast<Camera*>(node.getAttachedObject(name));
        if(cam)
        {
            cam->setMaterial(mat);
            return;
        }

        Entity* ent = dynamic_cast<Entity*>(node.getAttachedObject(name));
        CV_Assert(ent && "invalid entity");
        ent->setMaterial(mat);
    }

    void setEntityProperty(const String& name, int prop, const Scalar& value) CV_OVERRIDE
    {
        CV_Assert(prop == ENTITY_SCALE);
        SceneNode& node = _getSceneNode(sceneMgr, name);
        node.setScale(value[0], value[1], value[2]);
    }

    void getEntityProperty(const String& name, int prop, OutputArray value)
    {
        SceneNode& node = _getSceneNode(sceneMgr, name);
        switch(prop)
        {
        case ENTITY_SCALE:
        {
            Vector3 s = node.getScale();
            Mat_<Real>(1, 3, s.ptr()).copyTo(value);
            return;
        }
        case ENTITY_AABB_WORLD:
        {
            Entity* ent = dynamic_cast<Entity*>(node.getAttachedObject(name));
            CV_Assert(ent && "invalid entity");
            AxisAlignedBox aabb = ent->getWorldBoundingBox(true);
            Vector3 mn = aabb.getMinimum();
            Vector3 mx = aabb.getMaximum();
            Mat_<Real> ret(2, 3);
            Mat_<Real>(1, 3, mn.ptr()).copyTo(ret.row(0));
            Mat_<Real>(1, 3, mx.ptr()).copyTo(ret.row(1));
            ret.copyTo(value);
            return;
        }
        default:
            CV_Error(Error::StsBadArg, "unsupported property");
        }
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

    void getScreenshot(OutputArray frame) CV_OVERRIDE
    {
        frame.create(rWin->getHeight(), rWin->getWidth(), CV_8UC3);

        Mat out = frame.getMat();
        PixelBox pb(rWin->getWidth(), rWin->getHeight(), 1, PF_BYTE_RGB, out.ptr());
        rWin->copyContentsToMemory(pb, pb);

        // convert to OpenCV channel order
        cvtColor(out, out, COLOR_RGB2BGR);
    }

    void getDepth(OutputArray depth) CV_OVERRIDE
    {
        Camera* cam = sceneMgr->getCamera(title);
        if (!depthRTT)
        {
            // render into an offscreen texture
            // currently this draws everything twice as OGRE lacks depth texture attachments
            TexturePtr tex = TextureManager::getSingleton().createManual(
                title + "_Depth", RESOURCEGROUP_NAME, TEX_TYPE_2D, rWin->getWidth(),
                rWin->getHeight(), 0, PF_DEPTH, TU_RENDERTARGET);
            depthRTT = tex->getBuffer()->getRenderTarget();
            depthRTT->addViewport(cam);
            depthRTT->setAutoUpdated(false); // only update when requested
        }

        Mat tmp(depthRTT->getHeight(), depthRTT->getWidth(), CV_16U);
        PixelBox pb(depthRTT->getWidth(), depthRTT->getHeight(), 1, PF_DEPTH, tmp.ptr());
        depthRTT->update(false);
        depthRTT->copyContentsToMemory(pb, pb);

        // convert to NDC
        double alpha = 2.0/std::numeric_limits<uint16>::max();
        tmp.convertTo(depth, CV_64F, alpha, -1);

        // convert to linear
        float n = cam->getNearClipDistance();
        float f = cam->getFarClipDistance();
        Mat ndc = depth.getMat();
        ndc = -ndc * (f - n) + (f + n);
        ndc = (2 * f * n) / ndc;
    }

    void fixCameraYawAxis(bool useFixed, InputArray _up) CV_OVERRIDE
    {
        Vector3 up = Vector3::NEGATIVE_UNIT_Y;
        if (!_up.empty())
        {
            _up.copyTo(Mat_<Real>(3, 1, up.ptr()));
        }

        camNode->setFixedYawAxis(useFixed, up);
    }

    void setCameraPose(InputArray tvec, InputArray rot, bool invert) CV_OVERRIDE
    {
        Quaternion q;
        Vector3 t;
        _convertRT(rot, tvec, q, t, invert);

        if (!rot.empty())
            camNode->setOrientation(q*toOGRE);

        if (!tvec.empty())
            camNode->setPosition(t);
    }

    void getCameraPose(OutputArray R, OutputArray tvec, bool invert) CV_OVERRIDE
    {
        Matrix3 _R;
        // toOGRE.Inverse() == toOGRE
        (camNode->getOrientation()*toOGRE).ToRotationMatrix(_R);

        if (invert)
        {
            _R = _R.Transpose();
        }

        if (tvec.needed())
        {
            Vector3 _tvec = camNode->getPosition();

            if (invert)
            {
                _tvec = _R * -_tvec;
            }

            Mat_<Real>(3, 1, _tvec.ptr()).copyTo(tvec);
        }

        if (R.needed())
        {
            Mat_<Real>(3, 3, _R[0]).copyTo(R);
        }
    }

    void setCameraIntrinsics(InputArray K, const Size& imsize, float zNear, float zFar) CV_OVERRIDE
    {
        Camera* cam = sceneMgr->getCamera(title);

        if(zNear >= 0) cam->setNearClipDistance(zNear);
        if(zFar >= 0) cam->setFarClipDistance(zFar);
        if(!K.empty()) _setCameraIntrinsics(cam, K, imsize);
    }

    void setCameraLookAt(const String& target, InputArray offset) CV_OVERRIDE
    {
        SceneNode* tgt = sceneMgr->getEntity(target)->getParentSceneNode();

        Vector3 _offset = Vector3::ZERO;

        if (!offset.empty())
        {
            offset.copyTo(Mat_<Real>(3, 1, _offset.ptr()));
        }

        camNode->lookAt(tgt->_getDerivedPosition() + _offset, Ogre::Node::TS_WORLD);
    }
};

CV_EXPORTS_W void addResourceLocation(const String& path) { _extraResourceLocations.push_back(path); }

Ptr<WindowScene> createWindow(const String& title, const Size& size, int flags)
{
    if (!_app)
    {
        _app = makePtr<Application>(title.c_str(), size, flags);
        _app->initApp();
    }

    return makePtr<WindowSceneImpl>(_app, title, size, flags);
}

CV_EXPORTS_W int waitKey(int delay)
{
    CV_Assert(_app);

    _app->key_pressed = -1;
    _app->getRoot()->renderOneFrame();

    // wait for keypress, using vsync instead of sleep
    while(!delay && _app->key_pressed == -1)
        _app->getRoot()->renderOneFrame();

    return (_app->key_pressed != -1) ? (_app->key_pressed & 0xff) : -1;
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
    case MATERIAL_LINE_WIDTH:
#if OGRE_VERSION >= ((1 << 16) | (11 << 8) | 2)
        rpass->setLineWidth(val[0]);
#else
        CV_Error(Error::StsError, "needs OGRE 1.11.2+ for this");
#endif
        break;
    default:
        CV_Error(Error::StsBadArg, "invalid or non Scalar property");
        break;
    }
}

void setMaterialProperty(const String& name, int prop, const String& value)
{
    CV_Assert_N(prop >= MATERIAL_TEXTURE0, prop <= MATERIAL_TEXTURE3, _app);

    MaterialPtr mat = MaterialManager::getSingleton().getByName(name, RESOURCEGROUP_NAME);
    CV_Assert(mat);

    Pass* rpass = mat->getTechniques()[0]->getPasses()[0];

    size_t texUnit = prop - MATERIAL_TEXTURE0;
    CV_Assert(texUnit <= rpass->getTextureUnitStates().size());

    if (rpass->getTextureUnitStates().size() <= texUnit)
    {
        rpass->createTextureUnitState(value);
        return;
    }

    rpass->getTextureUnitStates()[texUnit]->setTextureName(value);
}

static bool setShaderProperty(const GpuProgramParametersSharedPtr& params, const String& prop,
                              const Scalar& value)
{
    const GpuConstantDefinition* def = params->_findNamedConstantDefinition(prop, false);

    if(!def)
        return false;

    Vec4f valf = value;

    switch(def->constType)
    {
    case GCT_FLOAT1:
        params->setNamedConstant(prop, valf[0]);
        return true;
    case GCT_FLOAT2:
        params->setNamedConstant(prop, Vector2(valf.val));
        return true;
    case GCT_FLOAT3:
        params->setNamedConstant(prop, Vector3(valf.val));
        return true;
    case GCT_FLOAT4:
        params->setNamedConstant(prop, Vector4(valf.val));
        return true;
    default:
        CV_Error(Error::StsBadArg, "currently only float[1-4] uniforms are supported");
        return false;
    }
}

void setMaterialProperty(const String& name, const String& prop, const Scalar& value)
{
    CV_Assert(_app);

    MaterialPtr mat = MaterialManager::getSingleton().getByName(name, RESOURCEGROUP_NAME);
    CV_Assert(mat);

    Pass* rpass = mat->getTechniques()[0]->getPasses()[0];
    bool set = false;
    if(rpass->hasGpuProgram(GPT_VERTEX_PROGRAM))
    {
        GpuProgramParametersSharedPtr params = rpass->getVertexProgramParameters();
        set = setShaderProperty(params, prop, value);
    }

    if(rpass->hasGpuProgram(GPT_FRAGMENT_PROGRAM))
    {
        GpuProgramParametersSharedPtr params = rpass->getFragmentProgramParameters();
        set = set || setShaderProperty(params, prop, value);
    }

    if(!set)
        CV_Error_(Error::StsBadArg, ("shader parameter named '%s' not found", prop.c_str()));
}

void updateTexture(const String& name, InputArray image)
{
    CV_Assert(_app);
    TexturePtr tex = TextureManager::getSingleton().getByName(name, RESOURCEGROUP_NAME);
    CV_Assert(tex);
    _createTexture(name, image.getMat());
}
}
}
