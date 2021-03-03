// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

namespace cv
{
namespace ovis
{
using namespace Ogre;

void createPlaneMesh(const String& name, const Size2f& size, InputArray image)
{
    CV_Assert(_app);

    // material
    MaterialPtr mat = MaterialManager::getSingleton().create(name, RESOURCEGROUP_NAME);

    Pass* rpass = mat->getTechniques()[0]->getPasses()[0];
    rpass->setCullingMode(CULL_NONE);
    rpass->setEmissive(ColourValue::White);

    if (!image.empty())
    {
        _createTexture(name, image.getMat());
        rpass->createTextureUnitState(name);
    }

    // plane
    MovablePlane plane(-Vector3::UNIT_Z, 0);
    MeshPtr mesh = MeshManager::getSingleton().createPlane(
        name, RESOURCEGROUP_NAME, plane, size.width, size.height, 1, 1, true, 1, 1, 1, -Vector3::UNIT_Y);
    mesh->getSubMesh(0)->setMaterialName(name);
}

void createPointCloudMesh(const String& name, InputArray vertices, InputArray colors)
{
    int color_type = colors.type();
    CV_Assert(_app);
    CV_CheckTypeEQ(vertices.type(), CV_32FC3, "vertices type must be Vec3f");
    CV_Assert(vertices.isContinuous());
    if (!colors.empty())
        CV_CheckType(color_type, color_type == CV_8UC3 || color_type == CV_8UC4, "unsupported type");

    // material
    MaterialPtr mat = MaterialManager::getSingleton().create(name, RESOURCEGROUP_NAME);
    Pass* rpass = mat->getTechniques()[0]->getPasses()[0];
    rpass->setEmissive(ColourValue::White);
    rpass->setPointSpritesEnabled(true);

    // mesh
    MeshPtr mesh = MeshManager::getSingleton().createManual(name, RESOURCEGROUP_NAME);
    SubMesh* sub = mesh->createSubMesh();
    sub->useSharedVertices = true;
    sub->operationType = RenderOperation::OT_POINT_LIST;
    sub->setMaterialName(name);

    int n = vertices.rows();

    mesh->sharedVertexData = new VertexData();
    mesh->sharedVertexData->vertexCount = n;
    VertexDeclaration* decl = mesh->sharedVertexData->vertexDeclaration;

    // vertex data
    HardwareBufferManager& hbm = HardwareBufferManager::getSingleton();

    Mat _vertices = vertices.getMat();

    int source = 0;
    HardwareVertexBufferSharedPtr hwbuf;

    decl->addElement(source, 0, VET_FLOAT3, VES_POSITION);
    hwbuf = hbm.createVertexBuffer(decl->getVertexSize(source), n, HardwareBuffer::HBU_STATIC_WRITE_ONLY);
    hwbuf->writeData(0, hwbuf->getSizeInBytes(), _vertices.ptr(), true);
    mesh->sharedVertexData->vertexBufferBinding->setBinding(source, hwbuf);

    // color data
    if (!colors.empty())
    {
        mat->setLightingEnabled(false);
        source += 1;

        Mat col4;
        cvtColor(colors, col4, color_type == CV_8UC3 ? COLOR_BGR2RGBA : COLOR_BGRA2RGBA);

        decl->addElement(source, 0, VET_COLOUR, VES_DIFFUSE);
        hwbuf =
            hbm.createVertexBuffer(decl->getVertexSize(source), n, HardwareBuffer::HBU_STATIC_WRITE_ONLY);
        hwbuf->writeData(0, hwbuf->getSizeInBytes(), col4.ptr(), true);
        mesh->sharedVertexData->vertexBufferBinding->setBinding(source, hwbuf);

        rpass->setVertexColourTracking(TVC_DIFFUSE);
    }

    AxisAlignedBox bounds(AxisAlignedBox::EXTENT_NULL);
    for (int i = 0; i < n; i++)
    {
        Vec3f v = _vertices.at<Vec3f>(i);
        bounds.merge(Vector3(v[0], v[1], v[2]));
    }
    mesh->_setBounds(bounds);
}

void createTriangleMesh(const String& name, InputArray vertices, InputArray normals, InputArray indices)
{
    CV_CheckTypeEQ(vertices.type(), CV_32FC3, "vertices type must be Vec3f");
    CV_Assert(vertices.isContinuous());

    if(!normals.empty())
    {
        CV_CheckTypeEQ(normals.type(), CV_32FC3, "normals type must be Vec3f");
        CV_Assert(normals.isContinuous());
        CV_Assert(normals.size() == vertices.size());
    }
    if(!indices.empty())
    {
        CV_CheckTypeEQ(indices.type(), CV_32S, "indices type must be int");
        CV_Assert(indices.isContinuous());
    }

    // default material
    auto mat = MaterialManager::getSingleton().create(name, RESOURCEGROUP_NAME);

    // mesh
    MeshPtr mesh = MeshManager::getSingleton().createManual(name, RESOURCEGROUP_NAME);
    SubMesh* sub = mesh->createSubMesh();
    sub->useSharedVertices = true;
    sub->operationType = RenderOperation::OT_TRIANGLE_LIST;
    sub->setMaterialName(name);

    int n = vertices.rows();

    mesh->sharedVertexData = new VertexData();
    mesh->sharedVertexData->vertexCount = n;
    VertexDeclaration* decl = mesh->sharedVertexData->vertexDeclaration;

    // vertex data
    HardwareBufferManager& hbm = HardwareBufferManager::getSingleton();

    Mat _vertices = vertices.getMat();

    int source = 0;
    HardwareVertexBufferSharedPtr hwbuf;

    decl->addElement(source, 0, VET_FLOAT3, VES_POSITION);
    hwbuf = hbm.createVertexBuffer(decl->getVertexSize(source), n, HardwareBuffer::HBU_STATIC_WRITE_ONLY);
    hwbuf->writeData(0, hwbuf->getSizeInBytes(), _vertices.ptr(), true);
    mesh->sharedVertexData->vertexBufferBinding->setBinding(source, hwbuf);

    // normals
    if (!normals.empty())
    {
        source += 1;

        Mat _normals = normals.getMat();
        decl->addElement(source, 0, VET_FLOAT3, VES_NORMAL);
        hwbuf =
            hbm.createVertexBuffer(decl->getVertexSize(source), n, HardwareBuffer::HBU_STATIC_WRITE_ONLY);
        hwbuf->writeData(0, hwbuf->getSizeInBytes(), _normals.ptr(), true);
        mesh->sharedVertexData->vertexBufferBinding->setBinding(source, hwbuf);
    }
    else
    {
        mat->setLightingEnabled(false);
    }

    // indices
    if (!indices.empty())
    {
        Mat _indices = indices.getMat();

        HardwareIndexBufferSharedPtr ibuf = HardwareBufferManager::getSingleton().createIndexBuffer(
            HardwareIndexBuffer::IT_32BIT, indices.total(), HardwareBuffer::HBU_STATIC_WRITE_ONLY);
        ibuf->writeData(0, ibuf->getSizeInBytes(), _indices.ptr(), true);

        sub->indexData->indexBuffer = ibuf;
        sub->indexData->indexStart = 0;
        sub->indexData->indexCount = indices.total();
    }

    AxisAlignedBox bounds(AxisAlignedBox::EXTENT_NULL);
    for (int i = 0; i < n; i++)
    {
        Vec3f v = _vertices.at<Vec3f>(i);
        bounds.merge(Vector3(v[0], v[1], v[2]));
    }
    mesh->_setBounds(bounds);
}

void createGridMesh(const String& name, const Size2f& size, const Size& segments)
{
    CV_Assert_N(_app, !segments.empty());

    // material
    MaterialPtr mat = MaterialManager::getSingleton().create(name, RESOURCEGROUP_NAME);
    Pass* rpass = mat->getTechniques()[0]->getPasses()[0];
    rpass->setEmissive(ColourValue::White);

    // mesh
    MeshPtr mesh = MeshManager::getSingleton().createManual(name, RESOURCEGROUP_NAME);
    SubMesh* sub = mesh->createSubMesh();
    sub->useSharedVertices = true;
    sub->operationType = RenderOperation::OT_LINE_LIST;
    sub->setMaterialName(name);

    int n = (segments.width + 1) * 2 + (segments.height + 1) * 2;

    mesh->sharedVertexData = new VertexData();
    mesh->sharedVertexData->vertexCount = n;
    VertexDeclaration* decl = mesh->sharedVertexData->vertexDeclaration;

    // vertex data
    HardwareBufferManager& hbm = HardwareBufferManager::getSingleton();

    int source = 0;
    HardwareVertexBufferSharedPtr hwbuf;
    decl->addElement(source, 0, VET_FLOAT2, VES_POSITION);
    hwbuf = hbm.createVertexBuffer(decl->getVertexSize(source), n, HardwareBuffer::HBU_STATIC_WRITE_ONLY);
    mesh->sharedVertexData->vertexBufferBinding->setBinding(source, hwbuf);

    Vector2 step = Vector2(size.width, size.height) / Vector2(segments.width, segments.height);

    Vec2f* data = (Vec2f*)hwbuf->lock(HardwareBuffer::HBL_DISCARD);

    for (int i = 0; i < segments.width + 1; i++)
    {
        data[i * 2] = Vec2f(-size.width / 2 + step.x * i, -size.height / 2);
        data[i * 2 + 1] = Vec2f(-size.width / 2 + step.x * i, size.height / 2);
    }

    data += (segments.width + 1) * 2;

    for (int i = 0; i < segments.height + 1; i++)
    {
        data[i * 2] = Vec2f(-size.width / 2, -size.height / 2 + step.y * i);
        data[i * 2 + 1] = Vec2f(size.width / 2, -size.height / 2 + step.y * i);
    }

    hwbuf->unlock();

    Vector3 sz(size.width, size.height, 0);
    mesh->_setBounds(AxisAlignedBox(-sz/2, sz/2));
}
}
}
