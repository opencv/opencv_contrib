// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#ifndef __OPENCV_KINFU_IMPL_HPP__
#define __OPENCV_KINFU_IMPL_HPP__

#include "precomp.hpp"
#include "kinfu_functions.hpp"

namespace cv {

class KinFu::Impl
{
public:
    Impl();
    virtual ~Impl() {};
    virtual OdometryFrame createOdometryFrame() const = 0;
    virtual bool update(InputArray depth) = 0;
    virtual void render(OutputArray image) const = 0;
    virtual void render(OutputArray image, const Matx44f& cameraPose) const = 0;
    virtual void reset() = 0;
    virtual void getCloud(OutputArray points, OutputArray normals) const = 0;
    virtual void getPoints(OutputArray points) const = 0;
    virtual void getNormals(InputArray points, OutputArray normals) const = 0;
    virtual const Affine3f getPose() const = 0;

public:
    VolumeSettings volumeSettings;
    OdometrySettings odometrySettings;
    Volume volume;
    Odometry odometry;
};

class KinFu_Common : public KinFu::Impl
{
public:
    KinFu_Common();
    ~KinFu_Common();
    virtual OdometryFrame createOdometryFrame() const override;
    virtual bool update(InputArray depth) override;
    virtual void render(OutputArray image) const override;
    virtual void render(OutputArray image, const Matx44f& cameraPose) const override;
    virtual void reset() override;
    virtual void getCloud(OutputArray points, OutputArray normals) const override;
    virtual void getPoints(OutputArray points) const override;
    virtual void getNormals(InputArray points, OutputArray normals) const override;
    virtual const Affine3f getPose() const override;
private:
    Vec3f lightPose = Vec3f::all(0.f);
    int frameCounter;
    Matx44f pose;
    OdometryFrame prevFrame;
    OdometryFrame renderFrame;
};


KinFu::KinFu()
{
    this->impl = makePtr<KinFu_Common>();
}

KinFu::~KinFu(){}

bool KinFu::update(InputArray depth) { return this->impl->update(depth); }
void KinFu::render(OutputArray image) const { this->impl->render(image); }
void KinFu::render(OutputArray image, const Matx44f& cameraPose) const { this->impl->render(image, cameraPose); }
void KinFu::reset() { this->impl->reset(); }
void KinFu::getCloud(OutputArray points, OutputArray normals) const { this->impl->getCloud(points, normals); }
void KinFu::getPoints(OutputArray points) const { this->impl->getPoints(points); }
void KinFu::getNormals(InputArray points, OutputArray normals) const { this->impl->getNormals(points, normals); }
const Affine3f KinFu::getPose() const { return this->impl->getPose(); }

} // namespace cv

#endif
