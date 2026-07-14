// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include <opencv2/v4d/v4d.hpp>

using namespace cv::v4d;

// based on: https://github.com/bkaradzic/bgfx/blob/07be0f213acd73a4f6845dc8f7b20b93f66b7cc4/examples/01-cubes/cubes.cpp
class BgfxDemoPlan : public Plan {
	struct PosColorVertex
	{
		float x_;
		float y_;
		float z_;
		uint32_t abgr_;

		static void init()
		{
			layout
				.begin()
				.add(bgfx::Attrib::Position, 3, bgfx::AttribType::Float)
				.add(bgfx::Attrib::Color0,   4, bgfx::AttribType::Uint8, true)
				.end();
		};

		inline static bgfx::VertexLayout layout;
	};

	inline static const PosColorVertex CUBE_VERTICES[] =
	{
		{-0.30f,  0.30f,  0.30f, 0xaa000000 },
		{ 0.30f,  0.30f,  0.30f, 0xaa0000ff },
		{-0.30f, -0.30f,  0.30f, 0xaa00ff00 },
		{ 0.30f, -0.30f,  0.30f, 0xaa00ffff },
		{-0.30f,  0.30f, -0.30f, 0xaaff0000 },
		{ 0.30f,  0.30f, -0.30f, 0xaaff00ff },
		{-0.30f, -0.30f, -0.30f, 0xaaffff00 },
		{ 0.30f, -0.30f, -0.30f, 0xaaffffff },
	};

	inline static const uint16_t CUBE_TRI_LIST[] =
	{
		0, 1, 2, // 0
		1, 3, 2,
		4, 6, 5, // 2
		5, 6, 7,
		0, 2, 4, // 4
		4, 2, 6,
		1, 5, 3, // 6
		5, 7, 3,
		0, 4, 1, // 8
		4, 5, 1,
		2, 3, 6, // 10
		6, 3, 7,
	};

	inline static const uint16_t CUBE_TRI_STRIP[] =
	{
		0, 1, 2,
		3,
		7,
		1,
		5,
		0,
		4,
		2,
		6,
		7,
		4,
		5,
	};

	inline static const uint16_t CUBE_LINE_LIST[] =
	{
		0, 1,
		0, 2,
		0, 4,
		1, 3,
		1, 5,
		2, 3,
		2, 6,
		3, 7,
		4, 5,
		4, 6,
		5, 7,
		6, 7,
	};

	inline static const uint16_t CUBE_LINE_STRIP[] =
	{
		0, 2, 3, 1, 5, 7, 6, 4,
		0, 2, 6, 4, 5, 7, 3, 1,
		0,
	};

	inline static const uint16_t CUBE_POINTS[] =
	{
		0, 1, 2, 3, 4, 5, 6, 7
	};

	inline static const char* PT_NAMES[]
	{
		"Triangle List",
		"Triangle Strip",
		"Lines",
		"Line Strip",
		"Points",
	};

	inline static const uint64_t PT_STATE[]
	{
		UINT64_C(0),
		BGFX_STATE_PT_TRISTRIP,
		BGFX_STATE_PT_LINES,
		BGFX_STATE_PT_LINESTRIP,
		BGFX_STATE_PT_POINTS,
	};

	struct Params {
		uint32_t width_;
		uint32_t height_;
		bgfx::VertexBufferHandle vbh_;
		bgfx::IndexBufferHandle ibh_[BX_COUNTOF(PT_STATE)];
		bgfx::ProgramHandle program_;
		int32_t pt_ = 0;

		bool red_ = true;
		bool green_ = true;
		bool blue_ = true;
		bool alpha_ = true;
	} params_;

	inline static int64_t time_offset_;

	Property<cv::Rect> vp_ = P<cv::Rect>(V4D::Keys::VIEWPORT);
public:
	BgfxDemoPlan(){

	}
	void setup() override {
		branch(BranchType::ONCE, always_)
				->plain([](int64_t& timeOffset) {
				timeOffset = bx::getHPCounter();
			}, RWS(time_offset_))
		->endBranch();

		bgfx([](const cv::Rect& vp, Params& params){
			params.width_ = vp.width;
			params.height_ = vp.height;
			// Set view 0 clear state.
			bgfx::setViewClear(0
				, BGFX_CLEAR_COLOR|BGFX_CLEAR_DEPTH
				, 0x00000000
				, 1.0f
				, 0
				);
			PosColorVertex::init();

			// Set view 0 default viewport.
			bgfx::setViewRect(0, vp.x, vp.y, uint16_t(vp.width), uint16_t(vp.height));

			// Create static vertex buffer.
			params.vbh_ = bgfx::createVertexBuffer(
				// Static data can be passed with bgfx::makeRef
				  bgfx::makeRef(CUBE_VERTICES, sizeof(CUBE_VERTICES) )
				, PosColorVertex::layout
				);

			// Create static index buffer for triangle list rendering.
			params.ibh_[0] = bgfx::createIndexBuffer(
				// Static data can be passed with bgfx::makeRef
				bgfx::makeRef(CUBE_TRI_LIST, sizeof(CUBE_TRI_LIST) )
				);

			// Create static index buffer for triangle strip rendering.
			params.ibh_[1] = bgfx::createIndexBuffer(
				// Static data can be passed with bgfx::makeRef
				bgfx::makeRef(CUBE_TRI_STRIP, sizeof(CUBE_TRI_STRIP) )
				);

			// Create static index buffer for line list rendering.
			params.ibh_[2] = bgfx::createIndexBuffer(
				// Static data can be passed with bgfx::makeRef
				bgfx::makeRef(CUBE_LINE_LIST, sizeof(CUBE_LINE_LIST) )
				);

			// Create static index buffer for line strip rendering.
			params.ibh_[3] = bgfx::createIndexBuffer(
				// Static data can be passed with bgfx::makeRef
				bgfx::makeRef(CUBE_LINE_STRIP, sizeof(CUBE_LINE_STRIP) )
				);

			// Create static index buffer for point list rendering.
			params.ibh_[4] = bgfx::createIndexBuffer(
				// Static data can be passed with bgfx::makeRef
				bgfx::makeRef(CUBE_POINTS, sizeof(CUBE_POINTS) )
				);

			// Create program from shaders.
			params.program_ = util::load_program("vs_cubes", "fs_cubes");

		}, vp_, RW(params_));
	}

	void infer() override {
		bgfx([](const Params& params, const int64_t timeOffset) {
			float time = (float)( (bx::getHPCounter()-timeOffset)/double(bx::getHPFrequency()));

			const bx::Vec3 at  = { 0.0f, 0.0f,   0.0f };
			const bx::Vec3 eye = { 0.0f, 0.0f, -35.0f };

			// Set view and projection matrix for view 0.
			{

				float view[16];
				bx::mtxLookAt(view, eye, at);

				float proj[16];
				bx::mtxProj(proj, 60.0f, float(params.width_)/float(params.height_), 0.1f, 100.0f, bgfx::getCaps()->homogeneousDepth);

				bgfx::setViewTransform(0, view, proj);

				// Set view 0 default viewport.
				bgfx::setViewRect(0, 0, 0, uint16_t(params.width_), uint16_t(params.height_) );
			}

			// This dummy draw call is here to make sure that view 0 is cleared
			// if no other draw calls are submitted to view 0.
			bgfx::touch(0);

			bgfx::IndexBufferHandle ibh = params.ibh_[params.pt_];
			uint64_t state = 0
				| (params.red_ ? BGFX_STATE_WRITE_R : 0)
				| (params.green_ ? BGFX_STATE_WRITE_G : 0)
				| (params.blue_ ? BGFX_STATE_WRITE_B : 0)
				| (params.alpha_ ? BGFX_STATE_WRITE_A : 0)
				| BGFX_STATE_WRITE_Z
				| BGFX_STATE_DEPTH_TEST_LESS
				| BGFX_STATE_CULL_CW
				| BGFX_STATE_MSAA
				| PT_STATE[params.pt_]
				;


			// Submit 11x11 cubes.
			for (uint32_t yy = 0; yy < 100; ++yy)
			{
				for (uint32_t xx = 0; xx < 100; ++xx)
				{
					float mtx[16];
					float angle = fmod(float(time) + sin((float(xx * yy / pow(170.0f, 2.0f)) * 2.0f - 1.0f) * CV_PI), 2.0f * CV_PI);
					bx::mtxRotateXYZ(mtx, angle, angle, angle);
					mtx[12] = ((xx / 100.0) * 2.0 - 1.0) * 30.0;
					mtx[13] = ((yy / 100.0) * 2.0 - 1.0) * 30.0;
					mtx[14] = 0.0f;

					// Set model matrix for rendering.
					bgfx::setTransform(mtx);

					// Set vertex and index buffer.
					bgfx::setVertexBuffer(0, params.vbh_);
					bgfx::setIndexBuffer(ibh);

					// Set render states.
					bgfx::setState(state);

					// Submit primitive for rendering to view 0.
					bgfx::submit(0, params.program_);
				}
			}

			// Advance to next frame. Rendering thread will be kicked to
			// process submitted rendering primitives.
			bgfx::frame();
		}, R(params_), CS(time_offset_));
	}
};


int main(int argc, char** argv) {
	cv::Ptr<V4D> runtime = V4D::init(cv::Rect(0,0, 1280, 720), "Bgfx Demo", AllocateFlags::BGFX | AllocateFlags::IMGUI);
    Plan::run<BgfxDemoPlan>(std::stoi(argv[1]));

    return 0;
}
