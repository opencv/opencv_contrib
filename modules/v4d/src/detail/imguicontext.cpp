// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "opencv2/v4d/v4d.hpp"
#if defined(OPENCV_V4D_USE_ES3) || defined(EMSCRIPTEN)
#   define IMGUI_IMPL_OPENGL_ES3
#endif

#define IMGUI_IMPL_OPENGL_LOADER_CUSTOM

#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

namespace cv {
namespace v4d {
namespace detail {
ImGuiContextImpl::ImGuiContextImpl(cv::Ptr<FrameBufferContext> fbContext) :
        mainFbContext_(fbContext) {
	FrameBufferContext::GLScope glScope(mainFbContext_, GL_FRAMEBUFFER);
	IMGUI_CHECKVERSION();
	context_ = ImGui::CreateContext();
	ImGui::SetCurrentContext(context_);

	ImGuiIO& io = ImGui::GetIO();
	(void)io;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
	ImGui::StyleColorsDark();

	ImGui_ImplGlfw_InitForOpenGL(mainFbContext_->getGLFWWindow(), false);
	ImGui_ImplGlfw_SetCallbacksChainForAllWindows(true);
#if !defined(OPENCV_V4D_USE_ES3)
	ImGui_ImplOpenGL3_Init("#version 330");
#else
	ImGui_ImplOpenGL3_Init("#version 300 es");
#endif
}

void ImGuiContextImpl::build(std::function<void(ImGuiContext*)> fn) {
    renderCallback_ = fn;
}

void ImGuiContextImpl::makeCurrent() {
    ImGui::SetCurrentContext(context_);
}

void ImGuiContextImpl::render(bool showFPS) {
	mainFbContext_->makeCurrent();
	ImGui::SetCurrentContext(context_);

	GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, 0));
#if !defined(OPENCV_V4D_USE_ES3)
	GL_CHECK(glDrawBuffer(GL_BACK));
#endif
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
	if (showFPS) {
		static bool open_ptr[1] = { true };
		static ImGuiWindowFlags window_flags = 0;
//            window_flags |= ImGuiWindowFlags_NoBackground;
		window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus;
		window_flags |= ImGuiWindowFlags_NoMove;
		window_flags |= ImGuiWindowFlags_NoScrollWithMouse;
		window_flags |= ImGuiWindowFlags_AlwaysAutoResize;
		window_flags |= ImGuiWindowFlags_NoSavedSettings;
		window_flags |= ImGuiWindowFlags_NoFocusOnAppearing;
		window_flags |= ImGuiWindowFlags_NoNav;
		window_flags |= ImGuiWindowFlags_NoDecoration;
		window_flags |= ImGuiWindowFlags_NoInputs;
		static ImVec2 pos(0, 0);
		ImGui::SetNextWindowPos(pos, ImGuiCond_Once);
		ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0f, 0.0f, 0.0f, 0.5f));
		ImGui::Begin("Display", open_ptr, window_flags);
		ImGui::Text("%.3f ms/frame (%.1f FPS)", (1000.0f / Global::fps()) , Global::fps());
		ImGui::End();
		ImGui::PopStyleColor(1);
		std::stringstream ss;
		TimeTracker::getInstance()->print(ss);
		std::string line;
		ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0f, 0.0f, 0.0f, 0.5f));
		ImGui::Begin("Time Tracking");
		while(getline(ss, line)) {
			ImGui::Text("%s", line.c_str());
		}
		ImGui::End();
		ImGui::PopStyleColor(1);
	}
	if (renderCallback_)
		renderCallback_(context_);
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	mainFbContext_->makeNoneCurrent();
}
}
}
}
