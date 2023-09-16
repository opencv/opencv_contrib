// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "opencv2/v4d/v4d.hpp"
#include "opencv2/v4d/detail/imguicontext.hpp"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>
#define GL_SILENCE_DEPRECATION
#if defined(OPENCV_V4D_USE_ES3) || defined(EMSCRIPTEN)
#include <GLES3/gl3.h>
#endif
#include <GLFW/glfw3.h>

namespace cv {
namespace v4d {
namespace detail {
ImGuiContext::ImGuiContext(FrameBufferContext& fbContext) :
        mainFbContext_(fbContext), glFbContext_(*fbContext.getV4D(), "ImGUI", fbContext) {
    V4D_INIT_PRIVATE(mainFbContext_.getV4D(), false);
#ifdef __EMSCRIPTEN__
    run_sync_on_main<24>([&,this](){
        mainFbContext_.initWebGLCopy(fbCtx().getIndex());
    });
#endif
}


void ImGuiContext::build(std::function<void(const cv::Size&)> fn) {
    renderCallback_ = fn;
}

bool show_demo_window = true;
void ImGuiContext::render() {
    run_sync_on_main<25>([&,this](){
#ifndef __EMSCRIPTEN__
        if(!fbCtx().isShared()) {
            UMat tmp;
            mainFbContext_.copyTo(tmp);
            fbCtx().copyFrom(tmp);
        }
#endif
        {
            fbCtx().getV4D()->makeCurrent();
            GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, 0));
            GL_CHECK(glDrawBuffer(GL_BACK));
            GL_CHECK(glViewport(0, 0, fbCtx().getV4D()->size().width, fbCtx().getV4D()->size().height));
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            ImGui::Begin("Display"); \
            ImGuiIO& io = ImGui::GetIO(); \
            ImGui::Text("%.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate); \
            ImGui::End();
            if(renderCallback_)
                renderCallback_(fbCtx().getV4D()->size());
            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        }
        if(!fbCtx().isShared()) {
#ifdef __EMSCRIPTEN__
            mainFbContext_.doWebGLCopy(fbCtx());
#else
            UMat tmp;
            fbCtx().copyTo(tmp);
            mainFbContext_.copyFrom(tmp);
#endif
        }
        GL_CHECK(glFinish());
    });
}

FrameBufferContext& ImGuiContext::fbCtx() {
    return glFbContext_;
}

}
}
}
