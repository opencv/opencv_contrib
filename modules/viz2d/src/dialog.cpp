// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "opencv2/viz2d/dialog.hpp"
#include <nanogui/layout.h>
#ifdef __EMSCRIPTEN__
#define VIZ2D_USE_ES3 1
#endif
#ifndef VIZ2D_USE_ES3
#  include <GL/glew.h>
#  define GLFW_INCLUDE_GLCOREARB
#else
#  define GLFW_INCLUDE_ES3
#  define GLFW_INCLUDE_GLEXT
#endif
#include <GLFW/glfw3.h>

namespace cv {
namespace viz {

std::function<bool(Dialog*, Dialog*)> Dialog::viz2DWin_Xcomparator([](Dialog* lhs, Dialog* rhs) {
    return lhs->position()[0] < rhs->position()[0];
});
std::set<Dialog*, decltype(Dialog::viz2DWin_Xcomparator)> Dialog::all_windows_xsorted_(
        viz2DWin_Xcomparator);

Dialog::Dialog(nanogui::Screen* screen, int x, int y, const string& title) :
        Window(screen, title), screen_(screen), lastDragPos_(x, y) {
    all_windows_xsorted_.insert(this);
    oldLayout_ = new nanogui::AdvancedGridLayout( { 10, 0, 10, 0 }, { });
    oldLayout_->set_margin(10);
    oldLayout_->set_col_stretch(2, 1);
    this->set_position( { x, y });
    this->set_layout(oldLayout_);
    this->set_visible(true);

    minBtn_ = this->button_panel()->add<nanogui::Button>("_");
    maxBtn_ = this->button_panel()->add<nanogui::Button>("+");
    newLayout_ = new nanogui::AdvancedGridLayout( { 10, 0, 10, 0 }, { });

    maxBtn_->set_visible(false);

    maxBtn_->set_callback([&, this]() {
        this->minBtn_->set_visible(true);
        this->maxBtn_->set_visible(false);

        for (auto* child : this->children()) {
            child->set_visible(true);
        }

        this->set_layout(oldLayout_);
        this->set_position(maximizedPos_);
        this->screen_->perform_layout();
        this->minimized_ = false;
    });

    minBtn_->set_callback([&, this]() {
        this->minBtn_->set_visible(false);
        this->maxBtn_->set_visible(true);

        for (auto* child : this->children()) {
            child->set_visible(false);
        }
        this->set_size( { 0, 0 });
        this->set_layout(newLayout_);
        this->screen_->perform_layout();
        int gap = 0;
        int x = 0;
        int w = width();
        int lastX = 0;
        this->maximizedPos_ = this->position();

        for (Dialog* win : all_windows_xsorted_) {
            if (win != this && win->isMinimized()) {
                x = win->position()[0];
                gap = lastX + x;
                if (gap >= w) {
                    this->set_position( { lastX, screen_->height() - this->height() });
                    break;
                }
                lastX = x + win->width() + 1;
            }
        }
        if (gap < w) {
            this->set_position( { lastX, screen_->height() - this->height() });
        }
        this->minimized_ = true;
    });
}

Dialog::~Dialog() {
    all_windows_xsorted_.erase(this);
}

bool Dialog::isMinimized() {
    return minimized_;
}

bool Dialog::mouse_drag_event(const nanogui::Vector2i& p, const nanogui::Vector2i& rel, int button,
        int mods) {
    if (m_drag && (button & (1 << GLFW_MOUSE_BUTTON_1)) != 0) {
        if (maxBtn_->visible()) {
            for (auto* win : all_windows_xsorted_) {
                if (win != this) {
                    if (win->contains(this->position())
                            || win->contains(
                                    { this->position()[0] + this->size()[0], this->position()[1]
                                            + this->size()[1] }) || win->contains( {
                                    this->position()[0], this->position()[1] + this->size()[1] })
                            || win->contains(
                                    { this->position()[0] + this->size()[0], this->position()[1] })
                            || this->contains(win->position())
                            || this->contains(
                                    { win->position()[0] + win->size()[0], win->position()[1]
                                            + win->size()[1] }) || this->contains( {
                                    win->position()[0], win->position()[1] + win->size()[1] })
                            || this->contains( { win->position()[0] + win->size()[0],
                                    win->position()[1] })) {
                        this->set_position(lastDragPos_);
                        return true;
                    }
                }
            }
        }
        lastDragPos_ = m_pos;
        bool result = nanogui::Window::mouse_drag_event(p, rel, button, mods);

        return result;
    }
    return false;
}

} /* namespace viz */
} /* namespace cv */
