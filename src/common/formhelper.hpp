// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_COMMON_FORMHELPER_HPP_
#define SRC_COMMON_FORMHELPER_HPP_
#include "dialog.hpp"

#include <nanogui/screen.h>
#include <nanogui/formhelper.h>

#include <string>

namespace cv {
namespace viz {
using std::string;
class FormHelper: public nanogui::FormHelper {
public:
    FormHelper(nanogui::Screen* screen);
    virtual ~FormHelper();

    Dialog* makeWindow(int x, int y, const string& title);
    nanogui::Label* makeGroup(const string& label);
    nanogui::detail::FormWidget<bool>* makeFormVariable(const string& name, bool& v,
            const string& tooltip = "", bool visible = true, bool enabled = true);
    template<typename T> nanogui::detail::FormWidget<T>* makeFormVariable(const string& name, T& v,
            const T& min, const T& max, bool spinnable, const string& unit, const string tooltip,
            bool visible = true, bool enabled = true) {
        auto var = this->add_variable(name, v);
        var->set_enabled(enabled);
        var->set_visible(visible);
        var->set_spinnable(spinnable);
        var->set_min_value(min);
        var->set_max_value(max);
        if (!unit.empty())
            var->set_units(unit);
        if (!tooltip.empty())
            var->set_tooltip(tooltip);
        return var;
    }

    nanogui::ColorPicker* makeColorPicker(const string& label, nanogui::Color& color,
            const string& tooltip = "", std::function<void(const nanogui::Color)> fn = nullptr,
            bool visible = true, bool enabled = true);
    template<typename T> nanogui::ComboBox* makeComboBox(const string& label, T& e,
            const std::vector<string>& items) {
        auto* var = this->add_variable(label, e, true);
        var->set_items(items);
        return var;
    }

    nanogui::Button* makeButton(const string& caption, std::function<void()> fn);
};

} /* namespace viz2d */
} /* namespace kb */

#endif /* SRC_COMMON_FORMHELPER_HPP_ */
