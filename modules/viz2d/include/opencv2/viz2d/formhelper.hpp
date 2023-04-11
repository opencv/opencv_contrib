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

/*!
 * Ansub-class of nanogui::FormHelper adding convenience calls not unlike highgui offers.
 */
CV_EXPORTS class FormHelper: public nanogui::FormHelper {
public:
    /*!
     * Creates a FormHelper.
     * @param screen The parent nanogui::screen.
     */
    CV_EXPORTS FormHelper(nanogui::Screen* screen);
    /*!
     * Default destructor.
     */
    CV_EXPORTS virtual ~FormHelper();
    /*!
     * Creates a dialog held by this form helper.
     * @param x The x position.
     * @param y The y position.
     * @param title The title.
     * @return A pointer to the newly created Dialog.
     */
    CV_EXPORTS Dialog* makeDialog(int x, int y, const string& title);
    /*!
     * Create a grouping label.
     * @param label The label text.
     * @return A pointer to the newly created Label.
     */
    CV_EXPORTS nanogui::Label* makeGroup(const string& label);
    /*!
     * Make a boolean form widget.
     * @param name The widget name.
     * @param v The state of the widget.
     * @param tooltip The tooltip.
     * @param visible The initial visibility.
     * @param enabled Indicates if the widget is initially enabled.
     * @return A pointer to the newly created boolean form widget.
     */
    CV_EXPORTS nanogui::detail::FormWidget<bool>* makeFormVariable(const string& name, bool& v,
            const string& tooltip = "", bool visible = true, bool enabled = true);

    /*!
     * Creates a form widget and deduces the type of widget by template parameter T.
     * @tparam T The underlying type of the form widget.
     * @param name The name of the widget.
     * @param v The current value.
     * @param min The minimum value.
     * @param max The maximum value.
     * @param spinnable Indicates if the widget is spinnable.
     * @param unit A string denoting a unit..
     * @param tooltip The tooltip.
     * @param visible The initial visibility.
     * @param enabled Indicates if the widget is initially enabled.
     * @return A pointer to the newly created form widget representing type T.
     */
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

    /*!
     * Creates a color picker widget
     * @param label The label of the widget
     * @param color The initial color
     * @param tooltip The widget tooltip
     * @param fn Custom color selection callback
     * @param visible Indicates if the widget is initially visible
     * @param enabled Indicates if the widget is initially enabled
     * @return A pointer to the newly created ColorPicker
     */
    CV_EXPORTS nanogui::ColorPicker* makeColorPicker(const string& label, nanogui::Color& color,
            const string& tooltip = "", std::function<void(const nanogui::Color)> fn = nullptr,
            bool visible = true, bool enabled = true);
    /*!
     * Create a ComboBox from an enumaration type T
     * @tparam T The enumation type to create the widget from
     * @param label The label text
     * @param e The value
     * @param items A vector of strings with one string per enumeration value
     * @return A pointer to the newly created ComboBox
     */
    template<typename T> nanogui::ComboBox* makeComboBox(const string& label, T& e,
            const std::vector<string>& items) {
        auto* var = this->add_variable(label, e, true);
        var->set_items(items);
        return var;
    }

    /*!
     * Create a Button Widget.
     * @param caption The caption
     * @param fn Button press callback
     * @return A pointer to the newly created Button
     */
    CV_EXPORTS nanogui::Button* makeButton(const string& caption, std::function<void()> fn);
};

} /* namespace viz */
} /* namespace cv */

#endif /* SRC_COMMON_FORMHELPER_HPP_ */
