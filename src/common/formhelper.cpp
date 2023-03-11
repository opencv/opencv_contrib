#include "formhelper.hpp"

namespace kb {
namespace viz2d {

FormHelper::FormHelper(nanogui::Screen* screen) : nanogui::FormHelper(screen) {
}

FormHelper::~FormHelper() {
}

Dialog* FormHelper::makeWindow(int x, int y, const string &title) {
    auto* win = new kb::viz2d::Dialog(m_screen, x, y, title);
    this->set_window(win);
    return win;
}

nanogui::Label* FormHelper::makeGroup(const string &label) {
    return add_group(label);
}

nanogui::detail::FormWidget<bool>* FormHelper::makeFormVariable(const string &name, bool &v, const string &tooltip, bool visible, bool enabled) {
    auto var = add_variable(name, v);
    var->set_enabled(enabled);
    var->set_visible(visible);
    if (!tooltip.empty())
        var->set_tooltip(tooltip);
    return var;
}

nanogui::ColorPicker* FormHelper::makeColorPicker(const string& label, nanogui::Color& color, const string& tooltip, std::function<void(const nanogui::Color)> fn, bool visible, bool enabled) {
    auto* colorPicker = add_variable(label, color);
    colorPicker->set_enabled(enabled);
    colorPicker->set_visible(visible);
    if (!tooltip.empty())
    colorPicker->set_tooltip(tooltip);
    if(fn)
        colorPicker->set_final_callback(fn);

    return colorPicker;
}

nanogui::Button* FormHelper::makeButton(const string& caption, std::function<void()> fn) {
    return add_button(caption, fn);
}

} /* namespace viz2d */
} /* namespace kb */
