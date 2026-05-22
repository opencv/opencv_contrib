// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef MODULES_V4D_INCLUDE_OPENCV2_V4D_DETAIL_EVENTS_HPP_
#define MODULES_V4D_INCLUDE_OPENCV2_V4D_DETAIL_EVENTS_HPP_

#include <GLFW/glfw3.h>
#include <opencv2/core.hpp>

namespace cv {
namespace v4d {
namespace event {

inline static thread_local GLFWwindow* current_window = nullptr;

struct WindowState {
    cv::Size size;
    cv::Point position;
    bool focused;
};

inline static thread_local WindowState window_state;

static GLFWwindow* get_current_glfw_window() {
	if(current_window == nullptr)
		CV_Error(cv::Error::StsBadArg, "No current glfw window set for event handling. You probably tried to call one of the cv::v4d::event functions outside a context-call.");
	return current_window;
}

static void set_current_glfw_window(GLFWwindow* window) {
	current_window = window;
}

// Define an enum class for the V4D keys
enum class Key {
    KEY_A,
    KEY_B,
    KEY_C,
    KEY_D,
    KEY_E,
    KEY_F,
    KEY_G,
    KEY_H,
    KEY_I,
    KEY_J,
    KEY_K,
    KEY_L,
    KEY_M,
    KEY_N,
    KEY_O,
    KEY_P,
    KEY_Q,
    KEY_R,
    KEY_S,
    KEY_T,
    KEY_U,
    KEY_V,
    KEY_W,
    KEY_X,
    KEY_Y,
    KEY_Z,
    KEY_0,
    KEY_1,
    KEY_2,
    KEY_3,
    KEY_4,
    KEY_5,
    KEY_6,
    KEY_7,
    KEY_8,
    KEY_9,
    KEY_SPACE,
    KEY_ENTER,
    KEY_BACKSPACE,
    KEY_TAB,
    KEY_ESCAPE,
    KEY_UP,
    KEY_DOWN,
    KEY_LEFT,
    KEY_RIGHT,
    KEY_HOME,
    KEY_END,
    KEY_PAGE_UP,
    KEY_PAGE_DOWN,
    KEY_INSERT,
    KEY_DELETE,
    KEY_F1,
    KEY_F2,
    KEY_F3,
    KEY_F4,
    KEY_F5,
    KEY_F6,
    KEY_F7,
    KEY_F8,
    KEY_F9,
    KEY_F10,
    KEY_F11,
    KEY_F12
};

enum class KeyEventType {
	NONE,
	PRESS,
	RELEASE,
    REPEAT,
    HOLD
};

inline static thread_local std::map<Key, bool> key_states;

constexpr Key get_v4d_key(int glfw_key) {
    switch (glfw_key) {
        case GLFW_KEY_A: return Key::KEY_A;
        case GLFW_KEY_B: return Key::KEY_B;
        case GLFW_KEY_C: return Key::KEY_C;
        case GLFW_KEY_D: return Key::KEY_D;
        case GLFW_KEY_E: return Key::KEY_E;
        case GLFW_KEY_F: return Key::KEY_F;
        case GLFW_KEY_G: return Key::KEY_G;
        case GLFW_KEY_H: return Key::KEY_H;
        case GLFW_KEY_I: return Key::KEY_I;
        case GLFW_KEY_J: return Key::KEY_J;
        case GLFW_KEY_K: return Key::KEY_K;
        case GLFW_KEY_L: return Key::KEY_L;
        case GLFW_KEY_M: return Key::KEY_M;
        case GLFW_KEY_N: return Key::KEY_N;
        case GLFW_KEY_O: return Key::KEY_O;
        case GLFW_KEY_P: return Key::KEY_P;
        case GLFW_KEY_Q: return Key::KEY_Q;
        case GLFW_KEY_R: return Key::KEY_R;
        case GLFW_KEY_S: return Key::KEY_S;
        case GLFW_KEY_T: return Key::KEY_T;
        case GLFW_KEY_U: return Key::KEY_U;
        case GLFW_KEY_V: return Key::KEY_V;
        case GLFW_KEY_W: return Key::KEY_W;
        case GLFW_KEY_X: return Key::KEY_X;
        case GLFW_KEY_Y: return Key::KEY_Y;
        case GLFW_KEY_Z: return Key::KEY_Z;
        case GLFW_KEY_0: return Key::KEY_0;
        case GLFW_KEY_1: return Key::KEY_1;
        case GLFW_KEY_2: return Key::KEY_2;
        case GLFW_KEY_3: return Key::KEY_3;
        case GLFW_KEY_4: return Key::KEY_4;
        case GLFW_KEY_5: return Key::KEY_5;
        case GLFW_KEY_6: return Key::KEY_6;
        case GLFW_KEY_7: return Key::KEY_7;
        case GLFW_KEY_8: return Key::KEY_8;
        case GLFW_KEY_9: return Key::KEY_9;
        case GLFW_KEY_SPACE: return Key::KEY_SPACE;
        case GLFW_KEY_ENTER: return Key::KEY_ENTER;
        case GLFW_KEY_BACKSPACE: return Key::KEY_BACKSPACE;
        case GLFW_KEY_TAB: return Key::KEY_TAB;
        case GLFW_KEY_ESCAPE: return Key::KEY_ESCAPE;
        case GLFW_KEY_UP: return Key::KEY_UP;
        case GLFW_KEY_DOWN: return Key::KEY_DOWN;
        case GLFW_KEY_LEFT: return Key::KEY_LEFT;
        case GLFW_KEY_RIGHT: return Key::KEY_RIGHT;
        case GLFW_KEY_END: return Key::KEY_END;
        case GLFW_KEY_PAGE_UP: return Key::KEY_PAGE_UP;
        case GLFW_KEY_PAGE_DOWN: return Key::KEY_PAGE_DOWN;
        case GLFW_KEY_INSERT: return Key::KEY_INSERT;
        case GLFW_KEY_DELETE: return Key::KEY_DELETE;
        case GLFW_KEY_F1: return Key::KEY_F1;
        case GLFW_KEY_F2: return Key::KEY_F2;
        case GLFW_KEY_F3: return Key::KEY_F3;
        case GLFW_KEY_F4: return Key::KEY_F4;
        case GLFW_KEY_F5: return Key::KEY_F5;
        case GLFW_KEY_F6: return Key::KEY_F6;
        case GLFW_KEY_F7: return Key::KEY_F7;
        case GLFW_KEY_F8: return Key::KEY_F8;
        case GLFW_KEY_F9: return Key::KEY_F9;
        case GLFW_KEY_F10: return Key::KEY_F10;
        case GLFW_KEY_F11: return Key::KEY_F11;
        case GLFW_KEY_F12: return Key::KEY_F12;
        default:
        	CV_Error_(cv::Error::StsBadArg, ("Invalid key: %d. Please ensure the key is within the valid range.", glfw_key));
        	return Key::KEY_F12;
    }
}

static KeyEventType get_key_event_type(int key) {
    Key v4d_key = get_v4d_key(key);
    int state = glfwGetKey(get_current_glfw_window(), key);
    switch (state) {
        case GLFW_PRESS:
            key_states[v4d_key] = true;
            return KeyEventType::PRESS;
        case GLFW_RELEASE:
            key_states[v4d_key] = false;
            return KeyEventType::RELEASE;
        case GLFW_REPEAT:
            return KeyEventType::REPEAT;
        default:
            return KeyEventType::NONE;
    }
}

static KeyEventType get_key_hold_event(Key key) {
    if (key_states[key]) {
        return KeyEventType::HOLD;
    } else {
        return KeyEventType::NONE;
    }
}

// Define an enum class for the V4D mouse buttons
enum class MouseButton {
    LEFT,
    RIGHT,
    MIDDLE,
    BUTTON_4,
    BUTTON_5,
    BUTTON_6,
    BUTTON_7,
    BUTTON_8
};

enum class MouseEventType {
	NONE,
	PRESS,
	RELEASE,
    MOVE,
    SCROLL,
    DRAG_START,
    DRAG,
    DRAG_END,
    HOVER_ENTER,
    HOVER_EXIT,
    DOUBLE_CLICK
};

// Define a static function that returns the mouse position as a cv::Point2d
static cv::Point2d get_mouse_position() {
    // Declare variables to store the mouse position
    double x, y;
    // Get the mouse position using glfwGetCursorPos
    glfwGetCursorPos(get_current_glfw_window(), &x, &y);
    // Return the mouse position as a cv::Point2d
    return cv::Point2d(x, y);
}

inline static thread_local std::map<MouseButton, bool> button_states;
inline static thread_local cv::Point2d last_position = get_mouse_position();
inline static thread_local cv::Point2d scroll_offset(0, 0);

static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    // Update the scroll offset
    scroll_offset = cv::Point2d(xoffset, yoffset);
}

constexpr static MouseButton get_v4d_mouse_button(int glfw_button) {
    switch (glfw_button) {
        case GLFW_MOUSE_BUTTON_LEFT: return MouseButton::LEFT;
        case GLFW_MOUSE_BUTTON_RIGHT: return MouseButton::RIGHT;
        case GLFW_MOUSE_BUTTON_MIDDLE: return MouseButton::MIDDLE;
        case GLFW_MOUSE_BUTTON_4: return MouseButton::BUTTON_4;
        case GLFW_MOUSE_BUTTON_5: return MouseButton::BUTTON_5;
        case GLFW_MOUSE_BUTTON_6: return MouseButton::BUTTON_6;
        case GLFW_MOUSE_BUTTON_7: return MouseButton::BUTTON_7;
        case GLFW_MOUSE_BUTTON_8: return MouseButton::BUTTON_8;
        default: CV_Error_(cv::Error::StsBadArg, ("Invalid mouse button: %d. Please ensure the button is within the valid range.", glfw_button));
    }
}

static MouseEventType get_mouse_event_type(int button) {
    MouseButton v4d_button = get_v4d_mouse_button(button);
    int state = glfwGetMouseButton(get_current_glfw_window(), button);
    switch (state) {
        case GLFW_PRESS:
            button_states[v4d_button] = true;
            return MouseEventType::PRESS;
        case GLFW_RELEASE:
            button_states[v4d_button] = false;
            return MouseEventType::RELEASE;
        default:
            return MouseEventType::NONE;
    }
}

static cv::Point2d get_mouse_scroll_offset() {
    return scroll_offset;
}

static MouseEventType get_mouse_scroll_event() {
    cv::Point2d current_offset = get_mouse_scroll_offset();
    if (current_offset != last_position) {
        last_position = current_offset;
        return MouseEventType::SCROLL;
    } else {
        return MouseEventType::NONE;
    }
}

static MouseEventType get_mouse_move_event() {
    cv::Point2d current_position = get_mouse_position();
    if (current_position != last_position) {
        last_position = current_position;
        return MouseEventType::MOVE;
    } else {
        return MouseEventType::NONE;
    }
}

static MouseEventType get_mouse_drag_event(MouseButton button) {
    cv::Point2d current_position = get_mouse_position();
    if (button_states[button] && current_position != last_position) {
        last_position = current_position;
        return MouseEventType::DRAG;
    } else {
        return MouseEventType::NONE;
    }
}

static MouseEventType get_mouse_hover_event() {
    cv::Point2d current_position = get_mouse_position();
    if (current_position != last_position) {
        last_position = current_position;
        return MouseEventType::HOVER_ENTER;
    } else {
        return MouseEventType::HOVER_EXIT;
    }
}

enum class WindowEvent {
    NONE,
    RESIZE,
    MOVE,
    FOCUS,
    UNFOCUS,
    CLOSE
};

static WindowEvent get_window_resize_event() {
    static WindowState last_state = window_state;

    if (window_state.size != last_state.size) {
        last_state.size = window_state.size;
        return WindowEvent::RESIZE;
    } else {
        return WindowEvent::NONE;
    }
}

static WindowEvent get_window_move_event() {
    static WindowState last_state = window_state;

    if (window_state.position != last_state.position) {
        last_state.position = window_state.position;
        return WindowEvent::MOVE;
    } else {
        return WindowEvent::NONE;
    }
}

static WindowEvent get_window_focus_event() {
    static WindowState last_state = window_state;

    if (window_state.focused && !last_state.focused) {
        last_state.focused = window_state.focused;
        return WindowEvent::FOCUS;
    } else if (!window_state.focused && last_state.focused) {
        last_state.focused = window_state.focused;
        return WindowEvent::UNFOCUS;
    } else {
        return WindowEvent::NONE;
    }
}

static cv::Size get_window_size() {
    int width, height;
    glfwGetWindowSize(get_current_glfw_window(), &width, &height);
    return cv::Size(width, height);
}

static cv::Point get_window_position() {
    int x, y;
    glfwGetWindowPos(get_current_glfw_window(), &x, &y);
    return cv::Point(x, y);
}

static bool get_window_focus() {
    int focused = glfwGetWindowAttrib(get_current_glfw_window(), GLFW_FOCUSED);
    return focused;
}

static void initialize_callbacks(GLFWwindow* window) {
    glfwSetScrollCallback(window, scroll_callback);
}

// Define an enum class for the V4D joystick buttons
enum class JoystickButton {
    BUTTON_A,
    BUTTON_B,
    BUTTON_X,
    BUTTON_Y,
    BUTTON_LB,
    BUTTON_RB,
    BUTTON_BACK,
    BUTTON_START,
    BUTTON_GUIDE,
    BUTTON_LEFT_THUMB,
    BUTTON_RIGHT_THUMB,
    BUTTON_DPAD_UP,
    BUTTON_DPAD_RIGHT,
    BUTTON_DPAD_DOWN,
    BUTTON_DPAD_LEFT
};

// Define an enum class for the V4D joystick axes
enum class JoystickAxis {
    AXIS_LEFT_X,
    AXIS_LEFT_Y,
    AXIS_RIGHT_X,
    AXIS_RIGHT_Y,
    AXIS_LEFT_TRIGGER,
    AXIS_RIGHT_TRIGGER
};

// Define a static function that returns the state of a joystick button
static bool get_joystick_button_state(int joystick, JoystickButton button) {
    int count;
    const unsigned char* buttons = glfwGetJoystickButtons(joystick, &count);
    if (buttons == nullptr) {
        CV_Error(cv::Error::StsBadArg, "Failed to get joystick buttons. Please ensure the joystick is connected and working properly.");
    }
    return buttons[static_cast<int>(button)];
}

// Define a static function that returns the name of a joystick
static const char* get_joystick_name(int joystick) {
    const char* name = glfwGetJoystickName(joystick);
    if (name == nullptr) {
        CV_Error(cv::Error::StsBadArg, "Failed to get joystick name. Please ensure the joystick is connected and working properly.");
    }
    return name;
}

// Define a static function that returns whether a joystick is present
static bool is_joystick_present(int joystick) {
    int present = glfwJoystickPresent(joystick);
    if (present != GLFW_TRUE && present != GLFW_FALSE) {
        CV_Error(cv::Error::StsBadArg, "Failed to check if joystick is present. Please ensure the joystick is connected and working properly.");
    }
    return present;
}

// Define a static function that sets the clipboard string
static void set_clipboard_string(const char* string) {
    if (string == nullptr) {
        CV_Error(cv::Error::StsNullPtr, "Cannot set clipboard string to null. Please provide a valid string.");
    }
    glfwSetClipboardString(get_current_glfw_window(), string);
}

// Define a static function that gets the clipboard string
static const char* get_clipboard_string() {
    const char* string = glfwGetClipboardString(get_current_glfw_window());
    if (string == nullptr) {
        CV_Error(cv::Error::StsNullPtr, "Failed to get clipboard string. Please ensure there is a string in the clipboard.");
    }
    return string;
}

}
}
}
#endif  // MODULES_V4D_INCLUDE_OPENCV2_V4D_DETAIL_EVENTS_HPP_
