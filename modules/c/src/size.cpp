#include <opencv2/c/size.hpp>

extern "C" {

    Size* cv_create_Size() {
        return new Size;
    }
    Size* cv_create_Size2(double width, double height) {
        return new Size(width, height);
    }
    Size* cv_Size_assignTo(Size* self, Size* other) {
        *self = *other;
        return self;
    }
    Size* cv_Size_fromPoint(Point* p) {
        return new Size(*p);
    }
    double cv_Size_area(Size* self) {
        return self->area();
    }
    double cv_Size_width(Size* self) {
        return self->width;
    }
    double cv_Size_height(Size* self) {
        return self->height;
    }
}
