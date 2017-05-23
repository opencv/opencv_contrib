#include <opencv2/c/rect.hpp>

extern "C" {
Rect* cv_create_Rect() {
    return new Rect;
}
Rect* cv_create_Rect4(int x, int y, int width, int height) {
    return new Rect(x, y, width, height);
}
Rect* cv_Rect_assignTo(Rect* self, Rect* r) {
    *self = *r;
    return self;
}
Rect* cv_Rect_clone(Rect* self) {
    return new Rect(self->x, self->y, self->width, self->height);
}
Point* cv_Rect_tl(Rect* self) {
    return new Point(self->tl());
}
Point* cv_Rect_br(Rect* self) {
    return new Point(self->br());
}
int cv_Rect_getX(Rect* self) {
    return self->x;
}
int cv_Rect_getY(Rect* self) {
    return self->y;
}
int cv_Rect_getWidth(Rect* self) {
    return self->width;
}
int cv_Rect_getHeight(Rect* self) {
    return self->height;
}
Size* cv_Rect_size(Rect* self) {
    return new Size(self->size());
}
int cv_Rect_contains(Rect* self, Point* pt) {
    return self->contains(*pt);
}
}
