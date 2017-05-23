#include <opencv2/c/point.hpp>

#define ADD_POINT_FUNC_IMPL(t, tn) \
    Point2##t * cv_create_Point2##t ( tn x,  tn y) { \
        return new Point2##t (x, y);\
    } \
    Point3##t * cv_create_Point3##t ( tn x,  tn y,  tn z) { \
        return new Point3##t (x, y, z);\
    } \
    tn cv_Point2##t##_getX( Point2##t * self) { \
        return self->x;\
    }\
    tn cv_Point2##t##_getY( Point2##t * self) { \
        return self->y;\
    }\
    tn cv_Point3##t##_getX( Point3##t * self) { \
        return self->x;\
    }\
    tn cv_Point3##t##_getY( Point3##t * self) { \
        return self->y;\
    }\
    tn cv_Point3##t##_getZ( Point3##t * self) { \
        return self->z;\
    }\
    tn cv_Point2##t##_dot( Point2##t * self, Point2##t * other) { \
        return self->dot(*other);\
    }\
    tn cv_Point3##t##_dot( Point3##t * self, Point3##t * other) { \
        return self->dot(*other);\
    }\
    Point3##t * cv_Point3##t##_cross(Point3##t * self, Point3##t * other) { \
        return new Point3##t (self->cross(*other));\
    }
    
extern "C" {
    ADD_POINT_FUNC_IMPL(i, int);
    ADD_POINT_FUNC_IMPL(f, float);
    ADD_POINT_FUNC_IMPL(d, double);
}
