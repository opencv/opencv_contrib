// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_COMMON_COUNTED_HPP__
#define __ZXING_COMMON_COUNTED_HPP__

#include <cstddef>
#include <algorithm>
namespace zxing {

/* base class for reference-counted objects */
class Counted {
private:
    unsigned int count_;

public:
    Counted() : count_(0) {}
    virtual ~Counted() {}
    Counted* retain() {
        count_++;
        return this;
    }
    void release() {
        count_--;
        if (count_ == 0) {
            count_ = 0xDEADF001;
            delete this;
        }
    }

    /* return the current count for denugging purposes or similar */
    int count() const { return count_; }
};

/* counting reference to reference-counted objects */
template <typename T>
class Ref {
private:
public:
    T* object_;
    explicit Ref(T* o = 0) : object_(0) { reset(o); }
    Ref(const Ref& other) : object_(0) { reset(other.object_); }

    template <class Y>
    Ref(const Ref<Y>& other) : object_(0) {
        reset(other.object_);
    }

    ~Ref() {
        if (object_) {
            object_->release();
        }
    }

    void reset(T* o) {
        if (o) {
            o->retain();
        }
        if (object_ != 0) {
            object_->release();
        }
        object_ = o;
    }
    Ref& operator=(const Ref& other) {
        reset(other.object_);
        return *this;
    }
    template <class Y>
    Ref& operator=(const Ref<Y>& other) {
        reset(other.object_);
        return *this;
    }
    Ref& operator=(T* o) {
        reset(o);
        return *this;
    }
    template <class Y>
    Ref& operator=(Y* o) {
        reset(o);
        return *this;
    }

    T& operator*() { return *object_; }
    T* operator->() const { return object_; }
    operator T*() const { return object_; }

    bool operator==(const T* that) { return object_ == that; }
    bool operator==(const Ref& other) const {
        return object_ == other.object_ || *object_ == *(other.object_);
    }
    template <class Y>
    bool operator==(const Ref<Y>& other) const {
        return object_ == other.object_ || *object_ == *(other.object_);
    }

    bool operator!=(const T* that) { return !(*this == that); }

    bool empty() const { return object_ == 0; }
};

}  // namespace zxing

#endif  // __ZXING_COMMON_COUNTED_HPP__
