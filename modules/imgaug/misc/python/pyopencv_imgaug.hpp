// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_AUG_MISC_PYTHON_HPP
#define OPENCV_AUG_MISC_PYTHON_HPP
typedef std::vector<cv::Ptr<cv::imgaug::Transform> > vector_Ptr_Transform;
typedef std::vector<cv::Ptr<cv::imgaug::det::Transform> > vector_Ptr_imgaug_det_Transform;

//template<>
//bool pyopencv_to(PyObject *o, std::vector<Ptr<cv::Transform> > &value, const ArgInfo& info){
//    return pyopencv_to_generic_vec(o, value, info);
//}
template<> struct pyopencvVecConverter<Ptr<cv::imgaug::Transform> >
{
    static bool to(PyObject* obj, std::vector<cv::Ptr<cv::imgaug::Transform> >& value, const ArgInfo& info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

};

template<> struct pyopencvVecConverter<Ptr<cv::imgaug::det::Transform> >
{
    static bool to(PyObject* obj, std::vector<cv::Ptr<cv::imgaug::det::Transform> >& value, const ArgInfo& info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

};

template<> struct PyOpenCV_Converter<unsigned long long>
{
    static bool to(PyObject* obj, unsigned long long& value, const ArgInfo& info){
        if(!obj || obj == Py_None)
            return true;
        if(PyLong_Check(obj)){
            value = PyLong_AsUnsignedLongLong(obj);
        }else{
            return false;
        }
        return value != (unsigned int)-1 || !PyErr_Occurred();
    }
};

#endif