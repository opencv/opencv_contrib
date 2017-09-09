#include <opencv2/dynamicfusion/cuda/device_array.hpp>
/////////////////////   implementations of DeviceArray ////////////////////////////////////////////
namespace cv
{
    namespace kfusion
    {
        namespace cuda
        {


            template<class T>
            DeviceArray<T>::DeviceArray() {}

            template<class T>
            DeviceArray<T>::DeviceArray(size_t size) : DeviceMemory(size * elem_size) {}

            template<class T>
            DeviceArray<T>::DeviceArray(T *ptr, size_t size) : DeviceMemory(ptr, size * elem_size) {}

            template<class T>
            DeviceArray<T>::DeviceArray(const DeviceArray &other) : DeviceMemory(other) {}

            template<class T>
            DeviceArray<T> &DeviceArray<T>::operator=(const DeviceArray &other) {
                DeviceMemory::operator=(other);
                return *this;
            }

            template<class T>
            void DeviceArray<T>::create(size_t size) { DeviceMemory::create(size * elem_size); }

            template<class T>
            void DeviceArray<T>::release() { DeviceMemory::release(); }

            template<class T>
            void DeviceArray<T>::copyTo(DeviceArray &other) const { DeviceMemory::copyTo(other); }

            template<class T>
            void DeviceArray<T>::upload(const T *host_ptr, size_t size) { DeviceMemory::upload(host_ptr, size * elem_size); }

            template<class T>
            void DeviceArray<T>::download(T *host_ptr) const { DeviceMemory::download(host_ptr); }

            template<class T>
            void DeviceArray<T>::swap(DeviceArray &other_arg) { DeviceMemory::swap(other_arg); }

            template<class T>
            DeviceArray<T>::operator T *() { return ptr(); }

            template<class T>
            DeviceArray<T>::operator const T *() const { return ptr(); }

            template<class T>
            size_t DeviceArray<T>::size() const { return sizeBytes() / elem_size; }

            template<class T>
            T *DeviceArray<T>::ptr() { return DeviceMemory::ptr<T>(); }

            template<class T>
            const T *DeviceArray<T>::ptr() const { return DeviceMemory::ptr<T>(); }

            template<class T>
            template<class A>
            void DeviceArray<T>::upload(const std::vector <T, A> &data) { upload(&data[0], data.size()); }

            template<class T>
            template<class A>
            void DeviceArray<T>::download(std::vector <T, A> &data) const {
                data.resize(size());
                if (!data.empty()) download(&data[0]);
            }

/////////////////////   implementations of DeviceArray2D ////////////////////////////////////////////

            template<class T>
            DeviceArray2D<T>::DeviceArray2D() {}

            template<class T>
            DeviceArray2D<T>::DeviceArray2D(int rows, int cols) : DeviceMemory2D(rows, cols * elem_size) {}

            template<class T>
            DeviceArray2D<T>::DeviceArray2D(int rows, int cols, void *data, size_t stepBytes) : DeviceMemory2D(rows, cols *
                                                                                                                     elem_size,
                                                                                                               data,
                                                                                                               stepBytes) {}

            template<class T>
            DeviceArray2D<T>::DeviceArray2D(const DeviceArray2D &other) : DeviceMemory2D(other) {}

            template<class T>
            DeviceArray2D <T> &DeviceArray2D<T>::operator=(const DeviceArray2D &other) {
                DeviceMemory2D::operator=(other);
                return *this;
            }

            template<class T>
            void DeviceArray2D<T>::create(int rows, int cols) { DeviceMemory2D::create(rows, cols * elem_size); }

            template<class T>
            void DeviceArray2D<T>::release() { DeviceMemory2D::release(); }

            template<class T>
            void DeviceArray2D<T>::copyTo(DeviceArray2D &other) const { DeviceMemory2D::copyTo(other); }

            template<class T>
            void DeviceArray2D<T>::upload(const void *host_ptr, size_t host_step, int rows, int cols) {
                DeviceMemory2D::upload(host_ptr, host_step, rows, cols * elem_size);
            }

            template<class T>
            void DeviceArray2D<T>::download(void *host_ptr, size_t host_step) const {
                DeviceMemory2D::download(host_ptr, host_step);
            }

            template<class T>
            template<class A>
            void DeviceArray2D<T>::upload(const std::vector <T, A> &data, int cols) {
                upload(&data[0], cols * elem_size, data.size() / cols, cols);
            }

            template<class T>
            template<class A>
            void DeviceArray2D<T>::download(std::vector <T, A> &data, int &elem_step) const {
                elem_step = cols();
                data.resize(cols() * rows());
                if (!data.empty()) download(&data[0], colsBytes());
            }

            template<class T>
            void DeviceArray2D<T>::swap(DeviceArray2D &other_arg) { DeviceMemory2D::swap(other_arg); }

            template<class T>
            T *DeviceArray2D<T>::ptr(int y) { return DeviceMemory2D::ptr<T>(y); }

            template<class T>
            const T *DeviceArray2D<T>::ptr(int y) const { return DeviceMemory2D::ptr<T>(y); }

            template<class T>
            DeviceArray2D<T>::operator T *() { return ptr(); }

            template<class T>
            DeviceArray2D<T>::operator const T *() const { return ptr(); }

            template<class T>
            int DeviceArray2D<T>::cols() const { return DeviceMemory2D::colsBytes() / elem_size; }

            template<class T>
            int DeviceArray2D<T>::rows() const { return DeviceMemory2D::rows(); }

            template<class T>
            size_t DeviceArray2D<T>::elem_step() const { return DeviceMemory2D::step() / elem_size; }

        }
    }
}