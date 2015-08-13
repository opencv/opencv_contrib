#ifndef __OPENCV_DNN_LAYER_HPP__
#define __OPENCV_DNN_LAYER_HPP__
#include <opencv2/dnn.hpp>

namespace cv
{
namespace dnn
{

//Layer factory allows to create instances of registered layers.
class CV_EXPORTS LayerRegister
{
public:

    typedef Ptr<Layer>(*Constuctor)(LayerParams &params);

    static void registerLayer(const String &type, Constuctor constructor);

    static void unregisterLayer(const String &type);

    static Ptr<Layer> createLayerInstance(const String &type, LayerParams& params);

private:
    LayerRegister();

    struct Impl;
    static Ptr<Impl> impl;
};

template<typename LayerClass>
Ptr<Layer> _layerDynamicRegisterer(LayerParams &params)
{
    return Ptr<Layer>(new LayerClass(params));
}

#define REG_RUNTIME_LAYER_FUNC(type, constuctorFunc) \
    LayerRegister::registerLayer(#type, constuctorFunc);

#define REG_RUNTIME_LAYER_CLASS(type, class) \
    LayerRegister::registerLayer(#type, _layerDynamicRegisterer<class>);

//allows automatically register created layer on module load time
struct _LayerStaticRegisterer
{
    String type;

    _LayerStaticRegisterer(const String &type, LayerRegister::Constuctor constuctor)
    {
        this->type = type;
        LayerRegister::registerLayer(type, constuctor);
    }

    ~_LayerStaticRegisterer()
    {
        LayerRegister::unregisterLayer(type);
    }
};

//registers layer constructor on module load time
#define REG_STATIC_LAYER_FUNC(type, constuctorFunc) \
static _LayerStaticRegisterer __LayerStaticRegisterer_##type(#type, constuctorFunc);

//registers layer class on module load time
#define REG_STATIC_LAYER_CLASS(type, class)                         \
Ptr<Layer> __LayerStaticRegisterer_func_##type(LayerParams &params) \
    { return Ptr<Layer>(new class(params)); }                       \
static _LayerStaticRegisterer __LayerStaticRegisterer_##type(#type, __LayerStaticRegisterer_func_##type);

}
}
#endif
