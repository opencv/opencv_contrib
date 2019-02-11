typedef TrackerSamplerCSC::Params TrackerSamplerCSC_Params;
typedef TrackerSamplerCS::Params TrackerSamplerCS_Params;
typedef TrackerSamplerPF::Params TrackerSamplerPF_Params;
typedef TrackerFeatureHAAR::Params TrackerFeatureHAAR_Params;
typedef TrackerMIL::Params TrackerMIL_Params;
typedef TrackerBoosting::Params TrackerBoosting_Params;
typedef TrackerMedianFlow::Params TrackerMedianFlow_Params;
typedef TrackerTLD::Params TrackerTLD_Params;
typedef TrackerKCF::Params TrackerKCF_Params;
typedef TrackerGOTURN::Params TrackerGOTURN_Params;
typedef TrackerCSRT::Params TrackerCSRT_Params;

template<>
PyObject* pyopencv_from(const std::string& value)
{
    return pyopencv_from(String(value));
}

template<>
bool pyopencv_to(PyObject* obj, std::string& value, const char* name)
{
    CV_UNUSED(name);
    if (PyString_Check(obj))
    {
        value = std::string(String(PyString_AsString(obj)));
        return true;
    }
    return false;
}
