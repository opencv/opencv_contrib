Exporting a template parameter file {#export_param_file}
==================

Goal
----

In this tutorial you will learn how to

-   create a simple parameter file template.

-----------
[Source Code](../samples/export_param_file.cpp)

## Explanation:

The class supports loading configuration parameters from a .yaml file using the method `loadParameters()`.
This is very useful for fine-tuning the class' parameters on the fly. To extract a template of this
parameter file you run the following code.
```
std::string parameterFileLocation = "./parameters.yaml";
Ptr<stereo::QuasiDenseStereo> stereo =  stereo::QuasiDenseStereo::create(cv::Size(5,5));
stereo->saveParameters(parameterFileLocation);
```
We make an instance of a `QuasiDenseStereo` object. Not specifying the second argument of the constructor,
makes the object to load default parameters.
By calling the method `saveParameters()`, we store the template file to the location specified by `parameterFileLocation`
