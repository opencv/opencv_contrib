Exporting a template parameter file {#tutorial_qds_export_parameters}
==================

Goal
----

In this tutorial you will learn how to

-   create a simple parameter file template.

@include ./samples/export_param_file.cpp

## Explanation:

The class supports loading configuration parameters from a .yaml file using the method `loadParameters()`.
This is very useful for fine-tuning the class' parameters on the fly. To extract a template of this
parameter file you run the following code.

We create an instance of a `QuasiDenseStereo` object. Not specifying the second argument of the constructor,
makes the object to load default parameters.
@snippet ./samples/export_param_file.cpp create
By calling the method `saveParameters()`, we store the template file to the location specified by `parameterFileLocation`
@snippet ./samples/export_param_file.cpp write
