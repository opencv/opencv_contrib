Quasi Dense Stereo (stereo module) {#tutorial_table_of_content_quasi_dense_stereo}
==========================================================

Quasi Dense Stereo is method for performing dense stereo matching. `QuasiDenseStereo` implements this process.
The code uses pyramidal Lucas-Kanade with Shi-Tomasi features to get the initial seed correspondences.
Then these seeds are propagated by using mentioned growing scheme.

-   @subpage tutorial_qds_quasi_dense_stereo

    Example showing how to get dense correspondences from a stereo image pair.

-   @subpage tutorial_qds_export_parameters

    Example showing how to genereate a parameter file template.
