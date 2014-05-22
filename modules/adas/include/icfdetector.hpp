class ICFDetector
{
public:
    /* Initialize detector

        min_obj_size — min possible object size on image in pixels (rows x cols)

        max_obj_size — max possible object size on image in pixels (rows x cols)

        scales_per_octave — number of images in pyramid while going
                            from scale x to scale 2x. Affects on speed
                            and quality of the detector
    */
    ICFDetector(Size min_obj_size,
                Size max_obj_size,
                int scales_per_octave = 8);

    /* Load detector from file, return true on success, false otherwise */
    bool load(const std::string& filename);

    /* Run detector on single image

        image — image for detection

        bboxes — output array of bounding boxes in format
                 Rect(row_from, col_from, n_rows, n_cols)

        confidence_values — output array of confidence values from 0 to 1.
            One value per bbox — confidence of detector that corresponding
            bbox contatins object
    */
    void detect(const Mat& image,
                std::vector<Rect>& bboxes,
                std::vector<float>& confidence_values) const;


    /* Train detector

        image_filenames — filenames of images for training

        labelling — vector of object bounding boxes per every image

        params — parameters for detector training
    */
    void train(const std::vector<std::string>& image_filenames,
               const std::vector<std::vector<Rect>>& labelling,
               const ICFDetectorParams& params = ICFDetectorParams());

    /* Save detector in file, return true on success, false otherwise */
    bool save(const std::string& filename);
};