Scene Text Recognition
======================

.. highlight:: cpp

OCRTesseract
------------
.. ocv:class:: OCRTesseract : public BaseOCR

OCRTesseract class provides an interface with the tesseract-ocr API (v3.02.02) in C++. Notice that it is compiled only when tesseract-ocr is correctly installed. 

.. note::

    * (C++) An example of OCRTesseract recognition combined with scene text detection can be found at the end_to_end_recognition demo: https://github.com/Itseez/opencv_contrib/blob/master/modules/text/samples/end_to_end_recognition.cpp
    * (C++) Another example of OCRTesseract recognition combined with scene text detection can be found at the webcam_demo: https://github.com/Itseez/opencv_contrib/blob/master/modules/text/samples/webcam_demo.cpp

OCRTesseract::create
--------------------
Creates an instance of the OCRTesseract class. Initializes Tesseract.

.. ocv:function:: Ptr<OCRTesseract> OCRTesseract::create(const char* datapath=NULL, const char* language=NULL, const char* char_whitelist=NULL, int oem=(int)tesseract::OEM_DEFAULT, int psmode=(int)tesseract::PSM_AUTO)

    :param datapath: the name of the parent directory of tessdata ended with "/", or NULL to use the system's default directory.
    :param language: an ISO 639-3 code or NULL will default to "eng".
    :param char_whitelist: specifies the list of characters used for recognition. NULL defaults to "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ".
    :param oem: tesseract-ocr offers different OCR Engine Modes (OEM), by deffault tesseract::OEM_DEFAULT is used. See the tesseract-ocr API documentation for other possible values.
    :param psmode: tesseract-ocr offers different Page Segmentation Modes (PSM) tesseract::PSM_AUTO (fully automatic layout analysis) is used. See the tesseract-ocr API documentation for other possible values.

OCRTesseract::run
-----------------
Recognize text using the tesseract-ocr API. Takes image on input and returns recognized text in the output_text parameter. Optionally provides also the Rects for individual text elements found (e.g. words), and the list of those text elements with their confidence values.

.. ocv:function:: void OCRTesseract::run(Mat& image, string& output_text, vector<Rect>* component_rects=NULL, vector<string>* component_texts=NULL, vector<float>* component_confidences=NULL, int component_level=0)

    :param image: Input image ``CV_8UC1`` or ``CV_8UC3``
    :param output_text: Output text of the tesseract-ocr.
    :param component_rects: If provided the method will output a list of Rects for the individual text elements found (e.g. words or text lines).
    :param component_text: If provided the method will output a list of text strings for the recognition of individual text elements found (e.g. words or text lines).
    :param component_confidences: If provided the method will output a list of confidence values for the recognition of individual text elements found (e.g. words or text lines).
    :param component_level: ``OCR_LEVEL_WORD`` (by default), or ``OCR_LEVEL_TEXT_LINE``.

OCRHMMDecoder
-------------
.. ocv:class:: OCRHMMDecoder : public BaseOCR

OCRHMMDecoder class provides an interface for OCR using Hidden Markov Models.

.. note::

    * (C++) An example on using OCRHMMDecoder recognition combined with scene text detection can be found at the webcam_demo sample: https://github.com/Itseez/opencv_contrib/blob/master/modules/text/samples/webcam_demo.cpp


OCRHMMDecoder::ClassifierCallback
---------------------------------
Callback with the character classifier is made a class. This way it hides the feature extractor and the classifier itself, so developers can write their own OCR code.

.. ocv:class:: OCRHMMDecoder::ClassifierCallback

The default character classifier and feature extractor can be loaded using the utility funtion ``loadOCRHMMClassifierNM`` and KNN model provided in https://github.com/Itseez/opencv_contrib/blob/master/modules/text/samples/OCRHMM_knn_model_data.xml.gz.

OCRHMMDecoder::ClassifierCallback::eval
---------------------------------------
The character classifier must return a (ranked list of) class(es) id('s)

.. ocv:function:: void OCRHMMDecoder::ClassifierCallback::eval( InputArray image, std::vector<int>& out_class, std::vector<double>& out_confidence)

    :param image: Input image ``CV_8UC1`` or ``CV_8UC3`` with a single letter.
    :param out_class: The classifier returns the character class categorical label, or list of class labels, to which the input image corresponds.
    :param out_confidence: The classifier returns the probability of the input image corresponding to each classes in ``out_class``.

OCRHMMDecoder::create
---------------------
Creates an instance of the OCRHMMDecoder class. Initializes HMMDecoder.

.. ocv:function:: Ptr<OCRHMMDecoder> OCRHMMDecoder::create(const Ptr<OCRHMMDecoder::ClassifierCallback> classifier, const std::string& vocabulary, InputArray transition_probabilities_table, InputArray emission_probabilities_table, decoder_mode mode = OCR_DECODER_VITERBI)

    :param classifier: The character classifier with built in feature extractor.
    :param vocabulary: The language vocabulary (chars when ascii english text). vocabulary.size() must be equal to the number of classes of the classifier.
    :param transition_probabilities_table: Table with transition probabilities between character pairs. cols == rows == vocabulary.size().
    :param emission_probabilities_table: Table with observation emission probabilities. cols == rows == vocabulary.size().
    :param mode: HMM Decoding algorithm. Only ``OCR_DECODER_VITERBI`` is available for the moment (http://en.wikipedia.org/wiki/Viterbi_algorithm).

OCRHMMDecoder::run
------------------
Recognize text using HMM. Takes image on input and returns recognized text in the output_text parameter. Optionally provides also the Rects for individual text elements found (e.g. words), and the list of those text elements with their confidence values.

.. ocv:function:: void OCRHMMDecoder::run(Mat& image, string& output_text, vector<Rect>* component_rects=NULL, vector<string>* component_texts=NULL, vector<float>* component_confidences=NULL, int component_level=0)

    :param image: Input image ``CV_8UC1`` with a single text line (or word).
    :param output_text: Output text. Most likely character sequence found by the HMM decoder.
    :param component_rects: If provided the method will output a list of Rects for the individual text elements found (e.g. words).
    :param component_text: If provided the method will output a list of text strings for the recognition of individual text elements found (e.g. words).
    :param component_confidences: If provided the method will output a list of confidence values for the recognition of individual text elements found (e.g. words).
    :param component_level: Only ``OCR_LEVEL_WORD`` is supported.

loadOCRHMMClassifierNM
----------------------
Allow to implicitly load the default character classifier when creating an OCRHMMDecoder object.

.. ocv:function:: Ptr<OCRHMMDecoder::ClassifierCallback> loadOCRHMMClassifierNM(const std::string& filename)

    :param filename: The XML or YAML file with the classifier model (e.g. OCRHMM_knn_model_data.xml)

The default classifier is based in the scene text recognition method proposed by Lukás Neumann & Jiri Matas in [Neumann11b]. Basically, the region (contour) in the input image is normalized to a fixed size, while retaining the centroid and aspect ratio, in order to extract a feature vector based on gradient orientations along the chain-code of its perimeter. Then, the region is classified using a KNN model trained with synthetic data of rendered characters with different standard font types.

.. [Neumann11b] Neumann L., Matas J.: Text Localization in Real-world Images using Efficiently Pruned Exhaustive Search, ICDAR 2011. The paper is available online at http://cmp.felk.cvut.cz/~neumalu1/icdar2011_article.pdf
