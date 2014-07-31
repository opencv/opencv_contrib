Scene Text Recognition
======================

.. highlight:: cpp

OCRTesseract
------------
.. ocv:class:: OCRTesseract

OCRTesseract class provides an interface with the tesseract-ocr API (v3.02.02) in C++. Notice that it is compiled only when tesseract-ocr is correctly installed. ::

    class CV_EXPORTS OCRTesseract
    {
    private:
        tesseract::TessBaseAPI tess;
    
    public:
        //! Default constructor
        OCRTesseract(const char* datapath=NULL, const char* language=NULL, const char* char_whitelist=NULL,
                     tesseract::OcrEngineMode oem=tesseract::OEM_DEFAULT, tesseract::PageSegMode psmode=tesseract::PSM_AUTO);
    
        ~OCRTesseract();
    
        /*!
        the key method. Takes image on input and returns recognized text in the output_text parameter
        optionally provides also the Rects for individual text elements (e.g. words) and a list of 
        ranked recognition alternatives.
        */
        void run(Mat& image, string& output_text, vector<Rect>* component_rects=NULL,
                 vector<string>* component_texts=NULL, vector<float>* component_confidences=NULL,
                 int component_level=0);
    };

To see the OCRTesseract combined with scene text detection, have a look at the end_to_end_recognition demo: https://github.com/Itseez/opencv_contrib/blob/master/modules/text/samples/end_to_end_recognition.cpp

OCRTesseract::OCRTesseract
--------------------------
Constructor.

.. ocv:function:: void OCRTesseract::OCRTesseract(const char* datapath=NULL, const char* language=NULL, const char* char_whitelist=NULL, tesseract::OcrEngineMode oem=tesseract::OEM_DEFAULT, tesseract::PageSegMode psmode=tesseract::PSM_AUTO);

    :param datapath: the name of the parent directory of tessdata ended with "/", or NULL to use the system's default directory.
    :param language: an ISO 639-3 code or NULL will default to "eng".
    :param char_whitelist: specifies the list of characters used for recognition. NULL defaults to "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ".
    :param oem: tesseract-ocr offers different OCR Engine Modes (OEM), by deffault tesseract::OEM_DEFAULT is used. See the tesseract-ocr API documentation for other possible values.
    :param psmode: tesseract-ocr offers different Page Segmentation Modes (PSM) tesseract::PSM_AUTO (fully automatic layout analysis) is used. See the tesseract-ocr API documentation for other possible values.

OCRTesseract::run
-----------------
Recognize text using the tesseract-ocr API. Takes image on input and returns recognized text in the output_text parameter. Optionally provides also the Rects for individual text elements found (e.g. words), and the list of those text elements with their confidence values.

.. ocv:function:: void OCRTesseract::run(Mat& image, string& output_text, vector<Rect>* component_rects=NULL, vector<string>* component_texts=NULL, vector<float>* component_confidences=NULL, int component_level=0);

    :param image: Input image ``CV_8UC1`` or ``CV_8UC3``
    :param output_text: Output text of the tesseract-ocr.
    :param component_rects: If provided the method will output a list of Rects for the individual text elements found (e.g. words or text lines).
    :param component_text: If provided the method will output a list of text strings for the recognition of individual text elements found (e.g. words or text lines).
    :param component_confidences: If provided the method will output a list of confidence values for the recognition of individual text elements found (e.g. words or text lines).
    :param component_level: ``OCR_LEVEL_WORD`` (by default), or ``OCR_LEVEL_TEXT_LINE``.
