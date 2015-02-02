The Chars74K Dataset
====================
.. ocv:class:: TR_chars

Implements loading dataset:

_`"The Chars74K Dataset"`: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/

.. note:: Usage

 1. From link above download dataset files: EnglishFnt\\EnglishHnd\\EnglishImg\\KannadaHnd\\KannadaImg.tgz, ListsTXT.tgz.

 2. Unpack them.

 3. Move .m files from folder ListsTXT/ to appropriate folder. For example, English/list_English_Img.m for EnglishImg.tgz.

 4. To load data, for example "EnglishImg", run: ./opencv/build/bin/example_datasets_tr_chars -p=/home/user/path_to_unpacked_folder/English/

**References:**

.. [Campos09] T. E. de Campos, B. R. Babu and M. Varma. Character recognition in natural images. In Proceedings of the International Conference on Computer Vision Theory and Applications (VISAPP), 2009

