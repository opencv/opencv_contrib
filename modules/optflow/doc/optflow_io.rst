Optical flow input / output
============================

Functions reading and writing .flo files in "Middlebury" format, see: [MiddleburyFlo]_ 

readOpticalFlow
-----------------
Read a .flo file

.. ocv:function:: Mat readOpticalFlow( const String& path )
    
    :param path: Path to the file to be loaded
    
The function readOpticalFlow loads a flow field from a file and returns it as a single matrix.
Resulting ``Mat`` has a type ``CV_32FC2`` - floating-point, 2-channel. 
First channel corresponds to the flow in the horizontal direction (u), second - vertical (v).

writeOpticalFlow
-----------------
Write a .flo to disk

.. ocv:function:: bool writeOpticalFlow( const String& path, InputArray flow )

    :param path: Path to the file to be written
    
    :param flow: Flow field to be stored
    
 The function stores a flow field in a file, returns ``true`` on success, ``false`` otherwise.

 The flow field must be a 2-channel, floating-point matrix (``CV_32FC2``). 
 First channel corresponds to the flow in the horizontal direction (u), second - vertical (v).
 
 
.. [MiddleburyFlo] http://vision.middlebury.edu/flow/code/flow-code/README.txt
 
 