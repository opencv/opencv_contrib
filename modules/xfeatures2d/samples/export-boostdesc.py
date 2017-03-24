#!/usr/bin/python

"""

/*********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright (c) 2016
 *
 * Balint Cristian <cristian dot balint at gmail dot com>
 *
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holders nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

/* export-boostdesc.py */
/* Export C headers from binary data */
// [http://infoscience.epfl.ch/record/186246/files/boostDesc_1.0.tar.gz]

"""

import sys
import struct



def float_to_hex(f):
  return struct.unpack( '<I', struct.pack('<f', f) )[0]

def main():

  # usage
  if ( len(sys.argv) < 3 ):
    print( "Usage: %s <binary-type (BGM, LBGM, BINBOOST)> <boostdesc-binary-filename>" % sys.argv[0] )
    sys.exit(0)


  if ( ( sys.argv[1] != "BGM" ) and
       ( sys.argv[1] != "LBGM" ) and
       ( sys.argv[1] != "BINBOOST" ) ):
    print( "Invalid type [%s]" % sys.argv[1] )
    sys.exit(0)

  # enum literals
  Assign = [ "ASSIGN_HARD",
             "ASSIGN_BILINEAR",
             "ASSIGN_SOFT",
             "ASSIGN_HARD_MAGN",
             "ASSIGN_SOFT_MAGN" ]

  # open binary data file
  f = open( sys.argv[2], 'rb' )

  # header
  print "/*"
  print " *"
  print " * Header exported from binary."
  print " * [%s %s %s]" % ( sys.argv[0], sys.argv[1], sys.argv[2] )
  print " *"
  print " */"

  # ini
  nDim = 1;
  nWLs = 0;

  # dimensionality (where is the case)
  if ( ( sys.argv[1] == "LBGM" ) or
       ( sys.argv[1] == "BINBOOST" ) ):
    nDim = struct.unpack( '<i', f.read(4) )[0]

  print
  print "// dimensionality of learner"
  print "static const int nDim = %i;" % nDim

  # week learners (where is the case)
  if ( sys.argv[1] != "BINBOOST" ):
    nWLs = struct.unpack( '<i', f.read(4) )[0]

  # common header
  orientQuant     = struct.unpack( '<i', f.read(4) )[0]
  patchSize       = struct.unpack( '<i', f.read(4) )[0]
  iGradAssignType = struct.unpack( '<i', f.read(4) )[0]

  print
  print "// orientations"
  print "static const int orientQuant = %i;" % orientQuant

  print
  print "// patch size"
  print "static const int patchSize = %i;" % patchSize

  print
  print "// gradient assignment type"
  print "static const int iGradAssignType = %s;" % Assign[iGradAssignType]

  arr_thresh = ""
  arr_orient = ""

  arr__y_min = ""
  arr__y_max = ""
  arr__x_min = ""
  arr__x_max = ""

  arr__alpha = ""
  arr___beta = ""


  dims = nDim
  if ( sys.argv[1] == "LBGM" ):
    dims = 1

  # iterate each dimension
  for d in range( 0, dims ):

    if ( sys.argv[1] == "BINBOOST" ):
      nWLs   = struct.unpack( '<i', f.read(4) )[0]

    if ( d == 0 ):
      print
      print "// number of weak learners"
      print "static const int nWLs = %i;" % nWLs

    # iterate each members
    for i in range( 0, nWLs ):

      # unpack structure array
      thresh = struct.unpack( '<f', f.read(4) )[0]
      orient = struct.unpack( '<i', f.read(4) )[0]

      y_min  = struct.unpack( '<i', f.read(4) )[0]
      y_max  = struct.unpack( '<i', f.read(4) )[0]
      x_min  = struct.unpack( '<i', f.read(4) )[0]
      x_max  = struct.unpack( '<i', f.read(4) )[0]

      alpha  = struct.unpack( '<f', f.read(4) )[0]

      beta = 0
      if ( sys.argv[1] == "BINBOOST" ):
        beta = struct.unpack( '<f', f.read(4) )[0]

      # first entry
      if ( d*dims + i == 0 ):

        arr_thresh += "\n"
        arr_thresh += "// threshold array (%s x %s)\n" % (dims,nWLs)
        arr_thresh += "static const unsigned int thresh[] =\n{\n"

        arr_orient += "\n"
        arr_orient += "// orientation array (%s x %s)\n" % (dims,nWLs)
        arr_orient += "static const int orient[] =\n{\n"

        arr__y_min += "\n"
        arr__y_min += "// Y min array (%s x %s)\n" % (dims,nWLs)
        arr__y_min += "static const int y_min[] =\n{\n"

        arr__y_max += "\n"
        arr__y_max += "// Y max array (%s x %s)\n" % (dims,nWLs)
        arr__y_max += "static const int y_max[] =\n{\n"

        arr__x_min += "\n"
        arr__x_min += "// X min array (%s x %s)\n" % (dims,nWLs)
        arr__x_min += "static const int x_min[] =\n{\n"

        arr__x_max += "\n"
        arr__x_max += "// X max array (%s x %s)\n" % (dims,nWLs)
        arr__x_max += "static const int x_max[] =\n{\n"

        arr__alpha += "\n"
        arr__alpha += "// alpha array (%s x %s)\n" % (dims,nWLs)
        arr__alpha += "static const unsigned int alpha[] =\n{\n"

        if ( sys.argv[1] == "BINBOOST" ):
          arr___beta += "\n"
          arr___beta += "// beta array (%s x %s)\n" % (dims,nWLs)
          arr___beta += "static const unsigned int beta[] =\n{\n"

      # last entry
      if ( i == nWLs - 1 ) and ( d == dims - 1):

        arr_thresh += " 0x%08x\n};" % float_to_hex(thresh)
        arr_orient += " 0x%02x\n};" % orient

        arr__y_min += " 0x%02x\n};" % y_min
        arr__y_max += " 0x%02x\n};" % y_max
        arr__x_min += " 0x%02x\n};" % x_min
        arr__x_max += " 0x%02x\n};" % x_max

        arr__alpha += " 0x%08x\n};" % float_to_hex(alpha)

        if ( sys.argv[1] == "BINBOOST" ):
          arr___beta += " 0x%08x\n};" % float_to_hex(beta)

        break

      # align entries
      if ( (d*dims + i + 1) % 8 ):

        arr_thresh += " 0x%08x," % float_to_hex(thresh)
        arr_orient += " 0x%02x," % orient

        arr__y_min += " 0x%02x," % y_min
        arr__y_max += " 0x%02x," % y_max
        arr__x_min += " 0x%02x," % x_min
        arr__x_max += " 0x%02x," % x_max

        arr__alpha += " 0x%08x," % float_to_hex(alpha)

        if ( sys.argv[1] == "BINBOOST" ):
          arr___beta += " 0x%08x," % float_to_hex(beta)

      else:

        arr_thresh += " 0x%08x,\n" % float_to_hex(thresh)
        arr_orient += " 0x%02x,\n" % orient

        arr__y_min += " 0x%02x,\n" % y_min
        arr__y_max += " 0x%02x,\n" % y_max
        arr__x_min += " 0x%02x,\n" % x_min
        arr__x_max += " 0x%02x,\n" % x_max

        arr__alpha += " 0x%08x,\n" % float_to_hex(alpha)

        if ( sys.argv[1] == "BINBOOST" ):
          arr___beta += " 0x%08x,\n" % float_to_hex(beta)

  # extra array (when LBGM)
  if ( sys.argv[1] == "LBGM" ):

    arr___beta += "\n"
    arr___beta += "// beta array (%s x %s)\n" % (nWLs,nDim)
    arr___beta += "static const unsigned int beta[] =\n{\n"

    for i in range( 0, nWLs ):
      for d in range( 0, nDim ):
        beta = struct.unpack( '<f', f.read(4) )[0]

        # last entry
        if ( i == nWLs-1 ) and ( d == nDim-1 ):
          arr___beta += " 0x%08x\n};" % float_to_hex(beta)
          break

        # align entries
        if ( (i*nDim + d + 1) % 8 ):
          arr___beta += " 0x%08x," % float_to_hex(beta)
        else:
          arr___beta += " 0x%08x,\n" % float_to_hex(beta)

  # release
  f.close()

  # dump on screen
  print arr_thresh
  print arr_orient

  print arr__y_min
  print arr__y_max
  print arr__x_min
  print arr__x_max

  print arr__alpha

  if ( ( sys.argv[1] == "LBGM" ) or
       ( sys.argv[1] == "BINBOOST" ) ):
    print arr___beta


if __name__ == "__main__":
    main()
