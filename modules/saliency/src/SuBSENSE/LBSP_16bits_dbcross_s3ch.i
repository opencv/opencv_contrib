// note: this is the LBSP 16 bit double-cross indiv RGB pattern as used in
// the original article by G.-A. Bilodeau et al.
// 
//  O   O   O          4 ..  3 ..  6
//    O O O           .. 15  8 13 ..
//  O O X O O    =>    0  9  X 11  1
//    O O O           .. 12 10 14 ..
//  O   O   O          7 ..  2 ..  5
//          (single/3x)            (single/3x)
//
// must be defined externally:
//		_t				(size_t, absolute threshold used for comparisons)
//		_ref			(uchar, 'central' value used for comparisons)
//		_data			(uchar*, triple-channel data to be covered by the pattern)
//		_y				(int, pattern rows location in the image data)
//		_x				(int, pattern cols location in the image data)
//		_c				(size_t, pattern channel location in the image data)
//		_step_row		(size_t, step size between rows, including padding)
//		_res			(ushort, 16 bit result vector)
//		absdiff_uchar	(function, returns the absolute difference between two uchars)

#ifdef _val
#error "definitions clash detected"
#else
#define _val(x,y,n) _data[_step_row*(_y+y)+3*(_x+x)+n]
#endif

_res = ((absdiff_uchar(_val(-1, 1, _c),_ref) > _t) << 15)
	 + ((absdiff_uchar(_val( 1,-1, _c),_ref) > _t) << 14)
	 + ((absdiff_uchar(_val( 1, 1, _c),_ref) > _t) << 13)
	 + ((absdiff_uchar(_val(-1,-1, _c),_ref) > _t) << 12)
	 + ((absdiff_uchar(_val( 1, 0, _c),_ref) > _t) << 11)
	 + ((absdiff_uchar(_val( 0,-1, _c),_ref) > _t) << 10)
	 + ((absdiff_uchar(_val(-1, 0, _c),_ref) > _t) << 9)
	 + ((absdiff_uchar(_val( 0, 1, _c),_ref) > _t) << 8)
	 + ((absdiff_uchar(_val(-2,-2, _c),_ref) > _t) << 7)
	 + ((absdiff_uchar(_val( 2, 2, _c),_ref) > _t) << 6)
	 + ((absdiff_uchar(_val( 2,-2, _c),_ref) > _t) << 5)
	 + ((absdiff_uchar(_val(-2, 2, _c),_ref) > _t) << 4)
	 + ((absdiff_uchar(_val( 0, 2, _c),_ref) > _t) << 3)
	 + ((absdiff_uchar(_val( 0,-2, _c),_ref) > _t) << 2)
	 + ((absdiff_uchar(_val( 2, 0, _c),_ref) > _t) << 1)
	 + ((absdiff_uchar(_val(-2, 0, _c),_ref) > _t));

#undef _val
		