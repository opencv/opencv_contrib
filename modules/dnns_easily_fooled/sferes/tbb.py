#! /usr/bin/env python
# encoding: utf-8
# JB Mouret - 2009

"""
Quick n dirty tbb detection
"""

import os, glob, types
import Options, Configure


def detect_tbb(conf):
	env = conf.env
	opt = Options.options

	if Options.options.no_tbb:
		print "TBB (multicore) is disabled"
		return 0
	if Options.options.tbb:
		conf.env['CPPPATH_TBB'] = [Options.options.tbb + '/include']
		conf.env['LIBPATH_TBB'] = [Options.options.tbb + '/lib']
	else:
		conf.env['CPPPATH_TBB'] = ['/opt/intel/include']
		conf.env['LIBPATH_TBB'] = ['/opt/intel/lib']

	res = Configure.find_file('tbb/parallel_for.h', conf.env['CPPPATH_TBB']+['/usr/include', '/usr/local/include'])
	conf.check_message('header','tbb/parallel_for.h', (res != '') , res)
	if (res == '') :
		return 0
	if Options.options.apple:
		res = Configure.find_file('libtbb.dylib', conf.env['LIBPATH_TBB'] + ['/usr/lib', '/usr/local/lib'])
	else:
		res = Configure.find_file('libtbb.so', conf.env['LIBPATH_TBB'] + ['/usr/lib', '/usr/local/lib'])

	conf.check_message('library','libtbb', (res != ''), res)
	if (res == '') :
		return 0
        conf.env['LIB_TBB'] = ['tbb']
	return 1

def detect(conf):
	tbb_found = detect_tbb(conf)
	if tbb_found != 0:
		conf.env['TBB_ENABLED'] = True
	else:
		conf.env['TBB_ENABLED'] = False
        	conf.env['LIB_TBB'] = []
	
	return detect_tbb(conf)

def set_options(opt):
	opt.add_option('--tbb', type='string', help='path to tbb', dest='tbb')
	opt.add_option('--no-tbb',  default=False, action='store_true',
		       help='disable tbb (multicore)', dest='no_tbb')
