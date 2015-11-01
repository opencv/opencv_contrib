#! /usr/bin/env python
# encoding: utf-8
# JB Mouret - 2009

"""
Quick n dirty eigen3 detection
"""

import os, glob, types
import Options, Configure

def detect_eigen3(conf):
	env = conf.env
	opt = Options.options

	conf.env['LIB_EIGEN3'] = ''
	conf.env['EIGEN3_FOUND'] = False
	if Options.options.no_eigen3:
		return 0
	if Options.options.eigen3:
		conf.env['CPPPATH_EIGEN3'] = [Options.options.eigen3]
		conf.env['LIBPATH_EIGEN3'] = [Options.options.eigen3]
	else:
		conf.env['CPPPATH_EIGEN3'] = ['/usr/include/eigen3', '/usr/local/include/eigen3', '/usr/include', '/usr/local/include']
		conf.env['LIBPATH_EIGEN3'] = ['/usr/lib', '/usr/local/lib']

	res = Configure.find_file('Eigen/Core', conf.env['CPPPATH_EIGEN3'])
	conf.check_message('header','Eigen/Core', (res != '') , res)
	if (res == '') :
		return 0
	conf.env['EIGEN3_FOUND'] = True
	return 1

def detect(conf):
	return detect_eigen3(conf)

def set_options(opt):
	opt.add_option('--eigen3', type='string', help='path to eigen3', dest='eigen3')
	opt.add_option('--no-eigen3', type='string', help='disable eigen3', dest='no_eigen3')
