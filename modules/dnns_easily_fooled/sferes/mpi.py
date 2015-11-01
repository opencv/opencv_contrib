#! /usr/bin/env python
# encoding: utf-8
# JB Mouret - 2009

"""
Quick n dirty mpi detection
"""

import os, glob, types
import Options, Configure


def detect_mpi(conf):
	env = conf.env
	opt = Options.options

	conf.env['LIB_MPI'] = ''
	conf.env['MPI_FOUND'] = False
	if Options.options.no_mpi :
		return
	if Options.options.mpi:
		conf.env['CPPPATH_MPI'] = Options.options.mpi + '/include'
		conf.env['LIBPATH_MPI'] = Options.options.mpi + '/lib'
	else:
		conf.env['CPPPATH_MPI'] = ['/usr/include/mpi', '/usr/local/include/mpi', '/usr/include', '/usr/local/include']
		conf.env['LIBPATH_MPI'] = ['/usr/lib', '/usr/local/lib', '/usr/lib/openmpi']

	res = Configure.find_file('mpi.h', conf.env['CPPPATH_MPI'] )
	conf.check_message('header','mpi.h', (res != '') , res)
	if (res == '') :
		return 0
	conf.env['MPI_FOUND'] = True
	conf.env['LIB_MPI'] = ['mpi_cxx','mpi']
	return 1

def detect(conf):
	return detect_mpi(conf)

def set_options(opt):
	opt.add_option("--no-mpi",
		       default=False, action='store_true',
		       help='disable mpi', dest='no_mpi')
	opt.add_option('--mpi', type='string', help='path to mpi', dest='mpi')
