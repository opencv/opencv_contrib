
#! /usr/bin/env python
# encoding: utf-8
# JB Mouret - 2009

"""
Quick n dirty ODE detection
"""

import os, glob, types
import Options, Configure, config_c

import commands

def detect_ode(conf):
    env = conf.env
    opt = Options.options
    ret = conf.find_program('ode-config')
    conf.check_message_1('Checking for ODE (optional)')
    if not ret:
        conf.check_message_2('not found', 'YELLOW')
        return 0
    conf.check_message_2('ok')
    res = commands.getoutput('ode-config --cflags --libs')
    config_c.parse_flags(res, 'ODE', env)
    return 1

def detect(conf):
    return detect_ode(conf)

def set_options(opt):
    pass
