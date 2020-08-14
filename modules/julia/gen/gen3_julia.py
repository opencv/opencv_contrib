#!/usr/bin/env python

# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html
# Copyright (C) 2020 by Archit Rungta


from __future__ import unicode_literals # Needed for python2

import hdr_parser, sys, re, os
from string import Template
from pprint import pprint
from collections import namedtuple
if sys.version_info[0] >= 3:
    from io import StringIO
else:
    from cStringIO import StringIO
import os, shutil

from parse_tree import *



submodule_template = Template('')
root_template = Template('')
with open("binding_templates_jl/template_cv2_submodule.jl", "r") as f:
    submodule_template = Template(f.read())
with open("binding_templates_jl/template_cv2_root.jl", "r") as f:
    root_template = Template(f.read())


class FuncVariant(FuncVariant):

    def get_complete_code(self, classname='', isalgo = False, iscons = False, gen_default = True, ns = ''):
        return 'const %s = OpenCV.%s_%s' %(self.mapped_name, ns, self.mapped_name)


def gen(srcfiles):
    namespaces, _ = gen_tree(srcfiles)

    jl_code = StringIO()
    for name, ns in namespaces.items():
        # cv_types.extend(ns.registered)
        jl_code = StringIO()
        nsname = '_'.join(name.split('::')[1:])

        # Do not duplicate functions. This should prevent overwriting of Mat function by UMat functions
        function_signatures = []
        if name != 'cv':
            for cname, cl in ns.classes.items():
                cl.__class__ = ClassInfo
                for mname, fs in cl.methods.items():
                    for f in fs:
                        f.__class__ = FuncVariant
                        if f.mapped_name in function_signatures:
                            print("Skipping entirely: ", f.name)
                            continue
                        jl_code.write('\n%s'  % f.get_complete_code(isalgo = cl.isalgorithm, ns=nsname))
                        function_signatures.append(f.mapped_name)
                for f in cl.constructors:
                    f.__class__ = FuncVariant
                    jl_code.write('\n%s'  % f.get_complete_code(classname = cl.mapped_name, isalgo = cl.isalgorithm, iscons = True, ns=nsname))
                    break
            for mname, fs in ns.funcs.items():
                for f in fs:
                    f.__class__ = FuncVariant
                    if f.mapped_name in function_signatures:
                        continue
                    jl_code.write('\n%s'  % f.get_complete_code(ns=nsname))
                    function_signatures.append(f.mapped_name)
        jl_code.write('\n')
        for mapname, cname in sorted(ns.consts.items()):
            jl_code.write('    const %s = OpenCV.%s_%s\n'%(cname, name.replace('::', '_'), cname))
            compat_name = re.sub(r"([a-z])([A-Z])", r"\1_\2", cname).upper()
            if cname != compat_name:
                jl_code.write('    const %s = OpenCV.%s_%s;\n'%(compat_name, name.replace('::', '_'), compat_name))

        imports = ''
        for namex in namespaces:
            if namex.startswith(name) and len(namex.split('::')) == 1 + len(name.split('::')):
                imports = imports + '\ninclude("%s_wrap.jl")'%namex.replace('::', '_')
        code = ''
        if name == 'cv':
            code = root_template.substitute(modname = name, code = jl_code.getvalue(), submodule_imports = imports)
        else:
            code = submodule_template.substitute(modname = name.split('::')[-1], code = jl_code.getvalue(), submodule_imports = imports)

        with open ('autogen_jl/%s_wrap.jl' % ns.name.replace('::', '_'), 'w') as fd:
            fd.write(code)



srcfiles = hdr_parser.opencv_hdr_list
if len(sys.argv) > 1:
    srcfiles = [l.strip() for l in sys.argv[1].split(';')]


gen(srcfiles)
