#!/usr/bin/env python3
# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.

# Copyright 2026, The FlagOS Contributors.
"""Check module sources and generated-file consistency without importing kernels.

Reads this module, checks Python syntax/notices and both generators, and compiles
CPU controls in a temporary directory with --cxx. Prints check results; exits 0
on success, nonzero on a failed control. Does not build OpenCV or use a GPU.
"""
import argparse
import ast
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default=os.environ.get('CXX', 'c++'),
                        help='C++17 compiler command, default CXX or c++')
    args = parser.parse_args()
    module = Path(__file__).resolve().parents[1]
    for path in sorted(module.rglob('*')):
        if path.suffix not in ('.cpp', '.hpp', '.py') or '__pycache__' in path.parts:
            continue
        source = path.read_text(encoding='utf-8')
        if ('license terms in the LICENSE file' not in source[:1300] or
                'http://opencv.org/license.html' not in source[:1300]):
            raise RuntimeError('missing project notice: ' + str(path.relative_to(module)))
        if path.suffix == '.py':
            ast.parse(source, filename=str(path))  # Do not import kernels or compiler.
    for generator in ('make_kernels_cpu_hip.py', 'port_hip_runtime.py'):
        subprocess.run([sys.executable, '-B', str(module / 'tools' / generator), '--check'], check=True)
    runtime = module / 'src' / 'runtime'
    with tempfile.TemporaryDirectory(prefix='opencv-triton-source-') as temporary:
        binary = Path(temporary) / 'protocol-check'
        command = shlex.split(args.cxx) + ['-std=c++17', '-O2', '-ffp-contract=off',
                   '-Wall', '-Wextra', '-Werror', '-I' + str(runtime),
                   str(module / 'test' / 'cpu' / 'test_protocol.cpp')]
        command += [str(runtime / name) for name in
                    ('resize_plan.cpp', 'provider.cpp', 'compile_protocol.cpp')]
        subprocess.run(command + ['-o', str(binary)], check=True)
        subprocess.run([str(binary), sys.executable, temporary], check=True)
        lifetime = Path(temporary) / 'lifetime-check'
        subprocess.run(shlex.split(args.cxx) + ['-std=c++17', '-O2', '-Wall', '-Wextra',
                       '-Werror', '-pthread', '-I' + str(runtime),
                       str(module / 'test' / 'cpu' / 'test_lifetime.cpp'), '-o', str(lifetime)], check=True)
        subprocess.run([str(lifetime)], check=True)
        subprocess.run([sys.executable, '-B', str(module / 'test' / 'cpu' / 'check_native_lifetime.py'),
                        '--cxx', args.cxx], check=True)
        graph = Path(temporary) / 'graph-request-check'
        subprocess.run(shlex.split(args.cxx) + ['-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                       '-I' + str(runtime), str(module / 'test' / 'cpu' / 'test_graph_request.cpp')]
                       + [str(runtime / name) for name in ('resize_plan.cpp', 'provider.cpp', 'compile_protocol.cpp')]
                       + ['-o', str(graph)], check=True)
        subprocess.run([str(graph)], check=True)
    print('PASS: source notices, Python syntax, HIP derivations and CPU protocol controls')
    print('Not checked: enabled OpenCV builds, GPU correctness/performance, installation or compiler JIT')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, RuntimeError, SyntaxError, subprocess.CalledProcessError) as error:
        print('FAIL:', error, file=sys.stderr)
        sys.exit(1)
