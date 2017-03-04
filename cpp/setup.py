#!/usr/bin/env python

from distutils.core import setup, Extension
import os
import subprocess

os.environ['CXX'] = '/usr/bin/g++'
os.environ['CC'] = '/usr/bin/g++'
cwd = os.path.dirname(os.path.realpath(__file__))

solver_module = Extension('_solver',
                        sources = [cwd+'/solver.i', cwd+'/solver.cpp'],
                        include_dirs = [cwd+'/Eigen/'],
                        swig_opts = ['-c++','-I '+cwd+'/solver.i'],
                        extra_compile_args = ['-Wall','-O2','-pg','-std=c++11', '-fopenmp', '-ftree-vectorize','-fPIC'],
                        extra_link_args = ['-lrt','-lgomp'] #required for older glibc versions
                           )

setup (name = 'Solver',
       version = '0.1',
       author      = "Abhijit Sharang",
       ext_modules = [solver_module],
       py_modules = ["solver"],
       )

