#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from os.path import join as pjoin
import warnings
import glob
from setuptools import setup, Extension, find_packages
from distutils.extension import Extension
from distutils.command.build_ext import build_ext
import subprocess
import numpy
import sys
import shutil

##########################################################

# Set Python package requirements for installation.   

install_requires = [
    'numpy>=1.7.0',
    'scipy>=0.12.0',
]

def find_in_path(name, path):
    "Find a file in a search path"
    #adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

# enforce these same requirements at packaging time
import pkg_resources
for requirement in install_requires:
    try:
        pkg_resources.require(requirement)
    except pkg_resources.DistributionNotFound:
        msg = 'Python package requirement not satisfied: ' + requirement
        msg += '\nsuggest using this command:'
        msg += '\n\tpip install -U ' + requirement.split('=')[0].rstrip('>')
        print (msg)
        raise (pkg_resources.DistributionNotFound)


def customize_compiler(self):
    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile

###########################
# Main setup configuration.

# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler(self.compiler)
        build_ext.build_extensions(self)

    
setup(
    name='sscResolution',
    version = open('VERSION').read().strip(),
        
    packages = find_packages(),
    include_package_data = True,
    
    cmdclass={'build_ext': custom_build_ext},
    
    # since the package has c code, the egg cannot be zipped
    zip_safe=False,    
    
    author='Eduardo X. Miqueles',
    author_email='eduardo.miqueles@lnls.br',
    
    description='Fourier correlation routines',
    keywords=['fourier', 'resolution', 'imaging', 'lnls','sirius'],
    url='http://www.',
    download_url='',
    
    license='BSD',
    platforms='Any',
    install_requires = install_requires,
    
    classifiers=['Development Status :: 4 - Beta',
                 'License :: OSI Approved :: BSD License',
                 'Intended Audience :: Science/Research',
                 'Intended Audience :: Education',
                 'Intended Audience :: Developers',
                 'Natural Language :: English',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3.0',
                 'Programming Language :: C',
                 'Programming Language :: C++']
    
)
