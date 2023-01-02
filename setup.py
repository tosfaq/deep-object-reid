# Copyright (c) 2018-2021 Kaiyang Zhou
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os.path as osp
from setuptools import setup, Extension, find_packages

from glob import glob

import numpy as np

repo_root = osp.dirname(osp.realpath(__file__))

def readme():
    with open(osp.join(repo_root, 'README.rst')) as f:
        content = f.read()
    return content


def find_version():
    version_file = osp.join(repo_root, 'torchreid/version.py')
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def numpy_include():
    try:
        numpy_include = np.get_include()
    except AttributeError:
        numpy_include = np.get_numpy_include()
    return numpy_include


def get_requirements(filename):
    requires = []
    links = []
    with open(osp.join(repo_root, filename), 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            if '-f http' in line:
                links.append(line)
            else:
                requires.append(line)
    return requires, links

req, links = get_requirements('requirements.txt')

setup(
    name='otxreid',
    # version=find_version(),
    version='0.3.1',
    description='A library for deep learning object re-ID and classification in PyTorch',
    author='Kaiyang Zhou, Intel Corporation',
    license='Apache-2.0',
    long_description=readme(),
    url='https://github.com/openvinotoolkit/deep-object-reid',
    dependency_links=links,
    packages=find_packages(include=('torchreid', 'torchreid.*', 'scripts', 'scripts.*')),
    include_package_data=True,
    install_requires=req,
    keywords=['Object Re-Identification', 'Image Classification', 'Deep Learning', 'Computer Vision'],
)
