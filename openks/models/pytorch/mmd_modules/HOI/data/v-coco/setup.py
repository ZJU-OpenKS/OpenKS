# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE_fast_rcnn for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

cmdclass = {}
ext_modules = [
    Extension(
        "cython_bbox",
        ["bbox.pyx"],
        extra_compile_args=["-Wno-cpp", "-Wno-unused-function"],
    )
]
cmdclass.update({'build_ext': build_ext})

setup(
    name='fast_rcnn',
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    include_dirs=[np.get_include()]
)
