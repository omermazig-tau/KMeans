from distutils.core import setup, Extension
import os

"""
The old way of doing things, using distutils.
In addition, a minimalist setup is shown.
"""

setup(name='mykmeanssp',
      version='1.0',
      description='kmeans c api for sp class',
      ext_modules=[Extension('mykmeanssp',
                             sources=[os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                                                   'CPart', 'kmeansApi.c')])])
