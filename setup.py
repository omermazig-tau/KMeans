from distutils.core import setup, Extension

"""
The old way of doing things, using distutils.
In addition, a minimalist setup is shown.
"""


setup(name='kmeans_c_api',
      version='1.0',
      description='kmeans_c_api for sp class',
      ext_modules=[Extension('kmeans_c_api', sources=['cPart/kmeansApi.c'])])
