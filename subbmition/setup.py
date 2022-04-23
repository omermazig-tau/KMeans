from distutils.core import setup, Extension

"""
The old way of doing things, using distutils.
In addition, a minimalist setup is shown.
"""

setup(name='mykmeanssp',
      version='1.0',
      description='kmeans c api',
      ext_modules=[
          Extension('mykmeanssp', sources=['kmeans.c', 'kmeansApi.c'])
      ]
      )

setup(name='spkmeans_api',
      version='1.0',
      description='spkmeans c api',
      ext_modules=[
          Extension('spkmeans_api', sources=['kmeans.c', 'spkmeans.c', 'spkmeansmodule.c'])
      ]
      )
