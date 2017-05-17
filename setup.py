#!/usr/bin/env python

from setuptools import setup,find_packages

setup(name='dmap',
    version='0.0.1',
    description="Diffusion Map Analysis Code",
    license='GPL',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Scientific/Engineering'
        ],
#    packages=find_packages(),
    packages=['dmap'],
    install_requires=['numpy','scipy'],

    )
