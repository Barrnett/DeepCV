#!/usr/bin/env python
# -*- coding:utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="deepcv",
    version="1.0.0",
    keywords=("pip", "pathtool", "timetool", "magetool", "mage"),
    description="Computer vision related algorithms",
    long_description="time and path tool",
    license="MIT Licence",

    url="https://github.com/Barrnett/DeepCV",
    author="zhuwenwen",
    author_email="stephenbarrnet@outlook.com",

    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    install_requires=[]
)
