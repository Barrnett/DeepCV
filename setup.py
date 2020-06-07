import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepray",
    version="0.1.1",
    author="Hailin Fu",
    author_email="hailinfufu@outlook.com",
    description="A new Modular, Scalable, Configurable, Easy-to-Use and Extend infrastructure for Deep Learning based Recommendation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fuhailin/deepray",
    packages=setuptools.find_packages(),
    classifiers=(
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ),
    license="Apache-2.0",
    keywords=['recommendation', 'deep learning', 'tensorflow2', 'keras'],
    python_requires='>=3.6',
)
