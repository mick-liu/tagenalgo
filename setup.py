
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tagenalgo",
    version="1.0.7",
    author="Mick Liu",
    author_email="mickey70636@gmail.com",
    description="A parameter optimization algorithm for trading strategy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mick-liu/tagenalgo",
    license='BSD',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
       'TA-Lib',
       'joblib'],
)