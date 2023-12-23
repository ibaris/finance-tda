<div align="center">
    <p>
        <img src="./data/logo/logo.png">
    </p>

<h2 align="center">Finance TDA</h4>
<h4 align="center">An example package. Generated with cookiecutter-pylibrary.</h4>
<h5 align="center">[v-2023.12.23]</h5>

<p align="center">
  <a href="#introduction">Introduction</a> •
  <a href="#installation">Installation</a> •
  <a href="#installation">Documentation</a> •
  <a href="#installation">Development</a>
</p>
</div>

# Overview

An example package. Generated with cookiecutter-pylibrary.


# Installation

```cmd
> pip install financetda
```

You can also install the stable version with

```cmd

>>> pip install https://github.com/ibaris/finance-tda/archive/main.zip

```

To install the in-development version, change the branch name main to the other
available branch names.

# Documentation

The documentation `code` documentation is in `build/docs`.

# Development

To run all the tests and to build the `code` documentation run

```cmd
>>> tox
```

Note, to combine the coverage data from all the tox environments run:

```cmd
>>> set PYTEST_ADDOPTS=--cov-append
>>> tox
```

for Windows and

```cmd
>>> PYTEST_ADDOPTS=--cov-append tox
```

for Linux.
