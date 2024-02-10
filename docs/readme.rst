========
Overview
========
Finance TDA [v-2024.2.3]


.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests

.. |docs| image:: https://readthedocs.org/projects/finance-tda/badge/?style=flat
    :target: https://finance-tda.readthedocs.io/
    :alt: Documentation Status

.. |commits-since| image:: https://img.shields.io/github/commits-since/ibaris/finance-tda/v2024.2.3.svg
    :alt: Commits since latest release
    :target: https://github.com/ibaris/finance-tda/compare/v2024.2.3...main


.. end-badges

An example package. Generated with cookiecutter-pylibrary.

Installation
============

::

    pip install fintda

You can also install the in-development version with::

    pip install https://github.com/ibaris/finance-tda/archive/main.zip


Documentation
=============


https://finance-tda.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
