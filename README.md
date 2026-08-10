# meqpy

[![Documentation Status](https://app.readthedocs.org/projects/meqpy/badge/?version=latest)](https://meqpy.readthedocs.io/en/latest/)

A code for solving master equations for STM

## Documentation

Documentation and tutorials are available at [meqpy.readthedocs.io](https://meqpy.readthedocs.io/en/latest/).

To build the docs locally:

```
pip install -r docs/requirements.txt
pip install -e .
cd docs && make html
```

then open `docs/build/html/index.html` in a browser.

## For developers

The package uses pre-commit hooks to check the style consistency of all commits.
To use those you need to first install the pre-commit package itself, e.g. with:

```
pip install .[dev]
```

and then install the pre-commit hooks with

```
pre-commit install
```

The pre-commit checks should now be automatically executed prior to each commit.

To run unit tests:

```
pytest -sv tests
```
