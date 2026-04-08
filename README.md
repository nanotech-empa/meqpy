# meqpy
A code for solving master equations for STM


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
