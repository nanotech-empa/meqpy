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

## Software attribution and scientific acknowledgements

Please credit [meqpy](https://github.com/nanotech-empa/meqpy) and its
[contributors](https://github.com/nanotech-empa/meqpy/graphs/contributors),
and identify the release or immutable commit used. Existing author,
contributor and funding acknowledgements remain authoritative.

See [CITATION.cff](CITATION.cff) for the software citation and identify the
exact release or commit used.

Preserve the copyright and permission notices in [LICENSE](LICENSE)
when copying or distributing copies or substantial portions of MIT-licensed
code. Separately licensed components retain their own notices. A name or
repository link does not replace the applicable licence notices.

The package metadata credits Andres Ortega-Guerrero, Gonçalo Catarina and Nils Krane. Preserve their authorship and cite the master-equation and tunnelling methods actually used. Standalone meqpy use does not imply use of AiiDA or AiiDAlab.

The [shared credit guide](https://github.com/nanotech-empa/.github/blob/main/CITATION_GUIDANCE.md) keeps software attribution, licence
compliance and scientific citation requests separate. Scientific citation
requests are not additional MIT licence conditions.
