# Contributing guide

Do you want to add new metrics, models, or other features to **seqme**?

This document aims at summarizing the most important information for getting you started on contributing to this project. We assume that you are already familar with git and with making pull requests on GitHub.

## Installing dev dependencies

In addition to the required dependencies needed to use this package, you need additional python packages to run tests and build the documentation. It's easy to set up the package for development using pip:

```shell
cd seqme
pip install -e ".[dev,doc]"
```

## Code-style

We use [`pre-commit`](#tooling) to enforce consistent code-styles. On every commit, pre-commit checks will either automatically fix issues with the code, or raise an error message.

To enable pre-commit locally, simply run

```bash
pre-commit install
```

in the root of the repository. Pre-commit will automatically download all dependencies when it is run for the first time.

To run pre-commit without calling `git commit`, do

```bash
pre-commit
```

## Writing tests

This package uses [`pytest`](#tooling) for automated testing. Please write tests in the [tests](https://github.com/szczurek-lab/seqme/tree/main/tests) directory for every function added to the package.

Most IDEs integrate with pytest and provide a GUI to run tests. Alternatively, you can run all tests from the
command line by executing

```bash
pytest
```

in the root of the repository.

Each file in the test directory has to follow the naming convention: `tests/test_foo.py`

```python
def helper(): # pytest does not check this function
    ...
def test_bar(): # pytest recognizes it as a test
    assert ...
```

Tests are run automatically with [`pre-commit`](#tooling).

## Code checks

Use [`ruff`](#tooling) to check and format code:

```shell
ruff check
ruff check --fix
ruff format
```

Use [`mypy`](#tooling) for typing errors:

```shell
mypy -p seqme
```

All checks are run automatically with [`pre-commit`](#tooling).

## Notebook stripping

Use [`nbstripout`](#tooling) to strip notebook metadata before commiting to repository:

```shell
find . -name '*.ipynb' -exec nbstripout --drop-empty-cells --keep-output {} +
```

This command is run automatically with [`pre-commit`](#tooling).

## Building sphinx docs

To build the sphinx docs, do

```shell
cd docs
make clean & make html
```

which generates `docs/_build` directory with HTML files.

Note: Any changes to the sphinx docs, will automatically be built by readthedocs when merged into the main branch.

## Tooling

- **Linter and formatter:** [`ruff`](https://docs.astral.sh/ruff/)
- **Static type checking**: [`mypy`](https://mypy.readthedocs.io/en/stable/#)
- **Testing**: [`pytest`](https://docs.pytest.org/en/stable/)
- **Pre-commit hooks:** [`pre-commit`](https://pre-commit.com/)
- **Notebook stripping**: [`nbstripout`](https://pypi.org/project/nbstripout/)
