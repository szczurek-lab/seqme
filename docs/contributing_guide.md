# Contributing

Want to add new metrics, models, or other features to **pepme**? This guide will help you get started.

## Installation

First, setup up the package for development.

```shell
pip install ".[dev,doc]"
```

## Development guide

### Testing

Write tests in the [tests](https://github.com/szczurek-lab/pepme/tree/main/tests) directory. Follow the [`pytest`](#tooling) guidlines. Each file in the directory has to follow the naming convention.

<h5 a><strong><code>tests/test_foo.py</code></strong></h5>

```python
def helper(): # pytest does not check this function
    ...
def test_bar(): # pytest recognizes it as a test
    assert ...
```

Remember it is crucial to add the `test` prefix to files and functions for [`pytest`](#tooling) to work correctly.

### Code checks

Use [`ruff`](#tooling) to check and format code:

```shell
ruff check
ruff check --fix
ruff format
```

Use [`mypy`](#tooling) for typing errors:

```shell
mypy -p pepme
```

Use [`pytest`](#tooling) to run tests:

```shell
pytest
```

All checks are run automatically with [`pre-commit`](#tooling).

### Notebook stripping

Use [`nbstripout`](#tooling) to strip notebook metadata before commiting to repository:

```bash
find . -name '*.ipynb' -exec nbstripout --drop-empty-cells --keep-output {} +
```

This command is run automatically with [`pre-commit`](#tooling).

### Tooling

- **Linter and formatter:** [`ruff`](https://docs.astral.sh/ruff/)
- **Static type checking**: [`mypy`](https://mypy.readthedocs.io/en/stable/#)
- **Testing**: [`pytest`](https://docs.pytest.org/en/stable/)
- **Pre-commit hooks:** [`pre-commit`](https://pre-commit.com/)
- **Notebook stripping**: [`nbstripout`](https://pypi.org/project/nbstripout/)
