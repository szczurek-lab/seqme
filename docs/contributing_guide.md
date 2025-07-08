# Contributing

Want to add new metrics, models, or other features? This guide will help you get started.

## Installation

First, setup up the package for development.

### 1. Install [`uv`](#tooling):

#### (Option 1) Systemwide install

```shell
pip install uv
```

And reload the bash session.

Running

```shell
which uv
```

should return path to the installed `uv` binary.

If you have troubles, visit the [official site](https://docs.astral.sh/uv/getting-started/installation/) for more information or install [`uv`](#tooling) locally as described below.

#### (Option 2) Local install for macOS (zsh) and Linux (bash) users

Run the following commands:

```shell
export UV_VENV_LOCATION={some path with enough storage to install Python packages}
python3 -m venv $UV_VENV_LOCATION/uv_venv
echo "export PATH=$UV_VENV_LOCATION/uv_venv/bin:\$PATH" >> $HOME/.$(basename $SHELL)rc
source $HOME/.$(basename $SHELL)rc
pip install uv
```

The above commands automatically append `export PATH=...` to `.bashrc` or `.zshrc` depending on your default shell. If you use different shell append the `export PATH=...` to the apropriate shell initialization script.

### 2. Install dependencies

```shell
uv sync
```

### 3. Activate the virtual environment

```shell
source .venv/bin/activate
```

### 4. Install pre-commit

```shell
pre-commit install
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
uv run ruff check
uv run ruff check --fix
uv run ruff format
```

Use [`mypy`](#tooling) for typing errors:

```shell
uv run mypy -p pepme
```

Use [`pytest`](#tooling) to run tests:

```shell
uv run pytest
```

All checks are run automatically with [`pre-commit`](#tooling).

### Dependencies

To add a library dependency use `uv add`, eg.:

```shell
uv add numpy
```

To add a development dependency use `uv add --dev`, eg.:

```shell
uv add --dev ruff
```

### Notebook stripping

Use [`nbstripout`](#tooling) to strip notebook metadata before commiting to repository:

```bash
find . -name '*.ipynb' -exec nbstripout --drop-empty-cells --keep-output {} +
```

This command is run automatically with [`pre-commit`](#tooling).

### Tooling

- **Project manager:** [`uv`](https://docs.astral.sh/uv/)
- **Linter and formatter:** [`ruff`](https://docs.astral.sh/ruff/)
- **Static type checking**: [`mypy`](https://mypy.readthedocs.io/en/stable/#)
- **Testing**: [`pytest`](https://docs.pytest.org/en/stable/)
- **Pre-commit hooks:** [`pre-commit`](https://pre-commit.com/)
- **Notebook stripping**: [`nbstripout`](https://pypi.org/project/nbstripout/)
