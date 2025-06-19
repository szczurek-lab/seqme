# pepme

todo: project description

## Example usage
```python
import pepme
...
```
For more examples visit [examples](/examples) directory.

## Installation
#### 1. Install [`uv`](#tooling):
```properties
curl -LsSf https://astral.sh/uv/install.sh | sh
```
If you have troubles, visit the [official site](https://docs.astral.sh/uv/getting-started/installation/) for more information.

#### 2. Install dependencies:
```properties
uv sync
```

# Development guide

## Testing
Write tests in the [tests](/tests) directory. Follow the [`pytest`](#tooling) guidlines. Each file in the directory has to follow the naming convention.
<h5 a><strong><code>tests/test_{name}.py</code></strong></h5>

```python
def helper(): # pytest does not check this function
    ...
def test_{subname}(): # pytest recognizes it as a test
    assert ...
```

Remember it is crucial to add the `test` prefix to files and functions for [`pytest`](#tooling) to work correctly.

## Code checks
Use [`ruff`](#tooling) to check and format code:
```properties
uv run ruff check
uv run ruff check --fix
uv run ruff format
```

Use [`mypy`](#tooling) for typing errors:
```properties
uv run mypy -p pepme
```

Use [`pytest`](#tooling) to run tests:
```properties
uv run pytest
```


All checks are run automatically with [`pre-commit`](#tooling).

## Dependencies
To add a library dependency use `uv add`, eg.:
```properties
uv add numpy
```
To add a development dependency use `uv add --dev`, eg.:
```properties
uv add --dev ruff
```

## Notebook stripping
Use [`nbstripout`](#tooling) to strip notebook metadata before commiting to repository:
```bash
find . -name '*.ipynb' -exec nbstripout --drop-empty-cells --keep-output {} +
```
This command is run automatically with [`pre-commit`](#tooling).

## Tooling
- __Project manager:__ [`uv`](https://docs.astral.sh/uv/)
- __Linter and formatter:__ [`ruff`](https://docs.astral.sh/ruff/)
- __Static type checking__: [`mypy`](https://mypy.readthedocs.io/en/stable/#)
- __Testing__: [`pytest`](https://docs.pytest.org/en/stable/)
- __Pre-commit hooks:__ [`pre-commit`](https://pre-commit.com/)
- __Notebook stripping__: [`nbstripout`](https://pypi.org/project/nbstripout/)