# Contributing to BQSKit

The Berkeley Quantum Synthesis Toolkit is an open-source superoptimizing compiler committed to making synthesis easy to use and quick to extend. As such, one of our primary goals is to allow everyone to join our community and contribute to the BQSKit project. This page describes how to do just that.

> [!TIP]
> If you encounter any issues or need help, please don't hesitate to reach
> out through a GitHub Issue.

## Getting Started

We welcome all contributions, but we envision two common contributions to BQSKit: extending the IR with new gates and implementing or altering algorithms in a compiler pass. If you plan to make a similar contribution, you can find documentation for the IR and supported algorithms under the API Reference section.

Before contributing, you will probably want to familiarize yourself with the codebase and documentation and set up a development environment.

### Development Environment

BQSKit primarily uses [uv](https://docs.astral.sh/uv/) to manage its development environment. You can then set up the development environment as follows:

```
git clone https://github.com/BQSKit/BQSKit.git
cd bqskit
uv sync
```

`uv sync` creates a `.venv` with BQSKit installed in editable mode, plus everything needed for development (pytest, ruff, pre-commit, etc.). You can then run anything inside that environment with `uv run <command>` (e.g. `uv run pytest`). If you're working on or testing documentation, `uv sync --extra docs` will additionally install the Sphinx toolchain.

To build and preview the docs locally after that:

```
cd docs
uv run make html
```

The rendered site is written to `docs/_build/html/`; open `index.html` there in a browser. Run `uv run make clean` first if you want a full rebuild (useful if you've renamed or removed pages, since Sphinx's incremental build can leave stale output otherwise).

`tox` can optionally be used to run the test suite against every supported Python version at once, similar to the version matrix in CI: `uvx --with tox-uv tox`.

#### Without uv

If you'd rather not use uv, you can alternatively use a standard python virtual environment workflow, using pip's native support for [PEP 735 dependency groups](https://peps.python.org/pep-0735/) (requires **pip >= 26.2**):

```
git clone https://github.com/BQSKit/BQSKit.git
cd bqskit
python -m venv .venv
source .venv/bin/activate  # .venv\Scripts\activate on Windows
pip install -e . --group dev
```

This installs BQSKit in editable mode, as well as the same development dependencies as `uv sync`. For the documentation toolchain, use the `docs` extra instead: `pip install -e '.[docs]'`.

> [!NOTE]
> This environment may differ than the one installed via `uv sync` as pip resolves `--group dev` fresh each time, rather than reading `uv.lock`. This means that it is not guaranteed to produce the same exact dependency versions CI uses. If you hit a test failure that only happens in a pip-managed environment, try reproducing it with `uv sync`.

## Guidelines

Please follow the below short list of guidelines when contributing.

### Pull Request Checklist

1. Please ensure the pre-commit checks pass on your branch. Run `uv run pre-commit install` once per clone to have ruff (lint + format) and mypy run automatically on every `git commit`. Before pushing, it's worth running `uv run pre-commit run --all-files` to check the whole repository at once, since the per-commit hooks only see the files in that commit.

2. Please ensure all tests are still passing, which can be checked with `uv run pytest`. If your change might behave differently across Python versions, you can additionally run the full suite against every supported interpreter with `uvx --with tox-uv tox`. If appropriate, appropriate, please add tests to ensure your change behaves correctly. See the testing section below for more information.

3. Please ensure that any added package, module, class, attribute, function, or method has an appropriate Google-style docstring. The documentation engine uses these to produce API references. If you have created a user-facing class, please add those to the autosummary list in the top-level package's `__init__.py` (e.g. `bqskit.ir.__init__`).

4. BQSKit is a type-annotated Python package, which helps catch some bugs early with static code analysis tools like [Mypy](http://mypy-lang.org/). You can see [PEP 484: Type Annotations](https://www.python.org/dev/peps/pep-0484/) for more information. Please annotate your contribution with types. Sometimes, this can be tricky. If you need help, please don't hesitate to ask.

## Testing

After making any changes, it is essential to ensure that all the previous tests still pass. Run `uv run pytest` to execute the test suite. If your change touches version-sensitive code, you can also run the full suite against every supported Python version with `uvx --with tox-uv tox` (see [Development Environment](#Development-Environment) above).

Additionally, you will want to write tests for any appropriate changes. Our test suite resides in the `tests` folder and uses a combination of `pytest` and `hypothesis`.

- [pytest](https://docs.pytest.org/en/6.2.x/)
- [Hypothesis](https://hypothesis.readthedocs.io/en/latest/)

Pytest is a framework for writing and running tests. Any Python method or
function that starts with `test_` in the `tests` folder will
be collected and run as part of the test suite. You can write normal Python
code and use assert statements in your tests. Although you can place your
test anywhere in the `tests` folder, please follow the same structure there
already. The `tests` directory structure closely follows the `bqskit`
package structure, which makes it easy to find tests. If you are not familiar
with pytest, we recommend you read a few of the tests included already and
ask any questions you may have.

Hypothesis is a powerful library that will intelligently generate inputs
to tests. Any test that starts with a `given` decorator uses Hypothesis
to generate inputs according to some strategy. BQSKit has several custom
strategies that can be found in `bqskit.utils.test` module. We recommend
using `hypothesis` to test complex functionality that may have corner cases.
