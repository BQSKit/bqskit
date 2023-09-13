# Contributing to BQSKit

The Berkeley Quantum Synthesis Toolkit is an open-source superoptimizing
compiler committed to making synthesis easy to use and quick to extend.
As such, one of our primary goals is to allow everyone to join our community
and contribute to the BQSKit project. This page describes how to do just
that.

> If you encounter any issues or need any help, please don't hesitate to reach
> out through a GitHub Issue.

## Getting Started

Before contributing, you will probably want to familiarize yourself with
the codebase and documentation and set up a development environment. We
welcome all contributions, but we envision two common contributions to
BQSKit: extending the IR with new gates and implementing or altering
algorithms in a compiler pass. If you plan to make a similar contribution,
you can find documentation for the IR and supported algorithms under the
API Reference section.

### Development Environment

You will want to install BQSKit from the source by cloning the repository
from GitHub:

```
git clone https://github.com/BQSKit/BQSKit.git
cd bqskit
pip install -e '.[dev]'
```

You can run `tox` to install all development packages, set up virtual
environments for all supported Python versions, perform all stylistic
checks and modifications, and run the test suite.

## Guidelines

Please follow the below short list of guidelines when contributing.

### Pull Request Checklist

1. Please ensure the pre-commit checks ran successfully on your branch. These
ensure your changes match the project's code style and perform other critical
analyses. To do this, either execute `pre-commit run --all-files` or `tox`
locally before pushing. Note that `tox` or `pre-commit` will make stylistic
modifications directly to your code.

2. Please ensure all tests are still passing, which can also be done
with `tox`. Also, if appropriate, please add tests to ensure your change
behaves correctly. See the testing section below for more information.

3. Please ensure that any added package, module, class, attribute, function,
or method has an appropriate Google-style docstring. The documentation
engine uses these to produce API references. If you have created a
user-facing class, please add those to the autosummary list in the top-level
package's `__init__.py`, e.g., `bqskit.ir.__init__`.

4. BQSKit is a type-annotated Python package, which helps catch some bugs
early with static code analysis tools like [Mypy](http://mypy-lang.org/).
You can see [PEP 484: Type Annotations](https://www.python.org/dev/peps/pep-0484/)
for more information. Please annotate your contribution with types.
Sometimes, this can be tricky. If you need help, please don't hesitate to ask.

## Testing

After any changes, it is essential to ensure that all the previous tests
still pass on all supported versions of Python. This can be done by running
the `tox` command after installing it. Additionally, you will want to write
tests for any appropriate changes. Our test suite resides in the `tests`
folder and uses a combination of `pytest` and `hypothesis`.

- [PyTest](https://docs.pytest.org/en/6.2.x/)
- [Hypothesis](https://hypothesis.readthedocs.io/en/latest/)

Pytest is a framework for writing and running tests. Any Python method or
function that starts with `test_` in the `tests` folder will
be collected and run as part of the test suite. You can write normal Python
code and use assert statements in your tests. Although you can place your
test anywhere in the `tests` folder, please follow the same structure there
already. The `tests` directory structure closely follows the `bqskit`
package structure, which makes it easy to find tests. If you are not familiar
with Pytest, we recommend you read a few of the tests included already and
ask any questions you may have.

Hypothesis is a powerful library that will intelligently generate inputs
to tests. Any test that starts with a `given` decorator uses Hypothesis
to generate inputs according to some strategy. BQSKit has several custom
strategies that can be found in `bqskit.utils.test` module. We recommend
using `hypothesis` to test complex functionality that may have corner cases.
