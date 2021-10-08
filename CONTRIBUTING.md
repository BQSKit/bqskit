# Contributing to BQSKit

The Berkeley Quantum Synthesis Toolkit is an open-source superoptimizing
compiler committed to making synthesis easy to use and quick to extend.
As such, it is one our main goals to allow everyone to join our community
and contribute to the BQSKit project. This page describes how to do just
that.

## Getting Started

You will probably want to familiarize yourself with the codebase and
documentation, and set up a development environment before making any
contribution. We will welcome all contributions, but we envision there
will be two common types of contributions to BQSKit: ones extending the
IR with new gates, and ones implementing algorithms in a compiler pass.
If you plan to make a similar contribution, you can find documentation
for the IR and supported algorithms under the API Reference section.

### Development Environment

You will want to install BQSKit from source, by cloning the repository from github:

```
git clone https://github.com/BQSKit/BQSKit.git
cd bqskit
pip install -e .
```

Once you clone the repository, you will want to install `tox`. The easiest
way is with pip:

```sh
pip install tox
```

You can run `tox` to install all development packages, setup virtual
environments for all supported versions of python, perform all stylistic
checks and modifications, and run the test suite.

## Guidelines

Please follow the below short list of guidelines when contributing.

### Pull Request Checklist

1. Please ensure pre-commit was run successfully on your branch. This will
ensure the code style of the project is matched in addition to other checks.
This can be done, by executing `tox` locally before pushing. Note that `tox`
or `pre-commit` will make stylistic modifications to your code.

2. Please ensure that all tests are still passing, which can also be done
with `tox`. Also, if appropriate, please add tests to ensure your change
behaves correctly. See the testing section below for more information.

3. Please ensure that any added package, module, class, attribute, function,
or method has an appropriate google-style docstring. These are used by the
documentation engine to produce api references. Additionally, if you have
created a user-facing class, please add those to the autosummary list in
top level package's `__init__.py`, e.g. `bqskit.ir.__init__`.

4. BQSKit is a type-annotated python package, which helps catch some bugs
early with static code analysis tools like [Mypy](http://mypy-lang.org/).
You can see [PEP 484: Type Annotations](https://www.python.org/dev/peps/pep-0484/)
for more information. Please do annotate your contribution with types.
Sometimes this can be tricky, if you need help, feel free to ask.


## Testing

After any changes, it is important to ensure that all the previous tests
still pass on all supported versions of Python. This can be done by running
the `tox` command after install it. Additionally, you will want to write
tests for any appropriate changes. Our test suite resides in the `tests`
folder and uses a combination of `pytest` and `hypothesis`.

- [PyTest](https://docs.pytest.org/en/6.2.x/)
- [Hypothesis](https://hypothesis.readthedocs.io/en/latest/)

Pytest is a framework for writing and running tests. Any python method or
function that starts with `test_` that resides in the `tests` folder will
be collected and run as part of the test suite. You can write normal python
code and use assert statements in your tests. Although you can place your
test anywhere in the `tests` folder, please do follow the same structure that is
there already. The `tests` directory structure closely follows the `bqskit`
package structure, which makes it easy to find tests. If you are not familiar
with Pytest, we recommend you read a few of the tests included already and
ask any questions you may have.

Hypothesis is a powerful library that will intelligently generate inputs
to tests. Any test that starts with a `given` decorator is using Hypothesis
to generate inputs according to some strategy. BQSKit has several custom
strategies that can be found in `bqskit.utils.test` module. We recommend
using `hypothesis` to test complex functionality that may have corner cases.
