Welcome to BQSKit's documentation!
==================================

BQSKit is a superoptimizing quantum compiler that aims to provide an
easy-to-use and quick-to-extend software package around quantum synthesis.
This is accomplished by first building a quantum circuit intermediate
representation designed to work efficiently with numerical optimizer
based synthesis algorithms, and second bundling a compiler infrastructure
and algorithm framework that can run many algorithms efficiently.


.. toctree::
   :caption: Introduction
   :maxdepth: 1

   intro/start
   intro/synthesis
   intro/CONTRIBUTING
   intro/license

.. toctree::
   :caption: BQSKit Tutorials
   :maxdepth: 1

   tutorials/Introduction to BQSKit IR.ipynb
   tutorials/Search Synthesis.ipynb
   tutorials/Searchless Synthesis.ipynb
   tutorials/Partitioning and Optimization.ipynb

.. toctree::
   :caption: API Reference
   :maxdepth: 1

   source/ir
   source/compiler
   source/passes
   source/qis
   source/utils
   source/exec
