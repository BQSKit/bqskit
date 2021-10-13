# Quantum Synthesis Explained

```{image} ../images/bottomup.png
```

### temp

Quantum synthesis is a form of compilation in which a unitary matrix is
converted into a circuit composed of only basis gates.
Topology-aware synthesis
Instantiation

TODO

Quantum synthesis techniques generate circuits from high
level mathematical descriptions of an algorithm. Thus, they
can provide a very powerful tool for circuit optimization,
hardware design exploration and algorithm discovery.


A quantum transformation (algorithm, circuit) on n-qubits
is represented by a unitary matrix U of size 2
n ×2
n. A circuit
is described by an evolution in space (application on qubits)
and time of gates.

The goal of circuit synthesis is to decompose unitaries from U(2n) into a product of terms, where
each individual term (e.g. from U(2) and U(4)) captures the
application of a quantum gate on individual qubits. The quality
of a synthesis algorithm is evaluated by the number of gates
in the circuit it produces, and by the distinguishability of the
solution from the original unitary

Circuit length provides one of the main optimality criteria
for synthesis algorithms: shorter circuits are better.

Synthesis algorithms use distance metrics to assess the
solution quality, and their goal is to minimize kUT − UC k,
where UT is the unitary that describes the target transformation
and UC is the computed solution. They choose an error
threshold  and use it for convergence, kUT − UC k ≤ .

In bqskit we use the HS by default...

Top-Down Synthesis: These algorithms follow prescribed,
simple rules to decompose large unitaries into a tensor product
of smaller terms or into a product of symmetric matrices. Figure 1a illustrates the Quantum Shannon Decomposition (QSD)
[9], which breaks an n-qubit unitary into four (n − 1)-qubit
unitaries and three multi-controlled rotations. Like most topdown methods, synthesis with QSD is quick, but circuit depth
grows exponentially. Overall, these techniques are memory
limited, rather than computationally limited.
The only known depth optimal rule based algorithm is the
KAK-decomposition [8], which is valid only for two-qubit
operations. Due to its optimality, KAK has been used in both
bottom-up and top-down synthesis, as well as in “traditional”
quantum compilation using peephole optimizations and circuit
mapping. For example, UniversalQ [23] implements multiple
top-down methods, some exposed directly by IBM Qiskit.
Their version of QSD stops when reaching two-qubit blocks
which are instantiated to native gates by KAK.
Bottom-Up Synthesis: These algorithms, described in Fig-
Unitary Unitary Unitary
Rz Ry
Unitary
Rz
Unitary

(a) Top-down synthesizers follow prescribed, simple rules to decompose large unitaries into smaller ones while maintaining equality.
= Unitary
...
...
...
...
...
(b) Bottom-up synthesizers start with an empty circuit and build up to equality.
Fig. 1: Quantum synthesizers are either top-down or bottom-up.
ure 1b, start with an empty circuit and attempt to place simple
building blocks until equality is formed.


QSearch [11] introduces an optimal depth, topology aware
synthesis algorithm that has been demonstrated to be extensible across native gate sets (e.g. {RX, RZ, CNOT},
{RX, RZ, SW AP}) and to multi-level systems such as
qutrits. The approach employed in QSearch is canonical for the
operation of other synthesis approaches that employ numerical
optimization.
Conceptually, the synthesis problem can be thought as a
search over a tree of possible circuit structures. A search
algorithm provides a principled way to walk the tree and
evaluate candidate solutions. For each candidate, a numerical
optimizer instantiates the function (parameters) of each gate
in order to minimize some distance objective function.
QSearch works by extending the circuit structure a layer
at a time. At each step the algorithm places a two-qubit
expansion operator in all legal placements. For the CNOT gate
set, the operator contains one CNOT gate and two U3(θ, φ, λ)
gates. QSearch then evaluates these candidates using numerical
optimization to instantiate all the single qubit gates in the
structure. An A* [24] heuristic determines which of the
candidates is selected for another layer expansion, as well as
the destination of backtracking steps. Figure 2a illustrates this
process for a three qubit circuit.
Although theoretically able to solve for any circuit size,
the scalability of QSearch is limited in practice to four qubit
programs due to several factors. The A* strategy determines
the number of solutions evaluated: at best this is linear in
depth, at worst it is exponential. Our examination of QSearch
performance indicates that its scalability is limited to four
qubits first due to the presence of too many deep backtracking
chains. Any technique to reduce the number of candidates,
especially when deep inside the solution space, is likely to
improve performance.
As each expansion operator has two single-qubit gates,
accounting for six3 parameters, circuit paramaterization grows
linearly with depth. Numerical optimizers scale at best with a
3
In practice, QSearch uses 5 parameters due to commutativity rules between
single qubit and CNOT gates.
(a) QSearch uses native gates in synthesis and searches for structure in their
circuit space. Each node in their circuit space is a valid circuit structure that
has edges to circuits deeper by a two-qubit native gate.
(b) QFAST uses block unitary matrices of arbitrary size. In this figure, twoqubit blocks are used and a similar tree is constructed.
Fig. 2: The two-qubit unitary blocks are more expressive than one, fixed
two-qubit gate followed by single-qubit rotations; Nodes 2 levels deep in the
unitary block tree can only be expressed with nodes 6 levels deep in the native
gate tree.
very high degree polynomial in parameters, making optimization of long circuits challenging.

QFAST Approach: From the above discussion, several trends
become apparent. Top-down methods scale to a larger number
of qubits than search based methods, and the quality of their
solution may be improved by incorporating optimal depth
techniques that work on more than two-qubits. The higher the
number of qubits handled by the native “bottom” synthesis,
probably the higher the impact on the quality of the solution.
Optimal search based techniques are limited in scalability first
by the search algorithm, second by the scalability of numerical optimization. QFAST improves the synthesis scalability
while providing good solutions through very simple intuitive
principles:
1) As small two-qubit building blocks may lack “computational power”, we use generic blocks spanning a
configurable number of qubits. See Figure 2b for an
example of the tree with two-qubit building blocks. In
this example, a depth two partial solution could express
circuits that are up to depth six4
in the Qsearch tree.
2) As the number of partial solutions and their evaluations
may hamper scalability, we conflate the numerical optimization and search problem. We do this by using
a continuous circuit space. At each step, the circuit
is expanded by one layer. Given an n-qubit circuit, a
layer encodes an arbitrary m-qubit operation on any mqubits, with m < n. Thus, our formulation does not
having a branching factor and solves combinatorially
less optimization problems.
