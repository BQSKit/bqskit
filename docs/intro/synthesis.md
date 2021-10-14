# Quantum Synthesis Explained


Quantum synthesis techniques generate circuits from high
level mathematical descriptions of an algorithm. Thus, they
can provide a very powerful tool for circuit optimization,
hardware design exploration and algorithm discovery. BQSKit is
a software suite that aims to expose these techniques in a
easy-to-use and quick-to-extend way.

## Circuit Synthesis

A quantum transformation (algorithm, circuit) on n-qubits
is represented by a unitary matrix U of size $2^n \times 2^n$.
A circuit is described by an evolution in space (application on qubits)
and time of gates.

The goal of circuit synthesis is to decompose unitaries from $U(2^n)$
into a product of terms, where each individual term (e.g. from $U(2)$
and $U(4)$) captures the application of a quantum gate on individual
qubits. The quality of a synthesis algorithm is evaluated by the number
of gates in the circuit it produces, and by the distinguishability of
the solution from the original unitary. Circuit length provides one of the
main optimality criteria for synthesis algorithms: shorter circuits are
better.

Synthesis algorithms use distance metrics to assess the
solution quality, and their goal is to minimize $|U_T - U_C|$,
where $U_T$ is the unitary that describes the target transformation
and $U_C$ is the computed solution. They choose an error
threshold $\epsilon$ and use it for convergence,
$|U_T - U_C| \le \epsilon$. In BQSKit, a metric based on
the Hilbert-Schmidt inner product is used by default:

$$| U_T - U_C | = \sqrt{1 - \frac{|Tr(U_T^\dagger U_C)|}{2^n}^2}$$

### Top-Down Synthesis

```{image} ../images/topdown.png
```

Top-down synthesis algorithms follow prescribed,
simple rules to decompose large unitaries into a tensor product
of smaller terms or into a product of symmetric matrices.
The above figure illustrates the Quantum Shannon Decomposition (QSD),
which breaks an n-qubit unitary into four (n − 1)-qubit
unitaries and three multi-controlled rotations.
Like most topdown methods, synthesis with QSD is quick, but circuit depth
grows exponentially. Overall, these techniques are memory
limited, rather than computationally limited.

The only known depth optimal rule based algorithm is the
KAK-decomposition, which is valid only for two-qubit
operations. Due to its optimality, KAK has been used in both
bottom-up and top-down synthesis, as well as in “traditional”
quantum compilation using peephole optimizations and circuit
mapping. For example, UniversalQ implements multiple
top-down methods, some exposed directly by IBM's Qiskit.
Their version of QSD stops when reaching two-qubit blocks
which are instantiated to native gates by KAK.

### Bottom-Up Synthesis

```{image} ../images/bottomup.png
```

Bottom-Up Synthesis: These algorithms, described in the above figure,
start with an empty circuit and attempt to place simple building
blocks until equality is formed.

QSearch introduces an optimal depth, topology aware
synthesis algorithm that has been demonstrated to be
extensible across native gate sets (e.g. {RX, RZ, CNOT},
{RX, RZ, SWAP}) and to multi-level systems such as
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
structure. An A* heuristic determines which of the
candidates is selected for another layer expansion, as well as
the destination of backtracking steps.

```{image} ../images/qsearchtree.png
```

Although theoretically able to solve for any circuit size,
the scalability of QSearch is limited in practice to four qubit
programs due to several factors. The A* strategy determines
the number of solutions evaluated: at best this is linear in
depth, at worst it is exponential. Our examination of QSearch
performance indicates that its scalability is limited to four
qubits first due to the presence of too many deep backtracking
chains.

The LEAP algorithm is an extension of QSearch with a 'prefixing'
technique designed to reduce the number of candidates, especially when
deep inside the solution space.

### QFAST

From the above discussion, several trends
become apparent. Top-down methods scale to a larger number
of qubits than search based methods, and the quality of their
solution may be improved by incorporating optimal depth
techniques that work on more than two-qubits. The higher the
number of qubits handled by the native “bottom” synthesis,
probably the higher the impact on the quality of the solution.
Optimal search based techniques are limited in scalability first
by the search algorithm, second by the scalability of numerical optimization.
QFAST improves the synthesis scalability
while providing comparable solutions through very simple intuitive
principles:

1) As small two-qubit building blocks may lack “computational power”,
QFAST uses generic blocks spanning a configurable number of qubits.

```{image} ../images/qfasttree.png
```

2) As the number of partial solutions and their evaluations
may hamper scalability, QFAST conflates the numerical optimization
and search problem. QFAST does this by using a continuous circuit space.
At each step, the circuit is expanded by one layer. Given an n-qubit
circuit, a layer encodes an arbitrary m-qubit operation on any m-qubits,
with $m < n$. Thus, QFAST's formulation does not
having a branching factor and solves combinatorially
less optimization problems.

While these do improve the scalability of synthesis, it does come at a
slight quality cost especially in shallow circuits.
For small, expected to be shallow circuits, LEAP is the preferred method.
For larger or expected to be deeper circuits, QFAST is the preferred method.
