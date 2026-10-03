"""Neurosymbolic finite and pushdown automata.

Public API:

    Alphabet, IndexedSet     -- indexed sets with stable name<->index map
    FiniteAutomaton          -- adapter over automata.fa.dfa.DFA with
                                tensor view of delta and a soft
                                forward-algorithm runtime
    PerceptionAdapter        -- nn.Module wrapper for one or more CNNs
    NSFA                     -- composition of FiniteAutomaton +
                                PerceptionAdapter; differentiable

PushdownAutomaton and NSPDA are added by `nsa.pda` and
`nsa.nspda` once the PDA adapter is in place.

The legacy Neutron / Mode / Transition / TS API (pre-refactor) remains
importable for backwards compatibility during the migration but is
deprecated.
"""

from nsa.alphabet import Alphabet, IndexedSet
from nsa.fa import FiniteAutomaton
from nsa.nsfa import NSFA
from nsa.nspda import NSPDA
from nsa.pda import PushdownAutomaton
from nsa.perception import PerceptionAdapter

__all__ = [
    "Alphabet",
    "FiniteAutomaton",
    "IndexedSet",
    "NSFA",
    "NSPDA",
    "PerceptionAdapter",
    "PushdownAutomaton",
]
