from .state import State
from .transition import Transition


class System:
    """
    Class to define a physical system for the master equation solver
    """

    def __init__(self, name="GenericSystem", states=None, **kwargs):
        self.name = name
        self.states = states if states is not None else []

    def add_state(self, state: State):
        self.states.append(state)

    def get_state(self, label):
        for state in self.states:
            if state.label == label:
                return state
        raise ValueError(f"State with label {label} not found in the system.")

    def add_transition(self, transition: Transition):
        self.transitions.append(transition)

    def add_transition_by_labels(self, label_i, label_j, rate):
        i = self.get_state(label_i)
        j = self.get_state(label_j)
        self.add_transition(Transition(i, j, rate))

    def __repr__(self):
        return f"System(name={self.name}, states={self.states})"
