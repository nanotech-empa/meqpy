class Transition:
    """
    Represents a transition between two states (i → j) with a given rate.
    """

    def __init__(self, state_i, state_j, rate, label=None):
        self.state_i = state_i
        self.state_j = state_j
        self.rate = float(rate)
        self.label = label or f"{state_i.label}->{state_j.label}"

    def add_transition(self, transition: Transition):
        self.transitions.append(transition)

    def __repr__(self):
        return f"Transition({self.label}, rate={self.rate})"
