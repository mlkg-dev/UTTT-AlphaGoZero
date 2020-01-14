

class Node:

    def __init__(self, state, parent, probability):
        self.state = state
        self.parent = parent
        self.policy = probability
        self.q_policy = 0
        self.net_value = 0
        self.sum_net_value = 0
        self.visit_counter = 0
        self.children = []
        self.evaluated = False

    def add_child(self, state, probability):
        self.children.append(Node(state, self, probability))
