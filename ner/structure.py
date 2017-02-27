__author__ = 'Tarrek Shaban'

ROOT_TAG = '@#r$%'


class NLPNode:
    # word : str
    # pos : str  (NOTE: entity not pos)
    def __init__(self, word=None, pos=None):
        self.word = word
        self.pos = pos

    def __str__(self):
        return '\t'.join([self.word, self.pos])


class NLPGraph:
    # nodes : list<NLPNode>
    def __init__(self, nodes=[]):
        self.root = create_root()
        self.nodes = nodes

    def next(self):
        if self._idx >= len(self.nodes):
            raise StopIteration

        node = self.nodes[self._idx]
        self._idx += 1
        return node

    def __iter__(self):
        self._idx = 0
        return self

    def __str__(self):
        return '\n'.join(map(str, self.nodes))

    def __len__(self):
        return len(self.nodes)


# returns an artificial root node : NLPNode
def create_root():
    return NLPNode(ROOT_TAG, ROOT_TAG)
