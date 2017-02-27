from structure import NLPGraph
from structure import NLPNode
import re
__author__ = 'Tarrek Shaban'

RE_TAB = re.compile('\t')


class TSVReader:
    # tsv_to : list<list<string> -> NLPGraph
    # fin : file input-stream
    def __init__(self, tsv_to, fin=None):
        self.tsv_to = tsv_to
        self.fin = fin

    def next(self):
        graph = self.n()
        if graph:
            return graph
        else:
            raise StopIteration

    def __iter__(self):
        return self

    # fin : file input-stream
    def open(self, fin):
        self.fin = fin

    def close(self):
        self.fin.close()

    def n(self):
        tokens = list()

        for line in self.fin:
            line = line.strip()
            if line:
                tokens.append(RE_TAB.split(line))
            elif tokens:
                break

        if tokens:
            return self.tsv_to(tokens=tokens)
        else:
            return None

    def next_all(self):
        return [graph for graph in self]


# tokens : list<list<string>>
def tsv_to_pos_graph(tokens, word_index=0, pos_index=3):
    return NLPGraph([NLPNode(t[word_index], t[pos_index]) for t in tokens])
