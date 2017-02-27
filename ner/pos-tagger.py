import sys
import random
from functools import partial
from itertools import islice
from reader import *
from model import *


# filename : str
def read_graphs(filename):
    # Builds a Reader type and then iterates over every entry, returning a graph
    return TSVReader(tsv_to_pos_graph, open(filename)).next_all()


# model : SparseModel
# graphs : list<NLPGraph>
def create_instances(model, graphs):
    xs = list()
    ys = list()
    correct = 0
    total = 0
    f = 0.0

    # we will say that p = c / g
    c = 0.0  # c is the num correct
    g = 0.0  # g is the num attempted

    # we will say that r = n / t
    n = 0.0  # n is the num of correct
    t = 0.0  # t is the num of true

    for graph in graphs:
        backup = [node.pos for node in graph]
        add_instances(model, graph, xs, ys)

        # working on getting data for F1 score here ====================================================================
        # precision  ---------------------------------------------------------------------------------------------------
        for pos, node in zip(backup, graph):
            # check to ensure it was attempted (i.e. that it was not guessed as a "O")
            if node.pos != "O":
                g += 1.0  # add one to g b/c whether right or wrong it is an attempted guess
                # check if the guess was correct
                if node.pos == pos:
                    c += 1.0  # add one to c b/c it was a correct guess

        # calculate the p score
        p = float(float(c)/float(g)) if g > 0.0 else 0.0  # prevents division by 0 errors

        # recall  ------------------------------------------------------------------------------------------------------
        for pos, node in zip(backup, graph):
            # check that the label should have been attempted (i.e. that it is not a "O")
            if pos != "O":
                print node.pos, pos
                t += 1.0  # add one to g b/c whether right or wrong it is either false negative or true positive
                # check to see if the guess was correct
                if node.pos == pos:
                    n += 1.0  # add one to c b/c it was a correct guess

        # calculate the r score
        r = float(float(n)/float(t)) if t > 0.0 else 0.0  # prevents division by 0 errors

        # calculate score ----------------------------------------------------------------------------------------------
        if p != 0 or r != 0:
            f = 2 * ((p * r)/(p + r))  # F1 Score = (p * r)/(p + r)
        else:
            f = 0.0

        # calculate old score ------------------------------------------------------------------------------------------
        for pos, node in zip(backup, graph):
            if node.pos == pos:
                correct += 1
            else:
                node.pos = pos

        total += len(graph)
    return xs, ys, correct, total, f


# model : SparseModel
# sentence : list of fields
def add_instances(model, graph, xs, ys):
    for i in range(len(graph)):
        x = [0]

        f = 0
        node = graph.nodes[i]
        x.append(model.index_x(str(f) + node.word))

        f += 1  # f == 1
        if i >= 2:
            node = graph.nodes[i-2]
            x.append(model.index_x(str(f) + node.word))

        f += 1  # f == 2
        if i >= 1:
            node = graph.nodes[i-1]
            x.append(model.index_x(str(f) + node.word))

        f += 1  # f == 3
        if i + 1 < len(graph):
            node = graph.nodes[i+1]
            x.append(model.index_x(str(f) + node.word))

        f += 1  # f == 4
        if i + 2 < len(graph):
            node = graph.nodes[i+2]
            x.append(model.index_x(str(f) + node.word))

        '''
        f += 1
        if i >= 2:
            node = graph.nodes[i - 2]
            x.append(model.index_x(str(f) + node.pos))

        f += 1
        if i >= 1:
            node = graph.nodes[i - 1]
            x.append(model.index_x(str(f) + node.pos))
        '''

        node = graph.nodes[i]  # get the node for this value

        # Array x is where we do our argmax on to determine y_hat ------------------------------------------------------
        x.sort()  # List x represents how we are going to store the information moving forward
        xs.append(x)  # xs is the sparse feature representation
        ys.append(model.index_y(node.pos))  # ys is the actual label

        # Guess the POS here and label node the resulting pos ----------------------------------------------------------
        node.pos = model.label(model.argmax(x))  # now node.pos is the predicted label (y_hat)


def perceptron(model, x, y, learning_rate):
    z = model.argmax(x)
    if y != z:
        model.update_weights(x, y,  learning_rate)
        model.update_weights(x, z, -learning_rate)


def update(model, xs, ys, algorithm):
    for x, y in zip(xs, ys):
        algorithm(model=model, x=x, y=y)

def main():
    # Set-up information ===============================================================================================
    trn_file = sys.argv[1]
    dev_file = sys.argv[2]
    learning_rate = 0.01
    max_x = 500000
    max_y = 50
    max_iter = 20
    mini_batch = 5

    print('Reading: '+trn_file)
    trn_graphs = read_graphs(trn_file)

    print('Reading: '+dev_file)
    dev_graphs = read_graphs(dev_file)

    print('Training:')
    model = SparseModel(max_x, max_y)

    for i in range(max_iter):
        random.shuffle(trn_graphs)

        for b_index in range(0, len(trn_graphs), mini_batch):
            e_index = b_index + mini_batch if b_index + mini_batch <= len(trn_graphs) else len(trn_graphs)
            xs, ys, correct, total, f = create_instances(model, islice(trn_graphs, b_index, e_index))
            update(model, xs, ys, partial(perceptron, learning_rate=learning_rate))

        xs, ys, correct, total, f = create_instances(model, dev_graphs)
        print('%3d: Old: %5.2f F1: %5.2f' % (i, 100.0*correct/total, f*100))

main()