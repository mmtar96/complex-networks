import numpy as np
import networkx as nx


''' MODELO DE ERDOS-RENYI '''

def er_model1(N, p): # number nodes and probability to be conected
    G_ER = nx.Graph()
    nodos = np.array(range(N))
    i = 0
    for i in range(len(nodos)):
        G_ER.add_node(nodos[i])
    for i in range(N):
        for j in range(N):
            if i < j:
                P = np.random.rand()
                if P < p:
                    G_ER.add_edge(i, j)
    print(G_ER.number_of_nodes())
    print(G_ER.number_of_edges())
    return (G_ER)

def er_model2(N, e): # number nodes and edges
    G_ER = nx.Graph()
    nodos = np.array(range(N))
    i = 0
    for i in range(len(nodos)):
        G_ER.add_node(nodos[i])
    for i in range(e):
        node1 = np.random.choice(nodos)
        node2 = np.random.choice(nodos)
        pos = int(list(np.where(nodos == node2))[0])
        if node1 == node2:
            pos = pos + 1
            node2 = nodos[pos]
        G_ER.add_edge(node1, node2)
    print(G_ER.number_of_nodes())
    print(G_ER.number_of_edges())
    return (G_ER)


''' MODELO DE CONFIGURACION '''

def config_model(G): # build a graf with a selected degree distribution
    G_C = nx.Graph()
    nodos = list(G.nodes())
    for i in range(len(nodos)):
        G_C.add_node(int(nodos[i]))
    pool = nodos
    pool_aux = np.array(pool)
    for p in nodos:
        p = int(p)
        while G_C.degree(p) < G.degree(p):
            for i in pool_aux:
                if G_C.degree(int(i)) == G.degree(int(i)):
                    pos = int(list(np.where(pool_aux == i))[0])
                    pool_aux = np.delete(pool_aux, pos)
            node_in = int(np.random.choice(pool_aux))
            if node_in == p:
                node_in = int(np.random.choice(pool_aux))
            if node_in == p:
                node_in = int(np.random.choice(pool_aux))
            if node_in == p:
                node_in = int(np.random.choice(pool_aux))
            if G_C.degree(node_in) < G.degree(node_in):
                G_C.add_edge(p, node_in)
    return(G_C)
