import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import statistics
import seaborn as sns
import community


''' NETWORK '''

def add_data(G, vertices, enlaces):
    nodos = vertices.iloc[:,0]
    for i in range(len(nodos)):
        G.add_node(int(nodos[i]))
    enlaces_def = []
    for i in range(len(enlaces)):
        enlaces_def.append((int(enlaces.iloc[i,2]), int(enlaces.iloc[i,3])))
    a = []
    for item in enlaces_def:
        if item not in a: # Not multiple edges
            a.append(item)
    for i in range(len(a)):
        G.add_edge(a[i][0], a[i][1])
    return G

def params(G):
    G_size = G.number_of_nodes()
    G_edges = G.number_of_edges()
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    mean_degree = statistics.mean(degree_sequence)
    Number_Components = nx.number_connected_components(G)
    print("Grid number nodes: ", G_size)
    print("Grid number edges: ", G_edges)
    print("Grid mean degree: ", mean_degree)
    print("Number of components: ", Number_Components)
    return G_size, G_edges, mean_degree, Number_Components
    
def paramsGcc(G, component):
    Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[component])
    Gcc_size = Gcc.number_of_nodes()
    Gcc_edges = Gcc.number_of_edges()
    degree_sequence_Gcc = sorted((d for n, d in Gcc.degree()), reverse=True)
    mean_degree_Gcc = statistics.mean(degree_sequence_Gcc)
    D_Gcc = nx.diameter(Gcc)
    print("Gcc number nodes: ", Gcc_size)
    print("Gcc number edges: ", Gcc_edges)
    print("Gcc mean degree: ", mean_degree_Gcc)
    print("Gcc diameter: ", D_Gcc)
    return Gcc, (Gcc_size, Gcc_edges, mean_degree_Gcc, D_Gcc)
    
def degree_sequence(G):    
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    fig = plt.figure("Degree of a random graph", figsize=(32, 16))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(6, 6)
    ax1 = fig.add_subplot(axgrid[:, :3])
    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Degree Rank Plot")
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Rank")
    ax2 = fig.add_subplot(axgrid[:, 3:])
    ax2.plot(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title("Degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("Number of Nodes")
    fig.tight_layout()
    plt.show()
    return degree_sequence

def plotgraf(G):
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    dmax = max(degree_sequence)
    fig = plt.figure("Degree of a graph", figsize=(16, 16))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 4)
    ax0 = fig.add_subplot(axgrid[0:3, :])
    pos = nx.spring_layout(G, seed=10396953)
    nx.draw_networkx_nodes(G, pos, ax=ax0, node_size=10)
    nx.draw_networkx_edges(G, pos, ax=ax0, alpha=0.4)
    ax0.set_title("All components of G")
    ax0.set_axis_off()
    ax1 = fig.add_subplot(axgrid[3:, :2])
    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Degree Rank Plot")
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Rank")
    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title("Degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("Number of Nodes")
    fig.tight_layout()
    plt.show()
        
def powerlaw(G): # Revisar esta funcion
    N = G.number_of_nodes()
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    a = 0
    for i in range(len(degree_sequence)):
        a = a + np.log((degree_sequence[i]/(min(degree_sequence)-0.5))) #La a no se calcula bien
    alpha = 1 + N*(1/a)
    b = 0
    for i in range(len(degree_sequence)):
        b = b + degree_sequence[i]**(-alpha)
    C = 1/b
    k = np.linspace(0, 15, 15)
    d = []
    i = 0
    for i in range(len(k)):
        value = C * (1/(k[i])**(alpha))
        d.append(value)
    plt.plot(d)


''' CENTRAL MEASUREMENTS '''

def eigenvector(G, iters=1000):
    E = nx.eigenvector_centrality(G, max_iter = iters)
    dataE = list(E.values())
    fig, ax = plt.subplots(figsize = ( 5 , 3 )) 
    sns.histplot(x=dataE, ax = ax, bins = 80)
    ax.set_xlabel( "Eigenvector value" , size = 8) 
    ax.set_ylabel( "Numbero de nodos" , size = 8 ) 
    plt.show()
    # best eigenvalue node
    list_of_key = list(E.keys())
    list_of_value = list(E.values())
    pos = list_of_value.index(max(list_of_value))
    print("El nodo de mayor valor es: ", list_of_key[pos])
    return E

def pagerank(G, iters=1000):
    PR = nx.pagerank(G, max_iter = 1000)
    dataPR = list(PR.values())
    fig, ax = plt.subplots(figsize = ( 5 , 3 )) 
    sns.histplot(x=dataPR, ax = ax, bins = 80)
    ax.set_xlabel( "Eigenvector value" , size = 8) 
    ax.set_ylabel( "Numbero de nodos" , size = 8 ) 
    plt.show()
    # best eigenvalue node
    list_of_key = list(PR.keys())
    list_of_value = list(PR.values())
    pos = list_of_value.index(max(list_of_value))
    print("El nodo de mayor valor es: ", list_of_key[pos])
    return PR

def closeness(G):
    C = nx.closeness_centrality(G)
    dataC = list(C.values())
    fig, ax = plt.subplots(figsize = ( 5 , 3 )) 
    sns.histplot(x=dataC, ax = ax, bins = 80)
    ax.set_xlabel( "Closeness value" , size = 8 ) 
    ax.set_ylabel( "Numbero de nodos" , size = 8 ) 
    plt.show() 
    # best closeness node
    list_of_key = list(C.keys())
    list_of_value = list(C.values())
    pos = list_of_value.index(max(list_of_value))
    print("El nodo de mayor valor es: ", list_of_key[pos])
    return C

def betweness(G):
    B = nx.betweenness_centrality(G)    
    dataB = list(B.values())
    fig, ax = plt.subplots(figsize = ( 5 , 3 )) 
    sns.histplot(x=dataB, ax = ax, bins = 80)
    ax.set_xlabel( "Betweenness value" , size = 8 ) 
    ax.set_ylabel( "Numbero de nodos" , size = 8 ) 
    plt.show() 
    # best betweness node
    list_of_key = list(B.keys())
    list_of_value = list(B.values())
    pos = list_of_value.index(max(list_of_value))
    print("El nodo de mayor valor es: ", list_of_key[pos])
    return B


''' CLUSTERING COEFFICIENT '''

def clustering(G):
    C = nx.clustering(G)    
    dataC = list(C.values())
    fig, ax = plt.subplots(figsize = ( 5 , 3 )) 
    sns.histplot(x=dataC, ax = ax, bins = 80)
    ax.set_xlabel( "Betweenness value" , size = 8 ) 
    ax.set_ylabel( "Numbero de nodos" , size = 8 ) 
    plt.show() 
    # best betweness node
    list_of_key = list(C.keys())
    list_of_value = list(C.values())
    pos = list_of_value.index(max(list_of_value))
    print("El nodo de mayor valor es: ", list_of_key[pos])
    return C


''' COMMUNITIES '''

def communities(G):
    partition = community.best_partition(G)
    size = float(len(set(partition.values())))
    pos = nx.kamada_kawai_layout(G)
    count = 0.
    for com in set(partition.values()) :
        count = count + 1.
        list_nodes = [nodes for nodes in partition.keys()
                                    if partition[nodes] == com]
        nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20,
                                    node_color = str(count / size))
    nx.draw_networkx_edges(G,pos, alpha=0.5)
    plt.show()
    return partition

def communities_detail(G, vertices):
    C = []
    partition = community.best_partition(G)
    list_of_value = np.array(list(partition.values()))
    comunities = []
    list_of_value_unique = np.unique(list_of_value)
    for i in list_of_value_unique:
        pos = list(np.where(list_of_value == i))[0]
        comunities.append(pos)
    for j in range(len(comunities)):
        for i in comunities[j]:
            print('########################################################')
            print('Comunidad grupo: ', j)
            print(vertices.iloc[i])
    C.append(comunities)


''' PERCOLATION '''

def uniform_percolation(G, percolation=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], sim=True):
    s_total = []
    n_nodes_total = []
    nodes = list(G.nodes())
    for i in range(10):
        s = []
        n_nodes = []
        for j in percolation:
            umbral = j
            G_per = G.copy()
            for i in nodes:
                if np.random.random() > umbral:
                    G_per.remove_node(i)
            G_per_cc = G_per.subgraph(sorted(nx.connected_components(G_per), key=len, reverse=True)[0])
            s.append(G_per_cc.number_of_nodes())   
            n_nodes.append(G_per.number_of_nodes())
        s_total.append(s)
        n_nodes_total.append(n_nodes)
    # mean for each simulation
    mean_s = [0]
    mean_n = [0]
    S = []
    for j in range(len(s_total[0])):
        per_sum = 0
        for i in s_total:
            per_sum = per_sum + i[j]
        mean_s.append(per_sum/len(s_total))
    for j in range(len(n_nodes_total[0])):
        per_sum_n = 0
        for i in n_nodes_total:
            per_sum_n = per_sum_n + i[j]
        mean_n.append(per_sum_n/len(n_nodes_total))
    for i in range(len(mean_n)):
        if mean_n[i] == 0:
            S = [0]
        else:
            S.append(mean_s[i]/mean_n[i])
    if sim == True:
        for j in percolation:
            umbral = j
            G_per = G.copy()
            nodes = list(G.nodes())
            for i in nodes:
                if np.random.random() > umbral:
                    G_per.remove_node(i)            
            G_per_cc = G_per.subgraph(sorted(nx.connected_components(G_per), key=len, reverse=True)[0])
            # VISUALIZACION G_PER
            degree_sequence = sorted((d for n, d in G_per.degree()), reverse=True)
            fig = plt.figure("Degree of a random graph", figsize=(16, 16))
            # Create a gridspec for adding subplots of different sizes
            axgrid = fig.add_gridspec(5, 4)
            ax0 = fig.add_subplot(axgrid[0:3, :])
            pos = nx.spring_layout(G_per, seed=10396953)
            nx.draw_networkx_nodes(G_per, pos, ax=ax0, node_size=10)
            nx.draw_networkx_edges(G_per, pos, ax=ax0, alpha=0.4)
            ax0.set_title("All components of G")
            ax0.set_axis_off()
            ax1 = fig.add_subplot(axgrid[3:, :2])
            ax1.plot(degree_sequence, "b-", marker="o")
            ax1.set_title("Degree Rank Plot")
            ax1.set_ylabel("Degree")
            ax1.set_xlabel("Rank")
            ax2 = fig.add_subplot(axgrid[3:, 2:])
            ax2.bar(*np.unique(degree_sequence, return_counts=True))
            ax2.set_title("Degree histogram")
            ax2.set_xlabel("Degree")
            ax2.set_ylabel("Number of Nodes")
            fig.tight_layout()
            plt.show()
            #VISUALIZACION G_PER_CC
            degree_sequence = sorted((d for n, d in G_per_cc.degree()), reverse=True)
            fig = plt.figure("Degree of a random graph", figsize=(16, 16))
            # Create a gridspec for adding subplots of different sizes
            axgrid = fig.add_gridspec(5, 4)
            ax0 = fig.add_subplot(axgrid[0:3, :])
            pos = nx.spring_layout(G_per_cc, seed=10396953)
            nx.draw_networkx_nodes(G_per_cc, pos, ax=ax0, node_size=10)
            nx.draw_networkx_edges(G_per_cc, pos, ax=ax0, alpha=0.4)
            ax0.set_title("All components of G")
            ax0.set_axis_off()
            ax1 = fig.add_subplot(axgrid[3:, :2])
            ax1.plot(degree_sequence, "b-", marker="o")
            ax1.set_title("Degree Rank Plot")
            ax1.set_ylabel("Degree")
            ax1.set_xlabel("Rank")
            ax2 = fig.add_subplot(axgrid[3:, 2:])
            ax2.bar(*np.unique(degree_sequence, return_counts=True))
            ax2.set_title("Degree histogram")
            ax2.set_xlabel("Degree")
            ax2.set_ylabel("Number of Nodes")
            fig.tight_layout()
            plt.show()
        return mean_s, mean_n, S