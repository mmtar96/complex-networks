import pandas as pd
import networkx as nx
import complex_network as cn
import random_graphs as rg

nodes = pd.read_csv('vertices.csv', delimiter = ',')
edges = pd.read_csv('enlaces.csv')

G = nx.Graph()

cn.add_data(G, nodes, edges) 
parameters = cn.params(G) 
degre_Sequence = cn.degree_sequence(G) 
cn.plotgraf(G)
Gcc, parametersGcc = cn.paramsGcc(G, 0) 
cn.powerlaw(Gcc) 

# metrics
E = cn.eigenvector(Gcc)
PR = cn.pagerank(Gcc)
C = cn.closeness(Gcc)
B = cn.betweness(Gcc)
clust = cn.clustering(Gcc)

# communities
part = cn.communities(Gcc)
cn.communities_detail(Gcc, nodes)

# percolation
mean_edges, mean_nodes, sizeGcc = cn.uniform_percolation(G) # 10 percolations

# create random graphs
GER1 = rg.er_model1(1000,2300)
GER1 = rg.er_model2(1000,2300)
GER1 = rg.config_model(G)

