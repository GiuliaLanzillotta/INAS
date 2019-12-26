"""
This module is responsible for creating, updating and saving the graph and the embedding
"""
from node2vec import Node2Vec
import networkx as nx
import numpy as np

class embedding():

    def __init__(self, layers, dimensions, walk_length, walks):
        self.layers = layers
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.walks = walks


    def build_graph(self,npl=5):
        """
        @param : npl = nodes per layer"""
        graph = nx.Graph()
        for l in range(self.layers):
            for i in range(npl):
                graph.add_node(npl * (l) + i + 1)
            graph.add_edge(npl* l + 1, npl * l + 2)
            graph.add_edge(npl * l + 1, npl * l + 5)
            graph.add_edge(npl * l + 3, npl * l + 5)
            graph.add_edge(npl * l + 3, npl * l + 4)
            if (l != 0):
                graph.add_edge(npl * l + 3, npl * (l - 1) + 3)
                graph.add_edge(npl * l + 4, npl * (l - 1) + 4)
                graph.add_edge(npl * l + 5, npl * (l - 1) + 5)
        self.graph = graph
        return graph

    def update_graph(self):
        #TODO
        return

    def train_embedding(self):
        node2vec = Node2Vec(self.graph, self.dimensions, self.walk_length, self.walks)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        self.model = model
        return model

    def get_embeddings(self):
        nodes = [x for x in self.model.wv.vocab]
        embeddings = np.array([self.model.wv[x] for x in nodes])
        self.embeddings = embeddings
        return embeddings