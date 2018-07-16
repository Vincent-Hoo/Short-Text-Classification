'''
这个代码是将文本-词二部图经过随机游走过程，生成序列，再将序列中的单词去掉，就得到文本序列
得到序列后，经过word2vec生成文本向量
'''
import logging
import sys
from io import open
from os import path
from time import time
from glob import glob
from six.moves import range, zip, zip_longest
from six import iterkeys
from collections import defaultdict, Iterable
from multiprocessing import cpu_count
import random
from random import shuffle
from itertools import product,permutations
from scipy.io import loadmat
from scipy.sparse import issparse

from concurrent.futures import ProcessPoolExecutor

from multiprocessing import Pool
from multiprocessing import cpu_count
from gensim.models import Word2Vec
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
logger = logging.getLogger("randwalk")

#parameter
sample_size = 127600


class Graph(defaultdict):
  """Efficient basic implementation of nx `Graph' â€“ Undirected graphs with self loops"""  
  def __init__(self):
    super(Graph, self).__init__(list)

  def nodes(self):
    return self.keys()

  def adjacency_iter(self):
    return self.iteritems()

  def subgraph(self, nodes={}):
    subgraph = Graph()
    
    for n in nodes:
      if n in self:
        subgraph[n] = [x for x in self[n] if x in nodes]
        
    return subgraph

  def make_undirected(self):
  
    t0 = time()

    for v in self.keys():
      for other in self[v]:
        if v != other:
          self[other].append(v)
    
    t1 = time()
    logger.info('make_directed: added missing edges {}s'.format(t1-t0))

    self.make_consistent()
    return self

  def make_consistent(self):
    t0 = time()
    for k in iterkeys(self):
      self[k] = list(sorted(set(self[k])))
    
    t1 = time()
    logger.info('make_consistent: made consistent in {}s'.format(t1-t0))

    self.remove_self_loops()

    return self

  def remove_self_loops(self):

    removed = 0
    t0 = time()

    for x in self:
      if x in self[x]: 
        self[x].remove(x)
        removed += 1
    
    t1 = time()

    logger.info('remove_self_loops: removed {} loops in {}s'.format(removed, (t1-t0)))
    return self

  def check_self_loops(self):
    for x in self:
      for y in self[x]:
        if x == y:
          return True
    
    return False

  def has_edge(self, v1, v2):
    if v2 in self[v1] or v1 in self[v2]:
      return True
    return False

  def degree(self, nodes=None):
    if isinstance(nodes, Iterable):
      return {v:len(self[v]) for v in nodes}
    else:
      return len(self[nodes])

  def order(self):
    "Returns the number of nodes in the graph"
    return len(self)    

  def number_of_edges(self):
    "Returns the number of nodes in the graph"
    return sum([self.degree(x) for x in self.keys()])/2

  def number_of_nodes(self):
    "Returns the number of nodes in the graph"
    return order()

  def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
    """ Returns a truncated random walk.

        path_length: Length of the random walk.
        alpha: probability of restarts.
        start: the start node of the random walk.
    """
    G = self
    if start:
      path = [start]
    else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rand.choice(list(G.keys()))]

    while len(path) < path_length:
      cur = path[-1]
      if len(G[cur]) > 0:
        if rand.random() >= alpha:
          path.append(rand.choice(G[cur]))
        else:
          path.append(path[0])
      else:
        break
    return [str(node)for node in path if node < sample_size ]

def load_edgelist(file_, undirected=True):
  G = Graph()
  with open(file_) as f:
    for l in f:
      x, y = l.strip().split()[:2]
      x = int(x)
      y = int(y)
      G[x].append(y)
      if undirected:
        G[y].append(x)
  
  G.make_consistent()
  return G

def build_deepwalk_corpus(G, num_paths, path_length, alpha=0,
                      rand=random.Random(0)):
  walks = []

  nodes = list(G.nodes())
  
  
  for cnt in range(num_paths):
    rand.shuffle(nodes)
    print(str(cnt+1) + "/" + str(num_paths))
    for node in nodes:
      walks.append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))
  
  return walks



def main():
    parser = ArgumentParser("document_randwalk", formatter_class = ArgumentDefaultsHelpFormatter, conflict_handler = "resolve")
    
    parser.add_argument("--dimension", default = 100, type = int)
    parser.add_argument("--input", required = True)
    parser.add_argument("--output", required = True)
    parser.add_argument("--walk-length", default = 20, type = int)
    parser.add_argument("--walk-num", default = 5, type = int)
    
    args = parser.parse_args()
    
    print('start reading graph')
    G = load_edgelist(args.input)
    
    print("start walking")
    walks = build_deepwalk_corpus(G, num_paths=args.walk_num, path_length=args.walk_length, alpha=0, rand=random.Random(0))
    window_size = 10
    
    print("start training model")
    model = Word2Vec(walks, size=args.dimension, window=window_size, min_count=0, sg=1, hs=1)
    model.wv.save_word2vec_format(args.output)



if __name__ == "__main__":
    sys.exit(main())
