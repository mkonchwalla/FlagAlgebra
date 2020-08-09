import sys
print("Python version: {}".format(sys.version))
import itertools
import numpy as np
import random
from collections import Counter

class Graph:
    def __init__(self,V=[], E=[]):
        self.Verticies= list(V)
        self.Edges = list(E) 

    def add_vertex(self,v):
        if v in self.Verticies:
            raise Exception("Vertex is already in the set")
        elif isinstance(v,Vertex):
            self.Verticies.append(v)

    def create_vertex(self,nbd=set() , name = "vertex" , label = None):
        v=Vertex(nbd,name,label)
        self+v

    def delete_vertex(self,v):
        if not v in self.Verticies:
            raise Exception("Vertex is not in the set")
        elif isinstance(v,Vertex):
            self.Verticies.pop(self.Verticies.index(v))
   
    def add_edge(self,v1,v2):
        if v1 in self.Verticies and v2 in self.Verticies:
            v1.neighbourhood.add(v2)
            v2.neighbourhood.add(v1)
            if (v1,v2) not in self.Edges:
                self.Edges.append((v1,v2))
        else:
            raise Exception("Both edges are not valid verticies in this vertex class")

    def delete_edge(self,v1,v2):
        if v1 in self.Verticies and v2 in self.Verticies and (v1,v2) in self.Edges:
            v1.neighbourhood.remove(v2)
            v2.neighbourhood.remove(v1)
            self.Edges.remove((v1,v2))
        else:
            raise Exception("Both edges are not valid verticies in this vertex class")

    def __add__(self,element):
        if isinstance(element, Edge):
            self.add_edge(element.v1,element.v2) 
        elif isinstance(element,Vertex):
            self.add_vertex(element)
        else:
            raise Exception("Invalid data structure")

    def __len__(self):
        return len(self.Verticies)

    def contains_copy(self,A):
        """
        This checks if G contains a copy of A.
        This works by taking all the k tuples of V and checking if the induced subgraph is isomorphic to A
        """
        k=len(A)
        G_v = self.Verticies[:]
        for k_tup in itertools.combinations(G_v,k):
            H=induced_subgraph(k_tup,self)
            if isIsomorphic(A,H):
                return True
        return False
    
    def __repr__(self):
        return str(self.Edges)
    
    def degrees(self):
        "Returns the degrees of each vertex in a Graph"
        d=dict(zip(self.Verticies,[0]*len(self)))
        for e in self.Edges:
            d[e[0]]+=1
            d[e[1]]+=1
        return d

    def neighbourhood(self): 
        """
        Returns the neighbourhood for each vertex in a Graph
        """
        d=dict(zip(self.Verticies,[[] for i in range(len(self))]))
        for e in self.Edges:
            d[e[0]].append(e[1])
            d[e[1]].append(e[0])
        return d
        
class Vertex: 
    def __init__(self, nbd=set(),name="vertex",label=None):
        self.neighbourhood = set(nbd)
        self.name=name
        self.label=label

    def __repr__(self):
        if self.label==None:
            return self.name
        else:
            return self.label
    
    def clear_neighbourhood(self):
        self.neighbourhood = set()

    def unlabel_vertex(self):
        self.label= None

class Edge:

    def __init__(self,v1,v2):
        self.head=v1
        self.tail = v2 
        self.edge= (v1,v2)


class DiGraph(Graph):
        def __init__(self,V=[], E=[]):
            super().__init__(V, E) 

class Flag(Graph):

    def __init__(self,V=[], E=[],sigma= None):
        super().__init__(V, E) 
        self.sigma= sigma
        self.labelled_verticies= {}
        self.unlabelled_verticies = self.Verticies[:]
        if sigma !=None and type(sigma)==dict:
            self.assign_label(dict(sigma))

    def add_vertex(self,v):
        if v in self.Verticies:
            raise Exception("Vertex is already in the set")
        elif isinstance(v,Vertex):
            self.Verticies.append(v)
            self.unlabelled_verticies.append(v)

    def mul(self,element):
        """
        Multiplies different graphs together to form partially labelled graphs. 
        """
        if isinstance(element,Flag): #Checks Flag
            if isIsomorphic( self.create_type(), element.create_type()): #Checks types are the same
               if set([v.label for v in self.labelled_verticies.values()]) == set([v.label for v in element.labelled_verticies.values()]): # Checks the same labels
                   F_B_V = element.labelled_verticies.values()
                   New_F=induced_subgraph(self.Verticies,self)                
                   corresponding_vertices=dict([(u,v) for u in F_B_V for v in New_F.labelled_verticies.values() if u.label==v.label] )
                   for v in F_B_V: 
                       corresponding_v = corresponding_vertices[v]
                       for u in element.Verticies: # Goes through the element verticies and adds verticies 
                           if u not in New_F.Verticies and u not in F_B_V: #If its not in the set already and not a labelled vertex
                               New_F.add_vertex(u)
                               if (u,v) in element.Edges or (v,u) in element.Edges:
                                New_F.add_edge(corresponding_v,u)
                   return New_F
               else: 
                    raise Exception("The labels are not the same")
            else: 
                raise Exception(" The Types are not Isomorphic")       # The idea here is to generate the new Flag by recreating it and generating the linear span of combinations with labels 
        else:  raise Exception("Invalid Structure – Not a flag")          

    def assign_label(self,sigma):
        """
        Assigns label to the flag given the embedding.
        Sigma should be a dictionary of the required Verticies and the label
        Example: sigma = {a: v4 , b :v6}
        """
        self.labelled_verticies={}
        self.sigma = sigma
        for l,v in sigma.items():
            v.label = l
            self.labelled_verticies[self.Verticies.index(v)]=v
        self.unlabelled_verticies=self.Verticies[:]
        [self.unlabelled_verticies.remove(elem) for elem in self.labelled_verticies.values() ]

    def unlabel_all(self):
        for v in self.labelled_verticies.values():
            v.unlabel_vertex()

    def create_type(self): 
        return induced_subgraph(self.labelled_verticies.values(), self) 

class DiFlag(Flag):
        def __init__(self,V=[], E=[],sigma=None):
            super().__init__(V, E,sigma) 
"""
Miscellaneous functions 
"""
def nCk(n,k):
    return np.math.factorial(n) /  (np.math.factorial(k)*np.math.factorial(n-k))

def nPk(n,k):
    return np.math.factorial(n) / np.math.factorial(n-k)

def find_key(value,dictionary):
     for k,v in dictionary.items():
        if v==value: return k

def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))
"""
The following section of code generates some common graphs: 
"""

def generate_complete_graph(n):
    V = [Vertex(name="v"+str(i)) for i in range(1,n+1)]
    G=Graph(V)
    for V1,V2 in itertools.combinations(G.Verticies,2):
        G.add_edge(V1,V2)
    return G

def generate_path_graph(n):
    V = [Vertex(name="v"+str(i)) for i in range(1,n+1)]
    G=Graph(V)
    edges=[(V[i],V[i+1]) for i in range(n-1)]
    for e in edges:
        G.add_edge(e[0],e[1])
    return G

def generate_cycle_graph(n):
    V = [Vertex(name="v"+str(i)) for i in range(1,n+1)]
    G=Graph(V)
    edges=[(V[i],V[i+1]) for i in range(n-1)]
    for e in edges:
        G.add_edge(e[0],e[1])
    G.add_edge(V[0],V[-1])
    return G

def generate_random_graph(n,p):
    """
    n - no of edges
    p - probability
    This follows from the Erdos Renyi Model"""
    V = [Vertex(name="v"+str(i)) for i in range(1,n+1)]
    G=Graph(V)
    for V1,V2 in itertools.combinations(G.Verticies,2):
        if random.random() <= p:
            G.add_edge(V1,V2)
    return G

def generate_hyper_cube(n):
    V=[Vertex(name=''.join([str(y) for y in x])) for x in itertools.product([0,1],repeat=n)]
    G=Graph(V)
    def differAtOneBitPos( a , b ): 
        def isPowerOfTwo( x ): 
            return x and (not(x & (x - 1)))
        return isPowerOfTwo(a ^ b) 
    for V1,V2 in itertools.combinations(V,2):
        if differAtOneBitPos(int(V1.name,2),int(V2.name,2)):
            G.add_edge(V1,V2)
    return G

def generate_complete_digraph(n):
    V = [Vertex(name="v"+str(i)) for i in range(1,n+1)]
    G=DiGraph(V)
    for V1,V2 in itertools.permutations(G.Verticies,2):
        G.add_edge(V1,V2)
    return G

def generate_dipath(n):
    V = [Vertex(name="v"+str(i)) for i in range(1,n+1)]
    G=DiGraph(V)
    edges=[(V[i],V[i+1]) for i in range(n-1)]
    for e in edges:
        G.add_edge(e[0],e[1])
    return G

def generate_directed_cycle(n):
    V = [Vertex(name="v"+str(i)) for i in range(1,n+1)]
    G=DiGraph(V)
    edges=[(V[i],V[i+1]) for i in range(n-1)]
    for e in edges:
        G.add_edge(e[0],e[1])
    G.add_edge(V[-1],V[0])
    return G

"""
Core functions
The following functions are the computations for the values to generate the Cauchy Schwartz Inequalities.
"""
def complement(H,G):
    l=H.Edges[:]
    l_c=[x for x in G.Edges if x not in l]
    return Graph(H.Verticies, l_c)


def generate_adjaceny_matrix(G):
    if isinstance(G,Flag) or isinstance(G,DiFlag):
        verts = list(G.labelled_verticies.values()) + list(G.unlabelled_verticies)[:]#This basically fixes the labelled vertices at the start for the adjacency matrix 
    else:
        verts = list(G.Verticies) #Bugfix - use list
    n=len(verts)
    M=np.zeros((n,n),dtype=int)
    if isinstance(G,DiGraph):
        for v,u in list(G.Edges):
            M[verts.index(v),verts.index(u)] = 1
    else:  
        for v,u in list(G.Edges):
            M[verts.index(u),verts.index(v)] = 1
            M[verts.index(v),verts.index(u)] = 1
    return M

def induced_subgraph(A,G):
    """
    A - The list of verticies composing of the subgraph.
    G - The graph 
    """
    if not set(A).issubset(set(G.Verticies)):  
        raise Exception("A contains an invalid vertex")
    else:
        A=list(A)
        [v.clear_neighbourhood() for v in A]
        if isinstance(G,Flag):
            H=Flag(V=A,sigma=G.sigma)
        else:
            H=Graph(V=A)
        for e in G.Edges:
            if e[0] in A and e[1] in A:
                H.add_edge(e[0],e[1])
        return H

def isIsomorphic(A,B): 
    """ 
    Checks if Graphs A and B are isomorphic by comparing permuations of adjacency matrix.
    """
    A_M = generate_adjaceny_matrix(A)
    B_M = generate_adjaceny_matrix(B)
    n=len(A_M)
    m=len(B_M)
    if n!=m or len(A.Edges)!=len(B.Edges):  
        return False
    if Counter(A.degrees().values()) != Counter(B.degrees().values()):
        return False
    for it in itertools.permutations(range(n)):
        P=np.array([[1 if j ==i else 0 for j in range(n)] for i in it]) # Permtuation matrix generated 
        if np.array_equal(A_M ,np.dot(np.dot(P, B_M), P.T)): 
            return True
    return False

def isFlagIsomorphic(F1,F2): 
    """ 
    This compares if 2 Flags are isomorphic.
    This fixes the types in the adjacency matrix and checks the permutations of the remaining combinations.
    """
    F1_M = generate_adjaceny_matrix(F1)
    F2_M = generate_adjaceny_matrix(F2)
    if isIsomorphic(F1.create_type(),F2.create_type()):
        n=len(F1_M)
        m=len(F2_M)
        s=len(F1.create_type())
        if n!=m or len(F1.Edges)!=len(F2.Edges):  
            return False
        if Counter(F1.degrees().values()) != Counter(F2.degrees().values()):
            return False
        for it in itertools.permutations(range(s,n)): #Requires labelling in the same order
            P=np.array([[1 if j ==i else 0 for j in range(n)] for i in list(range(s))+list(it)]) # Permtuation matrix generated 
            
            if np.array_equal(F1_M ,np.dot(np.dot(P, F2_M), P.T)): 
                return True
    return False

def edge_density(G):
     """
     This calculates the edge density of a graph, G
     """
     n=len(G)
     return (2.*len(G.Edges))/ (n*(n-1))

def induced_homomorphism_density(A,G):
    """
    This function calculates the induced homorphism density of a graph A in graph G. 
    This works by taking all k tuples of G and taking the subgraph and checking isomorphism.
    """
    k=len(A)
    n=len(G)
    G_v = G.Verticies[:]
    c=0 # the counter 
    for k_tup in itertools.combinations(G_v,k):
        H=induced_subgraph(k_tup,G)
        if isIsomorphic(A,H):
            c+=1
    return c/nCk(n,k)

def flag_density(F,G,theta):
    """
    F - Flag 
    G - Graph 
    Theta – which maps the labelled vertices of F to G
    """
    k=len(F)
    s=len(F.sigma)
    n=len(G)
    G_flag = Flag( V=G.Verticies , E = G.Edges, sigma = theta)
    G_v = G_flag.unlabelled_verticies[:]
    c=0 # the counter 
    for k_tup in itertools.combinations(G_v,k-s):
        H=induced_subgraph(list(G_flag.labelled_verticies.values())+list(k_tup),G_flag)
        if isFlagIsomorphic(F,H):
            c+=1
    return c/nCk(n-s,k-s)

def two_flag_density(F_i,F_j,G,theta):
    """
    F_i , F_j - Flags of the same size and with the same type and they fit in G.
    G - Graph 
    Theta – which maps the labelled vertices of F to G
    """
    k_1=len(F_i)
    k_2 = len(F_j)
    s=len(F_i.sigma)
    G_flag = Flag( V=G.Verticies , E = G.Edges, sigma = theta)
    G_v = G_flag.unlabelled_verticies[:]
    c=0 # the counter 
    total=0
    for k_tup in itertools.permutations(G_v,k_1 + k_2- 2*s):
        total+=1
        H1=induced_subgraph(list(G_flag.labelled_verticies.values())+list(k_tup)[:k_1 - s],G_flag)
        H2=induced_subgraph(list(G_flag.labelled_verticies.values())+list(k_tup)[k_1 - s:],G_flag)
        if isFlagIsomorphic(F_i,H1) and isFlagIsomorphic(F_j, H2):
            c+=1 
    return c/total

def E_theta(F_i,F_j,G): 
    labels=list(F_i.sigma.keys())
    s= len(labels)
    n=len(G)
    E_value=0
    for vertices in itertools.permutations(G.Verticies,s):
        theta=dict(zip(labels,list(vertices)))
        E_value+= two_flag_density(F_i,F_j,G,theta)
    return E_value / nPk(n,s)


"""
More generators
"""    

def generate_isomorphic_graphs_n_e(n,e):
    """
    Generates all graphs up to isomorphism on n vertices with e edges
    """
    V = [Vertex(name="v"+str(i)) for i in range(1,n+1)]
    unique_graphs=[]
    for edges in itertools.combinations(list(itertools.combinations(V,2)),e):
        G=Graph(V,E=[])
        for edge in edges:
            G.add_edge(edge[0],edge[1])
        if True in list(map(lambda x: isIsomorphic(x,G),unique_graphs)): 
            pass
        else: 
            unique_graphs.append(G)
    return unique_graphs

def generate_all_isomorphisms(n):
    graphs=[]
    E=nCk(n,2)
    for e in range(int(E+1)):
        V = [Vertex(name="v"+str(i)) for i in range(1,n+1)]
        unique_graphs=[] 
        for edges in itertools.combinations(list(itertools.combinations(V,2)),e):
            G=Graph(V,E=[])
            for edge in edges:
                G.add_edge(edge[0],edge[1])
            if True in list(map(lambda x: isIsomorphic(x,G),unique_graphs)): 
                pass
            else: 
                unique_graphs.append(G)
        graphs+=unique_graphs
    return graphs

def generate_subcubes(n):
    cube=generate_hyper_cube(n)
    Gs=[]
    i=0
    for edges in itertools.combinations(cube.Edges,5):
        i+=1
        if i%10==0: print(i)
        G=Graph(cube.Verticies)
        for edge in edges:
            G.add_edge(edge[0],edge[1])
        if True in list(map(lambda x: isIsomorphic(x,G),Gs)): 
            pass
        else: 
            print(G.Edges)
            Gs.append(G)
    return Gs

def generate_all_flags_isomorphisms(n,txt):
    "Not the best function but it works – Do not call and is not generalized copy and use when needed"
    graphs=[]
    f=open(txt,'w')
    k3=generate_complete_graph(3)
    i=0
    for e in range(int(E+1)):
        V = [Vertex(name="v"+str(i)) for i in range(1,n+1)]
        sigma = {'a':V[0],'b':V[1] , 'c': V[2]}
        unique_graphs=[]
        for edges in itertools.combinations(list(itertools.combinations(V,2)),e):
            if (V[0],V[2]) in edges or (V[1],V[2]) in edges or (V[0],V[1]) in edges:
                continue
            i+=1
            G=Flag(V,E=[],sigma=sigma)
            for edge in edges:
                G.add_edge(edge[0],edge[1])
            H=Graph(G.Verticies,G.Edges)
            if True in list(map(lambda x: isFlagIsomorphic(x,G),unique_graphs)): 
                pass
            elif not H.contains_copy(k3): 
                unique_graphs.append(G)
                f.write(str(G.Verticies) +str(G.Edges) +"\n")
                print(G.Verticies,G.Edges)
        graphs+=unique_graphs
    return graphs
