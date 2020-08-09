from main import *
import time

def generate_all_isomorphisms(n,txt):
    graphs=[]
    E=nCk(n,2)
    start = time.time()
    f=open(txt,'w')
    total = sum([nCk(nCk(n,2),e) for e in range(int(E+1))])
    i=0
    for e in range(int(E+1)):
        V = [Vertex(name="v"+str(i)) for i in range(1,n+1)]
        unique_graphs=[]
        for edges in itertools.combinations(list(itertools.combinations(V,2)),e):
            if i % int(total/10000)==0: 
                print("Computed {} out of {} calculations in {} seconds".format(i,total,time.time()-start)  )
            i+=1
            G=Graph(V,E=[])
            for edge in edges:
                G.add_edge(edge[0],edge[1])
            if True in list(map(lambda x: isIsomorphic(x,G),unique_graphs)): 
                pass
            else: 
                unique_graphs.append(G)
                f.write(str(G.Verticies) +str(G.Edges) +"\n")
                print(G.Verticies,G.Edges)
        graphs+=unique_graphs
    return graphs

# generate_all_isomorphisms(7,'iso7.txt')


def generate_all_flags_isomorphisms(n,txt):
    graphs=[]
    E=nCk(n,2)
    start = time.time()
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
generate_all_flags_isomorphisms(5,"test.txt")
