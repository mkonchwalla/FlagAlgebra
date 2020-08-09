from main import *

Gs=[]
flags=[]
n=5

K3=generate_complete_graph(3)
with open('iso7.txt','r') as f:
    for line in f:
        x=line.strip('\n').replace('][',']xxx[').split('xxx')
        x_v=x[1]
        l=x_v.replace('v','')
        l=eval(l)
        V = [Vertex(name="v"+str(i)) for i in range(1,n+1)]
        G=Graph(V)
        if l!=[]:
            for e in l:
                G.add_edge(V[e[0]-1],V[e[1]-1])
        if not contains_copy(K5,G):
            Gs.append(G)
            # print(G.Edges)

# unique_flags=[]
# with open('iso5.txt','r') as f:
#     for line in f:
#         x=line.strip('\n').replace('][',']xxx[').split('xxx')
#         x_v=x[1]
#         l=x_v.replace('v','')
#         l=eval(l)
#         V = [Vertex(name="v"+str(i)) for i in range(1,n+1)]
#         F=Flag(V,sigma={'1':V[0],'2':V[1],'3':V[2]})
#         if l!=[]:
#             for e in l:
#                 F.add_edge(V[e[0]-1],V[e[1]-1])
#         if True in list(map(lambda x: isFlagIsomorphic(x,F),unique_flags)): 
#             pass
#         else:
#             flags.append(F)
#             print(F.Verticies,F.Edges)

