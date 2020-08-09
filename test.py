from main import *

"""
Generating H's

"""

def c4_density(G):
    """
    This function calculates the induced homorphism density of a graph A in graph G. 
    This works by taking all k tuples of G and taking the subgraph and checking isomorphism.
    """
    C4=generate_cycle_graph(4)
    n=len(G)
    G_v = G.Verticies[:]
    c=0 # the counter 
    for k_tup in itertools.combinations(G_v,4):
        H=induced_subgraph(k_tup,G)
        if isIsomorphic(C4,H):
            c+=1
    return c/nCk(n,4)

C4=generate_cycle_graph(4)
cube=generate_hyper_cube(3)
Graphs=[]

with open('hc.txt','r') as f:
        for line in f:
            V=[Vertex(name=''.join([str(y) for y in x])) for x in itertools.product([0,1],repeat=3)]
            G=Graph(V)
            l=line.strip('\n').replace('000','"000"').replace('001','"001"').replace('010','"010"').replace('100','"100"').replace('011','"011"').replace('110','"110"').replace('101','"101"').replace('111','"111"')
            l=eval(l)
            dc=dict(zip([''.join([str(y) for y in x]) for x in itertools.product([0,1],repeat=3)],V))
            if l!=[]:
                for e in l:
                    G.add_edge(dc[e[0]],dc[e[1]])
            if not G.contains_copy(C4):
                Graphs.append(G)

def subcube_edge_density(G,n):
    return len(G.Edges)/(n*2**(n-1))

def subcube_two_flag_density(F_i,F_j,G,theta):  
    """
    F_i , F_j - Flags of the same size and with the same type and they fit in G.
    G - Graph 
    Theta – which maps the labelled vertices of F to G
    Assuming s=1
    """
    k_1=len(F_i)
    k_2 = len(F_j)
    s=len(F_i.sigma)
    G_flag = Flag( V=G.Verticies , E = G.Edges, sigma = theta)
    full_cube=Graph(G.Verticies)
    def differAtOneBitPos( a , b ): 
        def isPowerOfTwo( x ): 
            return x and (not(x & (x - 1)))
        return isPowerOfTwo(a ^ b) 
    for V1,V2 in itertools.combinations(full_cube.Verticies,2):
        if differAtOneBitPos(int(V1.name,2),int(V2.name,2)):
            full_cube.add_edge(V1,V2)
    G_v=full_cube.neighbourhood()[list(G_flag.labelled_verticies.values())[0]]
    # print(G_v,G_flag.neighbourhood()[list(G_flag.labelled_verticies.values())[0]])
    c=0 # the counter 
    total=0
    for k_tup in itertools.permutations(G_v,2):
        total+=1
        H1=induced_subgraph(list(G_flag.labelled_verticies.values())+list(k_tup)[:1],G_flag)
        H2=induced_subgraph(list(G_flag.labelled_verticies.values())+list(k_tup)[-1:],G_flag)
        if isFlagIsomorphic(F_i,H1) and isFlagIsomorphic(F_j, H2):
            c+=1 
    return c/total

def subcube_E_theta(F_i,F_j,G): 
    labels=list(F_i.sigma.keys())
    s= len(labels)
    n=len(G)
    E_value=0
    for vertices in itertools.permutations(G.Verticies,s):
        theta=dict(zip(labels,list(vertices)))
        E_value+= subcube_two_flag_density(F_i,F_j,G,theta)
    return E_value / 8

V = [Vertex(name="v"+str(i)) for i in range(1,3)]
f1 = Graph(V)
sigma=dict([ ("ab"[i],f1.Verticies[i]) for i in range(1)])
f1=Flag(f1.Verticies,f1.Edges,sigma)
print(f1.Verticies,f1.Edges)

f2 = generate_complete_graph(2)
sigma=dict([ ("ab"[i],f2.Verticies[i]) for i in range(1)])
f2=Flag(f2.Verticies,f2.Edges,sigma)
print(f2.Verticies,f2.Edges)

for i in range(len(Graphs)):
    G=Graphs[i]
    print('\n',G)
    mm="e{} = 1/24 *( {} ".format(i,round(subcube_edge_density(G,3)*24))
    for F1_index,F2_index in itertools.combinations_with_replacement(range(2),2):
        # print("F"+str(F1_index+1),"F"+str(F2_index+1))
        F1= [f1,f2][F1_index]
        F2= [f1,f2][F2_index]
        if E_theta(F1,F2,G)!=0.0 or False:
            if F1 != F2:
                # print(round(840*2*E_theta(F1,F2,G)),"p{0}{1} in graph G{2}".format(F1_index+1,F2_index+1,i+1))
                mm+=" + {2} q{0}{1} ".format(F1_index+1,F2_index+1,round(24*2*subcube_E_theta(F1,F2,G)))
            else:
                # print(round(840*E_theta(F1,F2,G)),"p{0}{1} in graph G{2}".format(F1_index+1,F2_index+1,i+1) )
                mm+=" + {2} q{0}{1} ".format(F1_index+1,F2_index+1,round(24*subcube_E_theta(F1,F2,G)))

    # for F1_index,F2_index in itertools.combinations_with_replacement(range(4),2):
    #     # print("F"+str(F1_index+1),"F"+str(F2_index+1))
    #     F1= [y1,y2,y3,y4][F1_index]
    #     F2= [y1,y2,y3,y4][F2_index]
    #     if E_theta(F1,F2,G)!=0.0 or False:
    #         if F1 != F2:
    #             # print(round(840*2*E_theta(F1,F2,G)),"p{0}{1} in graph G{2}".format(F1_index+1,F2_index+1,i+1))
    #             mm+=' + '+str(round(840*2*E_theta(F1,F2,G)) )+"q{0}{1} ".format(F1_index+1,F2_index+1)
    #         else:
    #             # print(round(840*E_theta(F1,F2,G)),"p{0}{1} in graph G{2}".format(F1_index+1,F2_index+1,i+1) )
    #             mm+=' + '+str(round(840*E_theta(F1,F2,G)) )+"q{0}{1} ".format(F1_index+1,F2_index+1)
    mm+=');'
    print(mm)
