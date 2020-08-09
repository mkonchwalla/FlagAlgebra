from main import *

BugFix=True

if BugFix:
    K8 = generate_complete_graph(8)
    A=[K8.Verticies[i] for i in range(4)]
    H=induced_subgraph(A,K8)
    R6=generate_random_graph(10,0.6)
    K4 = generate_complete_graph(4)
    C2=generate_path_graph(2) 

    if False:
        print(C2.Verticies,C2.Edges)
        print("R6",R6.Edges,R6.Verticies)
        print(edge_density(R6))
        print(induced_homomorphism_density(C2,R6))

if BugFix:
    F = Flag(K4.Verticies,K4.Edges)
    sigma=dict([ ("ab"[i],F.Verticies[i]) for i in range(2) ])
    F.assign_label(sigma)
    H=F.create_type()
    H.create_vertex(name="v6")
    H.add_edge(H.Verticies[0],H.Verticies[-1])


    if False: 
        G1=generate_complete_graph(3)
        G1.create_vertex(name="v4")

        G1.add_edge( G1.Verticies[2],G1.Verticies[3] )
        sigma=dict([ ("ab"[i],G1.Verticies[i]) for i in range(2) ])

        G1=Flag(G1.Verticies,G1.Edges,sigma=sigma)
        print(G1.Verticies)
        print(G1.Edges)
        H=G1.create_type()

        V = [Vertex(name="v"+str(i)) for i in range(5,9)]
        G2=Flag(V)
        G2.add_edge(G2.Verticies[0], G2.Verticies[2])
        G2.add_edge(G2.Verticies[0], G2.Verticies[3])
        G2.add_edge(G2.Verticies[1], G2.Verticies[2])
        G2.add_edge(G2.Verticies[1], G2.Verticies[3])
        G2.add_edge(G2.Verticies[2], G2.Verticies[3])

        sigma=dict([ ("ab"[i],G2.Verticies[i]) for i in range(2) ])
        G2.assign_label(sigma)
        print(isFlagIsomorphic(G1,G2))
 
if False: 
        G1=generate_complete_graph(3)
        G1.create_vertex(name="v4")
        G1.add_edge( G1.Verticies[0],G1.Verticies[3])
        theta=dict([ ("ab"[i],G1.Verticies[0]) for i in range(1) ])
        print(G1.Verticies,G1.Edges)

        G2 = Flag(G1.Verticies,G1.Edges,theta)

        # print(G2.sigma)
        # print(G2.Verticies)
        # print(G2.Edges)

        V = [Vertex(name="v"+str(i)) for i in range(1,3)]
        F = Graph(V)
        sigma=dict([ ("ab"[i],F.Verticies[i]) for i in range(1)])
        F=Flag(F.Verticies,F.Edges,sigma)
        print(F.Verticies,F.Edges)

        # print(flag_density(F,G1,theta))
        # # print(two_flag_density(F,F,G1,theta))
        # c4= generate_cycle_graph(4)
        print(E_theta(F,F,G1))
        # print(E_theta(F,F,c4))


if False: 
        V = [Vertex(name="v"+str(i)) for i in range(1,3)]
        f1 = Graph(V)
        sigma=dict([ ("ab"[i],f1.Verticies[i]) for i in range(1)])
        f1=Flag(f1.Verticies,f1.Edges,sigma)
        print(f1.Verticies,f1.Edges)
        f2 = generate_complete_graph(2)
        sigma=dict([ ("ab"[i],f2.Verticies[i]) for i in range(1)])
        f2=Flag(f2.Verticies,f2.Edges,sigma)
        print(f2.Verticies,f2.Edges)

        V = [Vertex(name="v"+str(i)) for i in range(1,4)]
        G1 = Graph(V)

        V = [Vertex(name="v"+str(i)) for i in range(1,4)]
        G2 = Graph(V)
        G2.add_edge(G2.Verticies[0], G2.Verticies[1])


        V = [Vertex(name="v"+str(i)) for i in range(1,4)]
        G3 = Graph(V)
        G3.add_edge(G3.Verticies[0], G3.Verticies[1])
        G3.add_edge(G3.Verticies[1], G3.Verticies[2])

        print(E_theta(f1,f2,G3))

        
if False: 
        G1=generate_complete_graph(3)
        G1.create_vertex(name="v4")
        G1.add_edge( G1.Verticies[0],G1.Verticies[3])
        theta=dict([ ("ab"[i],G1.Verticies[0]) for i in range(1) ])
        print(G1.Verticies,G1.Edges)

        G2 = Flag(G1.Verticies,G1.Edges,theta)

        # print(G2.sigma)
        # print(G2.Verticies)
        # print(G2.Edges)

        V = [Vertex(name="v"+str(i)) for i in range(1,3)]
        F = Graph(V)
        sigma=dict([ ("ab"[i],F.Verticies[i]) for i in range(1)])
        F=Flag(F.Verticies,F.Edges,sigma)
        print(F.Verticies,F.Edges)

        # print(flag_density(F,G1,theta))
        # # print(two_flag_density(F,F,G1,theta))
        # c4= generate_cycle_graph(4)
        print(E_theta(F,F,G1))
        # print(E_theta(F,F,c4))

K8 = generate_complete_graph(8)
A=[K8.Verticies[i] for i in range(4)]
H=induced_subgraph(A,K8)

K4=generate_complete_graph(4)

print(isIsomorphic(K4,H))