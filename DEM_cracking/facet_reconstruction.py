#coding: utf-8
import networkx as nx
import numpy as np
from itertools import combinations
import sys

def facet_reconstruction(problem):
    """ Function that will compute the simplex associated to every facet so as to reconstruct its value."""
    
    res_num = dict([])
    res_coord = dict([])
    connectivity_reconstruction = dict([])
    
    for f in problem.facet_num.keys():
        coord_bary,coord_num,connectivity_recon = bary_coord(f,problem)
        #Chacking computed reconstruction is correct
        assert len(coord_bary) == len(coord_num)
        try:
            assert(max(abs(coord_bary)) < 100.)
        except AssertionError:
            print('Problem in barycentric coordinates...')
            print(max(abs(coord_bary)))
            sys.exit()
        res_coord[f] = coord_bary
        res_num[f] = coord_num
        
        #for connectivity recon in case of facet breaking
        for k in list(connectivity_recon):
            n1,n2 = problem.facet_num.get(k)
            problem.Graph[n1][n2]['recon'].add(f) # if facet k breaks, recompute reconstruction on facet i
    return res_coord,res_num

def bary_coord(num_facet, problem):
    j = problem.facet_num.get(num_facet)
    if len(j) == 2: #facet is not on boundary
        c1,c2 = j[0],j[1] #numéro des cellules voisines de la face.
        voi = set(j)
        x = problem.Graph[c1][c2]['barycentre'] #position du barycentre de la face

        #loading number of all neighbouring cells of c1 and c2
        #first set of neighbors
        path_1 = nx.neighbors(problem.Graph, c1)
        path_1 = np.array(list(path_1))
        for_deletion = np.where(np.absolute(path_1) >= problem.nb_dof_cells // problem.d)
        path_1[for_deletion] = -1
        path_1 = set(path_1) - {-1,c2}
        #second set of neighbors
        path_2 = nx.neighbors(problem.Graph, c2)
        path_2 = np.array(list(path_2))
        for_deletion = np.where(np.absolute(path_2) >= problem.nb_dof_cells // problem.d)
        path_2[for_deletion] = -1
        path_2 = set(path_2) - {-1,c1}
        nei_to_nei = path_1 | path_2 #cells that are neighbour to c1 or c2 (excluding c1 and c2 and bnd facets...)
        nei_to_nei = list(nei_to_nei)

        #computing the two (or three) reconstructions
        list_positions = []
        for l in nei_to_nei:
            list_positions.append([problem.Graph.nodes[c1]['pos'], problem.Graph.nodes[c2]['pos'], problem.Graph.nodes[l]['pos']])

        #Computation of barycentric coordinates
        coords = np.array([])
        path = []
        for l,p in zip(list_positions,nei_to_nei):
            A = np.array(l)
            A = A[1:,:] - A[0,:]
            b = np.array(x - l[0])
            try:
                aux_coord_bary = np.linalg.solve(A.T,b)
            except np.linalg.LinAlgError: #singular matrix
                #print('Error')
                pass
            else:
                aux_coord_bary = np.append(1. - aux_coord_bary.sum(), aux_coord_bary)
                coords = np.concatenate((coords,aux_coord_bary))
                path += [c1,c2,p]

        #Averaging the barycentric coordinate if required
        if len(coords) > 0:
            chosen_coord_bary = coords / (len(coords) / (problem.dim+1))
            assert abs(chosen_coord_bary.sum() - 1.) < 1.e-5
        
            #getting dofs used in the CR reconstruction
            coord_num = []
            for l in path:
                coord_num.append(problem.Graph.nodes[l]['dof'])
            #Getting the link between dofs through facets in case of breaking
            aux_aux = set()
            intersection_1 = set(path_1)
            intersection_2 = set(path_2)
            for node in intersection_1:
                assert(problem.Graph.has_edge(c1,node)) #edge is actually in the graph
                num = int(problem.Graph[c1][node]['num'])
                breakable = problem.Graph[c1][node]['breakable']
                if breakable: #edge representing an inner facet
                    aux_aux.add(num)
            for node in intersection_2:
                assert(problem.Graph.has_edge(c2,node)) #edge is actually in the graph
                num = int(problem.Graph[c2][node]['num'])
                breakable = problem.Graph[c2][node]['breakable']
                if breakable: #edge representing an inner facet
                    aux_aux.add(num)
            aux_aux.discard(num_facet) #removing the number of the facet itself (no need to recompute its associated simplex if it breaks...)

        else: #break the facet ????
            chosen_coord_bary = np.ones(problem.d)
            coord_num = [problem.Graph.nodes[c1]['dof']]
            aux_aux = set()


    elif len(j) == 1: #facet is on boundary
        c1 = j[0]
        if num_facet >= 0:
            x = problem.Graph[c1][num_facet + problem.nb_dof_cells // problem.d]['barycentre'] #position du barycentre de la face
            neigh_c1 = set(nx.neighbors(problem.Graph, c1)) - {num_facet + problem.nb_dof_cells // problem.d}
            nei_to_nei = set(nx.single_source_shortest_path(problem.Graph, c1, cutoff=2)) - {num_facet + problem.nb_dof_cells // problem.d}
        else:
            x = problem.Graph[c1][num_facet - problem.nb_dof_cells // problem.d]['barycentre'] #position du barycentre de la face
            neigh_c1 = set(nx.neighbors(problem.Graph, c1)) - {num_facet - problem.nb_dof_cells // problem.d}
            nei_to_nei = set(nx.single_source_shortest_path(problem.Graph, c1, cutoff=2)) - {num_facet - problem.nb_dof_cells // problem.d}

        #Retirer les facettes de bord de nei_to_nei et neigh_c1...
        neigh_c1 = np.array(list(neigh_c1))
        for_deletion = np.where(np.absolute(neigh_c1) >= problem.nb_dof_cells // problem.d)
        neigh_c1[for_deletion] = -1
        neigh_c1 = set(neigh_c1) - {-1}

        nei_to_nei = np.array(list(nei_to_nei))
        for_deletion = np.where(np.absolute(nei_to_nei) >= problem.nb_dof_cells // problem.d)
        nei_to_nei[for_deletion] = -1
        nei_to_nei = set(nei_to_nei) - {-1}

        #Testing fragment
        fragment = False
        count = 0
        for n in neigh_c1:
            if abs(n) >= problem.nb_dof_cells // problem.d: #node represents a second facet on the boundary
                count += 1
        if count >= problem.dim:
            fragment = True

        if fragment: #single cell detaches.
            if num_facet > 0:
                dof_CR = problem.Graph[c1][num_facet + problem.nb_dof_cells // problem.d]['dof_CR']
            else:
                dof_CR = problem.Graph[c1][num_facet - problem.nb_dof_cells // problem.d]['dof_CR']
            chosen_coord_bary = [np.ones(problem.d)]
            coord_num = [problem.Graph.nodes[c1]['dof']]
            aux_aux = set()

        else: #Cell did not detach
            chosen_coord_bary = []
            coord_num = []
            aux_aux = set()

            for dof_num in combinations(nei_to_nei, problem.dim+1): #test reconstruction with a set of right size
                list_positions = []   
                for l in dof_num:
                    list_positions.append(problem.Graph.nodes[l]['pos'])

                #Computation of barycentric coordinates
                A = np.array(list_positions)
                A = A[1:,:] - A[0,:]
                b = np.array(x - list_positions[0])
                try:
                    aux_coord_bary = np.linalg.solve(A.T,b)
                except np.linalg.LinAlgError: #singular matrix
                    pass
                else:
                    if max(max(abs(aux_coord_bary)),1.-aux_coord_bary.sum()) < 10.:
                        chosen_coord_bary = np.append(1. - aux_coord_bary.sum(), aux_coord_bary)
                        coord_num = []
                        for l in dof_num:
                            assert len(problem.Graph.nodes[l]['dof']) > 0
                            coord_num.append(problem.Graph.nodes[l]['dof'])

                        #Il va falloir remplir le aux_aux...
                        aux_aux = set()

                        for couple in combinations(dof_num,2):
                            path = nx.shortest_path(problem.Graph,source=couple[0],target=couple[1])
                            for couple_bis in combinations(path,2):
                                if problem.Graph.has_edge(couple_bis[0],couple_bis[1]):
                                    num = int(problem.Graph[couple_bis[0]][couple_bis[1]]['num'])
                                    breakable = problem.Graph[couple_bis[0]][couple_bis[1]]['breakable']
                                    if breakable:
                                        aux_aux.add(num)

                        aux_aux.discard(num_facet) #removing the number of the facet itself (no need to recompute its associated simplex if it breaks...)

                        break #reconstruction is okay. We leave the loop.
                    else:
                        pass   

    return chosen_coord_bary,coord_num,aux_aux
