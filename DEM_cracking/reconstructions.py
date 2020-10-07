# coding: utf-8
import scipy.sparse as sp
from dolfin import *
from numpy import array,arange,append
from DEM_cracking.mesh_related import *
from DEM_cracking.facet_reconstruction import *

def DEM_to_DG_matrix(problem):
    """Creates a csr companion matrix to get the cells values of a DEM vector."""

    return sp.eye(problem.nb_dof_cells, n = problem.nb_dof_DEM, format='csr')

def DEM_to_DG_1_matrix(problem):
    dofmap_DG_0 = problem.DG_0.dofmap()
    dofmap_tens_DG_0 = problem.W.dofmap()
    elt_1 = problem.DG_1.element()
    dofmap_DG_1 = problem.DG_1.dofmap()
    nb_total_dof_DG_1 = dofmap_DG_1.global_dimension()
    
    matrice_resultat_1 = sp.dok_matrix((nb_total_dof_DG_1,problem.nb_dof_DEM)) #Matrice vide.
    matrice_resultat_2 = sp.dok_matrix((nb_total_dof_DG_1,problem.nb_dof_grad)) #Matrice vide.
    
    for c in cells(problem.mesh):
        index_cell = c.index()
        dof_position = dofmap_DG_1.cell_dofs(index_cell)

        #filling-in the matrix to have the constant cell value
        DG_0_dofs = dofmap_DG_0.cell_dofs(index_cell)
        for dof in dof_position:
            matrice_resultat_1[dof, DG_0_dofs[dof % problem.d]] = 1.

        #filling-in part to add the gradient term
        position_barycentre = problem.Graph.nodes[index_cell]['pos']
        pos_dof_DG_1 = elt_1.tabulate_dof_coordinates(c)
        tens_dof_position = dofmap_tens_DG_0.cell_dofs(index_cell)
        for dof,pos in zip(dof_position,pos_dof_DG_1): #loop on quadrature points
            diff = pos - position_barycentre
            for i in range(problem.dim):
                matrice_resultat_2[dof, tens_dof_position[(dof % problem.d)*problem.d + i]] = diff[i]

    matrice_resultat_1 = matrice_resultat_1.tocsr()
    matrice_resultat_2 = matrice_resultat_2.tocsr()
    return (matrice_resultat_1 +  matrice_resultat_2 * problem.mat_grad * problem.DEM_to_CR), matrice_resultat_1, matrice_resultat_2


def DEM_to_CR_matrix(problem):
    #Computing the facet reconstructions
    convex_coord,convex_num = facet_reconstruction(problem)

    #Storing the facet reconstructions in a matrix
    complete_matrix = sp.dok_matrix((problem.nb_dof_CR,problem.nb_dof_DEM)) #Empty matrix.
    trace_matrix = sp.dok_matrix((problem.nb_dof_CR,problem.nb_dof_DEM)) #Empty matrix.
    
    for x,y in problem.Graph.edges(): #looping through all facets of the mesh
        num_global_face = problem.Graph[x][y]['num']
        num_global_ddl = problem.Graph[x][y]['dof_CR']
        convexe_f = convex_num.get(num_global_face)
        convexe_c = convex_coord.get(num_global_face)

        Y = max(x,y)

        if abs(Y) >= problem.nb_dof_cells // problem.d and len(problem.Graph.nodes[Y]['dof']) > 0: #facet holds Dirichlet dof
            dof = problem.Graph.nodes[Y]['dof']
            dirichlet_components = problem.Graph.nodes[Y]['dirichlet_components']
            count = 0
            for num,dof_CR in enumerate(num_global_ddl):
                if num in dirichlet_components:
                    trace_matrix[dof_CR,dof[count]] = 1.
                    complete_matrix[dof_CR,dof[count]] = 1.
                    count += 1
                    
            for i,j in zip(convexe_f,convexe_c):
                if 0 not in problem.Graph.nodes[Y]['dirichlet_components']:
                    complete_matrix[num_global_ddl[0],i[0]] += j #because a single dof can be used twice with new symetric reconstruction
                if problem.d >=2 and 1 not in problem.Graph.nodes[Y]['dirichlet_components']:
                    complete_matrix[num_global_ddl[1],i[1]] += j
                if problem.d == 3 and 2 not in problem.Graph.nodes[Y]['dirichlet_components']:
                    complete_matrix[num_global_ddl[2],i[2]] += j
        else: #facet holds no Dirichlet dofs
            for i,j in zip(convexe_f,convexe_c):
                complete_matrix[num_global_ddl[0],i[0]] += j #because a single dof can be used twice with new symetric reconstruction
                if problem.d >= 2:
                    complete_matrix[num_global_ddl[1],i[1]] += j
                if problem.d == 3:
                    complete_matrix[num_global_ddl[2],i[2]] += j
            
        
    return complete_matrix.tocsr(), trace_matrix.tocsr()
