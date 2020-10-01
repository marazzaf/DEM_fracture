# coding: utf-8
import scipy.sparse as sp
from dolfin import *
from numpy import array,arange,append
from DEM_cracking.mesh_related import *
from DEM_cracking.facet_reconstruction import *
from itertools import combinations

def DEM_to_DG_matrix(problem):
    """Creates a csr companion matrix to get the cells values of a DEM vector."""

    return sp.eye(problem.nb_dof_cells, n = problem.nb_dof_DEM, format='csr')

def matrice_passage_ccG_DG_1(mesh_, nb_ddl_ccG_, d_, dim_, mat_grad_, passage_ccG_CR):
    if d_ == 1:
        EDG_0 = FunctionSpace(mesh_, 'DG', 0)
        EDG_1 = FunctionSpace(mesh_, 'DG', 1)
        tens_DG_0 = VectorFunctionSpace(mesh_, 'DG', 0)
    else:
        EDG_0 = VectorFunctionSpace(mesh_, 'DG', 0)
        EDG_1 = VectorFunctionSpace(mesh_, 'DG', 1)
        tens_DG_0 = TensorFunctionSpace(mesh_, 'DG', 0)
    dofmap_DG_0 = EDG_0.dofmap()
    dofmap_DG_1 = EDG_1.dofmap()
    dofmap_tens_DG_0 = tens_DG_0.dofmap()
    elt_0 = EDG_0.element()
    elt_1 = EDG_1.element()
    nb_total_dof_DG_1 = len(dofmap_DG_1.dofs())
    nb_ddl_grad = len(dofmap_tens_DG_0.dofs())
    matrice_resultat_1 = sp.dok_matrix((nb_total_dof_DG_1,nb_ddl_ccG_)) #Matrice vide.
    matrice_resultat_2 = sp.dok_matrix((nb_total_dof_DG_1,nb_ddl_grad)) #Matrice vide.
    
    for c in cells(mesh_):
        index_cell = c.index()
        dof_position = dofmap_DG_1.cell_dofs(index_cell)

        #filling-in the matrix to have the constant cell value
        DG_0_dofs = dofmap_DG_0.cell_dofs(index_cell)
        for dof in dof_position:
            matrice_resultat_1[dof, DG_0_dofs[dof % d_]] = 1.

        #filling-in part to add the gradient term
        position_barycentre = elt_0.tabulate_dof_coordinates(c)[0]
        pos_dof_DG_1 = elt_1.tabulate_dof_coordinates(c)
        tens_dof_position = dofmap_tens_DG_0.cell_dofs(index_cell)
        for dof,pos in zip(dof_position,pos_dof_DG_1): #loop on quadrature points
            diff = pos - position_barycentre
            for i in range(dim_):
                matrice_resultat_2[dof, tens_dof_position[(dof % d_)*d_ + i]] = diff[i]

    matrice_resultat_1 = matrice_resultat_1.tocsr()
    matrice_resultat_2 = matrice_resultat_2.tocsr()
    return (matrice_resultat_1 +  matrice_resultat_2 * mat_grad_ * passage_ccG_CR), matrice_resultat_1, matrice_resultat_2


def DEM_to_CR_matrix(problem):
    #Computing the facet reconstructions
    convex_num,convex_coord = facet_reconstruction(problem)

    #Storing the facet reconstructions in a matrix
    complete_matrix = sp.dok_matrix((nb_total_dof_CR,nb_ddl_ccG_)) #Empty matrix.
    trace_matrix = sp.dok_matrix((nb_total_dof_CR,nb_ddl_ccG_)) #Empty matrix.
    
    for x,y in G_.edges(): #looping through all facets of the mesh
        num_global_face = G_[x][y]['num']
        num_global_ddl = G_[x][y]['dof_CR']
        convexe_f = conv_num.get(num_global_face)
        convexe_c = conv_coord.get(num_global_face)

        Y = max(x,y)

        if abs(Y) >= nb_ddl_cells // d_ and len(G_.node[Y]['dof']) > 0: #facet holds Dirichlet dof
            dof = G_.node[Y]['dof']
            dirichlet_components = G_.node[Y]['dirichlet_components']
            count = 0
            for num,dof_CR in enumerate(num_global_ddl):
                if num in dirichlet_components:
                    trace_matrix[dof_CR,dof[count]] = 1.
                    complete_matrix[dof_CR,dof[count]] = 1.
                    count += 1
                    
            for i,j in zip(convexe_f,convexe_c):
                if 0 not in G_.node[Y]['dirichlet_components']:
                    complete_matrix[num_global_ddl[0],i[0]] += j #because a single dof can be used twice with new symetric reconstruction
                if d_ >=2 and 1 not in G_.node[Y]['dirichlet_components']:
                    complete_matrix[num_global_ddl[1],i[1]] += j
                if d_ == 3 and 2 not in G_.node[Y]['dirichlet_components']:
                    complete_matrix[num_global_ddl[2],i[2]] += j
        else: #facet holds no Dirichlet dofs
            for i,j in zip(convexe_f,convexe_c):
                complete_matrix[num_global_ddl[0],i[0]] += j #because a single dof can be used twice with new symetric reconstruction
                if d_ >= 2:
                    complete_matrix[num_global_ddl[1],i[1]] += j
                if d_ == 3:
                    complete_matrix[num_global_ddl[2],i[2]] += j
            
        
    return complete_matrix.tocsr(), trace_matrix.tocsr()
