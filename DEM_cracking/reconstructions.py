# coding: utf-8
import scipy.sparse as sp
from dolfin import *
from numpy import array,arange,append
from DEM.mesh_related import *
from itertools import combinations

def DEM_to_DG_matrix(problem):
    """Creates a csr companion matrix to get the cells values of a DEM vector."""

    return sp.eye(problem.nb_cell_dofs, n = problem.nb_dof_DEM, format='csr')

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

def gradient_matrix(problem):
    """Creates a matrix computing the cell-wise gradient from the facet values stored in a Crouzeix-raviart FE vector."""
    vol = CellVolume(problem.mesh)

    #variational form gradient
    u_CR = TrialFunction(problem.CR)
    Dv_DG = TestFunction(problem.W)
    a = inner(grad(u_CR), Dv_DG) / vol * dx
    A = assemble(a)
    row,col,val = as_backend_type(A).mat().getValuesCSR()
    return sp.csr_matrix((val, col, row))


def DEM_to_CR_matrix(problem):
    dofmap_CR = problem.CR.dofmap()
    nb_total_dof_CR = dofmap_CR.global_dimension()

    #computing the facet reconstructions
    
    
    #Computing the facet reconstructions
    convex_num,convex_coord = facet_interpolation(facet_num,pos_bary_cells,pos_ddl_vertex,dico_pos_bary_faces,problem.dim,problem.d)

    #Storing the facet reconstructions in a matrix
    matrice_resultat = sp.dok_matrix((nb_total_dof_CR,nb_dof_ccG)) #Matrice vide.
    for f in facets(problem.mesh):
        num_global_face = f.index()
        num_global_ddl = dofmap_CR.entity_dofs(problem.mesh, problem.dim - 1, array([num_global_face], dtype="uintp"))
        convexe_f = convex_num.get(num_global_face)
        convexe_c = convex_coord.get(num_global_face)

        if convexe_f != None: #Face interne, on interpolle la valeur !
            for i,j in zip(convexe_f,convexe_c):
                matrice_resultat[num_global_ddl[0],i[0]] = j
                if problem.d >= 2:
                    matrice_resultat[num_global_ddl[1],i[1]] = j
                if problem.d == 3:
                    matrice_resultat[num_global_ddl[2],i[2]] = j
        else: #Face sur le bord, on interpolle la valeur avec les valeurs aux vertex
            pos_init = vertex_associe_face.get(num_global_face)
            v1 = num_ddl_vertex[pos_init[0]]
            v2 = num_ddl_vertex[pos_init[1]]
            if problem.dim == 2:
                matrice_resultat[num_global_ddl[0], v1[0]] = 0.5
                matrice_resultat[num_global_ddl[0], v2[0]] = 0.5
                if problem.d == 2: #pb vectoriel
                    matrice_resultat[num_global_ddl[1], v1[1]] = 0.5
                    matrice_resultat[num_global_ddl[1], v2[1]] = 0.5
            if problem.dim == 3:
                v3 = num_ddl_vertex[pos_init[2]]
                matrice_resultat[num_global_ddl[0], v1[0]] = 1./3.
                matrice_resultat[num_global_ddl[0], v2[0]] = 1./3.
                matrice_resultat[num_global_ddl[0], v3[0]] = 1./3.
                if problem.d >= 2: #deuxième ligne
                    matrice_resultat[num_global_ddl[1], v1[1]] = 1./3.
                    matrice_resultat[num_global_ddl[1], v2[1]] = 1./3.
                    matrice_resultat[num_global_ddl[1], v3[1]] = 1./3.
                if problem.d == 3: #troisième ligne
                    matrice_resultat[num_global_ddl[2], v1[2]] = 1./3.
                    matrice_resultat[num_global_ddl[2], v2[2]] = 1./3.
                    matrice_resultat[num_global_ddl[2], v3[2]] = 1./3.
        
    return matrice_resultat.tocsr()
