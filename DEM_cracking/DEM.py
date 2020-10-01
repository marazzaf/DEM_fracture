# coding: utf-8

from dolfin import *
from scipy.sparse import csr_matrix,dok_matrix
import numpy as np
from DEM_cracking.reconstructions import compute_all_reconstruction_matrices,gradient_matrix
from DEM_cracking.mesh_related import *
from DEM_cracking.miscellaneous import Dirichlet_BC,schur_matrices

class DEMProblem:
    """ Class that will contain the basics of a DEM problem from the mesh and the dimension of the problem to reconstrucion matrices and gradient matrix."""
    def __init__(self, mesh, d, penalty, nz_vec_BC):
        self.mesh = mesh
        self.dim = self.mesh.geometric_dimension()
        self.d = d
        self.penalty = penalty

        #Define the necessary functionnal spaces depending on d
        if self.d == 1:
            self.CR = FunctionSpace(self.mesh, 'CR', 1)
            self.W = VectorFunctionSpace(self.mesh, 'DG', 0)
            self.DG_0 = FunctionSpace(self.mesh, 'DG', 0)
            self.DG_1 = FunctionSpace(self.mesh, 'DG', 1)
        elif self.d == self.dim:
            self.CR = VectorFunctionSpace(self.mesh, 'CR', 1)
            self.W = TensorFunctionSpace(self.mesh, 'DG', 0)
            self.DG_0 = VectorFunctionSpace(self.mesh, 'DG', 0)
            self.DG_1 = VectorFunctionSpace(self.mesh, 'DG', 1)
        else:
            raise ValueError('Problem is whether scalar or vectorial')

        #nb dofs
        self.nb_dof_cells = self.DG_0.dofmap().global_dimension()
        self.initial_nb_dof_CR = self.CR.dofmap().global_dimension()
        self.nb_dof_CR = self.initial_nb_dof_CR
        self.nb_dof_grad = self.W.dofmap().global_dimension()
        self.nb_dof_DEM = self.nb_dof_cells + len(nz_vec_BC)

        #gradient
        self.mat_grad = gradient_matrix(self)

        #useful mesh related
        self.facet_num = facet_neighborhood(self.mesh)
        self.Graph = connectivity_graph(self, nz_vec_BC)
        self.facet_to_facet = linked_facets(self) #designed to lighten research of potentialy failing facets close to a broken facet
        self.facets_cell = facets_in_cell(self)

        #DEM reconstructions
        self.DEM_to_DG = matrice_passage_ccG_DG(nb_ddl_cells,nb_ddl_ccG)
        self.DEM_to_CR,self.trace_matrix = matrice_passage_ccG_CR(problem)
        passage_ccG_to_DG_1,ccG_to_DG_1_aux_1,ccG_to_DG_1_aux_2 = matrice_passage_ccG_DG_1(mesh, nb_ddl_ccG, d, dim, mat_grad, passage_ccG_to_CR)
        print('Reconstruction matrices ok!')

        #Dirichlet conditions

    def for_dirichlet(self, A, boundary_dirichlet=None):
        hF = FacetArea(self.mesh)
        v_CG = TestFunction(self.CG)
        if boundary_dirichlet == None: #dependence on self.d ???
            form_dirichlet = inner(v_CG('+'),as_vector((1.,1.))) / hF * ds
        else:
            form_dirichlet = inner(v_CG('+'),as_vector((1.,1.))) / hF * ds(boundary_dirichlet)
        A_BC = Dirichlet_BC(form_dirichlet, self.DEM_to_CG)
        self.mat_not_D,self.mat_D = schur_matrices(A_BC)
        #A_D = mat_D * A * mat_D.T
        A_not_D = self.mat_not_D * A * self.mat_not_D.T
        B = self.mat_not_D * A * self.mat_D.T
        return A_not_D,B


def elastic_bilinear_form(mesh_, d_, DEM_to_CR_matrix, sigma=grad, eps=grad):
    dim = mesh_.geometric_dimension()
    if d_ == 1:
        U_CR = FunctionSpace(mesh_, 'CR', 1)
    elif d_ == dim:
        U_CR = VectorFunctionSpace(mesh_, 'CR', 1)
    else:
        raise ValueError('Problem is either scalar or vectorial (in 2d and 3d)')

    u_CR = TrialFunction(U_CR)
    v_CR = TestFunction(U_CR)

    #Mettre eps et sigma en arguments de la fonction ?
    if d_ == 1:
        a1 = eps(u_CR) * sigma(v_CR) * dx
    elif d_ == dim:
        a1 = inner(eps(u_CR), sigma(v_CR)) * dx
    else:
        raise ValueError('Problem is either scalar or vectorial (in 2d and 3d)')
    
    A1 = assemble(a1)
    row,col,val = as_backend_type(A1).mat().getValuesCSR()
    A1 = csr_matrix((val, col, row))
    return DEM_to_CR_matrix.T * A1 * DEM_to_CR_matrix

def penalty_term(nb_ddl_ccG_, mesh_, d_, dim_, mat_grad_, passage_ccG_CR_, G_, nb_ddl_CR_, nz_vec_BC):
    if d_ >= 2:
        U_DG = VectorFunctionSpace(mesh_, 'DG', 0)
        tens_DG_0 = TensorFunctionSpace(mesh_, 'DG', 0)
    else:
        U_DG = FunctionSpace(mesh_, 'DG', 0)
        tens_DG_0 = VectorFunctionSpace(mesh_, 'DG', 0)
        
    nb_ddl_cells = U_DG.dofmap().global_dimension()
    dofmap_tens_DG_0 = tens_DG_0.dofmap()
    nb_ddl_grad = dofmap_tens_DG_0.global_dimension()

    #creating jump matrix
    mat_jump_1 = sp.dok_matrix((nb_ddl_CR_,nb_ddl_ccG_))
    mat_jump_2 = sp.dok_matrix((nb_ddl_CR_,nb_ddl_grad))
    for (x,y) in G_.edges():
        num_global_ddl = G_[x][y]['dof_CR']
        coeff_pen = G_[x][y]['pen_factor']
        pos_bary_facet = G_[x][y]['barycentre'] #position barycentre of facet
        if abs(x) < nb_ddl_cells // d_ and abs(y) < nb_ddl_cells // d_: #Inner facet
            c1,c2 = x,y
            #filling-in the DG 0 part of the jump
            mat_jump_1[num_global_ddl[0]:num_global_ddl[-1]+1,d_ * c1 : (c1+1) * d_] = np.sqrt(coeff_pen)*np.eye(d_)
            mat_jump_1[num_global_ddl[0]:num_global_ddl[-1]+1,d_ * c2 : (c2+1) * d_] = -np.sqrt(coeff_pen)*np.eye(d_)

            for num_cell,sign in zip([c1,c2],[1., -1.]):
                #filling-in the DG 1 part of the jump...
                pos_bary_cell = G_.node[num_cell]['pos']
                diff = pos_bary_facet - pos_bary_cell
                pen_diff = np.sqrt(coeff_pen)*diff
                tens_dof_position = dofmap_tens_DG_0.cell_dofs(num_cell)
                for num,dof_CR in enumerate(num_global_ddl):
                    for i in range(dim_):
                        mat_jump_2[dof_CR,tens_dof_position[num*d_ + i]] = sign*pen_diff[i]

        #Penalty between facet reconstruction and cell value
        elif (abs(x) >= nb_ddl_cells // d_ or abs(y) >= nb_ddl_cells // d_) and len(set(num_global_ddl) & nz_vec_BC) > 0: #Outer facet
        
            if x >= 0 and y >= 0:
                num_cell = min(x,y)
                other = max(x,y)
            elif x <= 0 or y <= 0:
                num_cell = max(x,y)
                other = min(x,y)
        
            #selection dofs with Dirichlet BC
            coeff_pen = np.sqrt(coeff_pen)
            
            #cell part
            #filling-in the DG 0 part of the jump
            #dof = G_.node[num_cell]['dof']
            for pos,num_CR in enumerate(num_global_ddl): #should not be all num_global_dll but just what is in nz_vec_BC
            #for dof_CR,dof_c in zip(num_global_ddl,dof):
                if num_CR in nz_vec_BC:
                    mat_jump_1[num_CR,d_ * num_cell + pos] = coeff_pen
                    #mat_jump_1[dof_CR,dof_c] = coeff_pen
        
            #filling-in the DG 1 part of the jump
            pos_bary_cell = G_.node[num_cell]['pos']
            diff = pos_bary_facet - pos_bary_cell
            pen_diff = coeff_pen*diff
            tens_dof_position = dofmap_tens_DG_0.cell_dofs(num_cell)
            for num,dof_CR in enumerate(num_global_ddl):
                if dof_CR in nz_vec_BC:
                    for i in range(dim_):
                        mat_jump_2[dof_CR,tens_dof_position[num*d_ + i]] = pen_diff[i]

            #boundary facet part
            dof = G_.node[other]['dof']
            count = 0
            for num_CR in num_global_ddl:
                if num_CR in nz_vec_BC:
                    mat_jump_1[num_CR,dof[count]] = -coeff_pen
                    count += 1
                        

    mat_jump_1 = mat_jump_1.tocsr()
    mat_jump_2 = mat_jump_2.tocsr()
    mat_jump = mat_jump_1 + mat_jump_2 * mat_grad_ * passage_ccG_CR_
    return mat_jump.T * mat_jump, mat_jump_1, mat_jump_2#, bnd_1, bnd_2

def removing_penalty(mesh_, d_, dim_, nb_ddl_ccG_, mat_grad_, passage_ccG_CR_, G_, nb_ddl_CR_, cracking_facets, facet_num):
    if d_ >= 2:
        U_DG = VectorFunctionSpace(mesh_, 'DG', 0)
        tens_DG_0 = TensorFunctionSpace(mesh_, 'DG', 0)
    else:
        U_DG = FunctionSpace(mesh_, 'DG', 0)
        tens_DG_0 = VectorFunctionSpace(mesh_, 'DG', 0)
        
    nb_ddl_cells = U_DG.dofmap().global_dimension()
    dofmap_tens_DG_0 = tens_DG_0.dofmap()
    nb_ddl_grad = dofmap_tens_DG_0.global_dimension()

    #creating jump matrix
    mat_jump_1 = sp.dok_matrix((nb_ddl_CR_,nb_ddl_ccG_))
    mat_jump_2 = sp.dok_matrix((nb_ddl_CR_,nb_ddl_grad))

    for f in cracking_facets: #utiliser facet_num pour avoir les voisins ?
        assert(len(facet_num.get(f)) == 2)
        c1,c2 = facet_num.get(f) #must be two otherwise external facet broke
        num_global_ddl = G_[c1][c2]['dof_CR']
        coeff_pen = G_[c1][c2]['pen_factor']
        pos_bary_facet = G_[c1][c2]['barycentre'] #position barycentre of facet
        #filling-in the DG 0 part of the jump
        mat_jump_1[num_global_ddl[0]:num_global_ddl[-1]+1,d_ * c1 : (c1+1) * d_] = np.sqrt(coeff_pen)*np.eye(d_)
        mat_jump_1[num_global_ddl[0]:num_global_ddl[-1]+1,d_ * c2 : (c2+1) * d_] = -np.sqrt(coeff_pen)*np.eye(d_)

        for num_cell,sign in zip([c1,c2],[1., -1.]):
            #filling-in the DG 1 part of the jump...
            pos_bary_cell = G_.node[num_cell]['pos']
            diff = pos_bary_facet - pos_bary_cell
            pen_diff = np.sqrt(coeff_pen)*diff
            tens_dof_position = dofmap_tens_DG_0.cell_dofs(num_cell)
            for num,dof_CR in enumerate(num_global_ddl):
                for i in range(dim_):
                    mat_jump_2[dof_CR,tens_dof_position[(num % d_)*d_ + i]] = sign*pen_diff[i]

    return mat_jump_1.tocsr(), mat_jump_2.tocsr()
