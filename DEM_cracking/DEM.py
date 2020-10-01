# coding: utf-8
from dolfin import *
from scipy.sparse import csr_matrix,dok_matrix
import numpy as np
from DEM_cracking.mesh_related import *
from DEM_cracking.reconstructions import *
#from DEM_cracking.facet_interpolation import *
from DEM_cracking.miscellaneous import *

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
        self.DEM_to_DG = DEM_to_DG_matrix(self)
        self.DEM_to_CR,self.trace_matrix = DEM_to_CR_matrix(self)
        print('Reconstruction matrices ok!')

        #Penalty matrix
        self.mat_pen,self.mat_jump_1,self.mat_jump_2 = penalty_term(self, nz_vec_BC)
        self.mat_elas = self.elastic_bilinear_form(ref_elastic)

    def elastic_bilinear_form(self,ref_elastic):
        return  self.DEM_to_CR.T * self.mat_grad.T * ref_elastic * self.mat_grad * self.DEM_to_CR

    def adapting_elasticity_and_penalty(self,ref_elastic,cracking_facets):
        #Removing penalty terms
        mat_jump_1_aux,mat_jump_2_aux = removing_penalty(self, cracking_facets)
        self.mat_jump_1 -= mat_jump_1_aux
        self.mat_jump_2 -= mat_jump_2_aux
        passage_ccG_to_CR, mat_grad, nb_ddl_CR, facet_num, mat_D, mat_not_D = adapting_after_crack(cracking_facets, cracked_facets, d, dim, facet_num, nb_ddl_cells, nb_ddl_ccG, nb_ddl_CR, passage_ccG_to_CR, mat_grad, G, mat_D, mat_not_D)

        #Assembling new elastic term
        self.mat_elas = self.elastic_bilinear_form(ref_elastic)

        #Assembling new penalty term
        self.mat_jump_1.resize((self.nb_dof_CR,self.nb_dof_DEM))
        self.mat_jump_2.resize((self.nb_dof_CR,self.nb_dof_grad))
        mat_jump = self.mat_jump_1 + self.mat_jump_2 * self.mat_grad * self.DEM_to_CR
        self.mat_pen = mat_jump.T * mat_jump
        
        return

def ref_elastic_bilinear_form(problem, sigma=grad, eps=grad):
    Du = TrialFunction(problem.W)
    Dv = TestFunction(problem.W)

    a1 = inner(eps(Du), sigma(eps(Dv))) * dx    
    A1 = assemble(a1)
    row,col,val = as_backend_type(A1).mat().getValuesCSR()
    A1 = csr_matrix((val, col, row))
    return A1

def penalty_term(problem, nz_vec_BC):
    dofmap_tens_DG_0 = problem.W.dofmap()

    #creating jump matrix
    mat_jump_1 = sp.dok_matrix((problem.nb_dof_CR,problem.nb_dof_DEM))
    mat_jump_2 = sp.dok_matrix((problem.nb_dof_CR,problem.nb_dof_grad))
    for (x,y) in problem.Graph.edges():
        num_global_ddl = problem.Graph[x][y]['dof_CR']
        coeff_pen = problem.Graph[x][y]['pen_factor']
        pos_bary_facet = problem.Graph[x][y]['barycentre'] #position barycentre of facet
        if abs(x) < problem.nb_dof_cells // problem.d and abs(y) < problem.nb_dof_cells // problem.d: #Inner facet
            c1,c2 = x,y
            #filling-in the DG 0 part of the jump
            mat_jump_1[num_global_ddl[0]:num_global_ddl[-1]+1,problem.d * c1 : (c1+1) * problem.d] = np.sqrt(coeff_pen)*np.eye(problem.d)
            mat_jump_1[num_global_ddl[0]:num_global_ddl[-1]+1,problem.d * c2 : (c2+1) * problem.d] = -np.sqrt(coeff_pen)*np.eye(problem.d)

            for num_cell,sign in zip([c1,c2],[1., -1.]):
                #filling-in the DG 1 part of the jump...
                pos_bary_cell = problem.Graph.node[num_cell]['pos']
                diff = pos_bary_facet - pos_bary_cell
                pen_diff = np.sqrt(coeff_pen)*diff
                tens_dof_position = dofmap_tens_DG_0.cell_dofs(num_cell)
                for num,dof_CR in enumerate(num_global_ddl):
                    for i in range(problem.dim):
                        mat_jump_2[dof_CR,tens_dof_position[num*problem.d + i]] = sign*pen_diff[i]

        #Penalty between facet reconstruction and cell value
        elif (abs(x) >= problem.nb_dof_cells // problem.d or abs(y) >= problem.nb_dof_cells // problem.d) and len(set(num_global_ddl) & nz_vec_BC) > 0: #Outer facet
        
            if x >= 0 and y >= 0:
                num_cell = min(x,y)
                other = max(x,y)
            elif x <= 0 or y <= 0:
                num_cell = max(x,y)
                other = min(x,y)
        
            #selection dofs with Dirichlet BC
            coeff_pen = np.sqrt(coeff_pen)
            
            #cell part
            for pos,num_CR in enumerate(num_global_ddl): #should not be all num_global_dll but just what is in nz_vec_BC
                if num_CR in nz_vec_BC:
                    mat_jump_1[num_CR,problem.d * num_cell + pos] = coeff_pen
        
            #filling-in the DG 1 part of the jump
            pos_bary_cell = problem.Graph.node[num_cell]['pos']
            diff = pos_bary_facet - pos_bary_cell
            pen_diff = coeff_pen*diff
            tens_dof_position = dofmap_tens_DG_0.cell_dofs(num_cell)
            for num,dof_CR in enumerate(num_global_ddl):
                if dof_CR in nz_vec_BC:
                    for i in range(problem.dim):
                        mat_jump_2[dof_CR,tens_dof_position[num*problem.d + i]] = pen_diff[i]

            #boundary facet part
            dof = problem.Graph.node[other]['dof']
            count = 0
            for num_CR in num_global_ddl:
                if num_CR in nz_vec_BC:
                    mat_jump_1[num_CR,dof[count]] = -coeff_pen
                    count += 1
                        

    mat_jump_1 = mat_jump_1.tocsr()
    mat_jump_2 = mat_jump_2.tocsr()
    mat_jump = mat_jump_1 + mat_jump_2 * problem.mat_grad * problem.DEM_to_CR
    return mat_jump.T * mat_jump, mat_jump_1, mat_jump_2

def removing_penalty(problem, cracking_facets):    
    dofmap_tens_DG_0 = problem.W.dofmap()

    #creating jump matrix
    mat_jump_1 = sp.dok_matrix((problem.nb_dof_CR,problem.nb_dof_DEM))
    mat_jump_2 = sp.dok_matrix((problem.nb_dof_CR,problem.nb_dof_grad))

    for f in cracking_facets: #utiliser facet_num pour avoir les voisins ?
        assert len(self.facet_num.get(f)) == 2
        c1,c2 = self.facet_num.get(f) #must be two otherwise external facet broke
        num_global_ddl = problem.Graph[c1][c2]['dof_CR']
        coeff_pen = problem.Graph[c1][c2]['pen_factor']
        pos_bary_facet = problem.Graph[c1][c2]['barycentre'] #position barycentre of facet
        #filling-in the DG 0 part of the jump
        mat_jump_1[num_global_ddl[0]:num_global_ddl[-1]+1,d_ * c1 : (c1+1) * d_] = np.sqrt(coeff_pen)*np.eye(d_)
        mat_jump_1[num_global_ddl[0]:num_global_ddl[-1]+1,d_ * c2 : (c2+1) * d_] = -np.sqrt(coeff_pen)*np.eye(d_)

        for num_cell,sign in zip([c1,c2],[1., -1.]):
            #filling-in the DG 1 part of the jump...
            pos_bary_cell = problem.Graph.node[num_cell]['pos']
            diff = pos_bary_facet - pos_bary_cell
            pen_diff = np.sqrt(coeff_pen)*diff
            tens_dof_position = dofmap_tens_DG_0.cell_dofs(num_cell)
            for num,dof_CR in enumerate(num_global_ddl):
                for i in range(problem.dim):
                    mat_jump_2[dof_CR,tens_dof_position[(num % d_)*d_ + i]] = sign*pen_diff[i]

    return mat_jump_1.tocsr(), mat_jump_2.tocsr()
