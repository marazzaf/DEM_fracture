# coding: utf-8
import sys
sys.path.append('../..')
from facets import *
from scipy.sparse.linalg import eigsh,cg

# Form compiler options
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

# elastic parameters
nu = 0.3
E = 70e3
mu = Constant(E / (2.0*(1.0 + nu)))
lambda_ = Constant(E*nu / ((1.0 + nu)*(1.0 - 2.0*nu)))
penalty = 2*float(mu)
a = 0.8

size_ref = 2 #5 #10 #20 #1 for debug
mesh = RectangleMesh(Point(-0.5, -0.5), Point(0.5, 0.5), size_ref, size_ref, "crossed")
#size_ref = 1
#mesh = Mesh('mesh/circle_%i.xml' % size_ref)
bnd_facets = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
h = mesh.hmax()
ds = Measure('ds')(subdomain_data=bnd_facets)

# Mesh-related functions
vol = CellVolume(mesh) #Pour volume des particules voisines
hF = FacetArea(mesh)
h_avg = (vol('+') + vol('-'))/ (2. * hF('+'))
n = FacetNormal(mesh)

#Function spaces
U_DG = VectorFunctionSpace(mesh, 'DG', 0) #Pour déplacement dans cellules
U_DG_1 = VectorFunctionSpace(mesh, 'DG', 1) #Pour déplacement dans cellules
U_CR = VectorFunctionSpace(mesh, 'CR', 1) #Pour interpollation dans les faces
W = TensorFunctionSpace(mesh, 'DG', 0)

#useful
for_dim = Function(U_DG)
dim = for_dim.geometric_dimension()
d = dim
solution_u_DG = Function(U_DG,  name="disp DG")
solution_stress = Function(W, name="Stress")

#reference solution
x = SpatialCoordinate(mesh)
#Dirichlet BC
u_D = Expression(('0.5*a*(x[0]*x[0]+x[1]*x[1])', '0.5*a*(x[0]*x[0]+x[1]*x[1])'), a=a,degree=2)
volume_force = -a * (lambda_ + 3*mu) * as_vector((1.,1.))

def eps(v):
    return sym(v)

def sigma(eps_el):
    return lambda_ * tr(eps_el) * Identity(2) + 2.*mu * eps_el

# Define variational problem
u_CR = TrialFunction(U_CR)
v_CR = TestFunction(U_CR)
u_DG = TrialFunction(U_DG)
v_DG = TestFunction(U_DG)
Du_DG = TrialFunction(W)
Dv_DG = TestFunction(W)

#new for BC
l4 = inner(v_CR('+'), as_vector((1.,1.))) / hF * ds
L4 = assemble(l4)
vec_BC = L4.get_local()
nz_vec_BC = list(vec_BC.nonzero()[0])
nz_vec_BC = set(nz_vec_BC)

#Cell-centre Galerkin reconstruction
nb_ddl_cells = U_DG.dofmap().global_dimension()
print('nb cell dof : %i' % nb_ddl_cells)
nb_ddl_ccG = nb_ddl_cells
facet_num = new_facet_neighborhood(mesh)
nb_ddl_CR = U_CR.dofmap().global_dimension()
initial_nb_ddl_CR = nb_ddl_CR #will not change. Useful for reducing u_CR for cracking criterion
nb_facet_original = nb_ddl_CR // d
print('nb dof CR: %i' % nb_ddl_CR)
G = connectivity_graph(mesh, d, penalty, nz_vec_BC)
print('ok graph !')
nb_ddl_ccG = nb_ddl_cells + len(nz_vec_BC)
coord_bary,coord_num = smallest_convexe_bary_coord(mesh,facet_num,d,G) #, vertex_boundary)
print('Convexe ok !')
#matrice gradient
mat_grad = gradient_matrix(mesh, d)
print('gradient matrix ok !')
passage_ccG_to_CR,trace_matrix = matrice_passage_ccG_CR(mesh, coord_num, coord_bary, d, G, nb_ddl_ccG)
passage_ccG_to_DG = matrice_passage_ccG_DG(nb_ddl_cells, nb_ddl_ccG)
passage_ccG_to_DG_1,aux_1,aux_2 = matrice_passage_ccG_DG_1(mesh, nb_ddl_ccG, d, dim, mat_grad, passage_ccG_to_CR)
print('matrices passage ok !')
nb_ddl_grad = W.dofmap().global_dimension()
mat_not_D,mat_D = schur(nb_ddl_cells, nb_ddl_ccG)

#Variational problem
a1 = inner(sigma(eps(Du_DG)), Dv_DG) * dx #does not change with topological changes
A1 = assemble(a1)
row,col,val = as_backend_type(A1).mat().getValuesCSR()
A1 = sp.csr_matrix((val, col, row))

def elastic_term(mat_grad_, passage):
    return  passage.T * mat_grad_.T * A1 * mat_grad_ * passage

#Stress output
a47 = inner(sigma(eps(Du_DG)), Dv_DG) / vol * dx
A47 = assemble(a47)
row,col,val = as_backend_type(A47).mat().getValuesCSR()
mat_stress = sp.csr_matrix((val, col, row))

#Homogeneous Neumann BC
L = np.zeros(nb_ddl_CR)

#paraview outputs
file = File('convergence_no_crack/test_%i_.pvd' % size_ref)

mat_elas = elastic_term(mat_grad, passage_ccG_to_CR)
mat_pen,mat_jump_1,mat_jump_2 = penalty_term(nb_ddl_ccG, mesh, d, dim, mat_grad, passage_ccG_to_CR, G, nb_ddl_CR, nz_vec_BC)
passage_ccG_to_DG_1 = aux_1 + aux_2 * mat_grad * passage_ccG_to_CR

A = mat_elas + mat_pen

#Removing Dirichlet BC
A_not_D = mat_not_D * A * mat_not_D.T
rhs = -mat_not_D * A * trace_matrix.T * np.nan_to_num(interpolate(u_D, U_CR).vector().get_local()) + mat_not_D * passage_ccG_to_DG.T * assemble(inner(volume_force, v_DG) * dx).get_local()

#inverting system
u_reduced = spsolve(A_not_D, rhs)
#u_reduced,info = cg(A_not_D, rhs)
#assert(info == 0)

u = mat_not_D.T * u_reduced + trace_matrix.T * np.nan_to_num(interpolate(u_D, U_CR).vector().get_local())

#sys.exit()

#Post-processing
vec_u_CR = passage_ccG_to_CR * u
vec_u_DG = passage_ccG_to_DG * u

#output initial conditions
solution_u_DG.vector().set_local(vec_u_DG)
solution_u_DG.vector().apply("insert")
file.write(solution_u_DG, 0)
sol_DG_1 = Function(U_DG_1, name="disp DG 1")
sol_DG_1.vector().set_local(passage_ccG_to_DG_1 * u)
file.write(sol_DG_1, 0)

#write the convergence errors etc...
#Reference solution interpolated in ccG vector
u_ref_ccG = passage_ccG_to_DG.T * interpolate(u_D, U_DG).vector().get_local() + trace_matrix.T * interpolate(u_D, U_CR).vector().get_local()

#energy error
diff = u - u_ref_ccG
err_energy = np.dot(diff, A * diff)
err_energy = np.sqrt(0.5 * err_energy)
print('error energy: %.5e' % err_energy)

#L2 error
u_ref_DG_1 = Function(U_DG_1)
u_ref_DG_1.vector().set_local(passage_ccG_to_DG_1 * u_ref_ccG)
err_u_L2_DG_1 = errornorm(sol_DG_1, u_ref_DG_1, 'L2')
print('err L2 DG 1: %.5e' % err_u_L2_DG_1)

img = plot(sol_DG_1)
plt.colorbar(img)
plt.show()

img = plot(interpolate(u_D, U_DG_1))
plt.colorbar(img)
plt.show()

img = plot(sol_DG_1 - interpolate(u_D, U_DG_1))
plt.colorbar(img)
plt.show()


#solution_stress.vector().set_local(mat_stress * mat_grad * vec_u_CR)
#solution_stress.vector().apply("insert")
#file.write(solution_stress, 0)


#computation over
print('End of computation !')

