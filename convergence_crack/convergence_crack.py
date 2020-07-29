# coding: utf-8
import sys
sys.path.append('../')
from facets import *
from ufl.operators import sign
# Form compiler options
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

# elastic parameters
mu = 0.2
penalty = mu #mu ou 2*mu?

#geometry
a, R, l0 = 0.5, 0.025, 0.025
size_ref = 5
#mesh = Mesh("mesh_old/circle_%i.xml" % size_ref) #maillage domaine circulaire
#bnd_facets = MeshFunction("size_t", mesh, 'mesh_old/circle_%i_facet_region.xml' % size_ref)
#bnd_facets = MeshFunction("size_t", mesh, 1)
mesh = Mesh('mesh/circle_cracked_%i.xml' % size_ref) #pre-cracked
bnd_facets = MeshFunction("size_t", mesh, "mesh/circle_cracked_%i_facet_region.xml" % size_ref)
ds = Measure('ds')(subdomain_data=bnd_facets)

# Mesh-related functions
vol = CellVolume(mesh) #Pour volume des particules voisines
hF = FacetArea(mesh)
h_avg = (vol('+') + vol('-'))/ (2. * hF('+'))
n = FacetNormal(mesh)

#Function spaces
U_DG = FunctionSpace(mesh, 'DG', 0) #Pour déplacement dans cellules
U_DG_1 = FunctionSpace(mesh, 'DG', 1) #Pour déplacement dans cellules
U_CR = FunctionSpace(mesh, 'CR', 1) #Pour interpollation dans les faces
W = VectorFunctionSpace(mesh, 'DG', 0)

#useful
for_dim = Function(U_DG)
dim = for_dim.geometric_dimension()
d = 1 #scalar case
solution_u_DG = Function(U_DG,  name="disp DG")
solution_stress = Function(W, name="Stress")

#reference solution
x = SpatialCoordinate(mesh)
r = Expression('sqrt((x[0]) * (x[0]) + x[1] * x[1])', degree = 1)
X = x[1] / (x[0]) #!!!!!!!!!!!!!!!!!!!! #Modifier si maillage change !
c_theta = 1. / sqrt(1 + X**2.) * sign(x[0]) #!!!!!!!!!!!!!!!!!!!! #Modifier si maillage change !
s_theta = abs(X) / sqrt(1 + X**2.) * sign(x[1])
s_theta_2 = sqrt(0.5 * (1 - c_theta)) * sign(x[1])
c_theta_2 = sqrt(0.5 * (1 + c_theta))
tau = 1. #10. #100. #more than that ?
K_3 = tau * np.sqrt(a * np.pi)
u_sing = 2. * tau / mu * sqrt(r * a * 0.5) * s_theta_2
#u_sing = 2*K_3 / mu * sqrt(0.5*r/np.pi) * s_theta_2
e_r = as_vector((c_theta,s_theta))
e_theta = as_vector((-s_theta,c_theta))

#print('Ref value: %.5e' % (2. * tau / mu * sqrt(R * a * 0.5)))

def eps(v):
    return v

def sigma(eps_el):
    return mu * eps_el

# Define variational problem
u_CR = TrialFunction(U_CR)
v_CR = TestFunction(U_CR)
u_DG = TrialFunction(U_DG)
v_DG = TestFunction(U_DG)
Du_DG = TrialFunction(W)
Dv_DG = TestFunction(W)

#new for BC
#l4 = v_CR('+') / hF * ds(8) #not all boundary is Dirichlet
l4 = v_CR('+') / hF * ds #test...
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

count_output_energy_release = 0
count_output_disp = 0
count_output_crack = 0
cracked_facet_vertices = []
cracked_facets = set()

#Crack is put in the plate
cracking_facets = set()
#before the computation begins, we break the facets to have a crack of length 1
for (x,y) in G.edges():
    f = G[x][y]['dof_CR'][0] // d
    pos = G[x][y]['barycentre']
    if G[x][y]['breakable'] and np.absolute(pos[1]) < 1.e-15 and pos[0] < 0.:
        cracking_facets.add(f)
        n1,n2 = facet_num.get(f)
        cracked_facet_vertices.append(G[n1][n2]['vertices']) #position of vertices of the broken facet
        
#adapting after crack
passage_ccG_to_CR, mat_grad, nb_ddl_CR, facet_num, mat_D, mat_not_D = adapting_after_crack(cracking_facets, cracked_facets, d, dim, facet_num, nb_ddl_cells, nb_ddl_ccG, nb_ddl_CR, passage_ccG_to_CR, mat_grad, G, mat_D, mat_not_D)
out_cracked_facets('.', size_ref, 0, cracked_facet_vertices, dim) #paraview cracked facet file
cracked_facets.update(cracking_facets) #adding facets just cracked to broken facets
mat_pen,mat_jump_1,mat_jump_2 = penalty_term(nb_ddl_ccG, mesh, d, dim, mat_grad, passage_ccG_to_CR, G, nb_ddl_CR, nz_vec_BC)
passage_ccG_to_DG_1 = aux_1 + aux_2 * mat_grad * passage_ccG_to_CR
mat_elas = elastic_term(mat_grad, passage_ccG_to_CR)
A = mat_elas + mat_pen

#Homogeneous Neumann BC
L = np.zeros(nb_ddl_CR)

#paraview outputs
file = File('test_%i.pvd' % size_ref)

#Removing Dirichlet BC
A_not_D = mat_not_D * A * mat_not_D.T
rhs = -mat_not_D * A * trace_matrix.T * np.nan_to_num(local_project(u_sing, U_CR).vector().get_local())

#inverting system
u_reduced = spsolve(A_not_D, rhs)
#u_reduced,info = cg(A_not_D, rhs)
#assert(info == 0)

u = mat_not_D.T * u_reduced + trace_matrix.T * np.nan_to_num(local_project(u_sing, U_CR).vector().get_local())

#print(trace_matrix.T*local_project(u_sing, U_CR).vector().get_local())
#print(u[nb_ddl_cells:])
#print(trace_matrix.T*local_project(u_sing, U_CR).vector().get_local() - u)

#sys.exit()

#Post-processing
vec_u_CR = passage_ccG_to_CR * u
vec_u_DG = passage_ccG_to_DG * u

#print(max(vec_u_DG))
#print(min(vec_u_DG))
#print(max(local_project(u_sing, U_DG).vector().get_local()))
#print(min(local_project(u_sing, U_DG).vector().get_local()))
#print(max(vec_u_CR))
#print(min(vec_u_CR))
#print(max(local_project(u_sing, U_CR).vector().get_local()))
#print(min(local_project(u_sing, U_CR).vector().get_local()))

#output initial conditions
solution_u_DG.vector().set_local(vec_u_DG)
solution_u_DG.vector().apply("insert")
file.write(solution_u_DG, 0)
sol_DG_1 = Function(U_DG_1, name="disp DG 1")
sol_DG_1.vector().set_local(passage_ccG_to_DG_1 * u)
file.write(sol_DG_1, 0)

#write the convergence errors etc...
#Reference solution interpolated in ccG vector
u_ref_ccG = passage_ccG_to_DG.T * np.nan_to_num(local_project(u_sing, U_DG).vector().get_local()) + trace_matrix.T * np.nan_to_num(local_project(u_sing, U_CR).vector().get_local())

#energy error
diff = u - u_ref_ccG
err_energy = np.dot(diff, A * diff)
err_energy = np.sqrt(0.5 * err_energy)
print('error energy: %.5e' % err_energy)

#L2 error
u_ref_DG_1 = local_project(u_sing, U_DG_1)
for_para = Function(U_DG_1, name='Ref')
for_para.vector().set_local(u_ref_DG_1.vector().get_local())
file.write(u_ref_DG_1)
err_u_L2_DG_1 = errornorm(sol_DG_1, u_ref_DG_1, 'L2')
print('err L2 DG 1: %.5e' % err_u_L2_DG_1)
err_u_H1_DG_1 = errornorm(sol_DG_1, u_ref_DG_1, 'H10')
print('err H1 DG 1: %.5e' % err_u_H1_DG_1)

#sol_CR = Function(U_CR)
#sol_CR.vector().set_local(vec_u_CR)
#file.write(sol_CR)
#u_ref_CR = local_project(u_sing, U_CR)
#err_u_L2_CR = errornorm(sol_CR, u_ref_CR, 'L2')
#print('err L2 CR: %.5e' % err_u_L2_CR)
#err_u_H1_CR = errornorm(sol_CR, u_ref_CR, 'H10')
#print('err H1 CR: %.5e' % err_u_H1_CR)


solution_stress.vector().set_local(mat_stress * mat_grad * vec_u_CR)
solution_stress.vector().apply("insert")
file.write(solution_stress, 0)

#reference elastic energy
sigma_sing = tau * sqrt(a / (2 * r)) * (s_theta_2 * e_r + c_theta_2 * e_theta)
stress = local_project(sigma_sing, W)
err_stress = errornorm(solution_stress, stress, 'L2')
print('err stress: %.5e' % err_stress)
energy_sing = 0.5 * dot(sigma_sing, n) * u_sing * ds 
print('Ref elastic energy: %.5e' % assemble(energy_sing))
def ref_elastic_energy(w, w_):
    return inner(w, w_ / mu) * dx
print('Ref elastic energy bis: %.5e' % (0.5 * assemble(ref_elastic_energy(stress, stress))))

#img = plot(solution_u_DG)
#plt.colorbar(img)
#plt.show()
#
#img = plot(local_project(u_sing, U_DG))
#plt.colorbar(img)
#plt.show()

img = plot(sol_DG_1 - local_project(u_sing, U_DG_1))
plt.colorbar(img)
plt.show()

img = plot(solution_stress - stress)
plt.colorbar(img)
plt.show()

#pen_values = Function(U_CR)
#pen_values.vector().set_local(mat_pen * u)
#img = plot(pen_values)
#plt.colorbar(img)
#plt.show()

#img = plot(sol_CR)
#plt.colorbar(img)
#plt.show()
#
#img = plot(local_project(u_sing, U_CR))
#plt.colorbar(img)
#plt.show()
#
#img = plot(sol_CR - local_project(u_sing, U_CR))
#plt.colorbar(img)
#plt.show()


#computation over
print('End of computation !')

