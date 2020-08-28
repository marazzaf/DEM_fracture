# coding: utf-8
import sys
sys.path.append('../')
from facets import *
from scipy.sparse.linalg import eigsh,cg

# Form compiler options
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
#parameters['linear_algebra_backend'] = 'PETSc'

# elastic parameters
rho = Constant(1180.)
nu = 0.35
E = 3.09e9
mu = Constant(E / (2.0*(1.0 + nu)))
lambda_ = Constant(E*nu / ((1.0 + nu)*(1.0 - 2.0*nu)))
penalty = float(mu)
Gc = 300
Delta_u = 0.05e-3 #value from article

Ll, l0, H = 32e-3, 4.e-3, 16e-3
size_ref = 2 #5 #10 #20
mesh = Mesh('mesh_bis/plate_holes_2.xml') # % size_ref)
bnd_facets = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
h = mesh.hmax() #H / size_ref
c = np.sqrt(float(mu/rho))

# Sub domain for BC
def top(x, on_boundary):
    return near(x[1], H/2) and on_boundary

def down(x, on_boundary):
    return near(x[1], -H/2) and on_boundary

def right(x, on_boundary):
    return near(x[0], Ll) and on_boundary

def left(x, on_boundary):
    return near(x[0], 0.) and on_boundary

bnd_facets.set_all(0)
traction_boundary_1 = AutoSubDomain(top)
traction_boundary_1.mark(bnd_facets, 41)
traction_boundary_2 = AutoSubDomain(down)
traction_boundary_2.mark(bnd_facets, 42)
neumann_boundary_1 = AutoSubDomain(right)
neumann_boundary_1.mark(bnd_facets, 45)
neumann_boundary_2 = AutoSubDomain(left)
neumann_boundary_2.mark(bnd_facets, 47)
ds = Measure('ds')(subdomain_data=bnd_facets)

# Mesh-related functions
vol = CellVolume(mesh) #Pour volume des particules voisines
hF = FacetArea(mesh)
h_avg = (vol('+') + vol('-'))/ (2. * hF('+'))
n = FacetNormal(mesh)

#Function spaces
U_DG = VectorFunctionSpace(mesh, 'DG', 0) #Pour déplacement dans cellules
U_CR = VectorFunctionSpace(mesh, 'CR', 1) #Pour interpollation dans les faces
W = TensorFunctionSpace(mesh, 'DG', 0)
aux_CR = FunctionSpace(mesh, 'CR', 1) #Pour critère élastique. Reste en FunctionSpace
W_aux = VectorFunctionSpace(mesh, 'CR', 1)

#useful
for_dim = Function(U_DG)
dim = for_dim.geometric_dimension()
d = dim
solution_u_DG = Function(U_DG,  name="disp DG")
solution_v_DG = Function(U_DG,  name="vel DG")
solution_stress = Function(W, name="Stress")

#reference solution
x = SpatialCoordinate(mesh)
#quasi-ref solution
#Dirichlet BC
u_D = Expression(('0.', 'x[1]/fabs(x[1]) * disp'), disp = Delta_u,degree=2)
v_D = Expression(('0.', '0.'), degree=0)

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
l4 = v_CR('+')[1] / hF * (ds(41) + ds(42))
L4 = assemble(l4)
#vec_BC = trace_matrix.T * L4.get_local()
vec_BC = L4.get_local()
nz = vec_BC.nonzero()[0]
vec_BC[nz[0]-1] = 1. #fixing rigid movement
nz_vec_BC = list(vec_BC.nonzero()[0])
nz_vec_BC = set(nz_vec_BC)

#Cell-centred Galerkin reconstruction
nb_ddl_cells = U_DG.dofmap().global_dimension()
print('nb cell dof : %i' % nb_ddl_cells)
facet_num = new_facet_neighborhood(mesh)
nb_ddl_CR = U_CR.dofmap().global_dimension()
initial_nb_ddl_CR = nb_ddl_CR #will not change. Useful for reducing u_CR for cracking criterion
print('nb dof CR: %i' % nb_ddl_CR)
G = connectivity_graph(mesh, d, penalty, nz_vec_BC)
nb_ddl_ccG = nb_ddl_cells + len(nz_vec_BC)
print('ok graph !')
coord_bary,coord_num = smallest_convexe_bary_coord(mesh,facet_num,d,G)
print('Convexe ok !')
#matrice gradient
mat_grad = gradient_matrix(mesh, d)
print('gradient matrix ok !')
passage_ccG_to_CR,trace_matrix = matrice_passage_ccG_CR(mesh, coord_num, coord_bary, d, G, nb_ddl_ccG)
passage_ccG_to_DG = matrice_passage_ccG_DG(nb_ddl_cells,nb_ddl_ccG)
passage_ccG_to_DG_1,ccG_to_DG_1_aux_1,ccG_to_DG_1_aux_2 = matrice_passage_ccG_DG_1(mesh, nb_ddl_ccG, d, dim, mat_grad, passage_ccG_to_CR)
facet_to_facet = linked_facets(mesh,dim,G) #designed to lighten research of potentialy failing facets close to a broken facet
nb_ddl_grad = W.dofmap().global_dimension()
mat_not_D,mat_D = schur(nb_ddl_cells, nb_ddl_ccG)
print('matrices passage Schur ok')

#Variational problem
a1 = inner(sigma(eps(Du_DG)), Dv_DG) * dx #does not change with topological changes
A1 = assemble(a1)
row,col,val = as_backend_type(A1).mat().getValuesCSR()
A1 = sp.csr_matrix((val, col, row))

def elastic_term(mat_grad_, passage):
    return  passage.T * mat_grad_.T * A1 * mat_grad_ * passage

#average facet stresses
a50 = dot( dot( avg(sigma(eps(Du_DG))), n('-')), v_CR('-')) / hF('-') * dS #n('-')
A50 = assemble(a50)
row,col,val = as_backend_type(A50).mat().getValuesCSR()
average_stresses = sp.csr_matrix((val, col, row))

#for outputs:
#Stress output
a47 = inner(sigma(eps(Du_DG)), Dv_DG) / vol * dx
A47 = assemble(a47)
row,col,val = as_backend_type(A47).mat().getValuesCSR()
mat_stress = sp.csr_matrix((val, col, row))

#length of facets
aux = TestFunction(aux_CR)
areas = assemble(aux('+') * dS).get_local()

#normal to facets
aux = TestFunction(W_aux)
normals = assemble(inner(n('-'), aux('-')) / hF('-') * dS ).get_local()
normals = normals.reshape((initial_nb_ddl_CR // d,d))
tangents = np.array([normals[:,1],-normals[:,0]]).T

#Assembling mass matrix
M_lumped = mass_matrix(mesh, d, dim, rho, nb_ddl_ccG)  # M_lumped est un vecteur
print('Mass matrix assembled !')

#Homogeneous Neumann BC
L = np.zeros(nb_ddl_CR)

#paraview outputs
file = File('test/holes_%i_.pvd' % size_ref)

count_output_crack = 0
cracked_facet_vertices = []
broken_vertices = set()
last_broken_vertices = set()
cracked_facets = set()

#initial conditions
u = np.zeros(nb_ddl_ccG)
v = np.zeros(nb_ddl_ccG)

cracking_facets = set()
#before the computation begins, we break the facets to have a crack of length 1
for (x,y) in G.edges():
    f = G[x][y]['dof_CR'][0] // d
    pos = G[x][y]['barycentre']
    if G[x][y]['breakable'] and np.abs(pos[1]) < 1.e-15 and pos[0] < l0:
        cracking_facets.add(f)
        cracked_facet_vertices.append(G[x][y]['vertices']) #position of vertices of the broken facet
        
#adapting after crack
passage_ccG_to_CR, mat_grad, nb_ddl_CR, facet_num, mat_D, mat_not_D = adapting_after_crack(cracking_facets, cracked_facets, d, dim, facet_num, nb_ddl_cells, nb_ddl_ccG, nb_ddl_CR, passage_ccG_to_CR, mat_grad, G, mat_D, mat_not_D)
out_cracked_facets('test', size_ref, 0, cracked_facet_vertices, dim) #paraview cracked facet file
cracked_facets.update(cracking_facets) #adding facets just cracked to broken facets
mat_elas = elastic_term(mat_grad, passage_ccG_to_CR)
mat_pen,mat_jump_1,mat_jump_2 = penalty_term(nb_ddl_ccG, mesh, d, dim, mat_grad, passage_ccG_to_CR, G, nb_ddl_CR, nz_vec_BC)
A = mat_elas + mat_pen
L = np.concatenate((L, np.zeros(d * len(cracking_facets))))

#Imposing strongly Dirichlet BC
A_D = mat_D * A * mat_D.T
A_not_D = mat_not_D * A * mat_not_D.T
B = mat_not_D * A * mat_D.T
M_not_D = mat_not_D * M_lumped

#interpolation of Dirichlet BC
FF = interpolate(u_D, U_CR).vector().get_local()
F = mat_D * trace_matrix.T * FF

#taking into account exterior loads
#L_not_D = mat_not_D * trace_matrix.T * L
#L_not_D = L_not_D - B * F
L_not_D = -B*F

#inverting system
#u_reduced = spsolve(A_not_D, L_not_D)
u_reduced,info = cg(A_not_D, L_not_D)
assert(info == 0)
u = mat_not_D.T * u_reduced + mat_D.T * F

#Post-processing
vec_u_CR = passage_ccG_to_CR * u
vec_u_DG = passage_ccG_to_DG * u

#output initial conditions
solution_u_DG.vector().set_local(vec_u_DG)
solution_u_DG.vector().apply("insert")
file.write(solution_u_DG, 0)
solution_stress.vector().set_local(mat_stress * mat_grad * vec_u_CR)
solution_stress.vector().apply("insert")
file.write(solution_stress, 0)

#print('Intial elastic energy: %.5e' % (0.5 * np.dot(u, A * u)))

#definition of time-stepping parameters
T = 35e-6 #from article
t = 0.

# Time-stepping parameters
eig_M = min(M_not_D)
eigenvalues_K = eigsh(A_not_D, k=1, return_eigenvectors=False) #which='LM' (largest in magnitude, great)
eig_K = max(np.real(eigenvalues_K))
dt = np.sqrt(eig_M / eig_K)
print('dt: %.5e' % dt)

u_not_D = mat_not_D * u
v_not_D = mat_not_D * v
v_old = v

#Début boucle temporelle
while t < T:
    t += dt
    print('t: %.5e' % t)

    #Computing new displacement values for non-Dirichlet dofs
    u_not_D = u_not_D + dt * v_not_D

    #interpolation of Dirichlet BC
    #displacement
    FF = interpolate(u_D, U_CR).vector().get_local()
    F = mat_D * trace_matrix.T * FF

    #computing new disp values
    u = mat_not_D.T * u_not_D + mat_D.T * F

    #Post-processing
    vec_u_CR = passage_ccG_to_CR * u
    vec_u_DG = passage_ccG_to_DG * u
    stress = mat_stress * mat_grad * vec_u_CR
    stress_per_cell = stress.reshape((nb_ddl_cells // d,dim,d)) #For vectorial case
    #strain = mat_strain * mat_grad * vec_u_CR
    #strain_per_cell = strain.reshape((nb_ddl_cells // d,dim,d)) #For vectorial case

    #sorties paraview
    #if t % (T / 10) < dt:
    #    solution_u_DG.vector().set_local(vec_u_DG)
    #    solution_u_DG.vector().apply("insert")
    #    file.write(solution_u_DG, t)
    #    solution_stress.vector().set_local(mat_stress * mat_grad * vec_u_CR)
    #    solution_stress.vector().apply("insert")
    #    file.write(solution_stress, t)

    #for cracking
    cracking_facets = set()

    #Test of new cracking criterion
    stress_per_facet = average_stresses * mat_grad * vec_u_CR #plain stress
    stress_per_facet = stress_per_facet.reshape((initial_nb_ddl_CR // d,d)) #For vectorial case

    ##Regular version without taking into account local contraction
    #Gh = np.sum(stress_per_facet * stress_per_facet, axis=1)

    #Version taking into account local contraction
    normal_stress = np.sum(stress_per_facet * normals, axis=1)
    tangential_stress = np.sum(stress_per_facet * tangents, axis=1)
    local_contraction = np.where(normal_stress < 0.)
    normal_stress[local_contraction] = np.zeros_like(local_contraction)
    Gh = normal_stress * normal_stress + tangential_stress * tangential_stress
    assert(np.amin(Gh) >= 0.)

    #removing energy of already cracked facets
    Gh[list(cracked_facets)] = np.zeros(len(cracked_facets))
    Gh = np.pi / E * areas * Gh

    #breaking one facet at a time
    f = np.argmax(Gh)
    assert( f not in cracked_facets)
    cracking_facets = {f}
    print(Gh[f])
    #print(facet_num.get(f))
    c1,c2 = facet_num.get(f)
    print(G[c1][c2]['barycentre'])
    cracked_facet_vertices.append(G[c1][c2]['vertices']) #position of vertices of the broken facet

    ##breaking several facets at a time
    #cracking_facets = set(list(np.where(Gh > Gc)[0]))
    #assert(len(cracking_facets & cracked_facets) == 0)
    #for f in cracking_facets:
    #    print(Gh[f])
    #    c1,c2 = facet_num.get(f)
    #    print(G[c1][c2]['barycentre'])
    #    cracked_facet_vertices.append(G[c1][c2]['vertices']) #position of vertices of the broken facet
    

    #treatment if the crack propagates
    if len(cracking_facets) > 0:
        solution_u_DG.vector().set_local(vec_u_DG)
        solution_u_DG.vector().apply("insert")
        file.write(solution_u_DG, t)
        solution_stress.vector().set_local(mat_stress * mat_grad * vec_u_CR)
        solution_stress.vector().apply("insert")
        file.write(solution_stress, t)
        #sys.exit()

        #print('Dissipated energy cracking: %.5e' % dissipated_energy)
        #print('total energy: %.5e' % (dissipated_energy + 0.5*np.dot(v,M_lumped * v) + 0.5 * np.dot(u, A * u)))

        #storing the number of ccG dof before adding the new facet dofs
        old_nb_dof_ccG = nb_ddl_ccG

        #adapting after crack
        #penalty terms modification
        mat_jump_1_aux,mat_jump_2_aux = removing_penalty(mesh, d, dim, nb_ddl_ccG, mat_grad, passage_ccG_to_CR, G, nb_ddl_CR, cracking_facets, facet_num)
        mat_jump_1 -= mat_jump_1_aux
        mat_jump_2 -= mat_jump_2_aux
        #modifying mass and rigidity matrix beacuse of breaking facets...
        passage_ccG_to_CR, mat_grad, nb_ddl_CR, facet_num, mat_D, mat_not_D = adapting_after_crack(cracking_facets, cracked_facets, d, dim, facet_num, nb_ddl_cells, nb_ddl_ccG, nb_ddl_CR, passage_ccG_to_CR, mat_grad, G, mat_D, mat_not_D)
        out_cracked_facets('test',size_ref, count_output_crack, cracked_facet_vertices, dim) #paraview cracked facet file
        count_output_crack +=1

        #assembling new rigidity matrix and mass matrix after cracking
        mat_elas = elastic_term(mat_grad, passage_ccG_to_CR)
        passage_ccG_to_DG_1 = ccG_to_DG_1_aux_1 + ccG_to_DG_1_aux_2 * mat_grad * passage_ccG_to_CR #recomputed
        #penalty_terms modification
        mat_jump_1.resize((nb_ddl_CR,nb_ddl_ccG))
        mat_jump_2.resize((nb_ddl_CR,nb_ddl_grad))
        mat_jump = mat_jump_1 + mat_jump_2 * mat_grad * passage_ccG_to_CR
        mat_pen = mat_jump.T * mat_jump
        A = mat_elas + mat_pen #n'a pas de termes pour certains dof qu'on vient de créer
        L = np.concatenate((L, np.zeros(d * len(cracking_facets))))

        #Imposing strongly Dirichlet BC
        A_D = mat_D * A * mat_D.T
        A_not_D = mat_not_D * A * mat_not_D.T
        B = mat_not_D * A * mat_D.T
        M_not_D = mat_not_D * M_lumped

        ##recomputing time-step to adapt to the new rigidity matrix and the new mass matrix
        #eig_M = min(M_not_D)
        #eig_K = eigsh(A_not_D, k=1, return_eigenvectors=False) #which='LM' (largest in magnitude)
        #dt = np.sqrt(eig_M / eig_K)
        #print('dt: %.5e' % dt)

    cracked_facets.update(cracking_facets) #adding facets just cracked to broken facets

    #Finishing updates after cracking
    #Calcul des forces.
    #integral = mat_not_D * trace_matrix.T * L - A_not_D * u_not_D - B * F
    integral = -A_not_D * u_not_D - B * F
    Res = integral / M_not_D

    #interpolation of Dirichlet BC
    #velocity
    vel_interpolate = interpolate(v_D, U_CR).vector().get_local()
    vel_D = mat_D * trace_matrix.T * vel_interpolate

    #Computing new velocities
    v_not_D = mat_not_D * v_old + Res * dt
    v = mat_not_D.T * v_not_D + mat_D.T * vel_D
    v_old = v

#computation over
print('End of computation !')

