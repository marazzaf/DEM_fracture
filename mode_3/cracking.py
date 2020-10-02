# coding: utf-8
import sys
from dolfin import *
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg
from DEM_cracking.DEM import *
from DEM_cracking.miscellaneous import *
from DEM_cracking.cracking import *

# Form compiler options
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

# elastic parameters
mu = 0.2
penalty = mu
Gc = 0.01
k = 1.e-3 #loading speed...

Ll, l0, H = 5., 1., 1.
folder = 'structured'
size_ref = 5 #40 #20 #10
mesh = RectangleMesh(Point(0, H), Point(Ll, -H), size_ref*5, 2*size_ref, "crossed")
#folder = 'no_initial_crack'
#folder = 'unstructured'
#h = H / size_ref
#size_ref = 3
#mesh = Mesh('mesh/test.xml') #3
#size_ref = 2
#mesh = Mesh('mesh/cracked_plate_fine.xml')
#size_ref = 1
#mesh = Mesh('mesh/cracked_plate_coarse.xml')
h = mesh.hmax()
#print(h)

#scalar problem
d = 1

# Sub domain for BC
def upper_left(x, on_boundary):
    return near(x[0], 0.) and x[1] >= 0. and on_boundary

def lower_left(x, on_boundary):
    return near(x[0], 0.) and x[1] <= 0. and on_boundary

def right(x, on_boundary):
    return near(x[0], Ll) and on_boundary

#difining boundary
bnd_facets = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
bnd_facets.set_all(0)
traction_boundary_1 = AutoSubDomain(upper_left)
traction_boundary_1.mark(bnd_facets, 41)
traction_boundary_2 = AutoSubDomain(lower_left)
traction_boundary_2.mark(bnd_facets, 42)
clamped_boundary = AutoSubDomain(right)
clamped_boundary.mark(bnd_facets, 45)
ds = Measure('ds')(subdomain_data=bnd_facets)

# Define variational problem
U_CR = FunctionSpace(mesh, 'CR', 1)
hF = FacetArea(mesh)
v_CR = TestFunction(U_CR)

#new for BC
l4 = v_CR('+') / hF * (ds(41) + ds(42) + ds(45))
L4 = assemble(l4)
vec_BC = L4.get_local()
nz_vec_BC = list(vec_BC.nonzero()[0])
nz_vec_BC = set(nz_vec_BC)

#Creating the DEM problem
problem = DEMProblem(mesh, d, penalty, nz_vec_BC, mu)

#Imposing strongly Dirichlet BC
mat_not_D,mat_D = schur_matrices(problem)

def eps(v): #v is a gradient matrix
    return v

def sigma(eps_el):
    return mu * eps_el

#Variational problem
ref_elastic = ref_elastic_bilinear_form(problem, sigma, eps)
mat_elas = problem.elastic_bilinear_form(ref_elastic)

#Stresses output
problem.mat_stress = output_stress(problem, sigma, eps)

#useful
solution_u_DG = Function(problem.DG_0,  name="disp DG")
solution_stress = Function(problem.W, name="Stress")

#reference solution
x = SpatialCoordinate(mesh)
#Dirichlet BC
u_D = Expression('x[1]/fabs(x[1]) * k * t * H * (1 - x[0]/L)', L=Ll, H=H, k=k, t=0, degree=2)

#Homogeneous Neumann BC
L = np.zeros(problem.nb_dof_CR)

file = File('%s/anti_plane_%i_.pvd' % (folder,size_ref))

count_output_crack = 1
cracked_facet_vertices = []
cracked_facets = set()
length_cracked_facets = 0.
cells_with_cracked_facet = set()
not_breakable_facets = set()

cracking_facets = set()
cells_to_test = set()
#before the computation begins, we break the facets to have a crack of length 1
for (x,y) in problem.Graph.edges():
    f = problem.Graph[x][y]['dof_CR'][0] // d
    pos = problem.Graph[x][y]['barycentre']
    if problem.Graph[x][y]['breakable'] and abs(pos[1]) < 1.e-15 and pos[0] < l0:
        cracking_facets.add(f)
        cracked_facet_vertices.append(problem.Graph[x][y]['vertices']) #position of vertices of the broken facet
        cells_with_cracked_facet |= {x,y}
        #cells_to_test |= set(facet_num.get(f)) #verifying only one facet per cell breaks


#adapting after crack
old_mat_pen = problem.mat_pen
mat_jump_1_aux,mat_jump_2_aux = removing_penalty(problem, cracking_facets)
problem.mat_jump_1 -= mat_jump_1_aux
problem.mat_jump_2 -= mat_jump_2_aux
problem.adapting_after_crack(cracking_facets, cracked_facets) #Get problem as an output or make it a method of the class ?
out_cracked_facets(folder, size_ref, 0, cracked_facet_vertices, problem.dim) #paraview cracked facet file
cracked_facets.update(cracking_facets) #adding facets just cracked to broken facets
mat_elas = problem.elastic_bilinear_form(ref_elastic)
problem.mat_jump_1.resize((problem.nb_dof_CR,problem.nb_dof_DEM))
problem.mat_jump_2.resize((problem.nb_dof_CR,problem.nb_dof_grad))
problem.mat_jump = problem.mat_jump_1 + problem.mat_jump_2 * problem.mat_grad * problem.DEM_to_CR
mat_pen = problem.mat_jump.T * problem.mat_jump
A = mat_elas + mat_pen
L = np.concatenate((L, np.zeros(d * len(cracking_facets))))

#Updating facets that cannot be broken because they belong to a cell with an already broken facet
not_breakable_facets |= cracked_facets #already broken facets cannot break again
for c in cells_with_cracked_facet:
    not_breakable_facets |= problem.facets_cell.get(c)

#Imposing strongly Dirichlet BC
A_D = mat_D * A * mat_D.T
A_not_D = mat_not_D * A * mat_not_D.T
B = mat_not_D * A * mat_D.T

#definition of time-stepping parameters
T = 2. / k #1. / k
chi = 4.5
dt = h / (k * chi)
#print(dt)
u_D.t = 0.25 / k #il ne se passe rien avant...

#Début boucle temporelle
while u_D.t < T:
    u_D.t += dt
    print('BC disp: %.5e' % (k * u_D.t))
    inverting = True

    #interpolation of Dirichlet BC
    FF = interpolate(u_D, U_CR).vector().get_local()
    F = mat_D * problem.trace_matrix.T * FF

    #taking into account exterior loads
    #L_not_D = mat_not_D * matrice_trace_bord.T * L
    L_not_D = -B * F

    count = 0
    while inverting:
        #inverting system
        count += 1
        print('COUNT: %i' % count)
        u_reduced,info = cg(A_not_D, L_not_D)
        assert(info == 0)
        u = mat_not_D.T * u_reduced + mat_D.T * F

        #Post-processing
        vec_u_CR = problem.DEM_to_CR * u
        vec_u_DG = problem.DEM_to_DG * u
        #facet_stresses = average_stresses * mat_grad * vec_u_CR
        stresses = problem.mat_stress * problem.mat_grad * vec_u_CR
        stress_per_cell = stresses.reshape((problem.nb_dof_cells // problem.d,problem.dim))
        #strain = mat_strain * mat_grad * vec_u_CR
        #strain_per_cell = strain.reshape((nb_ddl_cells // d,dim))

        ##sorties paraview
        #if u_D.t % (T / 10) < dt:
        solution_u_DG.vector().set_local(vec_u_DG)
        solution_u_DG.vector().apply("insert")
        file.write(solution_u_DG, u_D.t)
        solution_stress.vector().set_local(problem.mat_stress * problem.mat_grad * vec_u_CR)
        solution_stress.vector().apply("insert")
        file.write(solution_stress, u_D.t)
        sys.exit()

        cracking_facets = set()
        
        #Computing new Gh
        Gh = problem.energy_release_rates(vec_u_CR, cracked_facets, not_breakable_facets)

        #Potentially cracking facet with biggest Gh
        idx = np.argpartition(Gh, -20)[-20:] #is 20 enough?
        indices = idx[np.argsort((-Gh)[idx])]

        #Chossing which facet to break
        for f in indices:
            if Gh[f] > Gc:
                #if len(test_1) == 0 and len(test_2) == 0:
                cracking_facets = {f}
                c1,c2 = facet_num.get(f)
                print(Gh[f])
                #print(G[c1][c2]['barycentre'])
                cracked_facet_vertices.append(G[c1][c2]['vertices']) #position of vertices of the broken facet
                cells_with_cracked_facet |= {c1,c2}
                not_breakable_facets |= (facets_cell.get(c1) | facets_cell.get(c2))
                #potentially_cracking_facets |= facet_to_facet.get(f) #updating set
                #cells_to_test |= set(facet_num.get(f))
                break #When we get a facet verifying the conditions, we stop the search and continue with the cracking process
            else:
                inverting = False

        if len(cracking_facets) > 0:
            #print(cracking_facets)
            solution_u_DG.vector().set_local(vec_u_DG)
            solution_u_DG.vector().apply("insert")
            file.write(solution_u_DG, u_D.t)
            solution_stress.vector().set_local(mat_stress * mat_grad * vec_u_CR)
            solution_stress.vector().apply("insert")
            file.write(solution_stress, u_D.t)

        #get correspondance between dof
        for f in cracking_facets:
            #sys.exit()
            n1,n2 = facet_num.get(f)
            cracked_facet_vertices.append(G[n1][n2]['vertices']) #position of vertices of the broken facet
            pos = G[n1][n2]['barycentre']
            #print('dof num: %i' % f)
            print('pos bary facet : (%f,%f)' % (pos[0], pos[1]))
            #potentially_cracking_facets -= (facets_cell.get(n1) | facets_cell.get(n2)) #so that no other facet of these two cells will break

#        #to be sure not to break facets of a cell that already has a broken facet
#        for f in cracked_facets:
#            c1 = facet_num.get(f)[0]
#            potentially_cracking_facets -= facets_cell.get(c1)

        #treatment if the crack propagates
        if len(cracking_facets) > 0:
            #storing the number of ccG dof before adding the new facet dofs
            old_nb_dof_ccG = nb_ddl_ccG

            count_output_crack +=1
            ##recomputing boundary of the crack
            #for n_facet in cracking_facets:
            #    crack.node[n_facet]['broken'] = True
            #bnd_crack = boundary_crack_new(crack,dim,cracking_facets,broken_vertices)

            #adapting after crack
            #penalty terms modification
            mat_jump_1_aux,mat_jump_2_aux = removing_penalty(mesh, d, dim, nb_ddl_ccG, mat_grad, passage_ccG_to_CR, G, nb_ddl_CR, cracking_facets, facet_num)
            mat_jump_1 -= mat_jump_1_aux
            mat_jump_2 -= mat_jump_2_aux
            passage_ccG_to_CR, mat_grad, nb_ddl_CR, facet_num, mat_D, mat_not_D = adapting_after_crack(cracking_facets, cracked_facets, d, dim, facet_num, nb_ddl_cells, nb_ddl_ccG, nb_ddl_CR, passage_ccG_to_CR, mat_grad, G, mat_D, mat_not_D)
            out_cracked_facets(folder, size_ref, count_output_crack, cracked_facet_vertices, dim) #paraview cracked facet file

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
            F = mat_D * trace_matrix.T * FF
            L_not_D = -B * F

        cracked_facets.update(cracking_facets) #adding facets just cracked to broken facets

#computation over
print('End of computation !')

