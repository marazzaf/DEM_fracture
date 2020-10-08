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
folder = 'test_structured'
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
#length_cracked_facets = 0.
cells_with_cracked_facet = set()
not_breakable_facets = set()

cracking_facets = set()
#cells_to_test = set()
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
problem.removing_penalty(cracking_facets)
problem.adapting_after_crack(cracking_facets, cracked_facets)
problem.update_penalty_matrix()
problem.elastic_bilinear_form(ref_elastic)
A = problem.mat_elas + problem.mat_pen
L = np.concatenate((L, np.zeros(d * len(cracking_facets))))

#After modifications
out_cracked_facets(folder, size_ref, 0, cracked_facet_vertices, problem.dim) #paraview cracked facet file
cracked_facets.update(cracking_facets) #adding facets just cracked to broken facets

#Updating facets that cannot be broken because they belong to a cell with an already broken facet
not_breakable_facets |= cracked_facets #already broken facets cannot break again
for c in cells_with_cracked_facet:
    not_breakable_facets |= problem.facets_cell.get(c)

#Imposing strongly Dirichlet BC
A_not_D,B = problem.schur_complement(A)

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
    F = problem.mat_D * problem.trace_matrix.T * FF

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
        u = problem.complete_solution(u_reduced,u_D)

        #Post-processing
        vec_u_CR = problem.DEM_to_CR * u
        vec_u_DG = problem.DEM_to_DG * u
        stresses = problem.mat_stress * problem.mat_grad * vec_u_CR
        stress_per_cell = stresses.reshape((problem.nb_dof_cells // problem.d,problem.dim))

        ##sorties paraview
        #if u_D.t % (T / 10) < dt:
        #solution_u_DG.vector().set_local(vec_u_DG)
        #solution_u_DG.vector().apply("insert")
        #file.write(solution_u_DG, u_D.t)
        #solution_stress.vector().set_local(problem.mat_stress * problem.mat_grad * vec_u_CR)
        #solution_stress.vector().apply("insert")
        #file.write(solution_stress, u_D.t)

        cracking_facets = set()
        
        #Computing new Gh
        #Gh = problem.energy_release_rates(vec_u_CR, cracked_facets, not_breakable_facets)
        Gh = problem.energy_release_rates_bis(vec_u_CR, vec_u_DG)

        #Potentially cracking vertex with biggest Gh
        idx = np.argpartition(Gh, -20)[-20:] #is 20 enough?
        indices = idx[np.argsort((-Gh)[idx])]

        #Choosing which facet to break
        for v in indices:
            if Gh[v] > Gc:
                f = kinking_criterion(problem, v, vec_u_CR, not_breakable_facets)
                #Updating
                cracking_facets = {f}
                c1,c2 = problem.facet_num.get(f)
                cells_with_cracked_facet |= {c1,c2}
                not_breakable_facets |= (problem.facets_cell.get(c1) | problem.facets_cell.get(c2))
                break #When we get a facet verifying the conditions, we stop the search and continue with the cracking process
            else:
                inverting = False

        if len(cracking_facets) > 0:
            solution_u_DG.vector().set_local(vec_u_DG)
            solution_u_DG.vector().apply("insert")
            file.write(solution_u_DG, u_D.t)
            solution_stress.vector().set_local(problem.mat_stress * problem.mat_grad * vec_u_CR)
            solution_stress.vector().apply("insert")
            file.write(solution_stress, u_D.t)

        #get correspondance between dof
        for f in cracking_facets:
            n1,n2 = problem.facet_num.get(f)
            cracked_facet_vertices.append(problem.Graph[n1][n2]['vertices']) #position of vertices of the broken facet
            pos = problem.Graph[n1][n2]['barycentre']
            print('pos bary facet : (%f,%f)' % (pos[0], pos[1]))


        #treatment if the crack propagates
        if len(cracking_facets) > 0:
            #adapting after crack
            problem.removing_penalty(cracking_facets)
            problem.adapting_after_crack(cracking_facets, cracked_facets)
            problem.update_penalty_matrix()
            problem.elastic_bilinear_form(ref_elastic)
            A = problem.mat_elas + problem.mat_pen
            L = np.concatenate((L, np.zeros(problem.d * len(cracking_facets))))

            #Crack output
            count_output_crack +=1
            out_cracked_facets(folder, size_ref, count_output_crack, cracked_facet_vertices, problem.dim) #paraview cracked facet file

            #Imposing strongly Dirichlet BC
            A_not_D,B = problem.schur_complement(A)
            F = problem.mat_D * problem.trace_matrix.T * FF
            L_not_D = -B * F

        cracked_facets.update(cracking_facets) #adding facets just cracked to broken facets

#computation over
print('End of computation !')

