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
size_ref = 20 #80 #40 #20 #10
mesh = RectangleMesh(Point(0, H), Point(Ll, -H), size_ref*5, 2*size_ref, "crossed")
bnd_facets = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
h = H / size_ref
print(h)

# Sub domain for BC
def upper_left(x, on_boundary):
    return near(x[0], 0.) and x[1] >= 0. and on_boundary

def lower_left(x, on_boundary):
    return near(x[0], 0.) and x[1] <= 0. and on_boundary

def right(x, on_boundary):
    return near(x[0], Ll) and on_boundary

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
problem = DEMProblem(mesh, 1, penalty, nz_vec_BC, mu)

#Load and non-homogeneous Dirichlet BC
def eps(v): #v is a gradient matrix
    return v

def sigma(eps_el):
    return mu * eps_el #mu est le mieux en QS...

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
u_D = Expression('x[1]/fabs(x[1]) * k * t * (1 - x[0]/L)', L=Ll, k=k, t=0, degree=2)

f_CR = TestFunction(U_CR)
areas = assemble(f_CR('+') * (dS + ds)).get_local() #For crack speeds

#length crack output
#length_crack = open('h_0_05/length_crack_%i.txt' % size_ref, 'w')
#length_crack = open('test/chi_%i.txt' % 1, 'w')
folder = 'constant_chi_4_5_bis'
length_crack = open(folder+'/length_crack_%i.txt' % size_ref, 'w')

count_output_crack = 0
cracked_facet_vertices = []
cracked_facets = set()
length_cracked_facets = 0.
broken_vertices = set()

cracking_facets = set()
#before the computation begins, we break the facets to have a crack of length 1
closest = 0
for (x,y) in problem.Graph.edges():
    f = problem.Graph[x][y]['num']
    pos = problem.Graph[x][y]['barycentre']
    if problem.Graph[x][y]['breakable'] and abs(pos[1]) < 1.e-15 and pos[0] < l0:
        cracking_facets.add(f)
        cracked_facet_vertices.append(problem.Graph[x][y]['vertices']) #position of vertices of the broken facet
        length_cracked_facets += areas[f]
        broken_vertices |= set(problem.Graph[x][y]['vertices_ind'])
        if pos[0] > l0 - h:
            closest = f
#adapting after crack
problem.removing_penalty(cracking_facets)
problem.adapting_after_crack(cracking_facets, cracked_facets)
problem.update_penalty_matrix()
problem.elastic_bilinear_form(ref_elastic)
A = problem.mat_elas + problem.mat_pen

#After modifications
out_cracked_facets(folder, size_ref, 0, cracked_facet_vertices, problem.dim) #paraview cracked facet file
cracked_facets.update(cracking_facets) #adding facets just cracked to broken facets

#Homogeneous Neumann BC
L = np.zeros(problem.nb_dof_CR)

#Imposing strongly Dirichlet BC
A_not_D,B = problem.schur_complement(A)

#sorties paraview avant début calcul
#file.write(solution_u_DG, 0)

#definition of time-stepping parameters
T = 1. / k
u_D.t = 0.24 / k #il ne se passe rien avant...
chi = 4.5 #450 #45 #4.5 #0.45
dt = h / (chi*k)
#dt = 5.6e-3 / k
#chi = h / 5.6e-3
print('chi: %.5e' % chi)
print('Delta u_D: %.5e' % (dt*k))

#Début boucle temporelle
while u_D.t < T:
    u_D.t += dt
    print('BC disp: %.5e' % (k * u_D.t))
    inverting = True

    #interpolation of Dirichlet BC
    FF = interpolate(u_D, U_CR).vector().get_local()
    F = problem.mat_D * problem.trace_matrix.T * FF

    #taking into account exterior loads
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

        cracking_facets = set()
        
        #Computing new Gh
        #Gh = problem.energy_release_rates(vec_u_CR, cracked_facets)
        Gh = problem.energy_release_rates_bis(vec_u_CR, vec_u_DG)

        #Test for Gh by vertex
        Gh_v = energy_release_rate_vertex(problem, broken_vertices, Gh)
        #to_print = -1
        c1 = problem.facet_num.get(closest)[0]
        #for v in problem.Graph[c1][problem.nb_dof_cells // problem.d + closest]['vertices_ind']:
        #    to_print = max(to_print, Gh_v[v])
        #print('Gh crack tip: %.5e' % to_print)

        #Testing crack advance #Should be modified for kinking
        pos_closest = problem.Graph[c1][problem.nb_dof_cells // problem.d + closest]['barycentre'][0]
        for f in problem.facet_to_facet.get(closest):
            if len(problem.facet_num.get(f)) == 2:
                n1,n2 = problem.facet_num.get(f)
                pos = problem.Graph[n1][n2]['barycentre']
                if pos[0] > pos_closest and np.absolute(pos[1]) < 1.e-15:
                    Gh = -1
                    for v in problem.Graph[n1][n2]['vertices_ind']:
                        Gh = max(Gh, Gh_v[v])
                    if Gh > Gc:
                        cracking_facets = {f}
                        closest = f #Update closest
                        broken_vertices |= set(problem.Graph[n1][n2]['vertices_ind'])
                        break
                    else:
                        inverting = False

##Finding which facet to break
#c1 = problem.facet_num.get(closest)[0]
#pos_closest = problem.Graph[c1][problem.nb_dof_cells // problem.d + closest]['barycentre'][0]
##Testing cracking
#for f in problem.facet_to_facet.get(closest):
#    if len(problem.facet_num.get(f)) == 2:
#        n1,n2 = problem.facet_num.get(f)
#        pos = problem.Graph[n1][n2]['barycentre']
#        if pos[0] > pos_closest and np.absolute(pos[1]) < 1.e-15:
#            #print('Gh facet: %.5e\n' % Gh[f])
#            if Gh[f] > Gc:
#                cracking_facets = {f}
#                closest = f #Update closest
#                broken_vertices |= set(problem.Graph[n1][n2]['vertices_ind'])
#                break
#            else:
#                inverting = False
#
            #get correspondance between dof
        for f in cracking_facets:
            n1,n2 = problem.facet_num.get(f)
            cracked_facet_vertices.append(problem.Graph[n1][n2]['vertices']) #position of vertices of the broken facet
            pos = problem.Graph[n1][n2]['barycentre']
            print('pos bary facet : (%f,%f)' % (pos[0], pos[1]))
            length_cracked_facets += areas[f]

        #treatment if the crack propagates
        if len(cracking_facets) > 0:
            #storing the number of ccG dof before adding the new facet dofs

            #output with length of the crack in 2d
            length_crack.write('%.5e %.5e\n' % (k * u_D.t, length_cracked_facets))
            
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
length_crack.close()
print('End of computation !')

