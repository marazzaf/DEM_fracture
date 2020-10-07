# coding: utf-8
import sys
from dolfin import *
import numpy as np
from ufl.operators import sign
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from DEM_cracking.DEM import *
from DEM_cracking.miscellaneous import *
from DEM_cracking.cracking import *

# Form compiler options
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

# elastic parameters
mu = 0.2
penalty = mu

#geometry
a, R, l0 = 0.5, 0.025, 0.025
size_ref = 1
#mesh = Mesh("mesh_old/circle_%i.xml" % size_ref) #maillage domaine circulaire
#bnd_facets = MeshFunction("size_t", mesh, 'mesh_old/circle_%i_facet_region.xml' % size_ref)
#bnd_facets = MeshFunction("size_t", mesh, 1)
mesh = Mesh('mesh/circle_cracked_%i.xml' % size_ref) #pre-cracked
bnd_facets = MeshFunction("size_t", mesh, "mesh/circle_cracked_%i_facet_region.xml" % size_ref)
ds = Measure('ds')(subdomain_data=bnd_facets)

#Function spaces
U_CR = FunctionSpace(mesh, 'CR', 1) #Pour interpollation dans les faces

#useful
d = 1 #scalar case

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
e_r = as_vector((c_theta,s_theta))
e_theta = as_vector((-s_theta,c_theta))

#print('Ref value: %.5e' % (2. * tau / mu * sqrt(R * a * 0.5)))

# Define BC
v_CR = TestFunction(U_CR)

#new for BC
hF = FacetArea(mesh)
l4 = v_CR('+') / hF * ds #test...
L4 = assemble(l4)
vec_BC = L4.get_local()
nz_vec_BC = list(vec_BC.nonzero()[0])
nz_vec_BC = set(nz_vec_BC)

#Creating the DEM problem
problem = DEMProblem(mesh, d, penalty, nz_vec_BC, mu)

def eps(v):
    return v

def sigma(eps_el):
    return mu * eps_el

#Variational problem
ref_elastic = ref_elastic_bilinear_form(problem, sigma, eps)
mat_elas = problem.elastic_bilinear_form(ref_elastic)

#Stresses output
problem.mat_stress = output_stress(problem, sigma, eps)

#To store solutions
solution_u_DG = Function(problem.DG_0,  name="disp DG")
solution_stress = Function(problem.W, name="Stress")

count_output_energy_release = 0
count_output_disp = 0
count_output_crack = 0
cracked_facet_vertices = []
cracked_facets = set()

#Crack is put in the plate
cracking_facets = set()
#before the computation begins, we break the facets to have a crack of length 1
for (x,y) in problem.Graph.edges():
    f = problem.Graph[x][y]['dof_CR'][0] // d
    pos = problem.Graph[x][y]['barycentre']
    if problem.Graph[x][y]['breakable'] and np.absolute(pos[1]) < 1.e-15 and pos[0] < 0.:
        cracking_facets.add(f)
        n1,n2 = facet_num.get(f)
        cracked_facet_vertices.append(problem.Graph[n1][n2]['vertices']) #position of vertices of the broken facet
        
#adapting after crack
problem.removing_penalty(cracking_facets)
problem.adapting_after_crack(cracking_facets, cracked_facets)
problem.update_penalty_matrix()
problem.elastic_bilinear_form(ref_elastic)
A = problem.mat_elas + problem.mat_pen

#Homogeneous Neumann BC
L = np.zeros(problem.nb_dof_CR)

#paraview outputs
file = File('test_%i.pvd' % size_ref)

#Imposing strongly Dirichlet BC
A_not_D,B = problem.schur_complement(A)

#Removing Dirichlet BC
rhs = -problem.mat_not_D * A * problem.trace_matrix.T * np.nan_to_num(local_project(u_sing, U_CR).vector().get_local())

#inverting system
u_reduced = spsolve(A_not_D, rhs)
u = problem.complete_solution(u_reduced,u_sing)

#Post-processing
vec_u_CR = problem.DEM_to_CR * u
vec_u_DG = problem.DEM_to_DG * u

#output initial conditions
solution_u_DG.vector().set_local(vec_u_DG)
solution_u_DG.vector().apply("insert")
file.write(solution_u_DG, 0)
sol_DG_1 = Function(problem.DG_1, name="disp DG 1")
sol_DG_1.vector().set_local(problem.DEM_to_DG_1 * u)
file.write(sol_DG_1, 0)

#write the convergence errors etc...
#Reference solution interpolated in ccG vector
u_ref_ccG = problem.DEM_interpolation(u_sing)

#energy error
diff = u - u_ref_ccG
err_energy = np.dot(diff, A * diff)
err_energy = np.sqrt(0.5 * err_energy)
print('error energy: %.5e' % err_energy)

#L2 error
u_ref_DG_1 = local_project(u_sing, problem.DG_1)
for_para = Function(problem.DG_1, name='Ref')
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


solution_stress.vector().set_local(problem.mat_stress * problem.mat_grad * vec_u_CR)
solution_stress.vector().apply("insert")
file.write(solution_stress, 0)

#reference elastic energy
sigma_sing = tau * sqrt(a / (2 * r)) * (s_theta_2 * e_r + c_theta_2 * e_theta)
stress = local_project(sigma_sing, problem.W)
err_stress = errornorm(solution_stress, stress, 'L2')
print('err stress: %.5e' % err_stress)
n = FacetNormal(problem.mesh)
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

img = plot(sol_DG_1 - local_project(u_sing, problem.DG_1))
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

