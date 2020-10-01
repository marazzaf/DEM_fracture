# coding: utf-8
from dolfin import *
import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx

def facet_neighborhood(mesh_):
    """Returns a dictionnary containing as key the a modified index for the facet and as values the list of indices of the cells (or cell) containing the facet. """
    scal_CR = FunctionSpace(mesh_, 'CR', 1) #Function spaces
    scal_DG = FunctionSpace(mesh_, 'DG', 0)
    area = FacetArea(mesh_) #Area of facets in mesh

    g_CR = TestFunction(scal_CR)
    f_DG = TrialFunction(scal_DG)

    signs = assemble( g_CR('+') * (f_DG('+') / area('+') - f_DG('-') / area('-')) * dS + g_CR('+') * f_DG('+') / area('+') * ds, form_compiler_parameters=None)
    row,col,val = as_backend_type(signs).mat().getValuesCSR()
    mat_signs = csr_matrix((val, col, row))
    mat_signs.eliminate_zeros()
    mat_signs = mat_signs.rint()

    res = dict()

    for i in range(mat_signs.shape[0]):
        non_zero = list(mat_signs[i,:].nonzero()[1])
        assert(len(non_zero) > 0 and len(non_zero) < 3) #because a facet has whether one or two neighbouring cells
        res[i] = non_zero #indexing with respect to scalar CR dof and not facet index...
    
    return res #just like facet_neighborhood but with everything indexed by the scalar_CR dof and not the index of the facet...

def connectivity_graph(problem, dirichlet_dofs):
    G = nx.Graph()
    count = 0

    #For numbers of facets
    dofmap_CR = problem.CR.dofmap()

    #useful auxiliary functions
    vol_c = CellVolume(problem.mesh) #Pour volume des particules voisines
    hF = FacetArea(problem.mesh)
    n = FacetNormal(problem.mesh)
    scalar_DG = FunctionSpace(problem.mesh, 'DG', 0) #for volumes
    f_DG = TestFunction(scalar_DG)
    scalar_CR = FunctionSpace(problem.mesh, 'CR', 1) #for surfaces
    f_CR = TestFunction(scalar_CR)

    #assembling penalty factor
    a_aux = problem.penalty * hF / vol_c * f_CR * ds + problem.penalty * (2.*hF('+'))/ (vol_c('+') + vol_c('-')) * f_CR('+') * dS
    pen_factor = assemble(a_aux).get_local()

    #computation of volumes, surfaces and normals
    volumes = assemble(f_DG * dx).get_local()
    assert(volumes.min() > 0.)
    areas = assemble(f_CR('+') * (dS + ds)).get_local()
    assert(areas.min() > 0.)

    #importing cell dofs
    for c in cells(problem.mesh): #Importing cells
        aux = list(np.arange(count, count+problem.d))
        count += problem.d
        #computing volume and barycentre of the cell
        vert = []
        vert_ind = []
        for v in vertices(c):
            vert.append( np.array(v.point()[:])[:problem.dim] )
            vert_ind.append(v.index())
        vol = volumes[c.index()]
        vert = np.array(vert)
        bary = vert.sum(axis=0) / vert.shape[0]
        #adding node to the graph
        G.add_node(c.index(), dof=aux, pos=bary, measure=vol, vertices=vert, bnd=False) #bnd=True if cell is on boundary of the domain
        
    #importing connectivity and facet dofs
    for f in facets(problem.mesh):
        aux_bis = [] #number of the cells
        for c in cells(f):
            aux_bis.append(c.index())
        num_global_ddl_facet = dofmap_CR.entity_dofs(problem.mesh, problem.dim - 1, np.array([f.index()], dtype="uintp")) #number of the dofs in CR
        #computing quantites related to the facets
        vert = []
        vert_ind = []
        for v in vertices(f):
            vert.append( np.array(v.point()[:])[:problem.dim] )
            vert_ind.append(v.index())
        area = areas[num_global_ddl_facet[0] // problem.d]
        #facet barycentre computation
        vert = np.array(vert)
        bary = vert.sum(axis=0) / vert.shape[0]
        #index of the edges of the facet
        Edges = set()
        if problem.dim == 3:
            for e in edges(f):
                Edges.add(e.index())

        #adding the facets to the graph
        if len(aux_bis) == 2: #add the link between two cell dofs
            #putting normals in the order of lowest cell number towards biggest cell number
            n1 = min(aux_bis[0],aux_bis[1])
            bary_n1 = G.node[n1]['pos']
            n2 = max(aux_bis[0],aux_bis[1])
            bary_n2 = G.node[n2]['pos']
         
            #adding edge
            G.add_edge(aux_bis[0],aux_bis[1], num=num_global_ddl_facet[0] // problem.d, recon=set([]), dof_CR=num_global_ddl_facet, measure=area, barycentre=bary, vertices=vert, pen_factor=pen_factor[num_global_ddl_facet[0] // problem.d], breakable=True) #, vertices_ind=vert_ind)
            
        elif len(aux_bis) == 1: #add the link between a cell dof and a boundary facet dof
            for c in cells(f): #only one cell contains the boundary facet
                bary_cell = G.node[c.index()]['pos']
            #computation of volume associated to the facet for mass matrix
            if problem.dim == 2:
                vol_facet = 0.5 * np.linalg.norm(np.cross(vert[0] - bary_cell, vert[1] - bary_cell))
            elif problem.dim == 3:
                vol_facet = np.linalg.norm(np.dot( np.cross(vert[0] - bary_cell, vert[1] - bary_cell), vert[2] - bary_cell )) / 6.

            #checking if adding "dofs" for Dirichlet BC
            nb_dofs = len(dirichlet_dofs & set(num_global_ddl_facet))
            aux = list(np.arange(count, count+nb_dofs))
            count += nb_dofs
            components = sorted(list(dirichlet_dofs & set(num_global_ddl_facet)))
            components = np.array(components) % problem.d
            
            #number of the dof is total number of cells + num of the facet
            G.add_node(problem.nb_dof_cells // problem.d + num_global_ddl_facet[0] // problem.d, pos=bary, dof=aux, dirichlet_components=components)
            G.add_edge(aux_bis[0], problem.nb_dof_cells // problem.d + num_global_ddl_facet[0] // problem.d, num=num_global_ddl_facet[0] // problem.d, dof_CR=num_global_ddl_facet, measure=area, barycentre=bary, vertices=vert, pen_factor=pen_factor[num_global_ddl_facet[0] // problem.d], breakable=False)
            G.node[aux_bis[0]]['bnd'] = True #Cell is on the boundary of the domain
                
    return G

def linked_facets(problem): #gives facets sharing a set of codimension 2
    res = dict()
    for f in facets(problem.mesh): #ne pas prendre l'index comme numéro !
        nei = []
        for c in cells(f): #cells neighbouring f
            nei.append(c.index())
        if len(nei) == 2: #for inner facets
            g = problem.Graph[nei[0]][nei[1]]['num'] #c'est le num de la facet pour nous !
            aux = set()
            if problem.dim == 2:
                for v in vertices(f):
                    for fp in facets(v):
                        nei_bis = []
                        for c in cells(fp):
                            nei_bis.append(c.index())
                        if len(nei_bis) == 2: #otherwise it is a boundary facet that cannot break
                            aux_facet_num = problem.Graph[nei_bis[0]][nei_bis[1]]['num']
                            aux.add(aux_facet_num)
                        
            elif problem.dim == 3:
                for e in edges(f):
                    for fp in facets(e):
                        nei_bis = []
                        for c in cells(fp):
                            nei_bis.append(c.index())
                        if len(nei_bis) == 2: #otherwise it is a boundary facet that cannot break
                            aux_facet_num = problem.Graph[nei_bis[0]][nei_bis[1]]['num']
                            aux.add(aux_facet_num)
            aux.remove(g)
            res[g] = aux #list #pas f !
    return res

def facets_in_cell(problem): #gives facets contained in every cell
    dofmap_CR = problem.CR.dofmap()
    
    res = dict()
    for c in cells(problem.mesh): 
        list_facet_nums = []
        for f in facets(c): #facets contained in c
            num_global_ddl_facet = dofmap_CR.entity_dofs(problem.mesh, problem.dim - 1, np.array([f.index()], dtype="uintp"))
            list_facet_nums.append(num_global_ddl_facet[0] // problem.d)
        res[c.index()] = set(list_facet_nums)
        
    return res
