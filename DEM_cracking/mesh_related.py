# coding: utf-8
from dolfin import *
from numpy import array

def facet_neighborhood(mesh_):
    """Returns a dictionnary containing as key the a modified index for the facet and as values the list of indices of the cells (or cell) containing the facet. """
    scal_CR = FunctionSpace(mesh_, 'CR', 1) #Function spaces
    scal_DG = FunctionSpace(mesh_, 'DG', 0)
    area = FacetArea(mesh_) #Area of facets in mesh

    g_CR = TestFunction(scal_CR)
    f_DG = TrialFunction(scal_DG)

    signs = assemble( g_CR('+') * (f_DG('+') / area('+') - f_DG('-') / area('-')) * dS + g_CR('+') * f_DG('+') / area('+') * ds, form_compiler_parameters=None)
    row,col,val = as_backend_type(signs).mat().getValuesCSR()
    mat_signs = sp.csr_matrix((val, col, row))
    mat_signs.eliminate_zeros()
    mat_signs = mat_signs.rint()

    res = dict()

    for i in range(mat_signs.shape[0]):
        non_zero = list(mat_signs[i,:].nonzero()[1])
        assert(len(non_zero) > 0 and len(non_zero) < 3) #because a facet has whether one or two neighbouring cells
        res[i] = non_zero #indexing with respect to scalar CR dof and not facet index...
    
    return res #just like facet_neighborhood but with everything indexed by the scalar_CR dof and not the index of the facet...

def connectivity_graph(mesh_, d_, penalty_, dirichlet_dofs):    G = nx.Graph()
    count = 0

    #useful mesh entities
    dim = mesh_.topology().dim()
    if d_ == 1:
        U_CR = FunctionSpace(mesh_, 'CR', 1)
        U_DG = FunctionSpace(mesh_, 'DG', 0)
    elif d_ >= 2:
        U_CR = VectorFunctionSpace(mesh_, 'CR', 1)
        U_DG = VectorFunctionSpace(mesh_, 'DG', 0)
    nb_ddl_cells = U_DG.dofmap().global_dimension()
    dofmap_CR = U_CR.dofmap()
    nb_ddl_CR = dofmap_CR.global_dimension()

    #useful auxiliary functions
    vol_c = CellVolume(mesh_) #Pour volume des particules voisines
    hF = FacetArea(mesh_)
    n = FacetNormal(mesh_)
    scalar_DG = FunctionSpace(mesh_, 'DG', 0) #for volumes
    f_DG = TestFunction(scalar_DG)
    scalar_CR = FunctionSpace(mesh_, 'CR', 1) #for surfaces
    f_CR = TestFunction(scalar_CR)
    vectorial_CR = VectorFunctionSpace(mesh_, 'CR', 1) #for normals
    v_CR = TestFunction(vectorial_CR)

    #assembling penalty factor
    a_aux = penalty_ * hF / vol_c * f_CR * ds + penalty_ * (2.*hF('+'))/ (vol_c('+') + vol_c('-')) * f_CR('+') * dS
    pen_factor = assemble(a_aux).get_local()

    #computation of volumes, surfaces and normals
    volumes = assemble(f_DG * dx).get_local()
    assert(volumes.min() > 0.)
    areas = assemble(f_CR('+') * (dS + ds)).get_local()
    assert(areas.min() > 0.)
    normals_aux = assemble( dot(n('-'), v_CR('-')) / hF('-') * dS + dot(n, v_CR) / hF * ds ).get_local() #(dS + ds)
    normals = normals_aux.reshape((nb_ddl_CR // d_, dim))

    #importing cell dofs
    for c in cells(mesh_): #Importing cells
        aux = list(np.arange(count, count+d_))
        count += d_
        #computing volume and barycentre of the cell
        vert = []
        vert_ind = []
        for v in vertices(c):
            vert.append( np.array(v.point()[:])[:dim] )
            vert_ind.append(v.index())
        vol = volumes[c.index()]
        vert = np.array(vert)
        bary = vert.sum(axis=0) / vert.shape[0]
        #adding node to the graph
        G.add_node(c.index(), dof=aux, pos=bary, measure=vol, vertices=vert, bnd=False) #bnd=True if cell is on boundary of the domain
        
    #importing connectivity and facet dofs
    for f in facets(mesh_):
        aux_bis = [] #number of the cells
        for c in cells(f):
            aux_bis.append(c.index())
        num_global_ddl_facet = dofmap_CR.entity_dofs(mesh_, dim - 1, np.array([f.index()], dtype="uintp")) #number of the dofs in CR
        #computing quantites related to the facets
        vert = []
        vert_ind = []
        for v in vertices(f):
            vert.append( np.array(v.point()[:])[:dim] )
            vert_ind.append(v.index())
        normal = normals[num_global_ddl_facet[0] // d_, :]
        area = areas[num_global_ddl_facet[0] // d_]
        #facet barycentre computation
        vert = np.array(vert)
        bary = vert.sum(axis=0) / vert.shape[0]
        #index of the edges of the facet
        Edges = set()
        if dim == 3:
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
            G.add_edge(aux_bis[0],aux_bis[1], num=num_global_ddl_facet[0] // d_, recon=set([]), dof_CR=num_global_ddl_facet, measure=area, barycentre=bary, normal=normal, vertices=vert, edges=Edges, pen_factor=pen_factor[num_global_ddl_facet[0] // d_], breakable=True) #, vertices_ind=vert_ind)
            
        elif len(aux_bis) == 1: #add the link between a cell dof and a boundary facet dof
            for c in cells(f): #only one cell contains the boundary facet
                bary_cell = G.node[c.index()]['pos']
            #computation of volume associated to the facet for mass matrix
            if dim == 2:
                vol_facet = 0.5 * np.linalg.norm(np.cross(vert[0] - bary_cell, vert[1] - bary_cell))
            elif dim == 3:
                vol_facet = np.linalg.norm(np.dot( np.cross(vert[0] - bary_cell, vert[1] - bary_cell), vert[2] - bary_cell )) / 6.

            #checking if adding "dofs" for Dirichlet BC
            nb_dofs = len(dirichlet_dofs & set(num_global_ddl_facet))
            aux = list(np.arange(count, count+nb_dofs))
            count += nb_dofs
            components = sorted(list(dirichlet_dofs & set(num_global_ddl_facet)))
            components = np.array(components) % d_
            
            #number of the dof is total number of cells + num of the facet
            G.add_node(nb_ddl_cells // d_ + num_global_ddl_facet[0] // d_, pos=bary, dof=aux, dirichlet_components=components)
            G.add_edge(aux_bis[0], nb_ddl_cells // d_ + num_global_ddl_facet[0] // d_, num=num_global_ddl_facet[0] // d_, dof_CR=num_global_ddl_facet, measure=area, barycentre=bary, normal=normal, vertices=vert, pen_factor=pen_factor[num_global_ddl_facet[0] // d_], breakable=False)
            G.node[aux_bis[0]]['bnd'] = True #Cell is on the boundary of the domain
                
    return G
