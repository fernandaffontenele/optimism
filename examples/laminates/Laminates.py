import numpy as onp  # as is "old numpy"
import jax

from optimism.JaxConfig import *
from optimism import EquationSolver
from optimism import FunctionSpace
from optimism import Interpolants
from optimism.material import J2Plastic
from optimism.material import Neohookean
from optimism import Mechanics
from optimism import Mesh
from optimism import Objective
from optimism import QuadratureRule
from optimism import SparseMatrixAssembler
from optimism import VTKWriter

# set up the mesh
w = 1.0
L = 1.0
numberLayers = 80
nodesInX = 321  # must be odd
nodesInY = 321
xRange = [0.0, w]
yRange = [0.0, L]
mesh = Mesh.construct_structured_mesh(nodesInX, nodesInY, xRange, yRange)


def triangle_centroid(vertices):
    return np.average(vertices, axis=0)

def determine_block_of_triangle(meshCoords, triangleConnectivity):
    vertices = meshCoords[triangleConnectivity, :]
    yCoord_rounded = np.floor(triangle_centroid(vertices)[0] * numberLayers/w)
    return np.where(yCoord_rounded % 2 == 0, 0, 1)

@jax.jit
def determine_blocks(coords, conns):
    return vmap(determine_block_of_triangle, (None, 0))(coords, conns)

blockIds = determine_blocks(mesh.coords, mesh.conns)

blocks = {'soft': np.flatnonzero(np.array(blockIds) == 0),
          'hard': np.flatnonzero(np.array(blockIds) == 1)}

mesh = Mesh.mesh_with_blocks(mesh, blocks)

# Create imperfection in the mesh
csi = 1e-2
new_coords_x = mesh.coords[:, 0] + csi * np.sin(onp.pi * mesh.coords[:, 0] / w) * np.arctan(8 * mesh.coords[:, 1] / L - 0.5)
updatedcoords = mesh.coords.at[:, 0].set(new_coords_x)
mesh = Mesh.mesh_with_coords(mesh, updatedcoords)

nodeTol = 1e-8
nodeSets = {'left': np.flatnonzero((mesh.coords[:, 0] + mesh.coords[:, 1]) < xRange[0] + nodeTol),
            'right': np.flatnonzero(mesh.coords[:, 0] > xRange[1] - nodeTol),
            'bottom': np.flatnonzero(mesh.coords[:, 1] < yRange[0] + nodeTol),
            'top': np.flatnonzero(mesh.coords[:, 1] > yRange[1] - nodeTol)}

mesh = Mesh.mesh_with_nodesets(mesh, nodeSets)

# set the essential boundary conditions and create the a DofManager to
# handle the indexing between unknowns and degrees of freedom.
EBCs = [Mesh.EssentialBC(nodeSet='left', field=0),
        Mesh.EssentialBC(nodeSet='bottom', field=1),
        Mesh.EssentialBC(nodeSet='top', field=1)]

fieldShape = mesh.coords.shape
dofManager = Mesh.DofManager(mesh, fieldShape, EBCs)

# write blocks and bcs to paraview output to check things are correct
writer = VTKWriter.VTKWriter(mesh, baseFileName='check_problem_setup')

writer.add_cell_field(name='block_id', cellData=blockIds,
                      fieldType=VTKWriter.VTKFieldType.SCALARS,
                      dataType=VTKWriter.VTKDataType.INT)

bcs = np.array(dofManager.isBc, dtype=np.int64)
writer.add_nodal_field(name='bcs', nodalData=bcs, fieldType=VTKWriter.VTKFieldType.VECTORS,
                       dataType=VTKWriter.VTKDataType.INT)

writer.write()

# create the function space
order = 2 * mesh.masterElement.degree
quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2 * (order - 1))
fs = FunctionSpace.construct_function_space(mesh, quadRule)

# create the material models
ESoft = 10.0
nuSoft = 0.48
props = {'elastic modulus': ESoft,
         'poisson ratio': nuSoft}
softMaterialModel = Neohookean.create_material_model_functions(props)

E = 10.0
nu = 0.48
Y0 = 0.01 * E
H = E / 100
props = {'elastic modulus': E,
         'poisson ratio': nu,
         'yield strength': Y0,
         'hardening model': 'linear',
         'hardening modulus': H}
hardMaterialModel = J2Plastic.create_material_model_functions(props)

materialModels = {'soft': softMaterialModel, 'hard': hardMaterialModel}

# mechanics functions
mechanicsFunctions = Mechanics.create_multi_block_mechanics_functions(fs, mode2D="plane strain",
                                                                      materialModels=materialModels)


# helper function to fill in nodal values of essential boundary conditions
def get_ubcs(p):
    appliedDisp = p[0]
    EbcIndex = (mesh.nodeSets['top'], 1)
    V = np.zeros(fieldShape).at[EbcIndex].set(appliedDisp)
    return dofManager.get_bc_values(V)


# helper function to go from unknowns to full DoF array
def create_field(Uu, p):
    return dofManager.create_field(Uu, get_ubcs(p))


# write the energy to minimize
def energy_function(Uu, p):
    U = create_field(Uu, p)
    internalVariables = p.state_data
    return mechanicsFunctions.compute_strain_energy(U, internalVariables)


def compute_energy_from_bcs(Uu, Ubc, internalVariables):
    U = dofManager.create_field(Uu, Ubc)
    return mechanicsFunctions.compute_strain_energy(U, internalVariables)


compute_bc_reactions = jax.jit(jax.grad(compute_energy_from_bcs, 1))

# Tell objective how to assemble preconditioner matrix
def assemble_sparse_preconditioner_matrix(Uu, p):
    U = create_field(Uu, p)
    internalVariables = p.state_data
    elementStiffnesses = mechanicsFunctions.compute_element_stiffnesses(U, internalVariables)
    return SparseMatrixAssembler.assemble_sparse_stiffness_matrix(
        elementStiffnesses, mesh.conns, dofManager)


def green_lagrange_strain(h):
    return 1/2 * (h + h.T + h.T @ h)


# solver settings
solverSettings = EquationSolver.get_settings(max_cumulative_cg_iters=100,
                                             max_trust_iters=1000,
                                             use_preconditioned_inner_product_for_cg=True)

precondStrategy = Objective.PrecondStrategy(assemble_sparse_preconditioner_matrix)

# initialize unknown displacements to zero
Uu = dofManager.get_unknown_values(np.zeros(fieldShape))

# set initial values of parameters
appliedDisp = 0.0
state = mechanicsFunctions.compute_initial_state()
p = Objective.Params(appliedDisp, state)

# Construct an objective object for the equation solver to work on
objective = Objective.ScaledObjective(energy_function, Uu, p, precondStrategy=precondStrategy)

outputForce = []
outputDisp = []

steps = 40
max_disp = 0.3
inc_disp = max_disp/steps
for i in range(1, steps):
    print('--------------------------------------')
    print('LOAD STEP ', i)

    # increment the applied displacement
    appliedDisp -= L * inc_disp
    p = Objective.param_index_update(p, 0, appliedDisp)

    # Find unknown displacements by minimizing the objective
    Uu = EquationSolver.nonlinear_equation_solve(objective, Uu, p, solverSettings)

    # update the state variables in the new equilibrium configuration
    U = create_field(Uu, p)
    state = mechanicsFunctions.compute_updated_internal_variables(U, p.state_data)
    p = Objective.param_index_update(p, 1, state)

    # write solution data to VTK file for post-processing
    writer = VTKWriter.VTKWriter(mesh, baseFileName='uniaxial_two_material' + str(i).zfill(3))

    U = create_field(Uu, p)
    writer.add_nodal_field(name='displacement', nodalData=U, fieldType=VTKWriter.VTKFieldType.VECTORS)

    Ubc = get_ubcs(p)
    rxnBc = compute_bc_reactions(Uu, Ubc, state)
    reactions = np.zeros(U.shape).at[dofManager.isBc].set(rxnBc)
    writer.add_nodal_field(name='reactions', nodalData=reactions, fieldType=VTKWriter.VTKFieldType.VECTORS)

    _, stresses = mechanicsFunctions.compute_output_energy_densities_and_stresses(
        U, state)
    cellStresses = FunctionSpace.project_quadrature_field_to_element_field(fs, stresses)
    writer.add_cell_field(name='stress', cellData=cellStresses,
                          fieldType=VTKWriter.VTKFieldType.TENSORS)

    dispGrads = FunctionSpace.compute_field_gradient(fs, U)
    compute_strains = jax.vmap(jax.vmap(green_lagrange_strain, 0), 0)
    strains = compute_strains(dispGrads)
    cellStrains = FunctionSpace.project_quadrature_field_to_element_field(fs, strains)
    writer.add_cell_field(name='strains', cellData=cellStrains,
                          fieldType=VTKWriter.VTKFieldType.TENSORS)

    eqpsField = state[:, :, J2Plastic.EQPS]
    cellEqpsField = FunctionSpace.project_quadrature_field_to_element_field(fs, eqpsField)
    writer.add_cell_field(name='eqps', cellData=cellEqpsField, fieldType=VTKWriter.VTKFieldType.SCALARS)

    writer.write()

    outputForce.append(float(-np.sum(reactions[mesh.nodeSets['top'], 1])))
    outputDisp.append(float(-p[0]))

    with open('arch_bc_Fd.npz', 'wb') as f:
        np.savez(f, force=np.array(outputForce),
                 displacement=np.array(outputDisp))
