import numpy as np
import stk


class AxialPaddlewheel(stk.metal_complex.MetalComplex):
    """
    Represents a metal complex topology graph.

    Two distinct metal building blocks, one with five functional
    groups, one with four.

    Binding ligand building blocks with two functional groups are
    required for this topology graph and axial ligand building
    blocks with one functional group are required.

    When using a :class:`dict` for initialization, a
    :class:`.BuildingBlock` needs to be assigned to each of the
    following numbers:
        | axial_metal: (0, )
        | other_metal: (1, )
        | ligands: (0, 1, 2, 3)
        | axial_ligand: (4, )

    See :class:`.MetalComplex` for more details and examples.

    """

    _metal_vertex_prototypes = (
        stk.metal_complex.vertices.MetalVertex(0, [0, 1, 0]),
        stk.metal_complex.vertices.MetalVertex(1, [0, -1, 0]),
    )
    _ligand_vertex_prototypes = (
        stk.metal_complex.vertices.BiDentateLigandVertex(2, [2, 0, 0]),
        stk.metal_complex.vertices.BiDentateLigandVertex(3, [0, 0, 2]),
        stk.metal_complex.vertices.BiDentateLigandVertex(4, [-2, 0, 0]),
        stk.metal_complex.vertices.BiDentateLigandVertex(5, [0, 0, -2]),
        stk.metal_complex.vertices.MonoDentateLigandVertex(6, [0, 2, 0]),
    )

    _edge_prototypes = (
        stk.Edge(
            id=0,
            vertex1=_metal_vertex_prototypes[0],
            vertex2=_ligand_vertex_prototypes[0],
            position=[0.1, 0.5, 0],
        ),
        stk.Edge(
            id=1,
            vertex1=_metal_vertex_prototypes[1],
            vertex2=_ligand_vertex_prototypes[0],
            position=[0.1, -0.5, 0],
        ),

        stk.Edge(
            id=2,
            vertex1=_metal_vertex_prototypes[0],
            vertex2=_ligand_vertex_prototypes[1],
            position=[0, 0.5, 0.1],
        ),
        stk.Edge(
            id=3,
            vertex1=_metal_vertex_prototypes[1],
            vertex2=_ligand_vertex_prototypes[1],
            position=[0, -0.5, 0.1],
        ),

        stk.Edge(
            id=4,
            vertex1=_metal_vertex_prototypes[0],
            vertex2=_ligand_vertex_prototypes[2],
            position=[-0.1, 0.5, 0],
        ),
        stk.Edge(
            id=5,
            vertex1=_metal_vertex_prototypes[1],
            vertex2=_ligand_vertex_prototypes[2],
            position=[-0.1, -0.5, 0],
        ),

        stk.Edge(
            id=6,
            vertex1=_metal_vertex_prototypes[0],
            vertex2=_ligand_vertex_prototypes[3],
            position=[0, 0.5, -0.1],
        ),
        stk.Edge(
            id=7,
            vertex1=_metal_vertex_prototypes[1],
            vertex2=_ligand_vertex_prototypes[3],
            position=[0, -0.5, -0.1],
        ),
        stk.Edge(
            id=8,
            vertex1=_metal_vertex_prototypes[0],
            vertex2=_ligand_vertex_prototypes[4],
            position=[0, 1.5, 0],
        ),
    )


def main():

    fgs = [
        stk.SmartsFunctionalGroupFactory(
            smarts='[#6]~[#8]~[#1]',
            bonders=(1, ),
            deleters=(2, ),
        ),
        stk.SmartsFunctionalGroupFactory(
            smarts='[#6]~[#8X1]',
            bonders=(1, ),
            deleters=(),
        ),
    ]
    ligand_bb = stk.BuildingBlock.init_from_file('org_carb.mol', fgs)
    axial_bb = stk.BuildingBlock(
        smiles='C(N)O',
        functional_groups=(stk.AlcoholFactory(), ),
    )
    axial_bb.write('axial_bb.mol')

    copper_atom = stk.BuildingBlock(
        smiles='[Cu+2]',
        functional_groups=(
            stk.SingleAtom(stk.Cu(0, charge=2))
            for i in range(5)
        ),
        position_matrix=np.array([[0, 0, 0]]),
    )

    # Use a custom Paddlewheel topology that has axial positions.
    copper_pw = stk.ConstructedMolecule(
        AxialPaddlewheel(
            metals={copper_atom: (0, 1)},
            ligands={
                ligand_bb: (0, 1, 2, 3),
                axial_bb: (4, ),
            },
            optimizer=stk.MCHammer(num_steps=150),
        )
    )
    copper_pw.write('axial_copper_pw.mol')

    copper_pw = stk.BuildingBlock.init_from_molecule(
        copper_pw, (stk.BromoFactory(), )
    )
    c_ligand_bb = stk.BuildingBlock.init_from_file(
        'metal_org.mol', (stk.BromoFactory(), )
    )
    cage = stk.ConstructedMolecule(
        topology_graph=stk.cage.M2L4Lantern(
            building_blocks=(
                copper_pw,
                c_ligand_bb,
            ),
            optimizer=stk.MCHammer(num_steps=2000),
        )
    )
    cage.write('axial_cage.mol')

    cage_bb = stk.BuildingBlock.init_from_molecule(
        molecule=cage,
        functional_groups=(stk.PrimaryAminoFactory(), ),
    )
    linker_bb = stk.BuildingBlock(
        smiles='C1(C(C(C1N)N)N)N',
        functional_groups=(stk.PrimaryAminoFactory(), ),
    )
    print(cage_bb, linker_bb)
    cof = stk.ConstructedMolecule(
        topology_graph=stk.cof.Square(
            building_blocks=(cage_bb, linker_bb),
            lattice_size=(2, 2, 1),
            # Setting scale_steps to False tends to lead to a
            # better structure.
            optimizer=stk.Collapser(scale_steps=False),
        ),
    )
    cof.write('axial_cof.mol')


if __name__ == '__main__':
    main()
