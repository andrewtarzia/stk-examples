import numpy as np
import stk


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
    print(ligand_bb)

    copper_atom = stk.BuildingBlock(
        smiles='[Cu+2]',
        functional_groups=(
            stk.SingleAtom(stk.Cu(0, charge=2))
            for i in range(4)
        ),
        position_matrix=np.array([[0, 0, 0]]),
    )

    copper_pw = stk.ConstructedMolecule(
        stk.metal_complex.Paddlewheel(
            metals={copper_atom: (0, 1)},
            ligands={ligand_bb: (0, 1, 2, 3)},
            optimizer=stk.MCHammer(num_steps=150),
        )
    )
    copper_pw.write('ligand_copper_pw.mol')

    copper_pw = stk.BuildingBlock.init_from_molecule(
        copper_pw, (stk.BromoFactory(), )
    )
    c_ligand_bb = stk.BuildingBlock.init_from_file(
        'ligand_org.mol', (stk.BromoFactory(), )
    )
    print(copper_pw, ligand_bb)
    cage = stk.ConstructedMolecule(
        topology_graph=stk.cage.M2L4Lantern(
            building_blocks=(
                copper_pw,
                c_ligand_bb,
            ),
            optimizer=stk.MCHammer(num_steps=2000),
        )
    )
    cage.write('ligand_cage.mol')

    cage_bb = stk.BuildingBlock.init_from_molecule(
        cage, (stk.PrimaryAminoFactory(), )
    )
    linker_bb = stk.BuildingBlock.init_from_file(
        'alde.mol', (stk.AldehydeFactory(), )
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
    cof.write('ligand_cof.mol')


if __name__ == '__main__':
    main()
