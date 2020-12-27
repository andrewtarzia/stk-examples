import stk
import time
import stko
import mchammer as mch
from os.path import exists
import sys

from atools import (
    build_conformers,
    AromaticCNCFactory,
    calculate_N_COM_N_angle,
    update_from_rdkit_conf,
)


def direction(mol):

    cids, confs = build_conformers(
        mol=mol,
        N=40,
        ETKDG_version='v3'
    )
    print(f'getting optimal conformer...')
    min_angle = 10000
    min_cid = -10
    # Need to define the functional groups.
    new_mol = stk.BuildingBlock.init_from_molecule(
        molecule=mol,
        functional_groups=[AromaticCNCFactory()]
    )
    for cid in cids:

        # Update stk_mol to conformer geometry.
        new_mol = update_from_rdkit_conf(
            stk_mol=new_mol,
            rdk_mol=confs,
            conf_id=cid
        )

        angle = calculate_N_COM_N_angle(new_mol)

        if angle < min_angle:
            min_cid = cid
            min_angle = angle
            mol = update_from_rdkit_conf(
                stk_mol=mol,
                rdk_mol=confs,
                conf_id=min_cid
            )

    return mol


def get_long_bond_ids(mol):
    """
    Find long bonds in stk.ConstructedMolecule.

    """

    long_bond_ids = []
    for bond_infos in mol.get_bond_infos():
        if bond_infos.get_building_block() is None:
            ba1 = bond_infos.get_bond().get_atom1().get_id()
            ba2 = bond_infos.get_bond().get_atom2().get_id()
            if ba1 < ba2:
                long_bond_ids.append((ba1, ba2))
            else:
                long_bond_ids.append((ba2, ba1))

    return tuple(long_bond_ids)


def merge_subunits_by_buildingblockid(mol, subunits):
    """
    Merge subunits in stk.Molecule by building block ids.

    """

    subunit_building_block_ids = {i: set() for i in subunits}
    for su in subunits:
        su_ids = subunits[su]
        for i in su_ids:
            atom_info = next(mol.get_atom_infos(atom_ids=i))
            subunit_building_block_ids[su].add(
                atom_info.get_building_block_id()
            )

    new_subunits = {}
    taken_subunits = set()
    for su in subunits:
        bb_ids = subunit_building_block_ids[su]
        if len(bb_ids) > 1:
            raise ValueError(
                'Subunits not made up of singular BuildingBlock'
            )
        bb_id = list(bb_ids)[0]
        if su in taken_subunits:
            continue

        compound_subunit = subunits[su]
        has_same_bb_id = [
            (su_id, bb_id) for su_id in subunits
            if list(subunit_building_block_ids[su_id])[0] == bb_id
            and su_id != su
        ]

        for su_id, bb_id in has_same_bb_id:
            for i in subunits[su_id]:
                compound_subunit.add(i)
            taken_subunits.add(su_id)
        new_subunits[su] = compound_subunit

    return new_subunits


def opter(mol, target_bond_length, name):
    stk_long_bond_ids = get_long_bond_ids(mol)
    bond_infos = []
    for i, bond in enumerate(mol.get_bonds()):
        ba1 = bond.get_atom1().get_id()
        ba2 = bond.get_atom2().get_id()
        if ba1 < ba2:
            bond_infos.append((i, ba1, ba2))
        else:
            bond_infos.append((i, ba2, ba1))

    mch_mol = mch.Molecule(
        atoms=(
            mch.Atom(
                id=atom.get_id(),
                element_string=atom.__class__.__name__,
            ) for atom in mol.get_atoms()
        ),
        bonds=(
            mch.Bond(id=i, atom1_id=j, atom2_id=k)
            for i, j, k in bond_infos
        ),
        position_matrix=mol.get_position_matrix(),
    )
    optimizer = mch.Optimizer(
        step_size=0.25,
        target_bond_length=target_bond_length,
        num_steps=1000,
    )
    subunits = mch_mol.get_subunits(
        bond_pair_ids=stk_long_bond_ids,
    )
    for su in subunits:
        mol.write(f'temp_{su}.mol', atom_ids=subunits[su])
    # Just get final step.
    mch_result = optimizer.get_result(
        mol=mch_mol,
        bond_pair_ids=stk_long_bond_ids,
        # Can merge subunits to match distinct BuildingBlocks in stk
        # ConstructedMolecule.
        subunits=merge_subunits_by_buildingblockid(mol, subunits),
    )
    mol = mol.with_position_matrix(
        mch_result.get_final_position_matrix()
    )
    return mol


def main():
    start_time = time.time()

    cycle_bb1 = stk.BuildingBlock.init_from_file(
        'cycle_bb1.mol',
        functional_groups=[stk.CarboxylicAcidFactory()],
    )
    cycle_bb2 = stk.BuildingBlock.init_from_file(
        'cycle_bb2.mol',
        functional_groups=[stk.PrimaryAminoFactory(
            deleters=(2, )
        )],
    )
    axle_bb = stk.BuildingBlock.init_from_file(
        'axle_bb.mol',
        functional_groups=[stk.BromoFactory()],
    )
    cap_bb = stk.BuildingBlock.init_from_file(
        'cap_bb.mol',
        functional_groups=[stk.BromoFactory()],
    )
    linker_bb = stk.BuildingBlock.init_from_file(
        'linker_bb.mol',
        functional_groups=[stk.BromoFactory()],
    )
    core_bb = stk.BuildingBlock.init_from_file(
        'core_bb.mol',
        functional_groups=[stk.BromoFactory()],
    )

    if exists('full_ligand.mol'):
        full_ligand = stk.BuildingBlock.init_from_file(
            'full_ligand.mol'
        )
    else:
        axle = stk.ConstructedMolecule(
            topology_graph=stk.polymer.Linear(
                building_blocks=(axle_bb, cap_bb),
                repeating_unit='AB',
                num_repeating_units=1,
            )
        )
        axle.write('axle.mol')

        # Build macrocycle.
        macrocycle = stk.ConstructedMolecule(
            topology_graph=stk.macrocycle.Macrocycle(
                building_blocks=(cycle_bb1, cycle_bb2),
                repeating_unit='AB',
                num_repeating_units=2,
            )
        )
        macrocycle.write('macrocycle.mol')

        # Build rotaxane.
        rotaxane = stk.ConstructedMolecule(
            topology_graph=stk.rotaxane.NRotaxane(
                axle=stk.BuildingBlock.init_from_molecule(axle),
                cycles=(
                    stk.BuildingBlock.init_from_molecule(macrocycle),
                ),
                repeating_unit='A',
                num_repeating_units=1
            )
        )

        rotaxane = stko.UFF(ignore_inter_interactions=False).optimize(
            rotaxane
        )
        rotaxane.write('rotaxane.mol')

        # Build cage ligand.
        ligand = stk.ConstructedMolecule(
            topology_graph=stk.polymer.Linear(
                building_blocks=(core_bb, linker_bb),
                repeating_unit='BAB',
                num_repeating_units=1,
            )
        )

        ligand = direction(ligand)
        ligand.write('ligand.mol')

        # Add rotaxane to ligand.
        full_ligand = stk.ConstructedMolecule(
            topology_graph=stk.polymer.Linear(
                building_blocks=(
                    stk.BuildingBlock.init_from_molecule(
                        ligand,
                        functional_groups=[stk.IodoFactory()],
                    ),
                    stk.BuildingBlock.init_from_molecule(
                        rotaxane,
                        functional_groups=[stk.IodoFactory()],
                    ),
                ),
                repeating_unit='AB',
                num_repeating_units=1,
            )
        )

        full_ligand = opter(
            full_ligand,
            target_bond_length=1.2,
            name='fl'
        )
        full_ligand = stko.UFF(
            ignore_inter_interactions=False
        ).optimize(
            full_ligand
        )
        full_ligand.write('full_ligand.mol')

    # Build cage.
    metal = stk.BuildingBlock(
        smiles='[Pd+2]',
        functional_groups=(
            stk.SingleAtom(stk.Pd(0, charge=2))
            for i in range(4)
        ),
        position_matrix=[[0., 0., 0.]],
    )
    cage = stk.ConstructedMolecule(
        topology_graph=stk.cage.M2L4Lantern(
            building_blocks=(
                metal,
                stk.BuildingBlock.init_from_molecule(
                    full_ligand,
                    functional_groups=[
                        stk.SmartsFunctionalGroupFactory(
                            smarts='[#6]~[#7X2]~[#6]',
                            bonders=(1, ),
                            deleters=(),
                        ),
                    ],
                ),
            ),
            reaction_factory=stk.DativeReactionFactory(
                stk.GenericReactionFactory(
                    bond_orders={
                        frozenset({
                            stk.GenericFunctionalGroup,
                            stk.SingleAtom
                        }): 9
                    }
                )
            )
        )
    )

    cage = opter(cage, target_bond_length=1.2, name='cg')
    cage.write('cage_c.mol')
    print(f'Time taken w/o xtb: {time.time()-start_time}s')

    cage = stko.XTBFF(
        xtb_path='/home/atarzia/software/xtb-6.3.1/bin/xtb',
        output_dir=f'xtbff_cage',
        num_cores=4,
        opt_level='normal',
        charge=4,
        unlimited_memory=True,
    ).optimize(cage)
    cage.write('cage.mol')

    print(f'Time taken: {time.time()-start_time}s')
    sys.exit()


if __name__ == '__main__':
    main()
