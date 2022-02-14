import os
import numpy as np
from scipy.spatial.distance import cdist
from itertools import product

import stk
import spindry as spd


class AlignmentPotential(spd.Potential):
    """
    Scale the size of the guest radii.
    """

    def __init__(self):
        super().__init__()

    def _potential(self, distance, sigmas):

        return (sigmas * distance) ** 2 - 0.1

    def _combine_atoms(self, atoms1, atoms2):

        len1 = len(atoms1)
        len2 = len(atoms2)
        _eps = 0.1

        mixed = np.zeros((len1, len2))
        for i in range(len1):
            for j in range(len2):
                a1e = atoms1[i].get_element_string()
                a2e = atoms2[j].get_element_string()
                # Ns like to be near Ns.
                if a1e == 'N' and a2e == 'N':
                    mixed[i, j] = _eps * 2
                # Ms like to be near Ms.
                elif a1e == 'Zn' and a2e == 'Zn':
                    mixed[i, j] = _eps
                else:
                    mixed[i, j] = 0

        return mixed

    def compute_potential(self, supramolecule):
        component_position_matrices = list(
            i.get_position_matrix()
            for i in supramolecule.get_components()
        )
        component_atoms = list(
            tuple(j for j in i.get_atoms())
            for i in supramolecule.get_components()
        )
        pair_dists = cdist(
            component_position_matrices[0],
            component_position_matrices[1],
        )
        sigmas = self._combine_atoms(
            component_atoms[0], component_atoms[1]
        )
        # Intro a cutoff to ensure no overlap (look at N-N distance).
        _cut = 4
        cut_dists = pair_dists.flatten()[pair_dists.flatten() < _cut]
        cut_sigs = sigmas.flatten()[pair_dists.flatten() < _cut]
        return np.sum(
            self._potential(
                distance=cut_dists,
                sigmas=cut_sigs,
            )
        )


def align_cages(name, xtal, comp):

    host_molecule = spd.Molecule(
        atoms=(
            spd.Atom(
                id=atom.get_id(),
                element_string=atom.__class__.__name__,
            ) for atom in xtal.get_atoms()
        ),
        bonds=(
            spd.Bond(
                id=i,
                atom_ids=(
                    bond.get_atom1().get_id(),
                    bond.get_atom2().get_id(),
                )
            ) for i, bond in enumerate(xtal.get_bonds())
        ),
        position_matrix=xtal.get_position_matrix(),
    )
    guest_molecule = spd.Molecule(
        atoms=(
            spd.Atom(
                id=atom.get_id()+host_molecule.get_num_atoms(),
                element_string=atom.__class__.__name__,
            ) for atom in comp.get_atoms()
        ),
        bonds=(
            spd.Bond(
                id=i,
                atom_ids=(
                    (
                        bond.get_atom1().get_id()
                        +host_molecule.get_num_atoms()
                    ),
                    (
                        bond.get_atom2().get_id()
                        +host_molecule.get_num_atoms()
                    ),
                )
            ) for i, bond in enumerate(comp.get_bonds())
        ),
        position_matrix=comp.get_position_matrix(),
    )

    supramolecule = spd.SupraMolecule.init_from_components(
        components=(host_molecule, guest_molecule),
    )

    cg = spd.Spinner(
        step_size=0.2,
        rotation_step_size=0.2,
        num_conformers=70,
        max_attempts=500,
        potential_function=AlignmentPotential(),
    )
    for conformer in cg.get_conformers(supramolecule):
        pass
    print(f'{name} final potential: {conformer.get_potential()}')
    conformer.write_xyz_file(
        f'output_dir/conf_{name}_final.xyz'
    )
    comps = list(conformer.get_components())
    xtal = xtal.with_position_matrix(comps[0].get_position_matrix())
    comp = comp.with_position_matrix(comps[1].get_position_matrix())

    return xtal, comp


def main():

    mol_pairs = [
        ('cc3.mol', 'cc3_2.mol'),
        ('4D2_C_optc.mol', '4d2_2.mol'),
        ('5B4_C_optc.mol', '5b4_2.mol',)
    ]

    for pair in mol_pairs:
        print(pair)
        xtal = pair[0].replace('.mol', '')
        comp = pair[1].replace('.mol', '')

        rots = (
            None, (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 0, 1), (1, 1, 0), (0, 1, 1)
        )
        angles = (
            np.radians(60), np.radians(90), np.radians(120),
            np.radians(180), np.radians(240), np.radians(270)
        )

        for r, rot in enumerate(product(rots, angles)):
            if os.path.exists(f'{xtal}_a_{r}_xtal.mol'):
                continue
            else:
                # Read in xtal structure.
                xtal_struct = stk.BuildingBlock.init_from_file(
                    f'{xtal}.mol'
                )
                xtal_struct = xtal_struct.with_centroid((0, 0, 0))

                # Read in comp structure.
                comp_struct = stk.BuildingBlock.init_from_file(
                    f'{comp}.mol'
                )
                comp_struct = comp_struct.with_centroid((0, 0, 0))
                if rot[0] is not None:
                    comp_struct = comp_struct.with_rotation_about_axis(
                        angle=rot[1],
                        axis=np.array(rot[0]),
                        origin=np.array((0, 0, 0)),
                    )
                elif r != 0:
                    continue

                xtal_struct.write(f'output_dir/{xtal}_u_{r}_xtal.mol')
                comp_struct.write(f'output_dir/{xtal}_u_{r}_comp.mol')
                print(f'---- doing: {xtal} - {r}: rotation {rot}')
                xtal_struct, comp_struct = align_cages(
                    name=f'{xtal}_{r}',
                    xtal=xtal_struct,
                    comp=comp_struct,
                )
                xtal_struct.write(f'output_dir/{xtal}_a_{r}_xtal.mol')
                comp_struct.write(f'output_dir/{xtal}_a_{r}_comp.mol')


if __name__ == "__main__":
    main()
