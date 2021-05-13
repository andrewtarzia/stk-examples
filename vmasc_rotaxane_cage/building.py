import numpy as np
from rdkit.Chem import AllChem as rdkit
from mendeleev import element
import stk
import time
import stko
from os.path import exists
import sys


def update_from_rdkit_conf(stk_mol, rdk_mol, conf_id):
    """
    Update the structure to match `conf_id` of `mol`.

    Parameters
    ----------
    struct : :class:`stk.Molecule`
        The molecule whoce coordinates are to be updated.

    mol : :class:`rdkit.Mol`
        The :mod:`rdkit` molecule to use for the structure update.

    conf_id : :class:`int`
        The conformer ID of the `mol` to update from.

    Returns
    -------
    :class:`.Molecule`
        The molecule.

    """

    pos_mat = rdk_mol.GetConformer(id=conf_id).GetPositions()
    return stk_mol.with_position_matrix(pos_mat)


def get_center_of_mass(molecule, atom_ids=None):
    """
    Return the centre of mass.

    Parameters
    ----------
    molecule : :class:`stk.Molecule`

    atom_ids : :class:`iterable` of :class:`int`, optional
        The ids of atoms which should be used to calculate the
        center of mass. If ``None``, then all atoms will be used.

    Returns
    -------
    :class:`numpy.ndarray`
        The coordinates of the center of mass.

    References
    ----------
    https://en.wikipedia.org/wiki/Center_of_mass

    """

    if atom_ids is None:
        atom_ids = range(molecule.get_num_atoms())
    elif not isinstance(atom_ids, (list, tuple)):
        # Iterable gets used twice, once in get_atom_positions
        # and once in zip.
        atom_ids = list(atom_ids)

    center = 0
    total_mass = 0.
    coords = molecule.get_atomic_positions(atom_ids)
    atoms = molecule.get_atoms(atom_ids)
    for atom, coord in zip(atoms, coords):
        mass = element(atom.__class__.__name__).atomic_weight
        total_mass += mass
        center += mass*coord
    return np.divide(center, total_mass)


def unit_vector(vector):
    """
    Returns the unit vector of the vector.

    https://stackoverflow.com/questions/2827393/
    angles-between-two-n-dimensional-vectors-in-python/
    13849249#13849249
    """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2, normal=None):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'::

        >>> angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966
        >>> angle_between((1, 0, 0), (1, 0, 0))
        0.0
        >>> angle_between((1, 0, 0), (-1, 0, 0))
        3.141592653589793

    https://stackoverflow.com/questions/2827393/
    angles-between-two-n-dimensional-vectors-in-python/
    13849249#13849249

    If normal is given, the angle polarity is determined using the
    cross product of the two vectors.

    """

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    if normal is not None:
        # Get normal vector and cross product to determine sign.
        cross = np.cross(v1_u, v2_u)
        if np.dot(normal, cross) < 0:
            angle = -angle
    return angle


def calculate_N_COM_N_angle(bb):
    """
    Calculate the N-COM-N angle of a ditopic building block.

    This function will not work for cages built from FGs other than
    metals + AromaticCNC and metals + AromaticCNN.

    Parameters
    ----------
    bb : :class:`stk.BuildingBlock`
        stk molecule to analyse.

    Returns
    -------
    angle : :class:`float`
        Angle between two bonding vectors of molecule.

    """

    fg_counts = 0
    fg_positions = []
    for fg in bb.get_functional_groups():
        if isinstance(fg, AromaticCNC):
            fg_counts += 1
            # Get geometrical properties of the FG.
            # Get N position - deleter.
            N_position, = bb.get_atomic_positions(
                atom_ids=fg.get_nitrogen().get_id()
            )
            fg_positions.append(N_position)

    if fg_counts != 2:
        raise ValueError(
            f'{bb} does not have 2 AromaticCNC or AromaticCNN '
            'functional groups.'
        )

    # Get building block COM.
    COM_position = get_center_of_mass(bb)

    # Get vectors.
    fg_vectors = [i-COM_position for i in fg_positions]

    # Calculate the angle between the two vectors.
    angle = np.degrees(angle_between(*fg_vectors))
    return angle


def build_conformers(mol, N, ETKDG_version=None):
    """
    Convert stk mol into RDKit mol with N conformers.

    ETKDG_version allows the user to pick their choice of ETKDG params.

    `None` provides the settings used in ligand_combiner and unsymm.

    Other options:
        `v3`:
            New version from DOI: 10.1021/acs.jcim.0c00025
            with improved handling of macrocycles.

    """
    molecule = mol.to_rdkit_mol()
    molecule.RemoveAllConformers()

    if ETKDG_version is None:
        cids = rdkit.EmbedMultipleConfs(
            mol=molecule,
            numConfs=N,
            randomSeed=1000,
            useExpTorsionAnglePrefs=True,
            useBasicKnowledge=True,
            numThreads=4,
        )

    elif ETKDG_version == 'v3':
        params = rdkit.ETKDGv3()
        params.randomSeed = 1000
        cids = rdkit.EmbedMultipleConfs(
            mol=molecule,
            numConfs=N,
            params=params
        )

    print(f'there are {molecule.GetNumConformers()} conformers')
    return cids, molecule


class AromaticCNCFactory(stk.FunctionalGroupFactory):
    """
    A subclass of stk.SmartsFunctionalGroupFactory.

    """

    def __init__(self, bonders=(1, ), deleters=()):
        """
        Initialise :class:`.AromaticCNCFactory`.

        """

        self._bonders = bonders
        self._deleters = deleters

    def get_functional_groups(self, molecule):
        generic_functional_groups = stk.SmartsFunctionalGroupFactory(
            smarts='[#6]~[#7X2]~[#6]',
            bonders=self._bonders,
            deleters=self._deleters
        ).get_functional_groups(molecule)
        for fg in generic_functional_groups:
            atom_ids = (i.get_id() for i in fg.get_atoms())
            atoms = tuple(molecule.get_atoms(atom_ids))
            yield AromaticCNC(
                carbon1=atoms[0],
                nitrogen=atoms[1],
                carbon2=atoms[2],
                bonders=tuple(atoms[i] for i in self._bonders),
                deleters=tuple(atoms[i] for i in self._deleters),
            )


class AromaticCNC(stk.GenericFunctionalGroup):
    """
    Represents an N atom in pyridine functional group.

    The structure of the functional group is given by the pseudo-SMILES
    ``[carbon][nitrogen][carbon]``.

    """

    def __init__(self, carbon1, nitrogen, carbon2, bonders, deleters):
        """
        Initialize a :class:`.Alcohol` instance.

        Parameters
        ----------
        carbon1 : :class:`.C`
            The first carbon atom.

        nitrogen : :class:`.N`
            The nitrogen atom.

        carbon2 : :class:`.C`
            The second carbon atom.

        bonders : :class:`tuple` of :class:`.Atom`
            The bonder atoms.

        deleters : :class:`tuple` of :class:`.Atom`
            The deleter atoms.

        """

        self._carbon1 = carbon1
        self._nitrogen = nitrogen
        self._carbon2 = carbon2
        atoms = (carbon1, nitrogen, carbon2)
        super().__init__(atoms, bonders, deleters)

    def get_carbon1(self):
        """
        Get the first carbon atom.

        Returns
        -------
        :class:`.C`
            The first carbon atom.

        """

        return self._carbon1

    def get_carbon2(self):
        """
        Get the second carbon atom.

        Returns
        -------
        :class:`.C`
            The second carbon atom.

        """

        return self._carbon2

    def get_nitrogen(self):
        """
        Get the nitrogen atom.

        Returns
        -------
        :class:`.N`
            The nitrogen atom.

        """

        return self._nitrogen

    def clone(self):
        clone = super().clone()
        clone._carbon1 = self._carbon1
        clone._nitrogen = self._nitrogen
        clone._carbon2 = self._carbon2
        return clone

    def with_atoms(self, atom_map):
        clone = super().with_atoms(atom_map)
        clone._carbon1 = atom_map.get(
            self._carbon1.get_id(),
            self._carbon1,
        )
        clone._nitrogen = atom_map.get(
            self._nitrogen.get_id(),
            self._nitrogen,
        )
        clone._carbon2 = atom_map.get(
            self._carbon2.get_id(),
            self._carbon2,
        )
        return clone

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self._carbon1}, {self._nitrogen}, {self._carbon2}, '
            f'bonders={self._bonders})'
        )


def direction(mol):

    cids, confs = build_conformers(
        mol=mol,
        N=40,
        ETKDG_version='v3'
    )
    print('getting optimal conformer...')
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
        rotaxane.write('rotaxane_uo.mol')
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
        ligand.write('ligand_uo.mol')
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
                optimizer=stk.MCHammer(),
            )
        )
        full_ligand.write('full_ligand_uo.mol')
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
            ),
            optimizer=stk.MCHammer(),
        )
    )
    cage.write('cage_c.mol')
    print(f'Time taken: {time.time()-start_time}s')
    sys.exit()


if __name__ == '__main__':
    main()
