import stk
import networkx as nx
import sys
import os


def get_disconnected_components(molecule):

    # Produce a graph from the molecule that does not include edges
    # where the bonds to be optimized are.
    mol_graph = nx.Graph()
    for atom in molecule.get_atoms():
        mol_graph.add_node(atom.get_id())

    # Add edges.
    for bond in molecule.get_bonds():
        pair_ids = (
            bond.get_atom1().get_id(), bond.get_atom2().get_id()
        )
        mol_graph.add_edge(*pair_ids)

    # Get atom ids in disconnected subgraphs.
    components = {}
    for c in nx.connected_components(mol_graph):
        c_ids = sorted(c)
        molecule.write('temp_mol.mol', atom_ids=c_ids)
        num_atoms = len(c_ids)
        newbb = stk.BuildingBlock.init_from_file('temp_mol.mol')
        os.system('rm temp_mol.mol')

        components[num_atoms] = newbb

    return components


def extract_host(molecule):
    components = get_disconnected_components(molecule)
    return components[max(components.keys())]


def extract_guest(molecule):
    components = get_disconnected_components(molecule)
    return components[min(components.keys())]


def main():
    if (not len(sys.argv) == 3):
        print(
            f'Usage: {__file__}\n'
            '   Expected 2 arguments: host_with_g_file new_guest_file'
        )
        sys.exit()
    else:
        host_with_g_file = sys.argv[1]
        new_guest_file = sys.argv[2]

    # Load in host.
    host_with_guest = stk.BuildingBlock.init_from_file(host_with_g_file)
    
    # Load in new guest.
    new_guest = stk.BuildingBlock.init_from_file(new_guest_file)
    
    # Split host and guest, assuming host has more atoms than guest.
    host = extract_host(host_with_guest)
    old_guest = extract_guest(host_with_guest)
    
    # Build new host-guest structure, with Spindry optimiser to 
    # do some conformer searching.
    new_host = stk.ConstructedMolecule(
        stk.host_guest.Complex(
            host=stk.BuildingBlock.init_from_molecule(host),
            guests=(stk.host_guest.Guest(new_guest), ),
            # There are options for the Spinner class, 
            # if the optimised conformer is crap.
            optimizer=stk.Spinner(),
        ),
    )
    
    # Write out new host guest.
    new_host.write('new_host_guest.mol')


if __name__ == '__main__':
    main()
