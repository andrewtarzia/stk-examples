import stk
import numpy as np


def unsymmetrical_ligand():
    core_c_bb = stk.BuildingBlock(
        smiles="[Br][C][I]",
        position_matrix=np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]]),
        functional_groups=(
            stk.SmartsFunctionalGroupFactory(
                smarts="[Br][C]",
                bonders=(0,),
                deleters=(),
                placers=(0, 1),
            ),
            stk.SmartsFunctionalGroupFactory(
                smarts="[I][C]",
                bonders=(0,),
                deleters=(),
                placers=(0, 1),
            ),
        ),
    )
    return core_c_bb


class M4L82(stk.cage.Cage):
    _non_linears = (
        stk.cage.NonLinearVertex(0, [0, 0, np.sqrt(6) / 2]),
        stk.cage.NonLinearVertex(1, [-1, -np.sqrt(3) / 3, -np.sqrt(6) / 6]),
        stk.cage.NonLinearVertex(2, [1, -np.sqrt(3) / 3, -np.sqrt(6) / 6]),
        stk.cage.NonLinearVertex(3, [0, 2 * np.sqrt(3) / 3, -np.sqrt(6) / 6]),
    )

    paired_wall_1_coord = (
        sum(
            vertex.get_position()
            for vertex in (_non_linears[0], _non_linears[1])
        )
        / 2
    )
    wall_1_shift = np.array((0.1, 0.1, -0.2))

    paired_wall_2_coord = (
        sum(
            vertex.get_position()
            for vertex in (_non_linears[2], _non_linears[3])
        )
        / 2
    )
    wall_2_shift = np.array((0.1, 0.1, 0.2))

    _vertex_prototypes = (
        *_non_linears,
        stk.cage.LinearVertex(
            id=4,
            position=paired_wall_1_coord + wall_1_shift,
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex.init_at_center(
            id=5,
            vertices=(_non_linears[0], _non_linears[2]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=6,
            vertices=(_non_linears[0], _non_linears[3]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=7,
            vertices=(_non_linears[1], _non_linears[2]),
        ),
        stk.cage.LinearVertex.init_at_center(
            id=8,
            vertices=(_non_linears[1], _non_linears[3]),
        ),
        stk.cage.LinearVertex(
            id=9,
            position=paired_wall_2_coord + wall_2_shift,
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            id=10,
            position=paired_wall_1_coord - wall_1_shift,
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            id=11,
            position=paired_wall_2_coord - wall_2_shift,
            use_neighbor_placement=False,
        ),
    )

    _edge_prototypes = (
        stk.Edge(0, _vertex_prototypes[0], _vertex_prototypes[4]),
        stk.Edge(1, _vertex_prototypes[0], _vertex_prototypes[5]),
        stk.Edge(2, _vertex_prototypes[0], _vertex_prototypes[6]),
        stk.Edge(3, _vertex_prototypes[0], _vertex_prototypes[10]),
        stk.Edge(4, _vertex_prototypes[1], _vertex_prototypes[4]),
        stk.Edge(5, _vertex_prototypes[1], _vertex_prototypes[7]),
        stk.Edge(6, _vertex_prototypes[1], _vertex_prototypes[8]),
        stk.Edge(7, _vertex_prototypes[1], _vertex_prototypes[10]),
        stk.Edge(8, _vertex_prototypes[2], _vertex_prototypes[5]),
        stk.Edge(9, _vertex_prototypes[2], _vertex_prototypes[7]),
        stk.Edge(10, _vertex_prototypes[2], _vertex_prototypes[9]),
        stk.Edge(11, _vertex_prototypes[2], _vertex_prototypes[11]),
        stk.Edge(12, _vertex_prototypes[3], _vertex_prototypes[6]),
        stk.Edge(13, _vertex_prototypes[3], _vertex_prototypes[8]),
        stk.Edge(14, _vertex_prototypes[3], _vertex_prototypes[9]),
        stk.Edge(15, _vertex_prototypes[3], _vertex_prototypes[11]),
    )


class M3L6New(stk.cage.Cage):

    _R, _theta = 1, 0

    _vertex_prototypes = (
        stk.cage.NonLinearVertex(
            id=0,
            position=[_R * np.cos(_theta), _R * np.sin(_theta), 0],
        ),
        stk.cage.NonLinearVertex(
            id=1,
            position=[
                _R * np.cos(_theta + (4 * np.pi / 3)),
                _R * np.sin(_theta + (4 * np.pi / 3)),
                0,
            ],
        ),
        stk.cage.NonLinearVertex(
            id=2,
            position=[
                _R * np.cos(_theta + (2 * np.pi / 3)),
                _R * np.sin(_theta + (2 * np.pi / 3)),
                0,
            ],
        ),
        stk.cage.LinearVertex(
            id=3,
            position=[
                _R * np.cos((_theta + 1 * np.pi / 3)),
                _R * np.sin((_theta + 1 * np.pi / 3)),
                0.5,
            ],
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            id=4,
            position=[
                _R * np.cos((_theta + 1 * np.pi / 3)),
                _R * np.sin((_theta + 1 * np.pi / 3)),
                -0.5,
            ],
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            id=5,
            position=[
                _R * np.cos((_theta + 1 * np.pi / 3) + (4 * np.pi / 3)),
                _R * np.sin((_theta + 1 * np.pi / 3) + (4 * np.pi / 3)),
                0.5,
            ],
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            id=6,
            position=[
                _R * np.cos((_theta + 1 * np.pi / 3) + (4 * np.pi / 3)),
                _R * np.sin((_theta + 1 * np.pi / 3) + (4 * np.pi / 3)),
                -0.5,
            ],
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            id=7,
            position=[
                _R * np.cos((_theta + 1 * np.pi / 3) + (2 * np.pi / 3)),
                _R * np.sin((_theta + 1 * np.pi / 3) + (2 * np.pi / 3)),
                0.5,
            ],
            use_neighbor_placement=False,
        ),
        stk.cage.LinearVertex(
            id=8,
            position=[
                _R * np.cos((_theta + 1 * np.pi / 3) + (2 * np.pi / 3)),
                _R * np.sin((_theta + 1 * np.pi / 3) + (2 * np.pi / 3)),
                -0.5,
            ],
            use_neighbor_placement=False,
        ),
    )

    _edge_prototypes = (
        stk.Edge(0, _vertex_prototypes[0], _vertex_prototypes[3]),
        stk.Edge(1, _vertex_prototypes[0], _vertex_prototypes[4]),
        stk.Edge(2, _vertex_prototypes[0], _vertex_prototypes[5]),
        stk.Edge(3, _vertex_prototypes[0], _vertex_prototypes[6]),
        stk.Edge(4, _vertex_prototypes[1], _vertex_prototypes[5]),
        stk.Edge(5, _vertex_prototypes[1], _vertex_prototypes[6]),
        stk.Edge(6, _vertex_prototypes[1], _vertex_prototypes[7]),
        stk.Edge(7, _vertex_prototypes[1], _vertex_prototypes[8]),
        stk.Edge(8, _vertex_prototypes[2], _vertex_prototypes[3]),
        stk.Edge(9, _vertex_prototypes[2], _vertex_prototypes[4]),
        stk.Edge(10, _vertex_prototypes[2], _vertex_prototypes[7]),
        stk.Edge(11, _vertex_prototypes[2], _vertex_prototypes[8]),
    )

    _num_windows = 2
    _num_window_types = 1


def topologies():
    return {
        "m2l4": {
            "fn": stk.cage.M2L4Lantern,
            "orientations": {
                "A": {2: 0, 3: 0, 4: 0, 5: 0},
                "B": {2: 1, 3: 0, 4: 0, 5: 0},
                "C": {2: 1, 3: 1, 4: 0, 5: 0},
                "D": {2: 1, 3: 0, 4: 1, 5: 0},
            },
        },
        "m3l6": {
            "fn": M3L6New,
            "orientations": {
                "1": {3: 1, 4: 0, 5: 1, 6: 0, 7: 0, 8: 1},
                "2": {3: 1, 4: 1, 5: 0, 6: 1, 7: 1, 8: 1},
                "3": {3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0},
                "4": {3: 1, 4: 0, 5: 1, 6: 1, 7: 1, 8: 1},
                "5": {3: 1, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1},
                "6": {3: 0, 4: 1, 5: 0, 6: 1, 7: 1, 8: 1},
                "7": {3: 1, 4: 0, 5: 0, 6: 1, 7: 0, 8: 1},
                "8": {3: 0, 4: 0, 5: 1, 6: 1, 7: 1, 8: 1},
                "9": {3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1},
            },
        },
        "m4l82": {
            "fn": M4L82,
            "orientations": {
                "1": {4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
            },
        },
    }


def main():
    bb = unsymmetrical_ligand()

    pd = stk.BuildingBlock(
        smiles="[Pd+2]",
        functional_groups=(
            stk.SingleAtom(stk.Pd(0, charge=2)) for i in range(4)
        ),
        position_matrix=[[0, 0, 0]],
    )

    # Construct Isomers of each cage topology.
    for topo in topologies():
        print(topo)
        tdict = topologies()[topo]
        c_orientations = tdict["orientations"]
        for c_a in c_orientations:
            c_orientation = c_orientations[c_a]
            name_ = f"{topo}_{c_a}"

            # Build cage.
            # if topo == "m4l82":
            opt = stk.NullOptimizer()
            # else:
            # opt = stk.MCHammer(target_bond_length=2.0, num_steps=1000)
            cage = stk.ConstructedMolecule(
                topology_graph=tdict["fn"](
                    building_blocks=(pd, bb),
                    optimizer=opt,
                    vertex_alignments=c_orientation,
                ),
            )
            cage.write(f"{name_}.mol")


if __name__ == "__main__":
    main()
