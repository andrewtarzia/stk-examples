import stk

four_c_bb = stk.BuildingBlock(
    smiles="[Br][C]([Br])([Br])[Br]",
    position_matrix=[
        [-2, 0, -1],
        [0, 0, 1],
        [0, -2, -1],
        [2, 0, 1],
        [0, 2, 1],
    ],
    functional_groups=(stk.BromoFactory(placers=(0, 1)),),
)
two_c_bb = stk.BuildingBlock(
    smiles="[Br][N][Br]",
    position_matrix=[
        [-2, 0, -1],
        [0, 0, 1],
        [0, -2, -1],
    ],
    functional_groups=(stk.BromoFactory(placers=(0, 1)),),
)


cage = stk.ConstructedMolecule(
    topology_graph=stk.cage.M2L4Lantern(
        building_blocks=(four_c_bb, two_c_bb),
    ),
)
cage.write("cage.mol")

cage = stk.ConstructedMolecule(
    topology_graph=stk.cage.M2L4Lantern(
        building_blocks=(four_c_bb, two_c_bb),
        optimizer=stk.MCHammer(),
    ),
)
cage.write("cage_opt.mol")

# Do STK docs, have unicode colours match to vertex.
