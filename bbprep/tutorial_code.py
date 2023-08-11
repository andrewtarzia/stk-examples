import stk
import bbprep
import time


def distanced():
    print("\ndoing distanced")
    bb = stk.BuildingBlock(
        smiles="C1=CC(=CN=C1)C2=NC=C(C=C2)C3=CC=NC=C3",
        functional_groups=stk.SmartsFunctionalGroupFactory(
            smarts="[#6]~[#7X2]~[#6]",
            bonders=(1,),
            deleters=(),
        ),
    )
    bb.write("distance_original.mol")
    original_fgs = tuple(bb.get_functional_groups())
    input(f"original: {original_fgs}")
    modified = bbprep.ClosestFGs().modify(
        building_block=bb,
        desired_functional_groups=2,
    )
    modified_fgs = tuple(modified.get_functional_groups())
    input(f"closest: {modified_fgs}")

    modified = bbprep.FurthestFGs().modify(
        building_block=bb,
        desired_functional_groups=2,
    )
    modified_fgs = tuple(modified.get_functional_groups())
    input(f"furthest: {modified_fgs}")
    print("done\n")
    input()


def planarfy():
    print("\ndoing planarfy")
    bb = stk.BuildingBlock(
        smiles=(
            "C1=CC=C2C=C(C=CC2=C1)C3=CC(=CC=C3)C4=CC=CC5=CC=CC=C54"
        ),
    )
    bb.write("planar_original.mol")
    input("see original")

    selector = bbprep.selectors.AllSelector()

    st = time.time()
    generator = bbprep.generators.ETKDG(num_confs=10)
    ensemble = generator.generate_conformers(bb)
    input(ensemble)

    process = bbprep.Planarfy(ensemble=ensemble, selector=selector)
    min_molecule = process.get_minimum()
    print("note that this gives a Conformer!")
    min_molecule.molecule.write("planar_etkdg_final.mol")
    min_score = process.calculate_score(
        min_molecule, process.get_minimum_id()
    )
    print(
        f"min score (etkdg): {min_score} ({round(time.time()-st, 2)}s)"
    )

    st = time.time()
    generator = bbprep.generators.TorsionScanner(
        target_torsions=bbprep.generators.TargetTorsion(
            smarts="[#6][#6]-!@[#6][#6]",
            expected_num_atoms=4,
            torsion_ids=(0, 1, 2, 3),
        ),
        angle_range=range(0, 362, 40),
    )
    ensemble = generator.generate_conformers(bb)

    process = bbprep.Planarfy(ensemble=ensemble, selector=selector)
    min_molecule = process.get_minimum()
    min_molecule.molecule.write("planar_scan_final.mol")
    min_score = process.calculate_score(
        min_molecule, process.get_minimum_id()
    )
    input(
        f"min score (scan): {min_score} ({round(time.time()-st, 2)}s)"
    )
    print("done\n")
    input()


def torsion():
    print("\ndoing torsion")
    bb = stk.BuildingBlock(smiles="C1=CC=NC(=C1)C2=CC=CC=N2")
    bb.write("torsion_original.mol")
    input("see original")

    selector = bbprep.selectors.BySmartsSelector(
        smarts="[#6][#7][#6][#6][#7][#6]",
        selected_indices=(1, 2, 3, 4),
    )
    generator = bbprep.generators.TorsionScanner(
        target_torsions=bbprep.generators.TargetTorsion(
            smarts="[#7][#6][#6][#7]",
            expected_num_atoms=4,
            torsion_ids=(0, 1, 2, 3),
        ),
        angle_range=range(0, 362, 20),
    )
    ensemble = generator.generate_conformers(bb)

    target1 = 120
    process = bbprep.TargetTorsion(
        ensemble=ensemble,
        selector=selector,
        target_value=target1,
    )
    best_molecule = process.get_best()
    best_molecule.molecule.write("torsion_target1.mol")

    target2 = 60
    process = bbprep.TargetTorsion(
        ensemble=ensemble,
        selector=selector,
        target_value=target2,
    )
    best_molecule = process.get_best()
    best_molecule.molecule.write("torsion_target2.mol")
    print("done\n")
    input()


def ditopic():
    print("\ndoing ditopic fitter")
    bb = stk.BuildingBlock(
        smiles="C1C=CN=CC=1C1=CC=C(C2C=NC=CC=2)C=C1",
        functional_groups=stk.SmartsFunctionalGroupFactory(
            smarts="[#6]~[#7X2]~[#6]",
            bonders=(1,),
            deleters=(),
        ),
    )
    bb.write("ditopic_original.mol")
    input("see original")

    generator = bbprep.generators.ETKDG(num_confs=30)
    ensemble = generator.generate_conformers(bb)

    process = bbprep.DitopicFitter(ensemble=ensemble)
    min_molecule = process.get_minimum()
    min_molecule.molecule.write("ditopic_final.mol")
    print("done\n")
    input()


def main():
    distanced()
    planarfy()
    torsion()
    ditopic()
    print("Thanks for watching!")


if __name__ == "__main__":
    main()
