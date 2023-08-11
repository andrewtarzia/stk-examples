import stk
import numpy as np
import stko


def main():

    mol_list = [
        ('caff', stk.BuildingBlock('CN1C=NC2=C1C(=O)N(C(=O)N2C)C')),
        ('pdmoc', stk.BuildingBlock.init_from_file('pdmoc_unopt.mol')),
        ('hg', stk.BuildingBlock.init_from_file('5A1_gmB_opt.mol')),
        ('hg2', stk.BuildingBlock.init_from_file('5A3_gmB_opt.mol')),
    ]

    _opt = stko.OpenBabel('uff')
    for name, mol in mol_list:
        if name == 'pdmoc':
            mol = _opt.optimize(mol)
        initial = mol.with_rotation_about_axis(
            1.34, np.array((0, 1, 1)), np.array((0, 0, 0)),
        )
        mol.write(f'{name}_unaligned.mol')
        initial.write(f'{name}_init.mol')


if __name__ == "__main__":
    main()
