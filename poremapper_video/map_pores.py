import stk
import pore_mapper as pm
import time


mol_files = ['4D2_C_optc.mol', '6C1_C_optc.mol']

for mol in mol_files:
    xyz_file = mol.replace('.mol', '.xyz')
    stk.BuildingBlock.init_from_file(mol).write(
        xyz_file
    )

    # Read in host from xyz file.
    host = pm.Host.init_from_xyz_file(path=xyz_file)
    host = host.with_centroid([0., 0., 0.])

    # Define calculator object.
    stime = time.time()
    calculator = pm.Inflater(bead_sigma=0.7)
    final_result = calculator.get_inflated_blob(host=host)
    print(f'run time: {time.time() - stime}')
    pore = final_result.pore
    blob = final_result.pore.get_blob()
    windows = pore.get_windows()
    print(final_result)
    print(
        f'step: {final_result.step}\n'
        f'num_movable_beads: {final_result.num_movable_beads}\n'
        f'windows: {windows}\n'
        f'blob: {blob}\n'
        f'pore: {pore}\n'
        f'blob_max_diam: {blob.get_maximum_diameter()}\n'
        f'pore_max_rad: {pore.get_maximum_distance_to_com()}\n'
        f'pore_mean_rad: {pore.get_mean_distance_to_com()}\n'
        f'pore_volume: {pore.get_volume()}\n'
        f'num_windows: {len(windows)}\n'
        f'max_window_size: {max(windows)}\n'
        f'min_window_size: {min(windows)}\n'
        f'asphericity: {pore.get_asphericity()}\n'
        f'shape anisotropy: {pore.get_relative_shape_anisotropy()}\n'
    )
    print()

    # Do final structure.
    host.write_xyz_file(xyz_file.replace('.xyz', '_final.xyz'))
    blob.write_xyz_file(xyz_file.replace('.xyz', '_blob.xyz'))
    pore.write_xyz_file(xyz_file.replace('.xyz', '_pore.xyz'))
