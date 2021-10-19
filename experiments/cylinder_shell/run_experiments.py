import os
import subprocess

# Values are intended for Suttner
# You might want to set this to the number of *physical* cores on your computer
# os.environ['RAYON_NUM_THREADS'] = "28"
# os.environ['OMP_NUM_THREADS'] = "28"
os.environ['RUSTFLAGS'] = "-C target-cpu=native"

data_dir = "data_cylinder_shell"

# We assume that the script is called from the root project directory
args = [
    "cargo",
    "run",
    "--",
    "--list-scenes"
]

subprocess.run(["cargo build --release"], shell=True).check_returncode()

scenes = [
    # "cylinder_shell_embedded_hex20_res1",
    # "cylinder_shell_embedded_hex20_res2",
    # "cylinder_shell_embedded_hex20_res3",
    # "cylinder_shell_embedded_hex20_res4",
    # "cylinder_shell_embedded_hex20_res5",
    # "cylinder_shell_embedded_hex20_res6_strength3",
    # "cylinder_shell_embedded_hex20_res6_strength5",
    # "cylinder_shell_embedded_hex20_res6_strength5_no_simp",
    "cylinder_shell_embedded_hex20_res7_strength3",
    "cylinder_shell_embedded_hex20_res7_strength5",
    # "cylinder_shell_embedded_hex20_res8",
    # "cylinder_shell_embedded_hex20_res10",
    # "cylinder_shell_embedded_hex8_res2",
    # "cylinder_shell_embedded_hex8_res3",
    # "cylinder_shell_embedded_hex8_res5",
    # "cylinder_shell_embedded_hex8_res10",
    # "cylinder_shell_embedded_hex8_res14",
    # "cylinder_shell_embedded_hex8_res16",
    # "cylinder_shell_embedded_hex8_res18",
    # "cylinder_shell_embedded_hex8_res20",
    # "cylinder_shell_embedded_hex8_res22",
    # "cylinder_shell_embedded_hex8_res24",
    # "cylinder_shell_embedded_hex8_res26",
    "cylinder_shell_embedded_hex8_res28",
    "cylinder_shell_embedded_hex8_res29",
    "cylinder_shell_embedded_hex8_res30",
    # "cylinder_shell_embedded_hex8_res32",
    # "cylinder_shell_fem_tet4_5k",
    # "cylinder_shell_fem_tet4_10k",
    # "cylinder_shell_fem_tet4_20k",
    # "cylinder_shell_fem_tet4_40k",
    "cylinder_shell_fem_tet10_5k_strength2",
    "cylinder_shell_fem_tet10_5k_strength3",
    "cylinder_shell_fem_tet10_5k_strength5",
    # "cylinder_shell_fem_tet10_10k",
    # # Run these denser meshes last
    # "cylinder_shell_fem_tet4_80k",
    "cylinder_shell_fem_tet10_20k",
    "cylinder_shell_fem_tet4_160k",
    "cylinder_shell_embedded_hex20_res7_strength5_no_simp",
    # "cylinder_shell_fem_tet4_320k",
]

for scene in scenes:
    print("\n\n\nRunning scene {}\n\n\n".format(scene))
    cmd = "target/release/dynamic_runner --scene {} --print-timings-output --output-dir {} --output-ply " \
          "--dump-stash"\
        .format(scene, data_dir)
    subprocess.run([cmd], shell=True)
