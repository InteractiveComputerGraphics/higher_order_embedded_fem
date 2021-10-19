import os
import subprocess

# Values are intended for Suttner
# You might want to set this to the number of *physical* cores on your computer
# os.environ['RAYON_NUM_THREADS'] = "28"
# os.environ['OMP_NUM_THREADS'] = "28"
os.environ['RUSTFLAGS'] = "-C target-cpu=native"

data_dir = "data_armadillo_slingshot"

# We assume that the script is called from the root project directory
args = [
    "cargo",
    "run",
    "--",
    "--list-scenes"
]

subprocess.run(["cargo build --release"], shell=True)

scenes = [
    "armadillo_slingshot_embedded_tet4_500",
    "armadillo_slingshot_embedded_tet4_1500",
    "armadillo_slingshot_embedded_tet4_3000",
    "armadillo_slingshot_embedded_tet10_500",
    "armadillo_slingshot_embedded_tet10_1000",
    "armadillo_slingshot_embedded_tet10_1500",
    "armadillo_slingshot_embedded_tet10_3000",
    "armadillo_slingshot_fem_tet4_500",
    "armadillo_slingshot_fem_tet4_1500",
    "armadillo_slingshot_fem_tet4_3000",
    "armadillo_slingshot_fem_tet10_500",
    "armadillo_slingshot_fem_tet10_1000",
    "armadillo_slingshot_fem_tet10_3000",

    # Save the more expensive sims for last
    "armadillo_slingshot_fem_tet4_5000",
    "armadillo_slingshot_embedded_tet4_5000",
    "armadillo_slingshot_fem_tet10_5000",
    "armadillo_slingshot_embedded_tet10_5000",
    "armadillo_slingshot_fem_tet4_full",
]

for scene in scenes:
    print("\n\n\nRunning scene {}\n\n\n".format(scene))
    cmd = "target/release/dynamic_runner --scene {} --print-timings-output --output-dir {} --output-ply" \
          " --output-fps 100 --dump-stash"\
        .format(scene, data_dir)
    subprocess.run([cmd], shell=True)

