import os
import subprocess

# Values are intended for Suttner
# You might want to set this to the number of *physical* cores on your computer
# os.environ['RAYON_NUM_THREADS'] = "28"
# os.environ['OMP_NUM_THREADS'] = "28"
os.environ['RUSTFLAGS'] = "-C target-cpu=native"

data_dir = "data_hollow_ball"

# We assume that the script is called from the root project directory
args = [
    "cargo",
    "run",
    "--",
    "--list-scenes"
]

subprocess.run(["cargo build --release"], shell=True)

scenes = [
    "hollow_ball_embedded_tet4_coarse",
    "hollow_ball_embedded_tet10_coarse",

    "hollow_ball_embedded_tet4_medium",
    "hollow_ball_embedded_tet10_medium",
    "hollow_ball_fem_tet4_medium",

    "hollow_ball_fem_tet4_fine",
    "hollow_ball_fem_tet10_medium",
    "hollow_ball_fem_tet4_coarse",
    "hollow_ball_fem_tet10_coarse",

    "hollow_ball_fem_tet10_fine",
]

for scene in scenes:
    print("\n\n\nRunning scene {}\n\n\n".format(scene))
    cmd = "target/release/dynamic_runner --scene {} --print-timings-output --output-dir {} --output-ply " \
          "--dump-stash --output-fps 100"\
        .format(scene, data_dir)
    subprocess.run([cmd], shell=True)

