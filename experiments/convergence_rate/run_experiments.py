import os
import subprocess

# Values are intended for Suttner
# You might want to set this to the number of *physical* cores on your computer
# os.environ['RAYON_NUM_THREADS'] = "28"
# os.environ['OMP_NUM_THREADS'] = "28"
os.environ['RUSTFLAGS'] = "-C target-cpu=native"

data_dir = "data_cylinder_shell"

# We assume that the script is called from the root project directory
command = "cargo run --release --bin fcm_convergence -- --resolutions 1 2 4 8 16 24 32 --reference-mesh hemisphere_50_uniform_refined2.msh"
subprocess.run([command], shell=True).check_returncode()
