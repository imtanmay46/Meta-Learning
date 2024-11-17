import os
import subprocess

def pre_install_dependencies():
    packages = [
        "torch",
        "numpy",
        "matplotlib",
        "tqdm",
        "scikit-learn",
        "mplcyberpunk",
        "pandas",
        "scipy",
        "seaborn",
        "pyarrow",
        "cloudpickle",
        "gdown"
    ]
    for package in packages:
        subprocess.call(["pip", "install", package])

def create_directories():
    os.makedirs("data", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

def download_arc_data():
    dataset_url = "https://github.com/fchollet/ARC/archive/master.zip"
    output_zip = "./data/arc_dataset.zip"
    
    os.makedirs("data", exist_ok=True)
    
    subprocess.run(["wget", dataset_url, "-O", output_zip])
    
    subprocess.run(["unzip", output_zip, "-d", "./data"])
    
    os.remove(output_zip)

def organize_arc_data():
    src_folder = "./data/ARC-AGI-master"
    dest_folder = "./data/arc_data"
    if os.path.exists(src_folder) and not os.path.exists(dest_folder):
        os.rename(src_folder, dest_folder)

def download_model():
    import gdown
    model_drive_id = "1F3D6LMvS0vjX8RMo0EUj5ix267PKh8DJ"
    model_output_path = "./saved_models/best_meta_model.pth"

    subprocess.run(["gdown", f"https://drive.google.com/uc?id={model_drive_id}", "-O", model_output_path])

pre_install_dependencies()
create_directories()
download_arc_data()
organize_arc_data()
download_model()