import os
import subprocess

def pre_install_dependencies():

    packages = [
        "scipy",
        "numpy",
        "pandas",
        "xgboost",
        "seaborn",
        "pyarrow",
        "numerapi",
        "imblearn",
        "catboost",
        "lightgbm",
        "matplotlib",
        "cloudpickle",
        "mplcyberpunk",
        "scikit-learn",
        "torch",
        "torchsummary",
        "tqdm",
        "gdown"
    ]

    for package in packages:
        subprocess.call(["pip", "install", package])

def create_directories():
    os.makedirs("data", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

def download_data():
    from numerapi import NumerAPI
    NumerAi = NumerAPI()
    dataset_files = NumerAi.list_datasets()

    for dataset_file in dataset_files:
        NumerAi.download_dataset(dataset_file, dest_path = os.path.join("./data/", dataset_file))

def download_saved_models():
    import gdown
    model_drive_ids = {
        "numerai_expert1": "1PwaT6rU5d70aOfvu_3Q1JXtDQ2bkHMuD",
        "numerai_expert2": "12L5pyOSYmrzvzMqz8ntJwrzIVqGPxDze",
        "numerai_expert3": "12x7hP0QcGNFtVQeTouGc0EOvNTHkz9T0",
        "numerai_expert4": "1-3hEmpEhSMdpho2J-lJjRZsi8gae1S1A",
        "numerai_expert5": "1sljotBCQ1ksaP64lS_97_8EeYTZ4g9IU",
        "numerai_expert6": "1uciYxGCtJdVM8X5mwGAN5dpBXdCYQeKQ",
        "numerai_meta_model": "1BfrNoSkRyXGNCouOm1ig2Bii28oI6psp"
    }
    
    for model_name, drive_id in model_drive_ids.items():
        model_output_path = os.path.join("./saved_models/", f"{model_name}.pkl")

        subprocess.run(["gdown", f"https://drive.google.com/uc?id={drive_id}", "-O", model_output_path])


pre_install_dependencies()
create_directories()
download_data()
download_saved_models()
