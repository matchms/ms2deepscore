import os
from ms2deepscore.train_new_model.visualize_results import create_all_plots

if __name__ == "__main__":

    model_folders = os.listdir("../../../../data/trained_models/")
    for model_folder in model_folders:
        print(model_folder)
        create_all_plots(model_folder_name=model_folder)