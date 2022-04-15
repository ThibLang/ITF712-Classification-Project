import os
import matplotlib.pyplot as plt


def save_fig(folder_path, fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(folder_path, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
