import os 
import yaml 
import argparse
import numpy as np
from matplotlib import pyplot as plt
from icecream import ic


def make_parser():
    parser = argparse.ArgumentParser("Visualize StreakData")
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default="datasets"
    )
    return parser


def read_config(path):
    with open(path, "r") as f:
        data = yaml.load(f.read(), Loader=yaml.FullLoader)
    return data 


def main(args):
    configs = []
    configs.append(["train", read_config(os.path.join(args.path, "train_config.yaml"))])
    configs.append(["valid", read_config(os.path.join(args.path, "valid_config.yaml"))])
    configs.append(["test", read_config(os.path.join(args.path, "test_config.yaml"))])
    lens = [len(config["config"]) for _, config in configs]
    max_len = np.max(lens)
    
    for j, (name, config) in enumerate(configs):
        sub_datasets = []
        for name_i in config["sub_datasets"]:
            gd = np.load(os.path.join(args.path, name_i, "groundtruth.npy"))
            sub_datasets.append(gd)
        for i, info in enumerate(config["config"]):
            plt.subplot(max_len, 3, 3 * i + j + 1)
            plt.title("{}-{}".format(name, i + 1))
            idx = info["sub_idx"]
            col = info["col_slice"]
            row = info["row_slice"]
            roi = sub_datasets[idx][row[0]:row[1], col[0]:col[1]]
            plt.imshow(roi.T, cmap="gray")
    plt.show()


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
    