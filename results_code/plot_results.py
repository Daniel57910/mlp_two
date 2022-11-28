import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob
# Set the path to the results folder


def main():
    residual_losses = os.path.join("result_sets", "vgg_38_residual")
    paths = glob.glob(os.path.join(residual_losses, "*.csv"))

    train_path = next(p for p in paths if "train" in p)
    val_path = next(p for p in paths if "val" in p)

    df = pd.read_csv(train_path)
    print(df.head(10))


# setup main
if __name__ == "__main__":
    main()