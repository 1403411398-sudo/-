import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline
from matplotlib.colors import LinearSegmentedColormap

def iris():
   
    iris = load_iris()
    X = iris.data[:, [1, 2]]  
    y = iris.target
    feature_names = [iris.feature_names[1], iris.feature_names[2]]
    class_names = iris.target_names  

    clf = make_pipeline(
        SplineTransformer(degree=3, n_knots=5), 
        LogisticRegression(max_iter=1000)
    )
    clf.fit(X, y)

    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),  
        np.linspace(y_min, y_max, 200)  
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    probs = clf.predict_proba(grid).reshape(xx.shape[0], xx.shape[1], 3)

    cmap_setosa = LinearSegmentedColormap.from_list("setosa_cmap", ["white", "#0000FF"])
    cmap_versicolor = LinearSegmentedColormap.from_list("versi_cmap", ["white", "#FF9900"])
    cmap_virginica = LinearSegmentedColormap.from_list("virgi_cmap", ["white", "#00CC00"])

    point_colors = {0: "#0000FF", 1: "#FF9900", 2: "#00CC00"}
    point_edge = "black"  
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Sepal Width vs Petal Length (Probability & Classification)", fontsize=14)
    ax = axes[0]
    im = ax.imshow(
        probs[:, :, 0].T, 
        extent=[x_min, x_max, y_min, y_max],
        origin="lower", cmap=cmap_setosa, alpha=0.8
    )
    for cls in range(3):
        mask = (y == cls)
        ax.scatter(X[mask, 0], X[mask, 1], color=point_colors[cls],
                   edgecolors=point_edge, s=30)
    ax.set_title(f"Class 0 ({class_names[0]}) Probability")
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax = axes[1]
    ax.imshow(
        probs[:, :, 1].T,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower", cmap=cmap_versicolor, alpha=0.8
    )

    for cls in range(3):
        mask = (y == cls)
        ax.scatter(X[mask, 0], X[mask, 1], color=point_colors[cls],
                   edgecolors=point_edge, s=30)
    ax.set_title(f"Class 1 ({class_names[1]}) Probability")
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax = axes[2]
    ax.imshow(
        probs[:, :, 2].T,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower", cmap=cmap_virginica, alpha=0.8
    )
  
    for cls in range(3):
        mask = (y == cls)
        ax.scatter(X[mask, 0], X[mask, 1], color=point_colors[cls],
                   edgecolors=point_edge, s=30)
    ax.set_title(f"Class 2 ({class_names[2]}) Probability")
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


    ax = axes[3]
   
    pred_cls = clf.predict(grid).reshape(xx.shape)
   
    class_cmap = LinearSegmentedColormap.from_list(
        "class_cmap", [point_colors[0], point_colors[1], point_colors[2]]
    )
    ax.imshow(
        pred_cls.T,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower", cmap=class_cmap, alpha=0.8
    )
    
    for cls in range(3):
        mask = (y == cls)
        ax.scatter(X[mask, 0], X[mask, 1], color=point_colors[cls],
                   edgecolors=point_edge, s=30)
    ax.set_title("Max Class")
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Probability")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    iris()