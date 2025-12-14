from sklearn.datasets import load_iris
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def iris3d():
    iris = load_iris()
    X = iris.data[:, [0, 2]]
    y = (iris.target == 2).astype(int)
    X_data = (X - X.mean(axis=0)) * 20
    
    model = SVC(kernel='rbf', gamma=0.005, C=50.0, probability=True, random_state=0)
    model.fit(X_data, y)
    
    x_min, x_max = -40.0, 40.0
    y_min, y_max = -40.0, 40.0
    z_min, z_max = -100.0, 100.0
    grid_res = 50
    xx = np.linspace(x_min, x_max, grid_res)
    yy = np.linspace(y_min, y_max, grid_res)
    X_surf, Y_surf = np.meshgrid(xx, yy)
    XY_grid = np.c_[X_surf.ravel(), Y_surf.ravel()]
    Probs = model.predict_proba(XY_grid)[:, 1]
    Z_score = (Probs - 0.5) * 200
    Z_surf = Z_score.reshape(X_surf.shape)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.cm.coolwarm

    ax.contourf(X_surf, Y_surf, Z_surf, zdir='z', offset=z_min, cmap=cmap, alpha=0.6)
    ax.contourf(Y_surf, Z_surf, Z_surf, zdir='x', offset=x_min, cmap=cmap, alpha=0.6)
    ax.contourf(X_surf, Z_surf, Z_surf, zdir='y', offset=y_max, cmap=cmap, alpha=0.6)

    Z_zero = np.zeros_like(X_surf)
    ax.plot_wireframe(X_surf, Y_surf, Z_zero, rstride=5, cstride=5, color='#000080', alpha=0.3, linewidth=0.5)

    ax.plot_surface(X_surf, Y_surf, Z_surf, cmap=cmap, rstride=1, cstride=1, alpha=0.6, linewidth=0, antialiased=False)
    ax.plot_wireframe(X_surf, Y_surf, Z_surf, rstride=5, cstride=5, color='gray', alpha=0.3, linewidth=0.5)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel('Sepal Length')
    ax.set_ylabel('Petal Length')
    ax.set_zlabel('Probability Score')
    ax.view_init(elev=30, azim=-60)
    plt.show()

if __name__ == "__main__":
    iris3d()