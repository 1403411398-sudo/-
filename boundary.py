from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

def run_3d_boundary_plot():
    iris = load_iris()
    X = iris.data[:, :3]
    y = (iris.target == 2).astype(int)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
    model.fit(X_scaled, y)
    
    coef = model.coef_[0]
    intercept = model.intercept_[0]
    
    x1_min, x1_max = -2.0, 2.0
    x2_min, x2_max = -2.0, 2.0
    x1_surf = np.linspace(x1_min, x1_max, 50)
    x2_surf = np.linspace(x2_min, x2_max, 50)
    x1_surf, x2_surf = np.meshgrid(x1_surf, x2_surf)
    
    epsilon = 1e-10
    if abs(coef[2]) < epsilon:
        x3_surf = np.zeros_like(x1_surf)
    else:
        x3_surf = -(coef[0] * x1_surf + coef[1] * x2_surf + intercept) / coef[2]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    cmap = mcolors.ListedColormap(['#0000FF', '#FF0000'])
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], 
               c=y, cmap=cmap, marker='o', s=30, edgecolors='k')
    
    ax.plot_surface(x1_surf, x2_surf, x3_surf, 
                    color='gray', alpha=0.3, linewidth=0, antialiased=False)
    
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_zlim(-2.0, 2.0)
    
    ax.set_xlabel('X1', fontsize=12)
    ax.set_ylabel('X2', fontsize=12)
    ax.set_zlabel('X3', fontsize=12)
    
    ax.view_init(elev=15, azim=135)
    
    plt.show()

if __name__ == "__main__":
    run_3d_boundary_plot()