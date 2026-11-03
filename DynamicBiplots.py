import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm

def Biplot():
    df = pd.read_excel("dataTensorPaperWomen.xlsx", header=0, index_col=0)

    countries = df.index.tolist()
    dimensions = ['Preventive Care', 'Emotional Health', 'Opinions of Health and Safety', 'Basic Needs', 'Individual Health']
    years = ['2020', '2021', '2022', '2023']

    X_blocks_scaled = []
    for i in range(4):
        block = df.iloc[:, i*5:(i+1)*5].values.astype(float)
        scaler = StandardScaler()
        block_scaled = scaler.fit_transform(block.T).T
        X_blocks_scaled.append(block_scaled)

    country_coords = []
    variable_coords = []
    explained_variance = []

    for X in X_blocks_scaled:
        pca = PCA(n_components=2)
        Y = pca.fit_transform(X)
        V = pca.components_
        country_coords.append(Y)
        variable_coords.append(V)
        explained_variance.append(pca.explained_variance_ratio_)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    colors = plt.colormaps['tab20']

    for i in range(4):
        ax = axes[i]
        Y = country_coords[i]
        V = variable_coords[i]
        var_ratio = explained_variance[i]

        used_positions = []
        for j, name in enumerate(countries):
            color = colors(j)
            ax.scatter(Y[j, 0], Y[j, 1], color=color, s=60, edgecolor='black')
            dx, dy = 0.05, 0.05
            px, py = Y[j, 0] + dx, Y[j, 1] + dy
            while any(np.linalg.norm(np.array([px, py]) - np.array(pos)) < 0.2 for pos in used_positions):
                dx += 0.03
                dy += 0.03
                px, py = Y[j, 0] + dx, Y[j, 1] + dy
            used_positions.append((px, py))
            ax.text(px, py, name, fontsize=8, color=color)

        for k, dim in enumerate(dimensions):
            x, y = V[0, k], V[1, k]
            ax.arrow(0, 0, x, y, color='darkred', alpha=0.9, head_width=0.05)
            offset_x = 0.15 * np.sign(x) + 0.05 * (-1)**k
            offset_y = 0.15 * np.sign(y) + 0.05 * ((-1)**(k+1))
            ax.text(x + offset_x, y + offset_y, dim, fontsize=9, color='darkred')

        ax.set_title(f"Biplot of the year {years[i]}", fontsize=10, pad=12)
        ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)", labelpad=10)
        ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
        ax.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35)
    plt.show()

def DynamicBiplotCountries():
    df = pd.read_excel("dataTensorPaperWomen.xlsx", header=0, index_col=0)

    countries = df.index.tolist()
    dimensions = ['Preventive Care', 'Emotional Health', 'Opinions of Health and Safety', 'Basic Needs', 'Individual Health']
    years = ['2020', '2021', '2022', '2023']

    X_blocks_scaled = []
    for i in range(4):
        block = df.iloc[:, i*5:(i+1)*5].values.astype(float)
        scaler = StandardScaler()
        block_scaled = scaler.fit_transform(block.T).T
        X_blocks_scaled.append(block_scaled)

    pca_ref = PCA(n_components=2)
    Y_ref = pca_ref.fit_transform(X_blocks_scaled[0])
    V_ref = pca_ref.components_
    var_ratio = np.round(pca_ref.explained_variance_ratio_ * 100, 2)

    trajectories = []
    for X in X_blocks_scaled:
        Y_proj = X @ V_ref.T
        trajectories.append(Y_proj)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.colormaps['tab20']

    for i, name in enumerate(countries):
        coords = np.array([traj[i] for traj in trajectories])
        color = colors(i / len(countries))
        ax.plot(coords[:, 0], coords[:, 1], marker='o', color=color, linewidth=2)
        for j, year in enumerate(years):
            dx, dy = 0.05, 0.05
            px, py = coords[j, 0] + dx, coords[j, 1] + dy
            ax.text(px, py, year, fontsize=8, color=color)
        ax.text(coords[-1, 0] + 0.1, coords[-1, 1] + 0.1, name, fontsize=9, color=color)

    for k, dim in enumerate(dimensions):
        x, y = V_ref[0, k], V_ref[1, k]
        ax.arrow(0, 0, x, y, color='darkred', alpha=0.9, head_width=0.05)
        offset_x = 0.15 * np.sign(x) + 0.05 * (-1)**k
        offset_y = 0.15 * np.sign(y) + 0.05 * ((-1)**(k+1))
        ax.text(x + offset_x, y + offset_y, dim, fontsize=9, color='darkred')

    ax.set_xlabel(f"PC1 ({var_ratio[0]}%)", labelpad=15)
    ax.set_ylabel(f"PC2 ({var_ratio[1]}%)")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def DynamicBiplotVariables():
    df = pd.read_excel("dataTensorPaperWomen.xlsx", header=0, index_col=0)

    countries = df.index.tolist()
    dimensions = ['Preventive Care', 'Emotional Health', 'Opinions of Health and Safety', 'Basic Needs', 'Individual Health']
    years = ['2020', '2021', '2022', '2023']

    X_blocks_scaled = []
    for i in range(4):
        block = df.iloc[:, i*5:(i+1)*5].values.astype(float)
        scaler = StandardScaler()
        block_scaled = scaler.fit_transform(block.T).T
        X_blocks_scaled.append(block_scaled)

    pca_ref = PCA(n_components=2)
    Y_ref = pca_ref.fit_transform(X_blocks_scaled[0])
    V_ref = pca_ref.components_
    var_ratio = np.round(pca_ref.explained_variance_ratio_ * 100, 2)

    variable_trajectories = []
    for X in X_blocks_scaled:
        V_proj = PCA(n_components=2).fit(X).components_
        variable_trajectories.append(V_proj)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.colormaps['tab10']

    for i, name in enumerate(countries):
        x, y = Y_ref[i, 0], Y_ref[i, 1]
        ax.scatter(x, y, color='gray', s=60, edgecolor='black')
        ax.text(x + 0.05, y + 0.05, name, fontsize=8, color='gray')

    for k, dim in enumerate(dimensions):
        color = colors(k)
        coords = np.array([[V[0, k], V[1, k]] for V in variable_trajectories])

        ax.arrow(0, 0, coords[0, 0], coords[0, 1], color=color, alpha=0.9,
                 width=0.004, head_width=0.02, length_includes_head=True)

        for j in range(1, len(years)):
            x0, y0 = coords[j-1]
            x1, y1 = coords[j]
            ax.plot([x0, x1], [y0, y1], color=color, linestyle='dashed', linewidth=1.2)
            ax.scatter(x1, y1, s=20, facecolors='white', edgecolors=color, linewidths=1.2)

        for j, year in enumerate(years):
            dx, dy = 0.05, 0.05
            px, py = coords[j, 0] + dx, coords[j, 1] + dy
            ax.text(px, py, year, fontsize=8, color=color)

        ax.text(coords[-1, 0] + 0.1, coords[-1, 1] + 0.1, dim, fontsize=9, color=color)

    ax.set_xlabel(f"PC1 ({var_ratio[0]}%)", labelpad=15)
    ax.set_ylabel(f"PC2 ({var_ratio[1]}%)")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

Biplot()
DynamicBiplotCountries()
DynamicBiplotVariables()
