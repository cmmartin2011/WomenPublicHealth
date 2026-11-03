import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import seaborn as sns

core = np.array([
    [[2.8198, 1.9641],
     [-1.4642, -2.0809],
     [3.4704, 2.0342]],
    [[3.9975, 2.7284],
     [4.6922, 2.2522],
     [4.0124, 2.4240]],
    [[5.1714, 3.3875],
     [5.4514, 1.7313],
     [0.3617, 0.0874]]
])

highlights = {
    5.4514: 1,
    5.1714: 2,
    4.6922: 3,
    4.0124: 4,
    3.9975: 5,
    3.4704: 6,
    3.3875: 7
}

def plot_core_with_labels(core_tensor, highlights):
    r0, r1, r2 = core_tensor.shape
    fig, axes = plt.subplots(1, r2, figsize=(5 * r2, 4), constrained_layout=True)
    if r2 == 1:
        axes = [axes]
    for k in range(r2):
        ax = axes[k]
        slice_ = core_tensor[:, :, k]
        annotated = np.empty_like(slice_, dtype=object)
        for i in range(r0):
            for j in range(r1):
                val = slice_[i, j]
                label = str(val)
                if val in highlights:
                    label += f"\n({highlights[val]})"
                annotated[i, j] = label
        sns.heatmap(slice_, annot=annotated, fmt="", cmap="coolwarm", center=0,
                    xticklabels=["DC1-B", "DC2-B", "DC3-B"],
                    yticklabels=["DC1-A", "DC2-A", "DC3-A"],
                    ax=ax, cbar=(k == r2 - 1), linewidths=0.5, linecolor='gray')
        ax.set_title(f"Core Slice: DC{k+1}-C", fontsize=12)
    plt.show()

# Matrices de cargas
coords_countries = np.array([
    [0.288389799, -0.010491617, -0.041720149],
    [0.364471875, -0.125546386, -0.288731029],
    [0.19135615, -0.727501546, 0.172431565],
    [0.441510478, 0.11091795, -0.658653486],
    [0.325056847, 0.171997566, 0.535603031],
    [0.329159905, 0.233132147, 0.317673731],
    [0.148996965, -0.503535037, 0.137733191],
    [0.386504916, 0.212028021, 0.127382294],
    [0.309061406, 0.159315149, 0.150860768],
    [0.267076023, -0.18640882, -0.074792408]
])

coords_indicators = np.array([
    [0.492474899, -0.371630106, -0.21313323],
    [-0.474676978, -0.365002077, 0.192120671],
    [0.362445952, 0.597657379, 0.604232877],
    [-0.37957669, 0.607670721, -0.572578024],
    [-0.506660349, -0.046973401, 0.474047371]
])

coords_years = np.array([
    [0.503397935, 0.820146391],
    [0.473171645, -0.000820179],
    [0.527745206, -0.446008133],
    [0.494149887, -0.358379645]
])

countries = ['Argentina','Brazil','Canada','Chile','Colombia','Ecuador','USA','Mexico','Peru','Uruguay']
indicators = ['Preventive Care','Emotional Health','Opinions of Health and Safety','Basic Needs','Individual Health']
years = ['2020','2021','2022','2023']

def plot_mode(coords, n_clusters, names, mode_label):
    x = coords[:, 0]
    y = coords[:, 1] if coords.shape[1] > 1 else np.zeros_like(x)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(coords)
    labels = kmeans.labels_

    offsets = {0: (0.015, 0.01), 1: (-0.03, 0.01), 2: (0.015, -0.02)}

    plt.figure(figsize=(7, 6))
    for i in range(len(names)):
        dx, dy = offsets.get(labels[i], (0.015, 0.01))
        plt.scatter(x[i], y[i], c=f'C{labels[i]}', s=90, edgecolors='black', linewidths=0.6)
        plt.text(x[i]+dx, y[i]+dy, names[i], fontsize=9, ha='left', va='center')

    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.xlabel(f'{mode_label}: Latent dimension 1')
    plt.ylabel(f'{mode_label}: Latent dimension 2')
    plt.tight_layout()
    plt.show()

plot_core_with_labels(core, highlights)

plot_mode(coords_countries, 3, countries, 'Countries')
plot_mode(coords_indicators, 4, indicators, 'Health Indicators')
plot_mode(coords_years, 2, years, 'Years')
