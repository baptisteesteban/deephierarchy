# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
#
# This page presents an example of usage of the Deep Hierarchy framework in
# order to build the region adjacency graph (RAG) of an image.

# %%

import sys

sys.path.append("../..")

# %%

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import inferno
from matplotlib.colors import Normalize
from numba import njit
from skimage.color import rgb2lab
from skimage.data import astronaut
from skimage.segmentation import mark_boundaries, slic

# %% [markdown]
#
# First of all, we import the different functions and classes required to
# compute the region adjacency graph.
# %%
from deephierarchy.core import C4
from deephierarchy.graph import build_rag, weight_rag_weighted_area_dist
from deephierarchy.labeling import compute_area, compute_sum, compute_sum_p

# %% [markdown]
#
# We load an image and we compute a partition using the SLIC superpixels
# algorithm. It has to be noted that we make the region labeling start at 0.
# Sometimes, the label 0 is used as a special label for some representation such
# as the background of the image. Since we look to build a partition, without
# any background notion, we make the labeling start at 0.

# %%

img = astronaut()
label_map = slic(img, start_label=0)

# %% [markdown]
#
# Then, we build the region adjacency graph from this partition. To this aim, we
# use a c4 connectivity. Note that the Deep Hierarchy framework also provides a
# c8 connectivity.

# %%

rag = build_rag(label_map, C4)

# %% [markdown]
#
# Each node of the RAG is linked to a region of the partition (or label map).
# Thus, we can compute, for each node, some region properties that may be used
# for feature extraction or edge weighting of the RAG. In our case, we want to
# compute the mean value of the partition regions, their spatial centroid or the
# area (the number of pixel in a region).

# %%

area = compute_area(label_map)
sum_v = compute_sum(label_map, rgb2lab(img))
sum_p = compute_sum_p(label_map)
centroid = sum_p / area[:, None]

# %% [markdown]
#
# Finally, we compute the edge weight of the rag using the following edge weight
# function<a href="#note1"><sup>[1]</sup></a>:
#
# $$
#
# \begin{align*}
# w(\mathcal{R}_1, \mathcal{R}_2) & = \mathcal{A}(\mathcal{R}_1) \times
# \|\mathcal{M}(\mathcal{R}_1) - \mathcal{M}(\mathcal{R}_1 \cup
# \mathcal{R}_2)\|_2\\
#  & + \mathcal{A}(\mathcal{R}_2) \times
# \|\mathcal{M}(\mathcal{R}_2) - \mathcal{M}(\mathcal{R}_1 \cup
# \mathcal{R}_2)\|_2\\
# \end{align*}
#
# $$
#
# with $\mathcal{M}(\mathcal{R})$ is the mean value of $\mathcal{R}$ and
# $\mathcal{A}(\mathcal{R})$ is the area (number of pixels) of $\mathcal{R}$.

# %%

rag_weights = weight_rag_weighted_area_dist(rag, area, sum_v)

# %% [markdown]
#
# To conclude, we display the region adjacency graph with edges colored
# according to their weight.

# %%

weight_set = np.unique(rag_weights[rag.adjacency_matrix])
min_w = weight_set.min()
max_w = weight_set.max()
norm = Normalize(vmin=min_w, vmax=max_w)

# %%

plt.figure(figsize=(10, 10))
plt.imshow(mark_boundaries(img, label_map))
for n1 in range(rag.num_nodes):
    for n2 in range(n1, rag.num_nodes):
        if rag.adjacency_matrix[n1, n2]:
            plt.plot(
                (centroid[n1, 1], centroid[n2, 1]),
                (centroid[n1, 0], centroid[n2, 0]),
                c=inferno(norm(rag_weights[n1, n2])),
            )
plt.scatter(centroid[:, 1], centroid[:, 0], c="r")
plt.show()

# %% [markdown]
#
# **References**
#
# <div id="note1">
#
# [1]: Luis Garrido, Philippe Salembier, David Garcia. *Extensive operators in
#     partition lattices for image sequence analysis*.
#     [10.1016/S0165-1684(98)00004-8](https://doi.org/10.1016/S0165-1684(98)00004-8)
#
# </div>
