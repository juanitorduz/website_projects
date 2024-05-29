import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")

u, v = np.mgrid[0 : 2 * np.pi : 100j, 0 : np.pi : 100j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)
ax.view_init(elev=30, azim=30)
ax.plot_surface(x, y, z, cmap="Spectral", rstride=3, cstride=3, edgecolor="k")
ax.set_box_aspect((1, 1, 0.87))
ax.grid(None)
ax.axis("off")
plt.savefig("sphere_spectral.png", dpi=500, bbox_inches="tight")
