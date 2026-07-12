import sys
import os
import pickle
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.colors as mcolors

with open('../sparse_operator/sparse_operator.pkl', 'rb') as fh:
    tensor = pickle.load(fh)

n  = 3
n3 = n**3

# map flat index → (k, l, m) label
labels = {}
for k in range(n):
    for l in range(n):
        for m in range(-l, l+1):
            labels[n*n*k + l*l + (m+l)] = (k, l, m)

total_nnz     = sum(mat.nnz for mat in tensor)
total_entries = n3 * n3 * n3

print(f'nonzeros:     {total_nnz} / {total_entries}  ({100 * total_nnz / total_entries:.2f}%)')

# global colour scale centred at zero
all_nonzero = np.concatenate([tensor[t].data for t in range(n3) if tensor[t].nnz > 0])
vmax      = np.abs(all_nonzero).max()
linthresh = np.abs(all_nonzero).min()
norm = mcolors.SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax, base=10)

cmap = mcm.RdBu_r.copy()
cmap.set_bad('white')

os.makedirs('./figures', exist_ok=True)

# 3 rows × 9 cols: row = radial index k, cols = all (l,m) for that k
ncols = n3 // n
nrows = n

fig, axes = plt.subplots(nrows, ncols, figsize=(14.5, 4.5),
                         constrained_layout=True)

for t in range(n3):
    row = t // ncols
    col = t % ncols
    ax  = axes[row, col]

    arr    = tensor[t].toarray()
    masked = ma.array(arr, mask=(arr == 0))

    ax.imshow(masked, cmap=cmap, norm=norm,
              aspect='equal', interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])

    k, l, m = labels[t]
    ax.set_title(f'({k},{l},{m})', fontsize=7, pad=2)

    nnz = tensor[t].nnz
    ax.text(0.97, 0.03, str(nnz),
            transform=ax.transAxes, fontsize=6,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

fig.supxlabel(r'$\psi_r$', fontsize=10)
fig.supylabel(r'$\psi_s$', fontsize=10)

sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.02, label='entry value')
# explicit ticks to avoid crowding near zero in the symlog scale
ticks = [-1e4, -1e2, -1e0, 1e0, 1e2, 1e4]
cbar.set_ticks(ticks)
cbar.set_ticklabels([r'$-10^4$', r'$-10^2$', r'$-10^0$',
                     r'$10^0$',  r'$10^2$',  r'$10^4$'])

fig_path = './figures/sparsity_coulomb.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f'saved {fig_path}')
