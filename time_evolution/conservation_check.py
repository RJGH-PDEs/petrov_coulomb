import sys
import os
import pickle
import numpy as np

# ── flat index (k,l,m) with n=3, L=2 ─────────────────────────────────────────
# ind(k,l,m) = 9k + l^2 + (m+l)
# mass      = ind(0,0,0) = 0
# Py        = ind(0,1,-1) = 1
# Pz        = ind(0,1,0)  = 2
# Px        = ind(0,1,1)  = 3
# energy    = ind(1,0,0)  = 9
MASS   = 0
PY     = 1
PZ     = 2
PX     = 3
ENERGY = 9

TAU    = 1e-6
COEFF_DIR = '../plot/coeff'
OUT_DIR   = './conservation'

# load mass inverse and recover M
with open('mass_inverse.pkl', 'rb') as fh:
    mi = pickle.load(fh)
M = np.linalg.inv(mi)

os.makedirs(OUT_DIR, exist_ok=True)

# find all saved snapshots
steps = sorted(
    int(f.replace('.pkl', ''))
    for f in os.listdir(COEFF_DIR)
    if f.endswith('.pkl')
)

rows = []
for step in steps:
    with open(f'{COEFF_DIR}/{step}.pkl', 'rb') as fh:
        coeff = pickle.load(fh)
    Mf = M @ coeff
    rows.append([step, step * TAU, Mf[MASS], Mf[PX], Mf[PY], Mf[PZ], Mf[ENERGY]])

out_path = f'{OUT_DIR}/coulomb.csv'
header = 'step,t,mass,px,py,pz,energy'
np.savetxt(out_path, rows, delimiter=',', header=header, comments='')
print(f'saved {out_path}  ({len(rows)} snapshots)')
