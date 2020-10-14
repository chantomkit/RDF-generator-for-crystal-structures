import math
import numpy as np
from pymatgen.ext.matproj import MPRester
from pymatgen.io.vasp.inputs import Poscar
import itertools
from scipy.spatial import distance
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt


atomnum_list = np.array([])
imgcell_list = np.array([])
rmax_list = np.array([])
v_list = np.array([])
type_list = np.array([], dtype=int)
lv_list = np.array([])

m = MPRester("rDTKpFTb78oIHdV0")

MPlist = np.load("MP_list.npy", allow_pickle=True)
for ids, spgrp in MPlist:
    spaceGroup = spgrp.astype(np.float).astype(np.int)
    inputs = ids
    print(spaceGroup, inputs)

fileNotUsed = 0
training_data = []
binNumber = np.array([])
types = {"Triclinic": 0, "Monoclinic": 0, "Orthorhombic": 0, "Tetragonal": 0, "Trigonal": 0, "Hexagonal": 0, "Cubic": 0}

for fileNumber, material_id in enumerate(inputs):
    structure = m.get_structure_by_material_id(material_id)
    poscar = Poscar(structure).get_string()

    # Basis vector of the unit cell of lattice [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)]
    lattice_vector = np.array([])
    # Atom type (Element1, Element2, ...)
    atom_type = np.array([])
    # Atom type (# of Element1, # of Element2, ...)
    atom_number = np.array([])
    # Atom position [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), ...]
    r = np.array([])

    line = 0
    print(f"Processing {fileNumber + 1} / {len(inputs)} files.")

    # -- Reading from poscar (vasp format)
    for i, line in enumerate(poscar.split("\n")):
        n_line = i
        # Lattice constant:
        if i == 1:
            lattice_cnst = float(line)
        # Lattice vector 1:
        elif i >= 2 and i <= 4:
            lattice_vector = np.append(lattice_vector, line.split(' '))
            # Removing redundant elements
            lattice_vector = lattice_vector[lattice_vector != ''].astype(np.float)
        elif i == 5:
            atom_type = np.append(atom_type, line.split(' '))
        # Atom number
        elif i == 6:
            atom_number = np.array(line.split(' '))
            # Removing redundant elements
            atom_number = atom_number[(atom_number != '') & (atom_number != '0')].astype(np.int)
        # Atom position
        elif i >= 8:
            r = np.append(r, line.split(' '))
            # Removing redundant elements
            for atom in atom_type:
                r = r[r != atom]
            r = r[r != ''].astype(np.float)
    n = np.sum(atom_number)
    print(f"Number of atoms: {n}")
    lattice_vector = lattice_vector.reshape((3, 3)) * lattice_cnst
    r = r.reshape((n_line - 8, 3))
    # print(lattice_cnst, lattice_vector, atom_type, atom_number, r)

    # -- For RDF
    RMAX = max([np.linalg.norm(lattice_vector[0] + lattice_vector[1] + lattice_vector[2]),
                np.linalg.norm(lattice_vector[0] - lattice_vector[1] + lattice_vector[2]),
                np.linalg.norm(-lattice_vector[0] + lattice_vector[1] + lattice_vector[2]),
                np.linalg.norm(-lattice_vector[0] - lattice_vector[1] + lattice_vector[2])])
    lattice_vector_norm = np.array([np.linalg.norm(lattice_vector[i]) for i in range(3)])
    ShiftMax = np.ceil(np.array([RMAX / v for v in lattice_vector_norm])).astype(int)
    RMIN = 0.0005
    binmax = 1000
    V = np.linalg.det(lattice_vector)
    n_total = np.prod(1 + 2 * ShiftMax) * n
    dr = RMAX / binmax

    # -- Find separations of atoms and count into histogram
    permu = list(itertools.product(np.arange(-ShiftMax[0], ShiftMax[0] + 1),
                                   np.arange(-ShiftMax[1], ShiftMax[1] + 1),
                                   np.arange(-ShiftMax[2], ShiftMax[2] + 1)))
    dist = distance.cdist(np.dot(r, lattice_vector)
                          , np.dot(np.array([r[i] + permu for i in range(n)]).reshape(-1, 3)
                                   , lattice_vector), 'euclidean').flatten()
    dist = dist[(dist <= RMAX) & (dist > RMIN)]
    bin_index = ((dist - RMIN) / dr).astype(int)
    count = np.histogram(bin_index, bins=np.arange(binmax + 1))[0].astype(int)

    if np.sum(count) == 0:
        print(f"Processed {fileNumber + 1} / {len(inputs)} files. (Bypassed: count = 0)")
        fileNotUsed += 1
        continue

    # -- RDF formula
    g = np.empty((binmax,), dtype=float)
    for x, cnt in enumerate(count):
        g[x] = (V / n_total) * (cnt / n_total / (4 * math.pi * dr * (x * dr + RMIN) ** 2))
    g = g / np.sum(g)

    # Generate RDF plots in "OUTPUT"
    # plt.plot(np.arange(binmax)*dr+RMIN, g)
    # plt.title(f"{spaceGroup[fileNumber]}")
    # plt.savefig(f"MP_OUTPUT/spgrp{spaceGroup[fileNumber]}-graph{fileNumber}.png")
    # plt.clf()

    cur_i = int(np.where(inputs == material_id)[0][0])
    # print(cur_i)
    if spaceGroup[cur_i] > 0 and spaceGroup[cur_i] <= 2:
        x = 0  # Triclinic
        types["Triclinic"] += 1
    elif spaceGroup[cur_i] > 2 and spaceGroup[cur_i] <= 15:
        x = 1  # Monoclinic
        types["Monoclinic"] += 1
    elif spaceGroup[cur_i] > 15 and spaceGroup[cur_i] <= 74:
        x = 2  # Orthorhombic
        types["Orthorhombic"] += 1
    elif spaceGroup[cur_i] > 74 and spaceGroup[cur_i] <= 142:
        x = 3  # Tetragonal
        types["Tetragonal"] += 1
    elif spaceGroup[cur_i] > 142 and spaceGroup[cur_i] <= 167:
        x = 4  # Trigonal
        types["Trigonal"] += 1
    elif spaceGroup[cur_i] > 167 and spaceGroup[cur_i] <= 194:
        x = 5  # Hexagonal
        types["Hexagonal"] += 1
    else:
        x = 6  # Cubic
        types["Cubic"] += 1
    # print([np.array(g), np.eye(7)[x]])
    training_data.append([g, np.eye(7)[x], np.eye(230)[spaceGroup[cur_i]-1]])
    # print(training_data, np.shape(training_data))
    print(f"Processed {fileNumber + 1} / {len(inputs)} files.")

    # -- For data analysis (Can be ignored)
    type_list = np.append(type_list, x)
    atomnum_list = np.append(atomnum_list, n)
    rmax_list = np.append(rmax_list, RMAX)
    v_list = np.append(v_list, V)
    imgcell_list = np.append(imgcell_list, np.prod(1 + 2 * ShiftMax))
    lv_list = np.append(lv_list, lattice_vector_norm)
    # --

# np.save(f"training_data_mp_{binmax}.npy", training_data)
print(f"File used: {len(training_data)}")
print(f"File not used: {fileNotUsed}")
print("Finished processing files")
print(f"Crystal system distribution: {types}")

# -- For data analysis (Can be ignored)
lv_list = np.reshape(lv_list, (-1, 3))

type_dist = np.zeros(7)
avg_atomnumdist = np.zeros(7)
avg_celldist = np.zeros(7)
avg_rmax = np.zeros(7)
avg_v = np.zeros(7)
avg_lv = np.zeros((7, 3))
for i, a_n, c, rm, v, lv in zip(type_list, atomnum_list, imgcell_list, rmax_list, v_list, lv_list):
    avg_atomnumdist[i] += a_n
    avg_celldist[i] += c
    avg_rmax[i] += rm
    avg_v[i] += v
    avg_lv[i][0] += lv.min()
    avg_lv[i][1] += np.median(lv)
    avg_lv[i][2] += lv.max()
    type_dist[i] += 1
avg_atomnumdist /= type_dist
avg_celldist /= type_dist
avg_rmax /= type_dist
avg_v /= type_dist
for i in range(len(type_dist)):
    avg_lv[i] /= type_dist[i]
print(avg_atomnumdist)
print(avg_celldist)
print(avg_celldist * avg_atomnumdist)
print(avg_rmax)
print(avg_v)
print(avg_celldist * avg_v)

fig = plt.figure(figsize=(18,6))
fig.suptitle("Material Project Dataset")

ax1 = plt.subplot2grid((2,4),(0,0), colspan=4)
w = 0.2
xint = np.arange(7)
ax1.bar(xint-w, avg_lv[:,0], width=w, color='b', align='center')
ax1.bar(xint, avg_lv[:,1], width=w, color='r', align='center')
ax1.bar(xint+w, avg_lv[:,2], width=w, color='g', align='center')
ax1.set_xticks(xint)
ax1.set_xticklabels(('Tri-', 'Mono-',  "Ortho-", "Tetra-", "Trig-", "Hexa-", "Cubic"))
plt.title("Average min/median/max lattice vector")

ax2 = plt.subplot2grid((2,4),(1,0), colspan=4)
xmax = rmax_list.max()
for i in range(7):
    indices = np.where(type_list == i)
    ds = rmax_list[indices]
    density = gaussian_kde(ds)
    xs = np.linspace(0, xmax, 200)
    density.covariance_factor = lambda: .25
    density._compute_covariance()
    plt.plot(xs, density(xs))
plt.legend(('Tri-', 'Mono-',  "Ortho-", "Tetra-", "Trig-", "Hexa-", "Cubic"))
plt.title("Gaussian density of rmax distribution")

plt.show()