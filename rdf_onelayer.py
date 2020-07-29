import math
import numpy as np
from os import listdir
import itertools
from scipy.spatial import distance
from matplotlib import pyplot as plt

# batchNum = 7
for batchNum in range(1, 8):
    if batchNum == 4:
        continue
    INPUT_FILES = listdir(f"./INPUT_{batchNum}")
    fileNumber = 0
    fileNotUsed = 0
    training_data = []
    types = {"Triclinic": 0, "Monoclinic": 0, "Orthorhombic": 0, "Tetragonal": 0, "Trigonal": 0, "Hexagonal": 0, "Cubic": 0}
    spgrp_dis = np.zeros(230, dtype=int)

    for filename in INPUT_FILES:
        # Basis vector of the unit cell of lattice [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)]
        lattice_vector = np.array([])
        # Atom type (# of Element1, # of Element2, ...)
        atom_number = np.array([])
        # Atom position [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), ...]
        r = np.array([])

        line = 0
        fileNumber += 1
        print(f"Processing {fileNumber} / {len(INPUT_FILES)} files.")
        spaceGroup = int(filename.split("-")[0])
        spgrp_dis[spaceGroup-1] += 1

        fp = open(f"INPUT_{batchNum}/{filename}", "r")
        for i, line in enumerate(fp.read().split("\n")):
            n_line = i
            # Lattice constant:
            if i == 1:
                lattice_cnst = float(line)
            # Lattice vector 1:
            elif i >= 2 and i <= 4:
                lattice_vector = np.append(lattice_vector, line.split(' '))
                # Removing redundant elements
                lattice_vector = lattice_vector[lattice_vector != ''].astype(np.float)
            # Atom number
            elif i == 6:
                atom_number = np.array(line.split(' '))
                # Removing redundant elements
                atom_number = atom_number[(atom_number != '') & (atom_number != '0')].astype(np.int)
            # Atom position
            elif i >= 9:
                r = np.append(r, line.split(' '))
                # Removing redundant elements
                r = r[(r != '') & (r != 'T')].astype(np.float)
        n = np.sum(atom_number)

        print(f"Number of atoms: {n}")
        lattice_vector = lattice_vector.reshape((3, 3)) * lattice_cnst
        r = r.reshape((n_line - 9, 3))

        # For RDF
        RMAX = max([np.linalg.norm(lattice_vector[0]+lattice_vector[1]+lattice_vector[2]),
                    np.linalg.norm(lattice_vector[0]-lattice_vector[1]+lattice_vector[2]),
                    np.linalg.norm(-lattice_vector[0]+lattice_vector[1]+lattice_vector[2]),
                    np.linalg.norm(-lattice_vector[0]-lattice_vector[1]+lattice_vector[2])])
        ShiftMax = 1 # This decides wrapping how many layers around the center cell
        ShiftIter = np.arange(-ShiftMax, ShiftMax + 1)
        RMIN = 0.0005
        binmax = 1000
        V = np.linalg.det(lattice_vector)
        n_total = (1 + 2 * ShiftMax) * n
        dr = RMAX / binmax

        # Find separations of atoms and count into histogram
        permu = list(itertools.product(ShiftIter, ShiftIter, ShiftIter))
        dist = distance.cdist(np.dot(r, lattice_vector),
                              np.dot(np.array([r[i] + permu for i in range(n)]).reshape(-1, 3), lattice_vector),
                              'euclidean').flatten()
        dist = dist[(dist <= RMAX) & (dist > RMIN)]
        bin_index = ((dist - RMIN) / dr).astype(int)
        count = np.histogram(bin_index, bins=np.arange(binmax + 1))[0].astype(int)

        if np.sum(count) == 0:
            print(f"Processed {fileNumber} / {len(INPUT_FILES)} files. (Bypassed: count = 0)")
            fileNotUsed += 1
            continue

        # RDF formula
        g = np.empty((binmax,), dtype=float)
        for x, cnt in enumerate(count):
            g[x] = (V / n_total) * (cnt / n_total / (4 * math.pi * dr * (x * dr + RMIN) ** 2))
        g = g / np.sum(g)

        # Generate RDF plots in "OUTPUT"
        # plt.plot(np.arange(binmax)*dr+RMIN, g)
        # plt.title(f"{spaceGroup}")
        # plt.savefig(f"OUTPUT/spgrp{spaceGroup}-graph{fileNumber}.png")
        # plt.clf()

        if spaceGroup > 0 and spaceGroup <= 2:
            x = 0  # Triclinic
            types["Triclinic"] += 1
        elif spaceGroup > 2 and spaceGroup <= 15:
            x = 1  # Monoclinic
            types["Monoclinic"] += 1
        elif spaceGroup > 15 and spaceGroup <= 74:
            x = 2  # Orthorhombic
            types["Orthorhombic"] += 1
        elif spaceGroup > 74 and spaceGroup <= 142:
            x = 3  # Tetragonal
            types["Tetragonal"] += 1
        elif spaceGroup > 142 and spaceGroup <= 167:
            x = 4  # Trigonal
            types["Trigonal"] += 1
        elif spaceGroup > 167 and spaceGroup <= 194:
            x = 5  # Hexagonal
            types["Hexagonal"] += 1
        else:
            x = 6  # Cubic
            types["Cubic"] += 1
        # print([np.array(g), np.eye(7)[x]])
        training_data.append([g, np.eye(7)[x], np.eye(230)[spaceGroup-1]])
        # print(training_data, np.shape(training_data))
        print(f"Processed {fileNumber} / {len(INPUT_FILES)} files.")

    print("Finished processing files")
    np.save(f"td_fullrdf/training_data_{batchNum}_{binmax}.npy", training_data)
    print("Finished saving.")
    print(f"File used: {len(training_data)}")
    print(f"File not used: {fileNotUsed}")
    print(f"Crystal system distribution: {types}")
    print(f"Space Group distribution: {spgrp_dis}")
