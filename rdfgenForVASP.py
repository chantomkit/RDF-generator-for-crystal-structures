import math
import numpy as np
from os import listdir
import itertools
from scipy.spatial import distance
from matplotlib import pyplot as plt


for batchNum in range(1, 8):
    if batchNum == 4:
        continue
    INPUT_FILES = listdir(f"./INPUT_{batchNum}")
    fileNumber = 0
    fileNotUsed = 0
    training_data = []
    binNumber = np.array([])
    types = {"Triclinic": 0, "Monoclinic": 0, "Orthorhombic": 0, "Tetragonal": 0, "Trigonal": 0, "Hexagonal": 0, "Cubic": 0}
    spgrp_dis = np.zeros(230, dtype=int)

    for filename in INPUT_FILES:
        # if fileNumber > 50:
            # break
        lattice_cnst = 0
        # Basis vector of the unit cell of lattice [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)]
        lattice_vector = np.array([])
        # Atom type (Element1, Element2, ...)
        atom_type = np.array([])
        # Atom type (# of Element1, # of Element2, ...)
        atom_number = np.array([])
        # Atom position [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), ...]
        r = np.array([])

        # print("Lattice constant = " + str(lattice_cnst))
        # print("Number of lines = " + str(line))
        # print("Lattice vectors = " + str(lattice_vector))
        # print("Atom type = " + str(atom_type))
        # print("Atom number = " + str(atom_number))
        # print("Atom positions = " + str(r))
        # print("Rescaled atom positions = " + str(r_rescaled))

        n_line = 0
        fileNumber += 1
        print(f"Processing {fileNumber} / {len(INPUT_FILES)} files.")
        spaceGroup = int(filename.split("-")[0])
        spgrp_dis[spaceGroup-1] += 1
        # print(filename, spaceGroup)

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
        lattice_vector = lattice_vector.reshape((3,3))
        r = r.reshape((n_line - 9, 3))

        # For RDF
        # Macro definitions for spherical shells and bins
        V = np.linalg.det(lattice_vector)
        neighbour_list = np.zeros((n, n))
        RMAX = max([np.linalg.norm(lattice_vector[0]+lattice_vector[1]+lattice_vector[2]),
                    np.linalg.norm(lattice_vector[0]-lattice_vector[1]+lattice_vector[2]),
                    np.linalg.norm(-lattice_vector[0]+lattice_vector[1]+lattice_vector[2]),
                    np.linalg.norm(-lattice_vector[0]-lattice_vector[1]+lattice_vector[2])]) / 2
        RMIN = 0.0005
        binmax = 1000
        # ratio = 1000
        dr = RMAX / binmax

        g = np.zeros((binmax,), dtype=float)

        # Find separations of atoms and apply PBC
        for i in range(0, n):
            for j in range(i + 1, n):
                delta_r = np.zeros(3)
                for k in range(3):
                    delta_r[k] = r[j, k] - r[i, k]
                    while delta_r[k] >= 0.5:
                        delta_r[k] -= 1
                    while delta_r[k] < -0.5:
                        delta_r[k] += 1
                neighbour_list[i, j] = np.linalg.norm(np.dot(delta_r, lattice_vector))
                neighbour_list[j, i] = neighbour_list[i, j]
        # Stores separations of i-th (includes original particle and PBC mirrored particles) and j-th particles
        # Classify into bins
        neighbour_list = neighbour_list[neighbour_list != 0].flatten()
        bin_index = ((neighbour_list - RMIN) / dr).astype(int)
        count = np.histogram(bin_index, bins=np.arange(binmax + 1))[0].astype(int)

        if np.sum(count) == 0:
            print(f"Processed {fileNumber} / {len(INPUT_FILES)} files. (Bypassed: count = 0)")
            fileNotUsed += 1
            continue
        # RDF formula
        g = np.empty((binmax,), dtype=float)
        for x, cnt in enumerate(count):
            g[x] = (V / n) * (cnt / n / (4 * math.pi * dr * (x * dr + RMIN) ** 2))
        g = g / np.sum(g)

        plt.plot(g)
        plt.title(f"{spaceGroup}")
        plt.savefig(f"OUTPUT_simrdf/spgrp{spaceGroup}-graph{fileNumber}.png")
        plt.clf()

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

        training_data.append([g, np.eye(7)[x], np.eye(230)[spaceGroup-1]])
        print(f"Processed {fileNumber} / {len(INPUT_FILES)} files.")

    print("Finished processing files")
    np.save(f"td_simrdf/training_data_{batchNum}_{binmax}.npy", training_data)
    print("Finished saving.")
    print(f"File used: {len(training_data)}")
    print(f"File not used: {fileNotUsed}")
    print(f"Crystal system distribution: {types}")
    print(f"Space Group distribution: {spgrp_dis}")
    # print("Now preparing training data")
    #
    # for i in range(len(training_data)):
    #     for j in range(int(max(binNumber))):
    #         if len(training_data[i][0]) < max(binNumber):
    #             training_data[i][0] = np.append(training_data[i][0], 0)
    #         else:
    #             break
    #     print(f"Processed {i + 1} / {len(training_data)} files.")

    # for i in range(len(training_data)):
    # print(training_data[0][0], training_data[0][1])
    # print(training_data)
