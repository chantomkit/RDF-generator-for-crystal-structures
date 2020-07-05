import math
import numpy as np
from os import listdir
from matplotlib import pyplot as plt

INPUT_FILES = listdir("./INPUT")
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
    # Rescaled position [(x1', y1', z1'), (x2', y2', z2'), (x3', y3', z3'), ...]
    r_rescaled = np.array([])

    # print("Lattice constant = " + str(lattice_cnst))
    # print("Number of lines = " + str(line))
    # print("Lattice vectors = " + str(lattice_vector))
    # print("Atom type = " + str(atom_type))
    # print("Atom number = " + str(atom_number))
    # print("Atom positions = " + str(r))
    # print("Rescaled atom positions = " + str(r_rescaled))

    line = 0
    fileNumber += 1
    print(f"Processing {fileNumber} / {len(INPUT_FILES)} files.")
    spaceGroup = int(filename.split("-")[0])
    spgrp_dis[spaceGroup-1] += 1
    # print(filename, spaceGroup)

    fp = open(f"INPUT/{filename}", "r")
    if fp.mode == "r":
        for x in fp:
            line = line + 1
    fp.close()

    fp = open(f"INPUT/{filename}", "r")
    for x in range(1, line + 1):
        # Lattice constant:
        if x == 2:
            lattice_cnst = float(fp.readline())
        # Lattice vector 1:
        elif x >= 3 and x <= 5:
            lattice_vector = np.append(lattice_vector, fp.readline().split(' '))
            # Removing redundant elements
            lattice_vector = lattice_vector[lattice_vector != '']
            lattice_vector = lattice_vector[lattice_vector != '\n']
            lattice_vector = lattice_vector.astype(np.float)
        # Atom type:
        elif x == 6:
            atom_type = np.append(atom_type, fp.readline().split(' '))
            # Removing redundant elements
            atom_type = atom_type[atom_type != '']
            atom_type = atom_type[atom_type != '\n']
        # Atom number
        elif x == 7:
            atom_number = np.append(atom_number, fp.readline().split(' '))
            # Removing redundant elements
            atom_number = atom_number[atom_number != '']
            atom_number = atom_number[atom_number != '\n']
            atom_number = atom_number[atom_number != '0']
            atom_number = atom_number.astype(np.int)
        # Atom position
        elif x >= 10:
            r = np.append(r, fp.readline().split(' '))
            # Removing redundant elements
            r = r[r != '']
            r = r[r != '\n']
            r = r[r != 'T']
            r = r[r != 'T\n']
            r = r.astype(np.float)

        else:
            fp.readline()

    n = np.sum(atom_number)
    print(f"Number of atoms: {n}")
    # if n > 100:
    #     print(f"Processed {fileNumber} / {len(INPUT_FILES)} files. (Bypassed: n too large)")
    #     fileNotUsed += 1
    #     continue
    lattice_vector = lattice_vector.reshape((3,3))
    r = r.reshape((line - 9, 3))

    # Rescaling atom position
    for i in range(n):
        dp = np.dot(r[i], lattice_vector)
        r_rescaled = np.append(r_rescaled, dp)
    r_rescaled = r_rescaled.reshape(r.shape)

    # For RDF
    # Macro definitions for spherical shells and bins
    neighbour_list = np.zeros((n, n))
    L = np.zeros(3)
    for i in range(3):
        L[i] = lattice_vector[i,0]**2 + lattice_vector[i,1]**2 + lattice_vector[i,2]**2
    diagonal = np.sum(lattice_vector, axis=0)
    RMAX = np.linalg.norm(diagonal)
    for i in L:
        if RMAX ** 2 < i:
            RMAX = float(np.sum(np.sqrt(L)))
            break
    RMIN = 0.0005
    binmax = 1000
    # ratio = 1000
    dr = RMAX / binmax

    count = np.zeros((binmax,), dtype=int)
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
            delta_r = np.dot(delta_r, lattice_vector)
            neighbour_list[i, j] = np.linalg.norm(delta_r)
            neighbour_list[j, i] = neighbour_list[i, j]
            # Stores separations of i-th (includes original particle and PBC mirrored particles) and j-th particles
            # Classify into bins
    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                continue
            if neighbour_list[i, j] <= RMAX:
                count[int((neighbour_list[i, j] - RMIN) / dr)] += 1
            # for x in range(binmax):
            #     R = float(RMIN + x * dr)
            #     if neighbour_list[i, j] >= R and neighbour_list[i, j] < (R + dr):
            #         count[x] = count[x] + 1
            #         break

    if np.linalg.norm(count) == 0:
        # print(count)
        print(f"Processed {fileNumber} / {len(INPUT_FILES)} files. (Bypassed: count = 0)")
        fileNotUsed += 1
        continue
    # RDF formula
    for x in range(binmax):
        count_mod = 2 * count[x] / n
        g[x] = (L[0] * L[1] * L[2] / n) * (count_mod / (4 * math.pi * dr * (x * dr + RMIN)**2))
    g = g / np.sum(g)
    plt.plot(g)
    plt.title(f"{spaceGroup}")
    plt.savefig(f"OUTPUT/spgrp{spaceGroup}-graph{fileNumber}.png")
    plt.clf()
    # print(np.sum(g))
    # print(binNumber)

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
np.save("training_data_1.npy", training_data)
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
