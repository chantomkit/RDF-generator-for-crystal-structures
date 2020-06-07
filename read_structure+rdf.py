import math
import numpy as np
import matplotlib.pyplot as plt

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
filename = '195-2-16-SiCl3-r0-id12-20191127-121044'
line = 0

fp = open(filename + ".vasp", "r")
if fp.mode == "r":
    for x in fp:
        line = line + 1
fp.close()

fp = open(filename + ".vasp", "r")
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
lattice_vector = lattice_vector.reshape((3,3))
r = r.reshape((line - 9, 3))

# Rescaling atom position
for i in range(n):
    dp = np.dot(r[i], lattice_vector)
    r_rescaled = np.append(r_rescaled, dp)
r_rescaled = lattice_cnst * r_rescaled
r_rescaled = r_rescaled.reshape(r.shape)

print("Lattice constant = " + str(lattice_cnst))
print("Number of lines = " + str(line))
print("Lattice vectors = " + str(lattice_vector))
print("Atom type = " + str(atom_type))
print("Atom number = " + str(atom_number))
print("Atom positions = " + str(r))
print("Rescaled atom positions = " + str(r_rescaled))

# For generating structure.xyz for VMD
# Not needed if not interested in generating .xyz file
fp = open("structure_" + filename + ".xyz", "w")
fp.write(str(n))
fp.write("\n\n")
for i in range(len(atom_number)):
    if i == 0:
        for j in range(0, atom_number[0]):
            fp.write(str(atom_type[i]) + " " + str(r_rescaled[j,0]) + " " + str(r_rescaled[j,1]) + " " + str(r_rescaled[j,2]) + "\n")
    elif i != 0:
        for j in range(sum(atom_number[:i]), sum(atom_number[:i+1])):
            fp.write(str(atom_type[i]) + " " + str(r_rescaled[j, 0]) + " " + str(r_rescaled[j, 1]) + " " + str(r_rescaled[j, 2]) + "\n")
fp.close()

# For RDF
# Macro definitions for spherical shells and bins
dr = 0.01
L = np.zeros(3)
for i in range(3):
    L[i] = lattice_vector[i,0]**2 + lattice_vector[i,1]**2 + lattice_vector[i,2]**2
diagonal = np.sum(lattice_vector, axis=0)
RMAX = math.sqrt(diagonal[0]**2 + diagonal[1]**2 + diagonal[2]**2) / 2
RMIN = 0.005
binmax = int(RMAX / dr)
count = np.zeros((binmax,), dtype=int)
g = np.zeros((binmax,), dtype=float)

# Find separations of atoms and apply PBC
for i in range(0, n):
    for j in range(i + 1, n):
        delta_r = np.array([r_rescaled[i,0] - r_rescaled[j,0], r_rescaled[i,1] - r_rescaled[j,1], r_rescaled[i,2] - r_rescaled[j,2]])
        # Stores separations of i-th (includes original particle and PBC mirrored particles) and j-th particles
        r_sq = np.zeros(7)
        for k in range(3):
            r_sq[0] = r_sq[0] + (delta_r[k])**2
            for l in range(3):
                r_sq[2*l+1] = r_sq[2*l+1] + (delta_r[k] + lattice_vector[l,k])**2
                r_sq[2*l+2] = r_sq[2*l+2] + (delta_r[k] - lattice_vector[l,k])**2
        #Classisify into bins
        for x in range(binmax):
            R = float(RMIN + x * dr)
            if r_sq.min() >= (R**2) and r_sq.min() < ((R + dr)**2):
                count[x] = count[x] + 1
                break

# RDF formula
for x in range(binmax):
    count[x] = 2 * count[x] / n
    g[x] = (L[0] * L[1] * L[2] / n) * (count[x] / (4 * math.pi * dr * (x * dr + RMIN)**2))

# Writing in .csv file
fp = open("rdf_" + filename + ".csv", "w")
for x in range(binmax):
    fp.write(str(x * dr + RMIN) + ", " + str(g[x]) + "\n")
fp.close()

# Plotting
x, y = np.loadtxt("rdf_" + filename + ".csv", delimiter=',', unpack=True)
plt.plot(x,y, color='k', markersize=0, linestyle='-', linewidth=1)
plt.title('RDF_' + filename)
plt.xlabel('r')
plt.ylabel('g(r)')
plt.savefig("rdf_" + filename + ".png", bbox_inches='tight')
plt.show()