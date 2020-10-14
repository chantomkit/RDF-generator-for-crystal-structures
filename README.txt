CUHK PHYSICS SUMMER INTERNSHIP PROGRAM 2020

# ABANDONED
- For RG2 strucutures:
- For PBC, it generates periodic boundary condition RDF
- For one layer, it generates cluster RDF with one (adjustable) layer of image cells around the center cell
- For auto layer, it generates full clutser RDF with auto filled layers around the center cell according to RMAX
#

For Material Project structures:
- (old) For matgen, it directly downloads poscar from MatPro and compute RDF
- For matgen2, it requires a dir of poscar files (and a space group list w.r.t. the poscar dir). If no space group list is provided, the program will auto-fetch the space groups of all file in dir and make a list. It can computes RDF locally once the setup is complete. 
