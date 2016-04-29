import sys
sys.path.append("S3DGLPy")
from PolyMesh import *
from Primitives3D import *
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr, cg, eigsh
import matplotlib.pyplot as plt
import scipy.io as sio


##############################################################
##                  Laplacian Mesh Editing                  ##
##############################################################

def getLaplacianMatrixHelp(mesh, anchorsIdx, weight, inPlace = False):
    N = len(mesh.vertices)
    if inPlace:  
        K = 0 
        quadIdxs = anchorsIdx
        anchorsIdx = []
    else:
        K = len(anchorsIdx)
        quadIdxs = []     
    X = [vtx.getVertexNeighbors() if vtx.ID not in quadIdxs else [] for vtx in mesh.vertices]
    I = [[index]*len(row) for index, row in enumerate(X)]
    J = [[nb.ID for nb in row] for row in X]
    V = [[weight(nb, mesh.vertices[index]) for nb in row] for index, row in enumerate(X)]
    D = [-sum(row) if index not in quadIdxs else 1 for index, row in enumerate(V)]
    I = [item for sublist in I for item in sublist] + range(N+K)
    J = [item for sublist in J for item in sublist] + range(N)
    J = np.concatenate((J, anchorsIdx))
    V = [item for sublist in V for item in sublist] + D + [1]*K
    L = sparse.coo_matrix((V, (I, J)), shape=(N+K, N)).tocsr()
    return L

def getLaplacianMatrixHelpSquareScale(mesh, weight):
    N = len(mesh.vertices)
    I = [[nb.ID for nb in vtx.getVertexNeighbors()] for vtx in mesh.vertices]
    J = [[index]*len(row) for index, row in enumerate(I)]
    I = [item for sublist in I for item in sublist] + range(N)
    J = [item for sublist in J for item in sublist] + range(N)
    V = [[weight(nb, vtx) for nb in vtx.getVertexNeighbors()] for vtx in mesh.vertices]    
    V = [[v/-sum(row) for v in row] for row in V]
    V = [item for sublist in V for item in sublist] + [1]*len(V)
    L = sparse.coo_matrix((V, (J, I)), shape=(N, N)).tocsr()
    return L

def umbrellaWeight(v1, v2):
        return -1.0

def cotangentWeight(v1, v2):
        edge = getEdgeInCommon(v1, v2)
        vtx = [[v for v in face.getVertices() if (v != v1) and (v != v2)][0] for face in [edge.f1, edge.f2] if face]
        pos = [[v1.getPos()-v3.getPos(), v2.getPos()-v3.getPos()] for v3 in vtx]
        cot = [np.dot(v3[0], v3[1])/np.linalg.norm(np.cross(v3[0], v3[1])) for v3 in pos]
        return -sum(cot)/len(cot)

#Purpose: To return a sparse matrix representing a Laplacian matrix with
#the graph Laplacian (D - A) in the upper square part and anchors as the
#lower rows
#Inputs: mesh (polygon mesh object), anchorsIdx (indices of the anchor points)
#Returns: L (An (N+K) x N sparse matrix, where N is the number of vertices
#and K is the number of anchors)
def getLaplacianMatrixUmbrella(mesh, anchorsIdx):
    return getLaplacianMatrixHelp(mesh, anchorsIdx, umbrellaWeight)

#Purpose: To return a sparse matrix representing a laplacian matrix with
#cotangent weights in the upper square part and anchors as the lower rows
#Inputs: mesh (polygon mesh object), anchorsIdx (indices of the anchor points)
#Returns: L (An (N+K) x N sparse matrix, where N is the number of vertices
#and K is the number of anchors)
def getLaplacianMatrixCotangent(mesh, anchorsIdx):
    return getLaplacianMatrixHelp(mesh, anchorsIdx, cotangentWeight)

#Purpose: Given a mesh, to perform Laplacian mesh editing by solving the system
#of delta coordinates and anchors in the least squared sense
#Inputs: mesh (polygon mesh object), anchors (a K x 3 numpy array of anchor
#coordinates), anchorsIdx (a parallel array of the indices of the anchors)
#Returns: Nothing (should update mesh.VPos)
def solveLaplacianMesh(mesh, anchors, anchorsIdx):
    L = getLaplacianMatrixCotangent(mesh, anchorsIdx)
    delta = np.array(L.dot(mesh.VPos))
    delta[-len(anchorsIdx):, :] = anchors
    for col in range(3):
        mesh.VPos[:, col] = lsqr(L, delta[:, col])[0]

#Purpose: Given a few RGB colors on a mesh, smoothly interpolate those colors
#by using their values as anchors and 
#Inputs: mesh (polygon mesh object), anchors (a K x 3 numpy array of anchor
#coordinates), anchorsIdx (a parallel array of the indices of the anchors)
#Returns: Nothing (should update mesh.VPos)
def smoothColors(mesh, colors, colorsIdx):
    L = getLaplacianMatrixUmbrella(mesh, colorsIdx)
    delta = np.zeros((L.shape[0],3))
    delta[-len(colorsIdx):, :] = colors
    for col in range(3):
        mesh.VPos[:, col] = lsqr(L, delta[:, col])[0]

#Purpose: Given a mesh, to smooth it by subtracting off the delta coordinates
#from each vertex, normalized by the degree of that vertex
#Inputs: mesh (polygon mesh object)
#Returns: Nothing (should update mesh.VPos)
def doLaplacianSmooth(mesh):
    L = getLaplacianMatrixHelpSquareScale(mesh, umbrellaWeight)
    mesh.VPos = np.subtract(mesh.VPos, L*mesh.VPos)

#Purpose: Given a mesh, to sharpen it by adding back the delta coordinates
#from each vertex, normalized by the degree of that vertex
#Inputs: mesh (polygon mesh object)
#Returns: Nothing (should update mesh.VPos)
def doLaplacianSharpen(mesh):
    L = getLaplacianMatrixHelpSquareScale(mesh, umbrellaWeight)
    mesh.VPos = np.add(mesh.VPos, L*mesh.VPos)

#Purpose: Given a mesh and a set of anchors, to simulate a minimal surface
#by replacing the rows of the laplacian matrix with the anchors, setting
#those "delta coordinates" to the anchor values, and setting the rest of the
#delta coordinates to zero
#Inputs: mesh (polygon mesh object), anchors (a K x 3 numpy array of anchor
#coordinates), anchorsIdx (a parallel array of the indices of the anchors)
#Returns: Nothing (should update mesh.VPos)
def makeMinimalSurface(mesh, anchors, anchorsIdx):
    L = getLaplacianMatrixHelp(mesh, anchorsIdx, umbrellaWeight, True)
    delta = np.zeros((len(mesh.vertices), 3))
    delta[anchorsIdx, :] = anchors
    for col in range(3):
        mesh.VPos[:, col] = lsqr(L, delta[:, col])[0]

##############################################################
##        Spectral Representations / Heat Flow              ##
##############################################################

#Purpose: Given a mesh, to compute first K eigenvectors of its Laplacian
#and the corresponding eigenvalues
#Inputs: mesh (polygon mesh object), K (number of eigenvalues/eigenvectors)
#Returns: (eigvalues, eigvectors): a tuple of the eigenvalues and eigenvectors
def getLaplacianSpectrum(mesh, K):
    L = getLaplacianMatrixHelp(mesh, [], umbrellaWeight)
    (eigvalues, eigvectors) = sparse.linalg.eigsh(L, K, which='LM', sigma = 0) 
    return (eigvalues, eigvectors)

#Purpose: Given a mesh, to use the first K eigenvectors of its Laplacian
#to perform a lowpass filtering
#Inputs: mesh (polygon mesh object), K (number of eigenvalues/eigenvectors)
#Returns: Nothing (should update mesh.VPos)
def doLowpassFiltering(mesh, K):
    L = getLaplacianMatrixHelp(mesh, [], cotangentWeight)
    vals, vecs = eigsh(L, K, which='LM',sigma=0)
    for col in range(3):
        mesh.VPos[:,col] = np.dot(np.dot(vecs, np.transpose(vecs)),mesh.VPos[:,col])


        
#Purpose: Given a mesh, to simulate heat flow by projecting initial conditions
#onto the eigenvectors of the Laplacian matrix, and then to sum up the heat
#flow of each eigenvector after it's decayed after an amount of time t
#Inputs: mesh (polygon mesh object), eigvalues (K eigenvalues), 
#eigvectors (an NxK matrix of eigenvectors computed by your laplacian spectrum
#code), t (the time to simulate), initialVertices (indices of the verticies
#that have an initial amount of heat), heatValue (the value to put at each of
#the initial vertices at the beginning of time
#Returns: heat (a length N array of heat values on the mesh)
def getHeat(mesh, eigvalues, eigvectors, t, initialVertices, heatValue = 100.0):
    N = mesh.VPos.shape[0]
    heat = np.zeros(N) #Dummy value
    return heat #TODO: Finish this

#Purpose: Given a mesh, to approximate its curvature at some measurement scale
#by recording the amount of heat that stays at each vertex after a unit impulse
#of heat is applied.  This is called the "Heat Kernel Signature" (HKS)
#Inputs: mesh (polygon mesh object), K (number of eigenvalues/eigenvectors to use)
#t (the time scale at which to compute the HKS)
#Returns: hks (a length N array of the HKS values)
def getHKS(mesh, K, t):
    N = mesh.VPos.shape[0]
    hks = np.zeros(N) #Dummy value
    return hks #TODO: Finish this

##############################################################
##                Parameterization/Texturing               ##
##############################################################

#Purpose: Given 4 vertex indices on a quadrilateral, to anchor them to the 
#square and flatten the rest of the mesh inside of that square
#Inputs: mesh (polygon mesh object), quadIdxs (a length 4 array of indices
#into the mesh of the four points that are to be anchored, in CCW order)
#Returns: nothing (update mesh.VPos)
def doFlattening(mesh, quadIdxs):
    if len(quadIdxs) != 4:
        print "please select 4 points"
        return
    L = getLaplacianMatrixHelp(mesh, quadIdxs, umbrellaWeight, True)
    delta = np.zeros((len(mesh.vertices), 3))
    delta[quadIdxs, :] = [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]
    for col in range(3):
        mesh.VPos[:, col] = lsqr(L, delta[:, col])[0]
    # mesh.VPos[:, 2] = 0

#Purpose: Given 4 vertex indices on a quadrilateral, to anchor them to the 
#square and flatten the rest of the mesh inside of that square.  Then, to 
#return these to be used as texture coordinates
#Inputs: mesh (polygon mesh object), quadIdxs (a length 4 array of indices
#into the mesh of the four points that are to be anchored, in CCW order)
#Returns: U (an N x 2 matrix of texture coordinates)
def getTexCoords(mesh, quadIdxs):
    N = mesh.VPos.shape[0]
    U = np.zeros((N, 2)) #Dummy value
    return U #TODO: Finish this

if __name__ == '__main__':
    print "TODO"
    # mesh = PolyMesh()
    # mesh.loadFile("meshes/homer.off")
    # doFlattening(mesh, [0, 1, 2, 3])
    # makeMinimalSurface(mesh, np.array([[0,0,0],[1,1,1]]), np.array([3,5]))
    # print [vtx.ID for vtx in mesh.vertices]
    # print getLaplacianMatrixHelp(mesh, [], umbrellaWeight) != getLaplacianMatrixHelp(mesh, [], cotangentWeight)
    # print getLaplacianMatrixUmbrella(mesh, np.array([3,5])) != getLaplacianMatrixCotangent(mesh, np.array([3,5]))