import numpy as np
import matplotlib.pyplot as plt



def SVD_spectra(data,num_components_to_keep):
    """
    Function to calculate the Singular Value Decomposition of a spectra 

     returns 
        -S-Matrix to plot the Singular Values and show their priority #plt.semilogy(S, marker='o', linestyle='-')
        -reconstructed data in same format as input data restricted to the num of components
        -U_vectors 
        -V-Vectors
    
    example:
    S,rdata,Uv,Vv = SVD_spectra(array[:,:],3)
    """
    U, S, V = np.linalg.svd(data, full_matrices=False)
    reconstructed_data = np.dot(U[:, :num_components_to_keep], np.dot(np.diag(S[:num_components_to_keep]), V[:num_components_to_keep, :]))
    U_vectors = []
    V_vectors = []
    for i in range(num_components_to_keep):
        U_vectors.append(U[:,i])
        V_vectors.append(V[i,:])

    return S,reconstructed_data,U_vectors,V_vectors









