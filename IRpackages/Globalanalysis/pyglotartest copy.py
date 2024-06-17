import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import lmfit
from scipy.optimize import minimize

def calculate_snthetic_data(C_matrix,S_matrix):
    AC = np.dot(C_matrix,S_matrix)
    return AC

def residue_DATA_FIT(A_matrix_data,AC_fitmatrix):
    #Calculate residue of DATA(A) and FITDATA(AC)
    DIFF = np.subtract(A_matrix_data,AC_fitmatrix)
    #residue = DIFF
    residue = np.linalg.norm(DIFF)**2
    return residue


def calculate_DAS(A, C):
    def objective_function(S_flat, A, C):
        """
        Objective function to minimize ||A - C*S||^2.
        """
        S = S_flat.reshape(C.shape[1], A.shape[1])  # Reshape the flattened S to its original shape
        res = np.linalg.norm(A - np.dot(C, S))**2
        print(res)
        return res
    """
    Minimize S numerically to solve ||A - C*S||^2.
    """
    # Initial guess for S
    S0 = np.zeros_like(V_vectors)
    # Flatten the initial guess for the optimization
    S0_flat = S0.flatten()
    # Minimize the objective function using numerical optimization
    resultS = minimize(objective_function, S0_flat, args=(A, C),options={'maxiter':1})#,tol=1)
    # Reshape the optimized flat S to its original shape
    S_optimized = resultS.x.reshape(C.shape[1], A.shape[1])

    if resultS.success == True:
        print('S optimized with:',resultS.message,resultS.nfev)#,result.x)
    return S_optimized

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













def generate_kinetic_model_parameters(model_matrix):
    """generate paramters for concentration profiles
    """
    # Check if the model matrix is square
    if np.shape(model_matrix)[0] != np.shape(model_matrix)[1]:
        raise ValueError("Model matrix must be square.")

    print('Model with shape:',np.shape(model_matrix))
    statenumber = int(np.shape(model_matrix)[0])
    
    #generate prameters for fitting and functions
    parameters_forLMFIT = list()
    statestring = 'ABCDEFGHIJKLMNOP'
    for i in range(statenumber):
        for j in range(statenumber):
            if model_matrix[i,j] != 0:
                parameters_forLMFIT.append(model_matrix[i,j])
    
    print('####PARAMETERS GENERATED:')
    return parameters_forLMFIT


def kinetic_derivatives(y,t,params_LMFIT,model_matrix):
    """ 
    build system of ODES to solve later
    -MODEL MATRIX: must be square for the global analysis to run
        example: for a reaction like A -> B -> C
          A B C
        A 0 1 0         
        B 0 0 0.1
        C 0 0 0
    following derivatives should result
        dA = -k_AB * y[0]
        dB = -k_BC * y[1] + k_AB * y[0]
        dC =  k_BC * y[1]
    """
    statenumber= np.shape(model_matrix)[0]
    statestring = 'ABCDEFGHIJKLMNOP'
    derivative_list= []
    
    for i in range(statenumber):
        derivative = 0
        for j in range(statenumber):
            if model_matrix[i,j]!=0:
                derivative = derivative - params_LMFIT[[i]] * y[i]
            if model_matrix[j,i]!=0:
                 derivative = derivative + params_LMFIT[j] * y[j]

        derivative_list.append(derivative)

    return derivative_list


def generate_Concentration_Profiles(t,y0,parameters_forLMFIT,model_matrix):
    y = sc.integrate.odeint(kinetic_derivatives,y0,t,args=(parameters_forLMFIT,model_matrix))
    return y









def OBJECTIVE_FIT(parameters_forLMFIT,A_matrix_data,y0,t,model_matrix):
    global C , S
    C = generate_Concentration_Profiles(t,y0,parameters_forLMFIT,model_matrix)
    S = calculate_DAS(A_matrix_data,C)
    AC = calculate_snthetic_data(C,S)
    residue = residue_DATA_FIT(A_matrix_data,AC)

    global residue_FIT,k1,k2
    residue_FIT.append(residue)
    k1.append(parameters_forLMFIT[0])
    k2.append(parameters_forLMFIT[1])

    return residue


def FIT(data,y0,t,parameters_forLMFIT,model_matrix):
    global result
    result = minimize(OBJECTIVE_FIT,parameters_forLMFIT,args=(data,y0,t,model_matrix),tol=1e-10,options={'maxiter':1000})
    if result.success == True:
        print('parameters optimized:',result.message,result.nfev)#,result.x)
    print(result)




def plot():
    fig = plt.figure(0,figsize=(12,6))
    plt.rc('font', size=10) #controls default text size
    plt.rcParams['xtick.major.pad']='1'
    plt.rcParams['ytick.major.pad']='1'
    grid = fig.add_gridspec(2, 3, hspace=0.2, wspace=0.2,bottom=0.05,top=0.95,left=0.05,right=0.95)
    subax1 = fig.add_subplot(grid[0,0])
    subax2 = fig.add_subplot(grid[0,2])
    svd = fig.add_subplot(grid[0,1])
    svdvectors = fig.add_subplot(grid[1,1])
    subax3 = fig.add_subplot(grid[1,0])
    subax4 = fig.add_subplot(grid[1,2])
    subax1.imshow(np.transpose(data), cmap='RdBu_r', aspect='auto', origin='lower',vmin=-1*np.max(np.abs(data)),vmax=1*np.max(np.abs(data)))
    fitdata = calculate_snthetic_data(C,S)
    subax2.imshow(np.transpose(fitdata), cmap='RdBu_r', aspect='auto', origin='lower',vmin=-1*np.max(np.abs(data)),vmax=1*np.max(np.abs(data)))
    
    svd.imshow(np.transpose(reconstructed_data), cmap='RdBu_r', aspect='auto', origin='lower',vmin=-1*np.max(np.abs(data)),vmax=1*np.max(np.abs(data)))
    for v in V_vectors:
        svdvectors.plot(wavenumbers,v)

    print('Cshape:',np.shape(C),'Sshape:',np.shape(S))
    i=0
    for state in np.transpose(C):
        i=i+1
        subax3.plot(timepoints,state,label=str(i))
    subax3.set_xscale('log')
    subax3.legend()
    
    i = 0
    for signal in S:
        i= i+1
        subax4.plot(wavenumbers,signal,label=str(i))
    subax4.legend()

    


    plt.show()


















residue_FIT=[]
k1 = []
k2 = []

timepoints = np.load('Setupfiles/Testdata/2024-01-29_A_delays.npy')
wavenumbers = np.load('Setupfiles/Testdata/2024-01-29_A_wavenumbers.npy')
data = np.load('Setupfiles/Testdata/2024-01-29_A_matrix.npy')
pixellow = 7
pixelhigh = 30
firstdelaystep=8
data = np.transpose(data[pixellow:pixelhigh,firstdelaystep:])
wavenumbers = wavenumbers[pixellow:pixelhigh]
timepoints = timepoints[firstdelaystep:]
print(np.shape(data),np.shape(timepoints),np.shape(wavenumbers))

model =        np.array([[0, 1, 0],
                         [0, 0, 0.1],
                         [0, 0, 0]])
y0=[1,0,0]


Sng,reconstructed_data,U_vectors,V_vectors = SVD_spectra(data,int(np.shape(model)[0]))
parameters = generate_kinetic_model_parameters(model)

FIT(data,y0,timepoints,parameters,model)
plot()


plt.plot(np.arange(len(residue_FIT)),residue_FIT)
plt.plot(np.arange(len(k1)),k1,label='kAB')
plt.plot(np.arange(len(k2)),k2,label='kBC')
plt.legend()
plt.show()