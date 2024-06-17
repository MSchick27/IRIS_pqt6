import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import lmfit
#from scipy.optimize import minimize

def calculate_snthetic_data(C_matrix,S_matrix):
    AC = np.dot(C_matrix,S_matrix)
    return AC

def residue_DATA_FIT(A_matrix_data,AC_fitmatrix):
    #Calculate residue of DATA(A) and FITDATA(AC)
    DIFF = np.subtract(A_matrix_data,AC_fitmatrix)
    #residue = np.linalg.norm(DIFF)**2
    residue = DIFF
    return residue

def calculate_DAS(A, C,parameters_forLMFIT):
    def objective_function(S_flat, A, C):
        """
        Objective function to minimize ||A - C*S||^2.
        """
        S = S_flat.reshape(C.shape[1], A.shape[1])  # Reshape the flattened S to its original shape
        return np.linalg.norm(A - np.dot(C, S))**2

    S = translate_sparamatrix(parameters_forLMFIT)
    return S

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








def translate_sparamatrix(parameters_forLMFIT):
    S = np.zeros_like(V_vectors)
    for i in range(len(S[:,0])):
        for j in range(len(S[0,:])):
            value = parameters_forLMFIT[str('S_'+str(i)+str(j))]#['value']
            S[i,j] = value
    return S






def generate_kinetic_model_parameters(model_matrix):
    """
    function to generate paramters for concentration profiles
    takes modelmatrix and generates the parameters needed to describe the physical model

    MODELmatrix must be square !
    -------------------------------
    returns:
        dictionary with all parameters:
            'k_ij' -> kinetic rates parameters
            example k_AB describes rate for A -> B
            
            'S_ij' -> Decay Associated Spectra parameters

    """
    # Check if the model matrix is square
    if np.shape(model_matrix)[0] != np.shape(model_matrix)[1]:
        raise ValueError("Model matrix must be square.")

    print('Model with shape:',np.shape(model_matrix))
    statenumber = int(np.shape(model_matrix)[0])
    
    #generate prameters for fitting and functions
    parameters_forLMFIT = dict()
    statestring = 'ABCDEFGHIJKLMNOP'
    for i in range(statenumber):
        for j in range(statenumber):
            if model_matrix[i,j] != 0:
                parameters_forLMFIT[str('k_'+str(statestring[i])+str(statestring[j]))]= {'value': model_matrix[i,j],'min':0,'vary':True}
    
    for i in range(statenumber):
        for j in range(len(wavenumbers)):
                parameters_forLMFIT[str('S_'+str(i)+str(j))]= {'value': 0,'vary':True}

    
    print('####PARAMETERS GENERATED:')
    print(parameters_forLMFIT)
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
    --------------------------
    returns list of dertivatives for the individual States eg. [dA,dB,...,dN] 
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
                derivative = derivative - params_LMFIT[str('k_'+str(statestring[i])+str(statestring[j]))] * y[i]
            if model_matrix[j,i]!=0:
                 derivative = derivative + params_LMFIT[str('k_'+str(statestring[j])+str(statestring[i]))] * y[j]

        derivative_list.append(derivative)

    return derivative_list

def generate_Concentration_Profiles(t,y0,parameters_forLMFIT,model_matrix):
    """ 
    function to integrate differential equations:
    inputs:
        t:  list/array          of timepoints, (delays)
        y0: list                concentrations at timepoint t=0, [A0,B0,...,N0] mostly [1,0,...,0]
        parameters for LMfit:   Parameteter object of lmfit model
        modelmatrix:            square matrix describing the model
    -----------------------------
    returns:
        list of concentrations profiles [A0,B0,C0
                                         A1,B1,C2
                                         An,Bn,Cn]
    """
    y = sc.integrate.odeint(kinetic_derivatives,y0,t,args=(parameters_forLMFIT,model_matrix))
    return y









def OBJECTIVE_FIT(parameters_forLMFIT,A_matrix_data,y0,t,model_matrix):
    global C , S
    C = generate_Concentration_Profiles(t,y0,parameters_forLMFIT,model_matrix)
    S = calculate_DAS(A_matrix_data,C,parameters_forLMFIT)
    AC = calculate_snthetic_data(C,S)
    residue = residue_DATA_FIT(A_matrix_data,AC)
    objectivetracking.append(np.linalg.norm(residue)**2)

    global parameters_forLMFIT_dict
    parameters_forLMFIT_dict =parameters_forLMFIT.valuesdict()
    for key in parameters_forLMFIT_dict:
        if 'k' in key:
            kparameter.append(parameters_forLMFIT_dict[key])

    return residue


def FIT(data,y0,t,parameters_forLMFIT,model_matrix):
    global result
    mini = lmfit.Minimizer(OBJECTIVE_FIT,params=parameters_forLMFIT,fcn_args=(data,y0,t,model_matrix))
    result = mini.minimize(max_nfev=3000)
    report = lmfit.report_fit(result,show_correl=False)
    print(report)











def plot():
    statestring = 'ABCDEFGHIJKLMNOP'
    fig = plt.figure(0,figsize=(12,8))
    plt.rc('font', size=10) #controls default text size
    plt.rcParams['xtick.major.pad']='1'
    plt.rcParams['ytick.major.pad']='1'
    grid = fig.add_gridspec(3, 3, hspace=0.2, wspace=0.2,bottom=0.05,top=0.95,left=0.05,right=0.95)

    subax1 = fig.add_subplot(grid[0,0])
    subax1.imshow(np.transpose(data), cmap='RdBu_r', aspect='auto', origin='lower',vmin=-1*np.max(np.abs(data)),vmax=1*np.max(np.abs(data)))
    subax1.set_title('Azide data')

    subax2 = fig.add_subplot(grid[0,2])
    fitdata = calculate_snthetic_data(C,S)
    subax2.imshow(np.transpose(fitdata), cmap='RdBu_r', aspect='auto', origin='lower',vmin=-1*np.max(np.abs(data)),vmax=1*np.max(np.abs(data)))
    subax2.set_title('Fitted data')

    svd = fig.add_subplot(grid[0,1])
    svd.imshow(np.transpose(reconstructed_data), cmap='RdBu_r', aspect='auto', origin='lower',vmin=-1*np.max(np.abs(data)),vmax=1*np.max(np.abs(data)))
    svd.set_title('SVD - reconstructed data')

    svdvectors = fig.add_subplot(grid[1,1])
    sng_values = fig.add_subplot(grid[2,1])
    sng_values.set_title('singular Values')
    for v in V_vectors:
        svdvectors.plot(wavenumbers,v)

    sng_values.plot(np.arange(len(Sng)),Sng,marker='o',color='#9467bd')
    sng_values.set_yscale('log')
    sng_values.grid(True)
    
    concentrations = fig.add_subplot(grid[2,2])
    i=0
    for state in np.transpose(C):
        concentrations.plot(timepoints,state,label=str(statestring[i]))
        i=i+1
    concentrations.set_xscale('log')
    concentrations.legend()
    concentrations.set_title('Concentrations')

    subax4 = fig.add_subplot(grid[1,2])
    i = 0
    for signal in S:
        subax4.plot(wavenumbers,signal,label=str(statestring[i]))
        i= i+1
    subax4.legend()
    subax4.set_title('DAS - signals')

    data_to_fit_residue = fig.add_subplot(grid[1,0])
    #data_to_fit_residue.imshow(np.subtract(np.transpose(data),np.transpose(fitdata)), cmap='RdBu_r', aspect='auto', origin='lower',vmin=-1*np.max(np.abs(data)),vmax=1*np.max(np.abs(data)))
    data_to_fit_residue.set_title('residues')
    data_to_fit_residue.plot(np.arange(len(objectivetracking)),objectivetracking,color='#d62728',label=str(f'{objectivetracking[-1]:.3g}'))#red
    data_to_fit_residue.set_yscale('log')
    data_to_fit_residue.legend()


    parameter_evaluation = fig.add_subplot(grid[2,0])
   
    i = 0
    for key in parameters_forLMFIT_dict:
        if 'k' in key:
            i = i+1
    k_param_number = i
    parameter_evaluation.plot(np.arange(len(objectivetracking)),kparameter[0::k_param_number])
    parameter_evaluation.plot(np.arange(len(objectivetracking)),kparameter[1::k_param_number])
    parameter_evaluation.plot(np.arange(len(objectivetracking)),kparameter[2::k_param_number])
    
    plt.show()

















timepoints =    np.load('Setupfiles/Testdata/2024-01-29_A_delays.npy')
wavenumbers =   np.load('Setupfiles/Testdata/2024-01-29_A_wavenumbers.npy')
data =          np.load('Setupfiles/Testdata/2024-01-29_A_matrix.npy')
#data =          np.load('Setupfiles/Testdata/exportedCH2_A_matrix.npy')
pixellow = 0
pixelhigh = 20
firstdelaystep=4
data = np.transpose(data[pixellow:pixelhigh,firstdelaystep:])
wavenumbers = wavenumbers[pixellow:pixelhigh]
timepoints = timepoints[firstdelaystep:]
print(np.shape(data),np.shape(timepoints),np.shape(wavenumbers))

""" model =        np.array([[0, 0.1, 0],
                         [0, 0, 0.01],
                         [0, 0, 0]]) """
y0=[1,0,0,1,0]
#   0,1,2,3,4
model=np.array([[0,  0.1,   0,     0,   0],
                [0,  0,   0.01,   0,   0],
                [0,  0,   0,     0,   0],
                [0,  0,   0,     0,   0.01],
                [0,  0,   0,     0,   0]])


Sng,reconstructed_data,U_vectors,V_vectors = SVD_spectra(data,int(np.shape(model)[0]))
parameters = generate_kinetic_model_parameters(model)
params = lmfit.Parameters()
for key in parameters:
    if 'k' in key:
        params.add(key,value=parameters[key]['value'],min=0,vary=parameters[key]['vary'])
    
    if 'S' in key:
        if 'S_2' in key:
            params.add(key,value=0,vary=False)
        elif 'S_3' in key:
            params.add(key,value=0,vary=False)

        else:
            params.add(key,value=parameters[key]['value'],vary=parameters[key]['vary'])
    


objectivetracking= list()
kparameter = list()
FIT(data,y0,timepoints,params,model)
plot()