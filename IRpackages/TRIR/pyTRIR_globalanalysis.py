
""" 
backend package for globalanalysis guy to calculate and fit data

 """
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import lmfit




class pyglobsis():
    def globalanalysis(self,model,y0,k_or_tau,data_array,delays,wavenumbers,zerospectra_components):
        """ 
        model
         k or tau
        data_array
        delays
        wavenumbers
        zerospectra_components:    list() of components where the spectra should be a flatline [3] -> component 3 spectra wont vary
        """
        if np.shape(model)[0] != np.shape(model)[1]:
            raise ValueError("Model matrix must be square.")
        components = np.shape(model)[0]
        self.S_svd,reconstructed_data,self.U_vectors,self.V_vectors = pyglobsis.SVD_spectra(data_array,components)
        ALL_parameters = pyglobsis.generate_kinetic_model_parameters(model,wavenumbers,components,zerospectra_components)

        self.objectivetracking= list()
        self.kparameter = list()
        pyglobsis.FIT(self,reconstructed_data,y0,delays,ALL_parameters,model)
        fitdata = pyglobsis.calculate_snthetic_data(self.C,self.S)
        
        return fitdata #something for now pls change




    def SVD_spectra(self,data,num_components_to_keep):
        """
        Function to calculate the Singular Value Decomposition of a spectra 
        returns 
        -S-Matrix to plot the Singular Values and show their priority #plt.semilogy(S, marker='o', linestyle='-')
        -reconstructed data in same format as input data restricted to the num of components
        -U_vectors 
        -V-Vectors
        example:
        S,rdata,Uv,Vv = SVD_spectra(array[:,:],3)

        is used to get a noise reduced data set for fitting as well as to see how many components contribute to the spectrum
        """
        U, S, V = np.linalg.svd(data, full_matrices=False)
        reconstructed_data = np.dot(U[:, :num_components_to_keep], np.dot(np.diag(S[:num_components_to_keep]), V[:num_components_to_keep, :]))
        U_vectors = []
        V_vectors = []
        for i in range(num_components_to_keep):
            U_vectors.append(U[:,i])
            V_vectors.append(V[i,:])

        return S,reconstructed_data,U_vectors,V_vectors


    def generate_kinetic_model_parameters(model_matrix,wavenumbers,components,zerospectra_components):
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
        #generate prameters for fitting and functions
        parameters_forLMFIT = dict()
        statestring = 'ABCDEFGHIJKLMNOP'
        for i in range(components):
            for j in range(components):
                if model_matrix[i,j] != 0:
                    parameters_forLMFIT[str('k_'+str(statestring[i])+str(statestring[j]))]= {'value': model_matrix[i,j],'min':0,'vary':True}
    
        for i in range(components):
            for j in range(len(wavenumbers)):
                    parameters_forLMFIT[str('S_'+str(i)+str(j))]= {'value': 0,'vary':True}

        print('####PARAMETERS GENERATED:')

        params = lmfit.Parameters()
        for key in parameters_forLMFIT:
            if 'k' in key:
                params.add(key,value=parameters_forLMFIT[key]['value'],min=0,vary=parameters_forLMFIT[key]['vary'])

            if 'S' in key:
                params.add(key,value=parameters_forLMFIT[key]['value'],vary=parameters_forLMFIT[key]['vary'])
                #set parameters vary False for Spectralcomponents in zerocomponents
                for comp in zerospectra_components:
                    params[str('S_'+str(comp))].set(0, vary=False)


        return params














    def FIT(self,data,y0,t,parameters_forLMFIT,model_matrix):
        mini = lmfit.Minimizer(pyglobsis.OBJECTIVE_FIT,params=parameters_forLMFIT,fcn_args=(data,y0,t,model_matrix))
        self.result = mini.minimize(max_nfev=3000)
        report = lmfit.report_fit(self.result,show_correl=False)
        print(report)

    def OBJECTIVE_FIT(self,parameters_forLMFIT,A_matrix_data,y0,t,model_matrix):
        self.C = pyglobsis.generate_Concentration_Profiles(t,y0,parameters_forLMFIT,model_matrix)
        self.S = pyglobsis.calculate_DAS(self,A_matrix_data,self.C,parameters_forLMFIT)
        AC = pyglobsis.calculate_snthetic_data(self.C,self.S)
        residue = pyglobsis.residue_DATA_FIT(A_matrix_data,AC)
        self.objectivetracking.append(np.linalg.norm(residue)**2)

        global parameters_forLMFIT_dict
        parameters_forLMFIT_dict =parameters_forLMFIT.valuesdict()
        for key in parameters_forLMFIT_dict:
            if 'k' in key:
                self.kparameter.append(parameters_forLMFIT_dict[key])

        return residue






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
        y = sc.integrate.odeint(pyglobsis.kinetic_derivatives,y0,t,args=(parameters_forLMFIT,model_matrix))
        return y

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




    def calculate_DAS(self,A, C,parameters_forLMFIT):
        """ def objective_function(S_flat, A, C):
        Objective function to minimize ||A - C*S||^2.
        Was the first idea to get S after i just put the S-parameters in the lmfit
        S = S_flat.reshape(C.shape[1], A.shape[1])  # Reshape the flattened S to its original shape
        return np.linalg.norm(A - np.dot(C, S))**2 """
    
        def translate_sparamatrix(parameters_forLMFIT):
            S = np.zeros_like(self.V_vectors)
            for i in range(len(S[:,0])):
                for j in range(len(S[0,:])):
                    value = parameters_forLMFIT[str('S_'+str(i)+str(j))]#['value']
                    S[i,j] = value
            return S

        S = translate_sparamatrix(parameters_forLMFIT)
        return S



    def calculate_snthetic_data(C_matrix,S_matrix):
        AC = np.dot(C_matrix,S_matrix)
        return AC

    def residue_DATA_FIT(A_matrix_data,AC_fitmatrix):
        #Calculate residue of DATA(A) and FITDATA(AC)
        DIFF = np.subtract(A_matrix_data,AC_fitmatrix)
        #residue = np.linalg.norm(DIFF)**2
        residue = DIFF
        return residue
