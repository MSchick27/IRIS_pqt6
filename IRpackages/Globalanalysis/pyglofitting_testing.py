
""" # Generate some example data
meas_data=[0.01781127, 0.00979093, 0.08709667,0.05138541, 0.06513939, 0.05224452,
 0.06324926, 0.11801531, 0.13707861, 0.13081697, 0.19798488, 0.18667978,
 0.22981461, 0.25921794, 0.30035197, 0.33356156, 0.31199069, 0.26298673,
 0.28073635, 0.19859076, 0.18277059, 0.15540159, 0.17053204, 0.11840765,
 0.12204991, 0.1787067,  0.13233532, 0.11169839, 0.0877237 , 0.08500605,
 0.08580032, 0.1320062,  0.09933906, 0.09053057, 0.0827318 , 0.0786071 ] """

#generate fake data
from test import generate_sample_data

meas_data = generate_sample_data(100,100)



import matplotlib.pyplot as plt
from lmfit import Model, Parameters
import lmfit
import pandas as pd
import scipy as sc
import numpy as np

model_simple = np.array([[0, 1, 0],
                         [0, 0, 0.1],
                         [0, 0, 0]])



def generate_kinetic_model_parameters(model_matrix,c0,s0):
    """Calculates the derivatives of the concentrations from a given model matrix

    inputs for the function are:
        -MODEL MATRIX: must be square for the global analysis to run
        example: for a reaction like A -> B -> C
          A B C
        A 0 1 0         
        B 0 0 0.1
        C 0 0 0

        -c0: list of values for the initial concentrations of the states
        example: c0=[1,0,0]

        -s0: list of initial guesses for the signal of the states (later coming from SVD)
    """
    # Check if the model matrix is square
    if np.shape(model_matrix)[0] != np.shape(model_matrix)[1]:
        raise ValueError("Model matrix must be square.")

    print('Model with shape:',np.shape(model_matrix))
    statenumber = int(np.shape(model_matrix)[0])
    
    #generate prameters for fitting and functions
    parameters_forLMFIT = dict()
    s_parameters = dict()
    k_parameters = dict()
    statestring = 'ABCDEFGHIJKLMNOP'
    for i in range(statenumber):
        s_parameters[str('S_'+statestring[i])] = {'value': s0[i], 'min':None,'vary':True}
        parameters_forLMFIT[str('S_'+statestring[i])] = {'value': s0[i], 'min':None,'vary':True}
        for j in range(statenumber):
            if model_matrix[i,j] != 0:
                k_parameters[str('k_'+str(statestring[i])+str(statestring[j]))]= {'value': model_matrix[i,j],'min':0,'vary':True}
                parameters_forLMFIT[str('k_'+str(statestring[i])+str(statestring[j]))]= {'value': model_matrix[i,j],'min':0,'vary':True}
    

    print('####PARAMETERS GENERATED:')#,'\n',c_parameters,'\n',s_parameters,'\n',k_parameters)
    print('ALL parameters saved in parameters for LMFIT dictionary')
    return parameters_forLMFIT
    




def kinetic_derivatives(y,t,params_LMFIT):
    """ 
    build system of ODES to solve later
    -MODEL MATRIX: must be square for the global analysis to run
        example: for a reaction like A -> B -> C
          A B C
        A 0 1 0         
        B 0 0 0.1
        C 0 0 0
    following derivatives should result
        dA = -k_AB * y[0]*s[A]
        dB = -k_BC * y[1]*s[B] + k_AB * y[0]*s[A]
        dC =  k_BC * y[1]*s[B]
          """
    model_matrix = model_simple
    statenumber= np.shape(model_matrix)[0]
    statestring = 'ABCDEFGHIJKLMNOP'
    derivative_list= []
    
    for i in range(statenumber):
        derivative = 0
        for j in range(statenumber):
            if model_matrix[i,j]!=0:
                derivative = derivative - params_LMFIT[str('k_'+str(statestring[i])*str(statestring[j]))]['value'] * y[i]
            if model_matrix[j,i]!=0:
                 derivative = derivative + params_LMFIT[str('k_'+str(statestring[j])*str(statestring[i]))]['value'] * y[j]

        derivative_list.append(derivative)

    return derivative_list

def solve_ODE_spectralresponse(t,y0,parameters_forLMFIT):
    y = sc.integrate.odeint(kinetic_derivatives,y0,t,args=(parameters_forLMFIT,))
    statestring='ABCDEFGHIJKLMNOP'
    model=y
    model = np.transpose(model)
    spec_response = np.zeros(len(model[0,:]))
    for i in range(len(model[:,0])):
        c = model[i,:] * parameters_forLMFIT[str('S_'+str(statestring[i]))]
        spec_response = np.add(spec_response,c)
    return spec_response

def residuals(parameters_forLMFIT,t,data):
    statestring='ABCDEFGHIJKLMNOP'
    y0 = [1,0,0]
    spec_response = solve_ODE_spectralresponse(t,y0,parameters_forLMFIT)
    return spec_response-meas_data





t = np.arange(100)
y0=[1,0,0]

parameters = generate_kinetic_model_parameters(model_matrix=model_simple,c0=[1,0,0],s0=[0.1,0.5,-0.1])
print(parameters)

params = lmfit.Parameters(parameters)
for key in parameters:
    params.add(key,value=parameters[key]['value'],min=parameters[key]['min'],vary=parameters[key]['vary'])

result = lmfit.minimize(residuals,params,args=(t,meas_data))
report = lmfit.report_fit(result)
print(report)



fig = plt.figure(0,figsize=(10,4))
plt.rc('font', size=10) #controls default text size
plt.rcParams['xtick.major.pad']='1'
plt.rcParams['ytick.major.pad']='1'
grid = fig.add_gridspec(1, 2, hspace=0.1, wspace=0.1,bottom=0.09,top=0.95,left=0.05,right=0.95)
subax1 = fig.add_subplot(grid[0,0])
subax2 = fig.add_subplot(grid[0,1])

fitted_data = solve_ODE_spectralresponse(t,y0,result.params)
subax1.plot(t,meas_data)
subax1.plot(t,fitted_data)

def solve_ODE_for_concentrations(t,y0,parameters_forLMFIT):
    y = sc.integrate.odeint(kinetic_derivatives,y0,t,args=(parameters_forLMFIT,))
    return y

concentrations = solve_ODE_for_concentrations(t,y0,result.params)
for states in np.transpose(concentrations):
    subax2.plot(t,states)

plt.show()
    



















