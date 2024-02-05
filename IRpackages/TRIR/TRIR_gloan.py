import numpy as np
import matplotlib.pyplot as plt

# Generate a synthetic 2D transient spectrum dataset (replace with your data)
# Here, we create a 2D array as an example
timepoints = np.load('Setupfiles/Testdata/2024-01-29_A_delays.npy')
wavenumbers = np.load('Setupfiles/Testdata/2024-01-29_A_wavenumbers.npy')
data = np.load('Setupfiles/Testdata/2024-01-29_A_matrix.npy')




pixellow = 7
pixelhigh = 30
firstdelaystep=8


data = data[pixellow:pixelhigh,firstdelaystep:]
wavenumbers = wavenumbers[pixellow:pixelhigh]
timepoints = timepoints[firstdelaystep:]


# Perform Singular Value Decomposition (SVD)
U, S, Vt = np.linalg.svd(data, full_matrices=False)
# Plot the singular values to assess the dimensionality
plt.figure(figsize=(8, 4))
plt.semilogy(S, marker='o', linestyle='-')
plt.title('Singular Values')
plt.xlabel('Component')
plt.ylabel('Singular Value')
plt.grid(True)
plt.show()




# Determine the number of components to keep based on the singular values
# You can use a threshold or visually inspect the singular values plot to decide
# In this example, we keep the top 10 components
num_components_to_keep = 1



# Reconstruct the dataset using the selected components
reconstructed_data = np.dot(U[:, :num_components_to_keep], np.dot(np.diag(S[:num_components_to_keep]), Vt[:num_components_to_keep, :]))
print(S[:num_components_to_keep])

# Plot the original and reconstructed 2D transient spectra
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(data, cmap='RdBu_r', aspect='auto', origin='lower',vmin=-1*np.max(np.abs(data)),vmax=1*np.max(np.abs(data)))
plt.title('Original 2D Transient Spectrum')
plt.xlabel('time')
plt.ylabel('wavenumber')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_data, cmap='RdBu_r', aspect='auto', origin='lower',vmin=-1*np.max(np.abs(data)),vmax=1*np.max(np.abs(data)))
plt.title('Reconstructed 2D Transient Spectrum')
plt.xlabel('time')
plt.ylabel('wavenumber')
plt.colorbar()

plt.tight_layout()
plt.show()


#now lets get the vectors from svd
""" plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for i in range(len(U[0,:num_components_to_keep])):
    plt.plot(wavenumbers,U[:,i])
plt.title('left vectors')
plt.xlabel('spectral role')
plt.ylabel('pixels')

plt.subplot(1, 2, 2)
for i in range(len(Vt[:num_components_to_keep,0])):
    plt.plot(timepoints,Vt[i,:])
plt.title('right vectors')
plt.xlabel('spectral role')
plt.ylabel('pixels')
plt.xscale('log')

plt.tight_layout()
plt.show() """









def model():

    return 1


def addpixel():
    print('pixel added')




















#to fit a spectra to the TRIR data. Let*s set a 

""" import numpy as np
import matplotlib.pyplot as plt
import pyDEsolver as pyDE


#a simple sequnetiell model to fit
def three_state_reaction(t,k1, k2, A, B, C):
    dA= -A*k1
    dB= +dA - B*k2
    dC= B*k2
    


def function(y,k1=1,k2=.1):
    return np.transpose([-y[0]*k1 ,  y[0]*k1-y[1]*k2,  y[1]*k2])

t= np.arange(0,20,0.1)
y = pyDE.ODEsolver.RK4_method(f=function,y0=[100,0,0],t=t)

plt.plot(t,y[0],label='A')
plt.plot(t,y[1],label='B')
plt.plot(t,y[2],label='C')
plt.legend()
plt.show() """



""" 
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the matrix representation of the kinetic model
k_AB = 1  # Rate constant for A -> B
k_BC = 0.1  # Rate constant for B -> C
kinetic_matrix = np.array([[0, k_AB, 0], [0, 0, k_BC], [0, 0, 0]])

# Define the initial concentrations of A, B, and C
initial_conditions = np.transpose(np.array([1.0, 0.0, 0.0]))  # [A0, B0, C0]

# Define the time points for integration
t_span = (0, 10)  # Integration time span

# Define a function that describes the kinetic model
def kinetic_model(t, concentrations):
    dCdt = np.dot(kinetic_matrix, concentrations)
    return dCdt

# Integrate the kinetic model using scipy's solve_ivp
solution = solve_ivp(kinetic_model, t_span, initial_conditions, t_eval=np.linspace(0, 10, 100))

# Plot the concentrations of A, B, and C over time
plt.figure(figsize=(8, 6))
plt.plot(solution.t, solution.y[0], label='A')
plt.plot(solution.t, solution.y[1], label='B')
plt.plot(solution.t, solution.y[2], label='C')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend() """
plt.show()




