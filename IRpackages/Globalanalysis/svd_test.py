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
num_components_to_keep = 5



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
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for i in range(len(U[0,:num_components_to_keep])):
    plt.plot(wavenumbers,U[:,i],label=str(i))
plt.title('left vectors')
plt.xlabel('spectral role')
plt.ylabel('pixels')
plt.legend()

plt.subplot(1, 2, 2)
for i in range(len(Vt[:num_components_to_keep,0])):
    plt.plot(timepoints,Vt[i,:],label=str(i))
plt.title('right vectors')
plt.xlabel('spectral role')
plt.ylabel('pixels')
plt.xscale('log')
plt.legend()

plt.tight_layout()
plt.show()

print(Vt[0,:])





















