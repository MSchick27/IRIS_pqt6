#generate fake data set

import numpy as np
import matplotlib.pyplot as plt


def generate_sample_data(t_max,num_points):
    # Define the rate constants
    k1 = 0.654 # Rate constant for A to B
    k2 = 0.243  # Rate constant for B to C

    # Define the initial concentrations
    A_0 = 1.0  # Initial concentration of A
    SA = 0.1
    B_0 = 0  # Initial concentration of B
    SB =0.7
    C_0 = 0.0  # Initial concentration of C
    SC =-0.1

    t = np.linspace(0, t_max, num_points)

    # Function to calculate concentrations at each time point
    def kinetic_reaction(t, k1, k2, A_0, B_0, C_0,SA,SB,SC):
        A = A_0 * np.exp(-k1 * t)
        B = (A_0 * k1 / (k1 - k2)) * (np.exp(-k2 * t) - np.exp(-k1 * t))
        C = A_0 - A - B
        A=A*SA
        B=B*SB
        C=C*SC
        return A, B, C

    # Calculate concentrations
    SignalA, SignalB, SignalC = kinetic_reaction(t, k1, k2, A_0, B_0, C_0,SA,SB,SC)
    fullsignal = np.add(SignalA,np.add(SignalB,SignalC))
    # Adding noise
    noise = np.random.normal(0, 0.02, len(fullsignal))  # Mean = 0, Standard deviation = 0.1
    noisy_list = fullsignal + noise

    return noisy_list

""" plt.plot(t,SignalA,label='A')
plt.plot(t,SignalB,label='B')
plt.plot(t,SignalC,label='C')
plt.plot(t,np.add(SignalA,np.add(SignalB,SignalC)),label='sum')
plt.plot(t,noisy_list,label='data')
plt.legend()
plt.show() """

# Print results
