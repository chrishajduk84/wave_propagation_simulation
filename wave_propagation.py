import numpy as np
import matplotlib.pyplot as plt

from .constants import c, epsilon_0

f = 1e9  # Frequency 1 GHz

# Wave parameters
omega = 2 * np.pi * f  # Angular frequency (rad/s)
k0 = omega / c  # Wave number in vacuum (rad/m)

# Reflection and Transmission Coefficients (complex)
R = 0.5 + 0j  # Reflection coefficient
T = 0.5 + 0j  # Transmission coefficient

# Define the space and time variables
x = np.linspace(-5, 5, 1000)  # Space range (meters)
t = 0  # Time (arbitrary)

# Incident wave (E_i)
E_i = np.exp(1j * (k0 * x - omega * t))

# Reflected wave (E_r)
E_r = R * np.exp(1j * (-k0 * x - omega * t))  # Reflection happens in the opposite direction

# Transmitted wave (E_t)
E_t = T * np.exp(1j * (k0 * x - omega * t))  # Propagating into the second medium

# Total electric field (incident + reflected + transmitted)
#E_total = E_i + E_r

# Plot the real part of the total electric field
plt.figure(figsize=(10, 6))
#plt.plot(x, np.real(E_total), label="Total Electric Field (E_total)", color='b')
plt.plot(x, np.real(E_i), label="Incident Wave (E_i)", color='g', linestyle='--')
plt.plot(x, np.real(E_r), label="Reflected Wave (E_r)", color='r', linestyle='--')
plt.plot(x, np.real(E_t), label="Transmitted Wave (E_t)", color='purple', linestyle='--')
plt.xlabel('Position (m)')
plt.ylabel('Electric Field (V/m)')
plt.title('Wave Propagation with Reflection and Transmission')
plt.legend()
plt.grid(True)
plt.show()

