import numpy as np
import matplotlib.pyplot as plt

# Define your two arrays
data1 = [1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3]
data2 = [1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 1, 3]
data1 = np.pad(data1, (0, len(data2) - len(data1)))

# Compute the FFT for both datasets
transformed_data1 = np.fft.fft(data1)
transformed_data2 = np.fft.fft(data2)

# Compute the frequency bins for both datasets
frequencies1 = np.fft.fftfreq(len(data1))
frequencies2 = np.fft.fftfreq(len(data2))

# Calculate the differences (will need to pad the shorter array for direct subtraction)
magnitude_difference = np.abs(transformed_data2) - np.abs(transformed_data1)
phase_difference = np.angle(transformed_data2) - np.angle(transformed_data1)

# Plot the results
plt.figure(figsize=(12, 8))

# Magnitude for data1
plt.subplot(3, 2, 1)
plt.stem(frequencies1, np.abs(transformed_data1), use_line_collection=True)
plt.title('Magnitude Spectrum of Data1')
plt.grid()

# Magnitude for data2
plt.subplot(3, 2, 2)
plt.stem(frequencies2, np.abs(transformed_data2), use_line_collection=True)
plt.title('Magnitude Spectrum of Data2')
plt.grid()

# Phase for data1
plt.subplot(3, 2, 3)
plt.stem(frequencies1, np.angle(transformed_data1), use_line_collection=True)
plt.title('Phase Spectrum of Data1')
plt.grid()

# Phase for data2
plt.subplot(3, 2, 4)
plt.stem(frequencies2, np.angle(transformed_data2), use_line_collection=True)
plt.title('Phase Spectrum of Data2')
plt.grid()

# Difference in magnitude
plt.subplot(3, 2, 5)
plt.stem(frequencies2, magnitude_difference, use_line_collection=True)
plt.title('Difference in Magnitude Spectrum')
plt.grid()

# Difference in phase
plt.subplot(3, 2, 6)
plt.stem(frequencies2, phase_difference, use_line_collection=True)
plt.title('Difference in Phase Spectrum')
plt.grid()

plt.tight_layout()
plt.show()
