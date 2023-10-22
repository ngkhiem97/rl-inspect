import numpy as np
import matplotlib.pyplot as plt

# Define your array of 10 numbers
data1 = [1, 1.1, 1.2, 1.3, 1.4]
data2 = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

# Compute the FFT
frequencies1 = np.fft.fftfreq(len(data1))
transformed_data1 = np.fft.fft(data1)
frequencies2 = np.fft.fftfreq(len(data2))
transformed_data2 = np.fft.fft(data2)

print('Frequencies 1:', transformed_data1)
print('Frequencies 2:', transformed_data2)

print('Data 1:', np.abs(transformed_data1))
print('Data 2:', np.abs(transformed_data2))

# Plot the magnitude of the frequency domain representation
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.stem(frequencies1, np.abs(transformed_data1), use_line_collection=True)
plt.title('Magnitude Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid()

# Plot the phase of the frequency domain representation
plt.subplot(2, 1, 2)
plt.stem(frequencies2, np.abs(transformed_data2), use_line_collection=True)
plt.title('Phase Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (Radians)')
plt.grid()

plt.tight_layout()
plt.show()
