import numpy as np
import matplotlib.pylab as plt
import padasip as pa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

###################### Example of usage of NLMS filter ######################
# # creation of data
# N = 500
# x = np.random.normal(0, 1, (N, 4)) # input matrix
# v = np.random.normal(0, 0.1, N) # noise
# d = 2*x[:,0] + 0.1*x[:,1] - 4*x[:,2] + 0.5*x[:,3] + v # target

# # identification
# f = pa.filters.FilterNLMS(n=4, mu=0.1, w="random")
# # y, e, w = f.run(d, x)
# # where y is output, e is the error and w is set of parameters at the end of the simulation.
# y, e, w = f.run(d, x) 

# # show results
# plt.figure(figsize=(15,9))
# plt.subplot(211);plt.title("Adaptation");plt.xlabel("samples - k")
# plt.plot(d,"b", label="d - target")
# plt.plot(y,"g", label="y - output");plt.legend()
# plt.subplot(212);plt.title("Filter error");plt.xlabel("samples - k")
# plt.plot(10*np.log10(e**2),"r", label="e - error [dB]");plt.legend()
# plt.tight_layout()
# plt.show()

def data_tap(x, M):
  '''
  Genera la matriz de taps MxN
  '''
  N = x.size
  x = np.array(x).reshape(N)
  x_M = np.zeros((N,M))
  
  # Zero-padding
  for m in range(M):
    x_M[m:,m] = x[:N-m]

  return x_M


fs = 1000  # Sampling frequency in Hz
t = np.arange(0, 10, 1/fs)  # 10 seconds duration

# --- EEG base signal ---
eeg = (50*np.sin(2*np.pi*10*t) +    # alpha 10 Hz
       30*np.sin(2*np.pi*4*t) +     # theta 4 Hz
       80*np.sin(2*np.pi*20*t))     # beta 20 Hz

# --- EMG noise (band-limited white noise 20-300 Hz) ---
emg_noise = np.random.randn(len(t))*200
b, a = butter(4, [20/(fs/2), 300/(fs/2)], btype='band')
emg = filtfilt(b, a, emg_noise)

# --- EOG noise (low freq 0-10 Hz) ---
eog_noise = np.random.randn(len(t))*100
b, a = butter(4, [0.1/(fs/2), 10/(fs/2)], btype='band')
eog = filtfilt(b, a, eog_noise)

# --- Total signal ---
total_signal = eeg + emg + eog

# --- Plot ---
plt.figure(figsize=(12,6))
plt.plot(t, total_signal, label='EEG + EMG + EOG', linewidth=0.8)
plt.plot(t, eeg, label='Clean EEG', alpha=0.6)
plt.xlim(0, 2)  # zoom on 2 seconds
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [µV]')
plt.legend()
plt.show()

M = 128 # Filterlength M
mu = 0.0000001 # Learning rate/step size
fs = 256 # Sampling frequency
s = 1000 # Signal duration in seconds

N = fs*s
t = np.arange(N)

### Example signals ###
eog_ref = np.random.normal(0, 0.1, N) 
emg_ref = np.random.normal(0, 0.1, N)
eeg = np.sin(2*np.pi*10*t/fs) + 0.5*np.random.normal(0, 1, N) + 0.5*np.convolve(eog_ref, np.ones(50)/50, mode='same')

### 1. EOG Filtering with LMS ###
x_1 = eog_ref 
x_1_tap = data_tap(x_1, M)
d_1 = eeg
# Create LMS-filter
f1 = pa.filters.FilterLMS(n = M, mu = mu, w='zeros') 
y1, e1, w1 = f1.run(d_1,x_1_tap)

### 2. EMG Filtering ###
x2 = emg_ref                # Reference Data: EMG
x2_M = data_tap(x2, M)
# Create NLMS-filter (NLMS is used at the end because it adapts more robustly to varying noise levels (typical in ECG signals))
f2 = pa.filters.FilterLMS(n = M, mu = mu, w='zeros')
y2, e2, w2 = f2.run(e1,x2_M)

# Questions: 
# Why same M and mu? Wouldnt it be better to have different values for each noise source?
# Different correlation/length:
# EOG is slow and correlated over many samples → needs longer tap length (M) to model baseline drift and slow eye-movement artifacts.
# EMG is high-frequency, localized bursts → shorter M often suffices.
# Its offline right now!