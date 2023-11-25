import numpy as np
import scipy
from scipy.signal import butter, filtfilt


# Function to create a high-pass filter
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

# Function to apply the high-pass filter
def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y



class AccIntegration:
    def __init__(self, 
                 time_limit: int = 100,
                 sampling_freq: int = 100, 
                 cuttoff_freq: int = 1,
                 saving_history: int = 100) -> None:
        self.dt = 1/sampling_freq
        self.time_limit = time_limit
        self.sampling_freq = sampling_freq
        self.cuttoff_freq = cuttoff_freq
        self.velocity = np.zeros((3,1))
        self.position = np.zeros((3,1))
        self.save_history = saving_history

    @property
    def _remove_large_data(self):
        self.velocity = self.velocity[:, -self.save_history:]
        self.position = self.position[:, -self.save_history:]

    def predict(self, imu_data: np.ndarray) -> np.ndarray:
        assert imu_data.shape[0] == 3,  "Need to be 3xn array."
        imu_data = imu_data[:, -self.time_limit:]
        imu_data = highpass_filter(imu_data, self.cuttoff_freq, self.sampling_freq, order=5)
        velocity = scipy.integrate.trapezoid(imu_data, dx = self.dt).reshape(3,1)
        self.velocity = np.append(self.velocity, velocity, axis = 1)
        self.velocity = scipy.signal.detrend(self.velocity)
        position = scipy.integrate.trapezoid(self.velocity[:, -self.time_limit:], dx = self.dt).reshape(3,1)
        self.position = np.append(self.position, position, axis = 1)
        # self.position = scipy.signal.detrend(self.position)
        self._remove_large_data
        return self.position[:, -1]
        
        



class GyroIntegration:
    def __init__(self, 
                 time_limit: int = 100,
                 sampling_freq: int = 100, 
                 cuttoff_freq: int = 5,
                 saving_history: int = 100) -> None:
        self.time_limit = time_limit
        self.sampling_freq = sampling_freq
        self.cuttoff_freq = cuttoff_freq
        self.angle = np.zeros((3,1))
        self.save_history = saving_history

    @property
    def _remove_large_data(self):
        self.angle = self.angle[:, -self.save_history:]

    def predict(self, gyro_data: np.ndarray) -> np.ndarray:
        assert gyro_data.shape[0] == 3,  "Need to be 3xn array."
        gyro_data = gyro_data[:, -self.time_limit:]
        gyro_data = highpass_filter(gyro_data, self.cuttoff_freq, self.sampling_freq, order=5)
        gyro_data = scipy.signal.detrend(gyro_data)
        angle = scipy.integrate.trapezoid(gyro_data).reshape(3,1)
        self.angle = np.append(self.angle, angle, axis = 1)
        self.angle = scipy.signal.detrend(self.angle)
        self._remove_large_data
        return self.angle[:, -1]


if __name__ == "__main__":

    # example usage
    AccIntergtation = AccIntergration()
    imu_data = np.random.rand(3,100)
    position = AccIntergtation.predict(imu_data)

    GyroIntergtation = GyroIntegration()
    gyro_data = np.random.rand(3,100)
    angle = GyroIntergtation.predict(gyro_data)
    


