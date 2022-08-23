from math import ceil
import torch
import numpy as np
from torch.utils.data import Dataset


def fill_zeros(signal, slp, freq=4, epoch_length=128, normalize=True):
    if normalize:
        signal = signal - np.mean(signal)
        signal = signal / np.std(signal)
    filled_length = 30 * 1200 + epoch_length - 30
    real_length = signal.size // freq
    half_left = (epoch_length - 30) // 2
    if real_length > filled_length or slp.size >= 1200:
        half_left_i = ceil(half_left)
        left = half_left_i * 30 - half_left
        signal = signal[half_left_i * freq: (left + filled_length) * freq]
        slp = slp[half_left_i: half_left_i + 1200]
        if signal.size == filled_length and slp.size == 1200:
            return signal, slp, (0, 0)
        else:
            signal = signal[half_left_i * freq:]
            num_slp = min(slp.size, signal.size // freq // 30)
            signal = signal[:num_slp * 30 * freq]
            slp = slp[:num_slp]
            real_length = signal.size // freq

    elif filled_length - real_length < epoch_length:
        real_length = signal.size // freq
    filled_signal = np.zeros(filled_length * freq, dtype=np.float32)
    start_i = (filled_length - real_length) // 2
    start_nepoch = (start_i - half_left) // 30
    start_i = start_nepoch * 30 + half_left
    filled_signal[start_i * freq: start_i * freq + real_length * freq] = signal

    filled_slp = np.zeros(1200, dtype=np.int64)
    filled_slp[start_nepoch: start_nepoch + slp.size] = slp

    return filled_signal, filled_slp, (start_nepoch, start_nepoch + slp.size)


def reshape_tensor(signal, slp, freq=4, device='cpu', epoch_length=128):
    signal, slp, start_end = fill_zeros(signal, slp, freq, epoch_length)
    signal = torch.from_numpy(signal).float()
    slp = torch.from_numpy(slp)
    signal = signal.unfold(0, epoch_length * freq, 30 * freq)
    return signal.to(device), slp.to(device), start_end


class dataset_cnn(Dataset):
    """Dataset class for reading data from h5 file
    The h5 file should be organized as follows:
    /rri: rri signals of all patients
    /mad: mad signals of all patients (optional)
    /slp: sleep stage labels of all patients
    Parameters
    ----------
    h5file: h5py.File object
        h5 file containing the data
    patient_id: List 
        list of patient id you want to read, by default None, which means all patients
    signals: dict 
        dict of signals you want to read, the values are signals' frequency, by default {'rri': 4}
    epoch_length: int
        length of each epoch, by default 128
    device: str or torch.device, by default 'cpu'
    """
    def __init__(self, h5file, patient_id=None, signals={'rri': 4}, epoch_length=128, device='cpu'):
        self.h5file = h5file
        self.signals = signals
        self.epoch_length = epoch_length
        self.device = device
        if patient_id is None:
            self.patient_id = list(self.h5file['slp'].keys())
        else:
            self.patient_id = patient_id

    def __len__(self):
        return len(self.patient_id)

    def __getitem__(self, idx):
        patient = self.pid[idx]
        signals = []
        for key in self.signals:
            signal = self.h5file[key][patient]
            freq = signals[key]
            slp = self.h5file['slp'][patient]
            signal, slp, _ = reshape_tensor(signal, slp, freq, self.device, self.epoch_length)
            signals.append(signal)
        return signals, slp


if __name__ == "__main__":
    test_rri = np.random.rand(3600 * 7)
    test_slp = np.random.randint(0, 4, 120 * 7)
    reshaped_rri, reshaped_slp, _ = reshape_tensor(test_rri, test_slp, 1, 'cpu', 30)
    print(reshaped_rri.shape)
