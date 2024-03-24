from math import ceil
import torch
import numpy as np
from torch.nn.functional import softmax


def reshape_signal(signal, freq, num_windows=1200, windows_size=128, overlap=98, segment_step=240, remove_mean=False):
    """
    Reshape a signal whose shape is (n * freq) to (num_windows, windows_size)

    If the signal's size is smaller than num_windows * (windows_size - overlap) + overlap,
    then the signal will be padded with zeros.

    If the signal's size is larger than num_windows * (windows_size - overlap) + overlap,
    then the signal will be splitted into multiple segments with step = 1 hour

    Parameters
    ----------
    signal
        signal
    freq
        sampling frequency
    num_windows
        number of windows
    windows_size
        size of each window / in seconds
    overlap
        overlap between two windows / in seconds
    segment_step
        step between two segments / in windows
    """
    if remove_mean:
        signal = signal - signal.mean()
    step = windows_size - overlap
    left = (windows_size - step)
    num_droped = step - left % step
    if num_droped != 0 and num_droped != 30:
        signal = signal[int(num_droped * freq): - int(num_droped * freq)]
    signal = signal.unfold(0, int(windows_size * freq), int(step * freq))
    if signal.shape[0] < num_windows:
        zero_start = signal.shape[0]
        signal = torch.cat([signal, torch.zeros(num_windows - signal.shape[0],
                                                int(windows_size * freq), 
                                                device=signal.device) - 1], dim=0)
        return signal.reshape(1, signal.shape[0], signal.shape[1]), zero_start + 1
    else:
        zero_start = -1
        signals = []
        segment_overlap = num_windows - segment_step
        num_segments = (signal.shape[0] - segment_overlap) / segment_step
        for i in range(ceil(num_segments)):
            if i * segment_step + num_windows > signal.shape[0]:
                zero_start = signal.shape[0] - i * segment_step
                signals.append(torch.cat([signal[i * segment_step:],
                                          torch.zeros(num_windows - zero_start,
                                                      int(windows_size * freq),
                                                      device=signal.device) - 1],
                                         dim=0))

            else:
                signals.append(signal[i * segment_step: i * segment_step + num_windows])

        return torch.stack(signals), zero_start


def num_segs(signal, freq, num_windows=1200, windows_size=128, overlap=98, segment_step=240):
    step = windows_size - overlap
    signal = signal.unfold(0, int(windows_size * freq), int(step * freq))
    segment_overlap = num_windows - segment_step
    num_segments = (signal.shape[0] - segment_overlap) / segment_step
    return ceil(num_segments)


def SleepDataset(h5file, features=['rri', 'mad'],
                 target='stage', pid=None, remove_mean={'rri': True, 'mad': False},
                 offset={'rri':None, 'mad':None},
                 batchsize=16, shuffle=False,
                 **reshape_kwargs):
    if pid is None:
        pid = list(h5file[target].keys())
    if shuffle:
        import random
        random.shuffle(pid)
    for i in range(ceil(len(pid) / batchsize)):
        signals = {}
        for feature in features:
            try:
                rm = remove_mean['rri']
            except KeyError:
                rm = False
            offset_signal = 0
            if offset[feature] is not None:
                offset_signal = np.random.uniform(*offset[feature])
            signals[feature] = [
                reshape_signal(torch.from_numpy(h5file[feature][p][:] + offset_signal),
                               h5file[feature].attrs['freq'], remove_mean=rm,
                               **reshape_kwargs) for p in pid[i * batchsize: (i + 1) * batchsize]]
            seg_lengths = [len(s[0]) for s in signals[feature]]
            zero_starts = [s[1] for s in signals[feature]]
            signals[feature] = [s[0] for s in signals[feature]]
            signals[feature] = torch.cat(signals[feature], dim=0)
        targets = [reshape_signal(torch.from_numpy(h5file[target][p][:]), 1 / 30,
                                  windows_size=150, overlap=120)[0] for p in pid[i * batchsize: (i + 1) * batchsize]]
        targets = torch.cat(targets, dim=0)[:, :, 2]
        yield signals, targets, seg_lengths, zero_starts


def predict_stage(net, rri, mad=None, out_prob=False):
    """Predict sleep stage from a neural nework"""
    net.eval()
    reshaped_rri, zero_start = reshape_signal(rri, 4)
    if mad is not None:
        reshaped_mad, zero_start = reshape_signal(mad, 1)
        reshaped_mad = reshaped_mad
        if reshaped_mad.shape[0] > reshaped_rri.shape[0]:
            shape_diff = reshaped_mad.shape[0] - reshaped_rri.shape[0]
            reshaped_rri = torch.cat([reshaped_rri,
                                      torch.zeros(shape_diff, reshaped_rri.shape[1], reshaped_rri.shape[2],
                                                  device=reshaped_rri.device, dtype=reshaped_rri.dtype)])
    else:
        reshaped_mad = None
    reshaped_rri = reshaped_rri

    with torch.no_grad():
        predict_score = net(reshaped_rri, reshaped_mad).reshape(reshaped_rri.shape[0], 
                                                                reshaped_rri.shape[1], -1)
    result = []
    predict_result = torch.argmax(predict_score, dim=-1)


    if out_prob:
        predict_prob = softmax(predict_score, dim=-1)
        result_prob = []
    for i in range(predict_result.shape[0]):
        if i == predict_result.shape[0] - 1:
            if i != 0:
                result.append(predict_result[i][-240:zero_start])
            else:
                result.append(predict_result[i][:zero_start])
        elif i == 0:
            result.append(predict_result[i])
        else:
            result.append(predict_result[i][-240:])
        if out_prob:
            if i == predict_result.shape[0] - 1:
                if i != 0:
                    result_prob.append(predict_prob[i][-240:zero_start])
                else:
                    result_prob.append(predict_prob[i][:zero_start])
            elif i == 0:
                result_prob.append(predict_prob[i])
            else:
                result_prob.append(predict_prob[i][-240:])

    if out_prob:
        return torch.cat(result), torch.cat(result_prob)
    return torch.cat(result)
