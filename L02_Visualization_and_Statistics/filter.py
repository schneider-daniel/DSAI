from scipy import signal

def filter_butterworth(x, order=4, cut_off_freq=0.2, sample_rate=100.0):
    """
    Returns a filtered signal using butterworth filter. \n
    :param x: signal to filter
    :type x: ndarray
    :param order: order of the filter
    :type order: int
    :param cut_off_freq: cut-off frequency of the filter
    :type cut_off_freq: float
    :param sample_rate: sample rate of the signal
    :type sample_rate: float
    :return: filtered signal
    :rtype: ndarray
    """
    a, b = signal.butter(order, cut_off_freq, 'low', fs=sample_rate)
    return signal.filtfilt(a, b, x.copy())