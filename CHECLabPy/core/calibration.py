import numpy as np
from os import environ
from os.path import join, exists


def get_auto_tf_paths(reader):
    if "TC_CONFIG_PATH" not in environ:
        raise OSError("TC_CONFIG_PATH not defined in environment")
    base = environ["TC_CONFIG_PATH"]
    paths = [""]*len(reader.n_modules)
    for tm in range(reader.n_modules):
        sn = reader.get_sn(tm)
        if sn >= 0:
            tf_path = join(base, "SN{:04}_tf.tcal".format(sn))
            if not exists(tf_path):
                print("WARNING TF path does not exist: {}".format(tf_path))
            paths[tm] = tf_path
    return paths


class Calibrator:
    def __init__(self, pedestal_path, tf_path=""):
        """
        Apply the low-level calibration to the waveforms
        """
        try:
            from target_calib import Calibrator, CalibratorMultiFile
        except ModuleNotFoundError:
            msg = ("Cannot find target_calib module, please follow "
                   "installation instructions from https://forge.in2p3.fr/"
                   "projects/gct/wiki/Installing_CHEC_Software")
            raise ModuleNotFoundError(msg)

        self._calibrated = None

        if pedestal_path:
            if isinstance(tf_path, list):
                self.calibrator = CalibratorMultiFile(pedestal_path, tf_path)
            self.calibrator = Calibrator(pedestal_path, tf_path)
            self.calibrate = self.real_calibrate
        else:
            print("WARNING No pedestal path supplied, "
                  "r1 samples will equal r0 samples.")
            self.calibrate = self.fake_calibrate

    def fake_calibrate(self, waveforms, fci):
        """
        Don't perform any calibration on the waveforms, just return the
        waveforms.

        Parameters
        ----------
        waveforms : ndarray
        fci : ndarray
            First cell id for each pixel
        """
        return waveforms.astype(np.float32)

    def real_calibrate(self, waveforms, fci):
        """
        Apply the TargetCalib calibration to the waveforms

        Parameters
        ----------
        waveforms : ndarray
        fci : ndarray
            First cell id for each pixel
        """
        if self._calibrated is None:
            self._calibrated = np.zeros(waveforms.shape, dtype=np.float32)

        self.calibrator.ApplyEvent(waveforms, fci, self._calibrated)
        return self._calibrated
