import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Constant Parameters setup
COMPRESSED_DIR = "./compressed"
CACHE_DIR = "./cache"

CUTS_NUMBER = 315

TRAIN_DIR_NAME = ["c1", "c4", "c6"]
TEST_DIR_NAME = ["c2", "c3", "c5"]

TRAIN_DIR_PATH = [os.path.join(COMPRESSED_DIR, dirName) for dirName in TRAIN_DIR_NAME]
TEST_DIR_PATH = [os.path.join(COMPRESSED_DIR, dirName) for dirName in TEST_DIR_NAME]

# Allowable subsampling rate : 144.23
# 50000 / (10400/60 * 2) = 144.23
RESAMPLING_RATE = 144 // 2

MAX_OUTPUT_LENGTH = 2048


class Data:
    def __init__(self):
        self.run_initial_setup()

    def run_initial_setup(self):
        """
        run initialization for next data retrieve
        :return: void
        """
        print("Run initialization")
        if not os.path.exists(CACHE_DIR):
            print("Can't find a cache directory", CACHE_DIR, "and try to make a new folder")
            os.mkdir(CACHE_DIR)

    def get_test_data_by_path(self, cutIdx: int):
        """
        get PHM 2010 data by directory path
        :param cutIdx: The milling index
        :return: np.array
        """
        cache_data_path = os.path.join(CACHE_DIR, "cache_cut_%d.npy" % cutIdx)
        if os.path.exists(cache_data_path):
            print("Get data from cache")
            return np.load(cache_data_path)

        # initial shape
        data = np.zeros(shape=[CUTS_NUMBER, MAX_OUTPUT_LENGTH, 7])

        for runIndex in range(1, CUTS_NUMBER + 1):

            runDataPath = os.path.join(COMPRESSED_DIR, "c%d" % cutIdx, "c_%d_%03d.csv" % (cutIdx, runIndex))
            print("Read data from", runDataPath)
            pdData = pd.read_csv(runDataPath, header=None)
            npData = pdData.to_numpy()
            reSamplingData = npData[::RESAMPLING_RATE, :]
            if (reSamplingData.shape[0] > MAX_OUTPUT_LENGTH):
                data[runIndex - 1] = reSamplingData[:MAX_OUTPUT_LENGTH, :]
            else:
                data[runIndex - 1, :reSamplingData.shape[0], :] = reSamplingData
        # back it up
        np.save(cache_data_path, data)
        return data

    def get_train_data_by_path(self, cutIdx: int):
        """
        get PHM 2010 data by directory path
        :param cutIdx: The milling index
        :return: np.array, np.array
        """
        cache_signal_path = os.path.join(CACHE_DIR, "cache_train_signal_cut_%d.npy" % cutIdx)
        cache_wear_path = os.path.join(CACHE_DIR, "cache_train_wear_cut_%d.npy" % cutIdx)
        if os.path.exists(cache_signal_path) and os.path.exists(cache_wear_path):
            print("Get data from cache")
            return np.load(cache_signal_path), np.load(cache_wear_path)
        # Read wear data first
        wearDataPath = os.path.join(COMPRESSED_DIR, "c%d_wear.csv" % cutIdx)
        wearNpData = pd.read_csv(wearDataPath)
        wearData = wearNpData.to_numpy()[:, 1:4]
        print(wearData.shape)
        signalList = None
        wearList = []
        for runIndex in range(1, CUTS_NUMBER + 1):
            # initial shape
            data = np.zeros(shape=[MAX_OUTPUT_LENGTH, 7])
            runWear = wearData[runIndex - 1, :]
            runDataPath = os.path.join(COMPRESSED_DIR, "c%d" % cutIdx, "c_%d_%03d.csv" % (cutIdx, runIndex))
            print("Read [Train] data from", runDataPath)
            pdData = pd.read_csv(runDataPath, header=None)
            npData = pdData.to_numpy()
            dataLength = npData.shape[0]
            reSamplingSequence = np.arange(0, dataLength, RESAMPLING_RATE)
            lastDigit = reSamplingSequence[-1]
            # resound it
            for i in range(min(dataLength - lastDigit, 20)):
                # print("Resound shift",i,reSamplingSequence[-1],dataLength, RESAMPLING_RATE)
                shiftSamplingSequence = reSamplingSequence + i
                shiftSamplingData = npData[shiftSamplingSequence, :]
                # print("shift shape",shiftSamplingData.shape)
                # append it
                if (shiftSamplingData.shape[0] > MAX_OUTPUT_LENGTH):
                    data = shiftSamplingData[:MAX_OUTPUT_LENGTH, :]
                else:
                    data[:shiftSamplingData.shape[0], :] = shiftSamplingData

                if signalList is None:
                    signalList = np.array([data])
                else:
                    signalList = np.append(signalList, np.array([data]),axis=0)

                wearList.append(runWear)
            print(data.shape, signalList.shape)

        signalData = np.array(signalList)
        wearData = np.array(wearList)
        # back it up
        np.save(cache_signal_path, signalData)
        np.save(cache_wear_path, wearData)
        return signalData, wearData


if __name__ == "__main__":
    data = Data()
    signal, wear = data.get_train_data_by_path(1)
    signal, wear = data.get_train_data_by_path(4)
    signal, wear = data.get_train_data_by_path(6)
    print(signal.shape, wear.shape)
