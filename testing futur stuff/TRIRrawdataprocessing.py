from scipy.fft import rfft, fft, next_fast_len
from scipy.stats import linregress
from scipy.signal import find_peaks, peak_widths,savgol_filter
from scipy.signal.windows import get_window
from scipy.ndimage import uniform_filter, median_filter
import matplotlib.pyplot as plt

import numpy as np
from numpy import ndarray
import logger

def bubble_filter_segmented(raw_data:ndarray, pixel_idx:ndarray, framesize=1000, width:float=None):
    ''' Removes outliers based on channel std of data segment

    Args:
        raw_data: 2d array of raw data (axis 0: ADC channels, axis 1: shots)
        pixel_idx: 1d array with the indices of the ADC pixel channels
            e. g. [0,...,63] in the I-lab
        width: relative width of the "good" corridor around the
            channel segment mean, e.g. 0.05 means a data point must lie within
            +/- 5% of the mean
            if not given, an estimate is made based on the standard deviation

    Returns:
        filtered array
    '''
    filtered = np.ndarray(raw_data.shape)
    pointer = 0
    len_ = raw_data.shape[1]
    newframesize = framesize
    while 0 < len_%newframesize < framesize //2:
        newframesize += 1
    #logger.info("Frame size requested: {}, used: {}.".format(framesize,newframesize))
    framesize = newframesize
    for i in range((raw_data.shape[1]-1)//framesize+1):
        frame = raw_data[:,i*framesize:(i+1)*framesize]
        if width is None:
            width = 5*(frame[pixel_idx,:].std(axis=1) / frame[pixel_idx,:].mean(axis=1))[:,np.newaxis]
        ref = frame[pixel_idx,:].mean(axis=1)[:,np.newaxis]
        lo = ref * (1-width)
        hi = ref * (1+width)
        lo_f = np.all(frame[pixel_idx,:] > lo[pixel_idx],axis=0)
        hi_f = np.all(frame[pixel_idx,:] < hi[pixel_idx],axis=0)
        filtered_frame = frame[:,np.logical_and(lo_f,hi_f)]
        fflen = filtered_frame.shape[1]
        filtered[:,pointer:pointer+fflen] = filtered_frame
        pointer += fflen
    #logger.info("{} shots discarded by segmented filter".format(raw_data.shape[1]-filtered[:,:pointer].shape[1]))
    return filtered[:,:pointer],ref,lo,hi





def bubble_filter_uniform_new(raw_data: ndarray, pixel_idx: ndarray, stdfactor:float=None, bubble_width: int=50):
    ''' Removes outliers based on floating-average smoothed data

    Args:
        raw_data: 2d array of raw data (axis 0: ADC channels, axis 1: shots)
        pixel_idx: 1d array with the indices of the ADC pixel channels
            e. g. [0,...,63] in the I-lab
        stdfactor: relative width of the "good" corridor around the
            smoothed data, e.g. 3.5 means a data point must lie within
            +/- 3.5x standard deviation of the average curve
            if not given, a standard value of 5 is used and logged
        bubble_width: how many data points will one bubble affect. This
            sets the width of the moving average filter window.

    Returns:
        filtered array
    '''
    pxdata = raw_data[pixel_idx,:]
    if stdfactor is None:
        stdfactor = 5
        #logger.info("No multiplier for standard deviation given, using default {}.".format(stdfactor))
    width = stdfactor*pxdata.std(axis=1)[:,np.newaxis]
    ref = uniform_filter(pxdata,size=(1,2*bubble_width))
    lo_f = np.all(pxdata > ref-width,axis=0)
    hi_f = np.all(pxdata < ref+width,axis=0)
    both_f = np.logical_and(lo_f,hi_f)
    #logger.info("{} shots discarded by uniform filter".format(np.count_nonzero(both_f == False)))
    return raw_data[:,both_f]


def bubble_filter_median(raw_data: ndarray, pixel_idx: ndarray, width:float=None, bubble_width: int=50):
    ''' Removes outliers based on median smoothed data

    Args:
        raw_data: 2d array of raw data (axis 0: ADC channels, axis 1: shots)
        pixel_idx: 1d array with the indices of the ADC pixel channels
            e. g. [0,...,63] in the I-lab
        width: relative width of the "good" corridor around the
            smoothed data, e.g. 0.05 means a data point must lie within
            +/- 5% of the average curve
            if not given, an estimate is made based on the standard deviation
        bubble_width: how many data points will one bubble affect. This
            sets the width of the filter window.

    Returns:
        filtered array
    '''

    if width is None:
        width = 5*(raw_data[pixel_idx,:].std(axis=1) / raw_data[pixel_idx,:].mean(axis=1))[:,np.newaxis]
        #logger.info("No width given, using estimate per channel. Max is {}".format(width.max()))
    ref = median_filter(raw_data,size=(1,2*bubble_width))
    lo = ref * (1-width)
    hi = ref * (1+width)
    lo_f = np.all(raw_data[pixel_idx,:] > lo[pixel_idx],axis=0)
    hi_f = np.all(raw_data[pixel_idx,:] < hi[pixel_idx],axis=0)
    filtered = raw_data[:,np.logical_and(lo_f,hi_f)]
    #logger.info("{} shots discarded by median filter".format(raw_data.shape[1]-filtered.shape[1]))
    return filtered

def bubble_filter_uniform(raw_data: ndarray, pixel_idx: ndarray, width:float=None, bubble_width: int=50):
    ''' Removes outliers based on floating-average smoothed data

    Args:
        raw_data: 2d array of raw data (axis 0: ADC channels, axis 1: shots)
        pixel_idx: 1d array with the indices of the ADC pixel channels
            e. g. [0,...,63] in the I-lab
        width: relative width of the "good" corridor around the
            smoothed data, e.g. 0.05 means a data point must lie within
            +/- 5% of the average curve
            if not given, an estimate is made based on the standard deviation
        bubble_width: how many data points will one bubble affect. This
            sets the width of the filter window.

    Returns:
        filtered array
    '''

    if width is None:
        width = 5*(raw_data[pixel_idx,:].std(axis=1) / raw_data[pixel_idx,:].mean(axis=1))[:,np.newaxis]
    ref = uniform_filter(raw_data,size=(1,2*bubble_width))
    lo = ref * (1-width)
    hi = ref * (1+width)
    lo_f = np.all(raw_data[pixel_idx,:] > lo[pixel_idx],axis=0)
    hi_f = np.all(raw_data[pixel_idx,:] < hi[pixel_idx],axis=0)
    filtered = raw_data[:,np.logical_and(lo_f,hi_f)]
    return filtered



def bubblefilter_IRIS(raw_data: ndarray, pixel_idx: ndarray, window:int=301, polyorder:int=5 , derivative_order:int=0, width:float=None, bubble_width: int=50):
    #for pix in pixel_idx:
    tracer = savgol_filter(raw_data[pixel_idx,:],window_length=window,polyorder=polyorder, deriv=derivative_order,axis=1)
    if width==None:
        width = 4*(raw_data[pixel_idx,:].std(axis=1) / raw_data[pixel_idx,:].mean(axis=1))[:,np.newaxis]
    else:
        width = width*(raw_data[pixel_idx,:].std(axis=1) / raw_data[pixel_idx,:].mean(axis=1))[:,np.newaxis]
    
    lo = tracer- width
    hi = tracer + width
    lo_f = np.all(raw_data[pixel_idx,:] > lo[pixel_idx],axis=0)
    hi_f = np.all(raw_data[pixel_idx,:] < hi[pixel_idx],axis=0)
    filtered = raw_data[:,np.logical_and(lo_f,hi_f)]



    #fig = plt.figure()
    #plt.plot(raw_data[pix,:],label='raw')
    #plt.plot(tracer[pix,:],label='tracer')
    #plt.errorbar(x=np.arange(len(tracer[pix,:])),y=tracer[pix,:],yerr=width[15],color='grey',alpha=0.1)
    #plt.plot(filtered[pix,:],label='filtered')
    #plt.show()
    return filtered



















from sklearn.cluster import KMeans,DBSCAN,SpectralClustering,MeanShift
def bubblefilter_clusters_kmeans(raw_data: ndarray, pixel_idx: ndarray,clusternum: int=3):
        TRACER = savgol_filter(raw_data[pixel_idx,:],window_length=501,polyorder=5, deriv=0,axis=1)
        DIFF = np.subtract(raw_data,TRACER)
        DIFF = np.abs(DIFF)

        """ wcss = []  # within-cluster sum of squares
        # Calculate WCSS for different numbers of clusters
        for i in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(np.transpose(DIFF))#.reshape(-1, 1))
            wcss.append(kmeans.inertia_)  # inertia_ contains WCSS

        fig = plt.figure()
        plt.plot(range(1, max_clusters + 1), wcss)
        plt.title('Elbow Method')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        plt.show() """

        kmeans = KMeans(n_clusters=clusternum,init='k-means++',tol=1e-4,algorithm='elkan')
        kmeans.fit(np.transpose(DIFF))
        cluster_labels = kmeans.labels_

        fig = plt.figure()
        plt.scatter(np.arange(len(raw_data[15,:])),DIFF[15,:],c=cluster_labels,s=1)
        plt.title('KMEANS')
        plt.xlabel('shots')
        plt.ylabel('Volts')
        plt.show()

        target_cluster = 0
        filtered  = np.transpose(raw_data)[cluster_labels == target_cluster]
        filtered = np.transpose(filtered)
        return filtered

def bubblefilter_clusters_spectralclustering(raw_data: ndarray, pixel_idx: ndarray ,n_clusters:int=3):
        TRACER = savgol_filter(raw_data[pixel_idx,:],window_length=501,polyorder=5, deriv=0,axis=1)
        DIFF = np.subtract(raw_data,TRACER)

        spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
        spectral.fit(np.transpose(DIFF))
        cluster_labels = spectral.labels_

        fig = plt.figure()
        plt.scatter(np.arange(len(raw_data[15,:])),raw_data[15,:],c=cluster_labels,s=1)
        plt.title('Spectral')
        plt.xlabel('shots')
        plt.ylabel('Volts')
        plt.show()

        target_cluster = 0
        filtered  = np.transpose(raw_data)[cluster_labels == target_cluster]
        filtered = np.transpose(filtered)
        return filtered

def bubblefilter_clusters_meanshift(raw_data: ndarray, pixel_idx: ndarray ,bandwidth:float=None):
        TRACER = savgol_filter(raw_data[pixel_idx,:],window_length=501,polyorder=5, deriv=0,axis=1)
        DIFF = np.subtract(raw_data,TRACER)

        meanshift = MeanShift(bandwidth=bandwidth)
        meanshift.fit(np.transpose(DIFF))
        cluster_labels = meanshift.labels_

        fig = plt.figure()
        plt.scatter(np.arange(len(raw_data[15,:])),raw_data[15,:],c=cluster_labels,s=1)
        plt.title('Mean Shift')
        plt.xlabel('shots')
        plt.ylabel('Volts')
        plt.show()

        target_cluster = 0
        filtered  = np.transpose(raw_data)[cluster_labels == target_cluster]
        filtered = np.transpose(filtered)
        return filtered
        

def bubblefilter_clusters_dbscan(raw_data: ndarray, pixel_idx: ndarray,eps: int=0.1 ,min_samples:int=50):
        TRACER = savgol_filter(raw_data[pixel_idx,:],window_length=501,polyorder=5, deriv=0,axis=1)
        DIFF = np.subtract(raw_data,TRACER)

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(np.transpose(DIFF))
        cluster_labels = dbscan.labels_

        target_cluster = 0
        filtered  = np.transpose(raw_data)[cluster_labels == target_cluster]
        filtered = np.transpose(filtered)
        return filtered

        




