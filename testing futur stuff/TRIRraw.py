import os
import sys
import numpy as np
from time import strftime,sleep
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg,NavigationToolbar2QT
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


#from PyQt6 import QtCore, QtGui, QtWidgets,uic
import tkinter as tk
from tkinter import filedialog
from TRIRrawdataprocessing import *



#own packages
#from IRpackages.TRIR import pyTRIR_pack,pyTRIR_bgcorr

proc_dtype='float64'



class import_raw():
    def openrawdata():
        global jsondataset
        '''Function called when import button is pressed triggers a lot of other functions to load data'''
        print('asking values from GUI as INPUT')
        #instance.TRIRpixelnumber = instance.findChild(QtWidgets.QLineEdit,'TRIRpixelnum')
        #detectorsize = instance.TRIRpixelnumber.text()
        #instance.TRIRscanstring = instance.findChild(QtWidgets.QLineEdit,'scannumberedit')
        #scanstring = instance.TRIRscanstring.text()
        #instance.TRIRdelaystring = instance.findChild(QtWidgets.QLineEdit,'delaynumberedit')
        #delaystring = instance.TRIRdelaystring.text()
        #instance.TRIRweighting = instance.findChild(QtWidgets.QComboBox,'TRIRcombobox')
        #funcoptstring = instance.TRIRweighting.currentText()
        #print('IMPORT METHOD=', funcoptstring,'WITH (detectorsize,scans,delays)=',str(tuple((detectorsize,scanstring,delaystring))))
        #instance.file_dialog = QtWidgets.QFileDialog()
        #instance.file_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)  # Set the mode to select an existing file
        #instance.file_dialog.setNameFilter("All files (*.*)")  # Set filters for file all types
        #if instance.file_dialog.exec() == QtWidgets.QFileDialog.DialogCode.Accepted:
        #    file_path = instance.file_dialog.selectedFiles()
        detectorsize = 32
        scanstring =':'
        delaystring=':'
        funcoptstring= 's2s_signal'
        print('IMPORT METHOD=', funcoptstring,'WITH (detectorsize,scans,delays)=',str(tuple((detectorsize,scanstring,delaystring))))

        #dspath = filedialog.askdirectory()
        #dspath= '/Users/mschick/Desktop/AKB/AQUISITION_DATA/I-LAB_TRIR/CZUPPY_TRIR/20240410_cz_3-8mj_000'
        dspath= '/Users/mschick/Desktop/AKB/AQUISITION_DATA/I-LAB_TRIR/CZUPPY_TRIR/20240415_ms4muj_30deg_000'
        print('#####DATA storage from:',dspath)

        data= import_raw.read_raw_data(38,41,dspath)
        print('#####DATA import shape:', np.shape(data))
        #
        global ref,lo,hi,filtereddata,uniformfiltered,medianfiltered,filtered_iris
        filtereddata,ref,lo,hi = bubble_filter_segmented(raw_data=data[0:63,:], pixel_idx=np.arange(0,63), framesize=1000)
        uniformfiltered = bubble_filter_uniform(raw_data=data[0:63,:], pixel_idx=np.arange(0,63),width=0.20,bubble_width=500)#, stdfactor:float=None, bubble_width: int=50)
        #uniformfiltered = bubble_filter_uniform_new(raw_data=data[0:63,:], pixel_idx=np.arange(0,63),stdfactor=1, bubble_width=50)
        #medianfiltered = bubble_filter_median(raw_data=data[0:63,:], pixel_idx=np.arange(0,63),width=1,bubble_width=500)
        filtered_iris = bubblefilter_IRIS(raw_data=data[0:63,:], pixel_idx=np.arange(0,63), window=251, polyorder=5 , derivative_order=0)
        filtered_iris = bubblefilter_IRIS(raw_data=filtered_iris, pixel_idx=np.arange(0,63), window=101, polyorder=5 , derivative_order=0,width=5)
        shotnumberraw = np.shape(data)[1]
        shotnumberuniform = np.shape(uniformfiltered)[1] /shotnumberraw*100
        shotnumbersavgol = np.shape(filtered_iris)[1] /shotnumberraw*100

        global cluster_filtered_kmeans,cluster_filtered_mean
        cluster_filtered_kmeans = bubblefilter_clusters_kmeans(raw_data=data[0:63,:], pixel_idx=np.arange(0,63))
        shotnumberkmeans = np.shape(cluster_filtered_kmeans)[1] /shotnumberraw*100
        #cluster_filtered_mean = bubblefilter_clusters_meanshift(raw_data=data[0:63,:], pixel_idx=np.arange(0,63))
        #shotnumberdbmean = np.shape(cluster_filtered_mean)[1] /shotnumberraw*100
        #cluster_filtered_spectral = bubblefilter_clusters_spectralclustering(raw_data=data[0:63,:], pixel_idx=np.arange(0,63))
        #shotnumberspectral = np.shape(cluster_filtered_spectral)[1] /shotnumberraw*100
        print('#####FILTER PERFORMANCE -> passed data percentage: ','uniform:',f'{shotnumberuniform:.4g}','%','savgol:',f'{shotnumbersavgol:.4g}','%','kmeans:',f'{shotnumberkmeans:.4g}','%')#,'dbscan:',f'{shotnumberdbmean:.4g}','%')
        import_raw.plotshot(data)


    def read_raw_data(delay_idx,scan_idx,ds_path):
        ds_root = ds_path.split('/')[-1]            #MAC SPECIAL /      Windows to: \
        print('#####',ds_root)
        filepath = os.path.join(ds_path,'raw_data','delay'+str(delay_idx).zfill(3))
        filename = 's'+str(scan_idx).zfill(6)+'_d'+str(delay_idx).zfill(3)+'_'+ds_root+'_raw.npz'
        data = np.load(os.path.join(filepath,filename))['arr_0'].astype(proc_dtype)
        return data


    def plotshot(data):
        fig = plt.figure()
        pix =25
        plt.plot(data[pix,:],label='central pixel')#,alpha=0.1)
        plt.plot(uniformfiltered[pix,:],label='uniform')
        #plt.plot(filtered_iris[15,:],label='savgol')
        plt.plot(cluster_filtered_kmeans[15,:],label='kmeans')
        #plt.plot(cluster_filtered_mean[15,:],label='mean shift')

        #plt.plot([0,4000],[ref[15],ref[15]],color='r')
        #plt.plot([0,4000],[lo[15],lo[15]],color='k')
        #plt.plot([0,4000],[hi[15],hi[15]],color='k')

        #plt.plot([0,40000],[median[15],median[15]],color='b')

        plt.legend()
        plt.show()


import_raw.openrawdata()