import numpy as np
from time import strftime,sleep
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg,NavigationToolbar2QT
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from PyQt6 import QtCore, QtGui, QtWidgets,uic

import cv2
from matplotlib.animation import FuncAnimation


class CAM_GUI():
    def initer(instance):
        print('startup CAM')
        '''this part is added to build the plotting canvas inside the ui'''

        #Canvas
        if 'CheckerCAM' in locals() or 'CheckerCAM' in globals():
            print("no need to regenerate canvas")
        else:
            print("building canvas")
            global CheckerCAM
            CheckerCAM=1
            instance.CAMcanvasframeviewer = instance.findChild(QtWidgets.QFrame,'camframe')
            instance.CAMhorizontalLayoutviewer= QtWidgets.QVBoxLayout(instance.CAMcanvasframeviewer)
            instance.CAMhorizontalLayoutviewer.setObjectName('CAMcanvas_viewer_layout')
            #the canvas
            instance.Feedlabel= QtWidgets.QLabel()
            instance.CAMhorizontalLayoutviewer.addWidget(instance.Feedlabel)
            
            """ instance.CAM_fig = plt.figure(4,figsize=(10, 10))
            plt.rc('font', size=4) #controls default text size
            plt.rcParams['xtick.major.pad']='1'
            plt.rcParams['ytick.major.pad']='1'

            instance.camgrid = instance.CAM_fig.add_gridspec(6, 6, hspace=0.01, wspace=0.01,bottom=0.09,top=0.95,left=0.05,right=0.95)
            instance.cam = instance.CAM_fig.add_subplot(instance.camgrid[1:, 1:])
            instance.cam_y_hist = instance.CAM_fig.add_subplot(instance.camgrid[1:, 0],sharey=instance.cam)
            instance.cam_x_hist = instance.CAM_fig.add_subplot(instance.camgrid[0, 1:-1], sharex=instance.cam)
       
            instance.cam.set_xlabel('pixl')
            instance.cam_y_hist.set_ylabel('pix')
            instance.cam_x_hist.xaxis.set_visible(False)
            
            instance.CAMcanvasviewer = FigureCanvasQTAgg(instance.CAM_fig) 
            instance.CAMtoolbar= NavigationToolbar2QT(instance.CAMcanvasviewer,instance.CAMcanvasframeviewer)
            print('test')
            #add canvas to widget
            instance.CAMhorizontalLayoutviewer.addWidget(instance.CAMcanvasviewer)
            instance.CAMhorizontalLayoutviewer.addWidget(instance.CAMtoolbar) """


        instance.camstartbutton = instance.findChild(QtWidgets.QPushButton,'start')
        instance.camstartbutton.clicked.connect(lambda:CAM_GUI.CancelFeed(instance))
        instance.camstopbutton = instance.findChild(QtWidgets.QPushButton,'stop')
        instance.camstartbutton.clicked.connect(lambda:CAM_GUI.CancelFeed(instance))
        instance.camchoose = instance.findChild(QtWidgets.QComboBox,'camchoose')

        instance.Worker1 = Worker1(instance)
        instance.Worker1.ImageUpdate.connect(instance.ImageUpdateSlot)
        instance.Worker1.start()
        #instance.Worker1.ImageUpdate.connect(CAM_GUI.ImageUpdateSlot)
        
        

    @QtCore.pyqtSlot(QtGui.QImage)
    def ImageUpdateSlot(instance,Image):
        instance.Feedlabel.setPixmap(QtGui.QPixmap.fromImage(Image))

    def CancelFeed(instance):
        instance.Worker1.stop()
        print('stoppend aquisation')

        



    


class Worker1(QtCore.QThread):
    ImageUpdate = QtCore.pyqtSignal(QtGui.QImage)
    def run(self):
        self.ThreadActive = True
        Capture = cv2.VideoCapture(0)
        while self.ThreadActive:
            ret, frame = Capture.read()
            if ret:
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FlippedImage = cv2.flip(Image, 1)
                ConvertToQtFormat = QtGui.QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QtGui.QImage.Format.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)
    def stop(self):
        self.ThreadActive = False
        self.quit()
    
   
        

    