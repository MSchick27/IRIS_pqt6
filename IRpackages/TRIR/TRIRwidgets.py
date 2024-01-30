import sys
import numpy as np
from time import strftime,sleep
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg,NavigationToolbar2QT
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



from PyQt6 import QtCore, QtGui, QtWidgets,uic

#own packages
from IRpackages.TRIR import pyTRIR_pack,pyTRIR_bgcorr

def getjsonfile():
    return jsondataset


class TRIR_widgets_defining():
    def initer(instance):
        print('building TRIR module with instance as self variable')
        '''this part is added to build the plotting canvas inside the ui'''
        #Frame for the canvas
        instance.TRIRcanvasframe = instance.findChild(QtWidgets.QFrame,'TRIRcanvas')
        #Canvas
        if 'CheckerTRIR' in locals() or 'CheckerTRIR' in globals():
            print("no need to regenerate canvas")
        else:
            print("building canvas")
            global CheckerTRIR
            CheckerTRIR=1
            instance.horizontalLayout= QtWidgets.QVBoxLayout(instance.TRIRcanvasframe)
            instance.horizontalLayout.setObjectName('canvas_h_layout')
            #the canvas
            instance.fig = plt.figure(0,figsize=(10, 10))
            plt.rc('font', size=4) #controls default text size
            plt.rcParams['xtick.major.pad']='1'
            plt.rcParams['ytick.major.pad']='1'

            instance.grid = instance.fig.add_gridspec(11, 11, hspace=0.01, wspace=0.01,bottom=0.09,top=0.95,left=0.05,right=0.95)
            instance.data_ax = instance.fig.add_subplot(instance.grid[0:3, 0:3])
            instance.data_ax.set_xlabel('time [ps]',labelpad=.1)
            instance.data_ax.set_title('raw',pad=.1)
            instance.bg_ax = instance.fig.add_subplot(instance.grid[0:3, 4:7],sharey=instance.data_ax,sharex=instance.data_ax)
            instance.bg_ax.set_title('background',pad=.1)
            instance.rms_bg_ax = instance.fig.add_subplot(instance.grid[0:3, 8:11])
            instance.rms_bg_ax.set_title('std deviation',pad=.1)
            instance.rms_bg_ax.set_xlabel('scans',labelpad=.1)
            instance.rms_bg_ax.set_ylabel('pixels',labelpad=.1)
            instance.noise_ax = instance.fig.add_subplot(instance.grid[4:7, 0:3])
            instance.noise_ax.set_title('noise',pad=.1)
            instance.noise_ax.set_xlabel('scans',labelpad=.1)
            instance.noise_ax.set_ylabel('pixels',labelpad=.1)
            instance.noiseall_ax = instance.fig.add_subplot(instance.grid[8:11, 0:3])
            instance.noiseall_ax.set_title('trace of all scans',pad=.1)
            instance.noiseall_ax.set_xlabel('scans',labelpad=.1)
            instance.noiseall_ax.set_ylabel('Pixels',labelpad=.1)
            instance.diff_ax = instance.fig.add_subplot(instance.grid[4:11, 4:11],sharey=instance.data_ax,sharex=instance.data_ax)
            instance.diff_ax.set_title('Diff Signal',pad=.1)
            instance.diff_ax.set_xlabel('time',labelpad=.1)
            instance.diff_ax.set_ylabel('wn',labelpad=.1)
        
            instance.canvas = FigureCanvasQTAgg(instance.fig) 
            instance.toolbar= NavigationToolbar2QT(instance.canvas,instance.TRIRcanvasframe)

            #add canvas to widget
            instance.horizontalLayout.addWidget(instance.canvas)
            instance.horizontalLayout.addWidget(instance.toolbar)


        #adding relevance to buttons (commands)
        instance.TRIRimportbutton = instance.findChild(QtWidgets.QPushButton,'TRIRimport')
        instance.TRIRimportbutton.clicked.connect(lambda:TRIR_widgets_defining.TRIRimportfunc(instance))
        instance.TRIRimportbutton.setShortcut('Ctrl+I')
        instance.TRIRimportbutton.setToolTip('Shortcut: Ctrl+ I')

        instance.TRIRreloadbutton = instance.findChild(QtWidgets.QPushButton,'reloadbutton')
        instance.TRIRreloadbutton.clicked.connect(lambda:TRIR_widgets_defining.import_reloaded(instance))

        instance.TRIRplottingbutton =instance.findChild(QtWidgets.QPushButton,'TRIRrefreshbutton')
        instance.TRIRplottingbutton.clicked.connect(lambda:TRIR_widgets_defining.TRIR_plot(instance))

        instance.TRIRscanonebutton =instance.findChild(QtWidgets.QPushButton,'subscanone')
        instance.TRIRscanonebutton.clicked.connect(lambda:TRIR_widgets_defining.subtractfirstscan(instance))

        instance.TRIRgeneratebackground =instance.findChild(QtWidgets.QPushButton,'genbackground')
        instance.TRIRgeneratebackground.clicked.connect(lambda:TRIR_widgets_defining.generate_background_fit(instance))

        instance.TRIRexportdata =instance.findChild(QtWidgets.QPushButton,'exportbutton')
        instance.TRIRexportdata.clicked.connect(lambda:TRIR_widgets_defining.exportdatacomp(instance))

        instance.TRIRexportdata_npy =instance.findChild(QtWidgets.QPushButton,'exportbutton_2')
        instance.TRIRexportdata_npy.clicked.connect(lambda:TRIR_widgets_defining.exportdata_npyfile(instance))


        #defining shortcuts for fast work
        """  shortcut = QtGui.QKeySequence(QtCore.Key_M)
        instance.shortcut = QtGui.QShortcut(shortcut, instance)
        instance.shortcut.activated.connect(lambda:TRIR_widgets_defining.TRIRimportfunc(instance)) """



















    '''####################################################################################
    This part from now will be all about defining functions for the different buttons above
    ####################################################################################'''

    def clearcanva(instance):
        #print('cleared frame')
        instance.data_ax.cla()
        instance.bg_ax.cla()
        instance.rms_bg_ax.cla()
        instance.diff_ax.cla()
        instance.noise_ax.cla()
        instance.noiseall_ax.cla()

        instance.data_ax.set_xlabel('time [ps]',labelpad=.1)
        instance.data_ax.set_title('raw',pad=.1)
        #instance.bg_ax = instance.fig.add_subplot(instance.grid[0:3, 4:7],sharey=instance.data_ax,sharex=instance.data_ax)
        instance.bg_ax.set_title('background',pad=.1)
        #instance.rms_bg_ax = instance.fig.add_subplot(instance.grid[0:3, 8:11])
        instance.rms_bg_ax.set_title('std deviation',pad=.1)
        instance.rms_bg_ax.set_xlabel('scans',labelpad=.1)
        instance.rms_bg_ax.set_ylabel('pixels',labelpad=.1)
        #instance.noise_ax = instance.fig.add_subplot(instance.grid[4:7, 0:3])
        instance.noise_ax.set_title('noise',pad=.1)
        instance.noise_ax.set_xlabel('scans',labelpad=.1)
        instance.noise_ax.set_ylabel('pixels',labelpad=.1)
        #instance.noiseall_ax = instance.fig.add_subplot(instance.grid[8:11, 0:3])
        instance.noiseall_ax.set_title('trace of all scans',pad=.1)
        instance.noiseall_ax.set_xlabel('scans',labelpad=.1)
        instance.noiseall_ax.set_ylabel('Pixels',labelpad=.1)
        #instance.diff_ax = instance.fig.add_subplot(instance.grid[4:11, 4:11],sharey=instance.data_ax,sharex=instance.data_ax)
        instance.diff_ax.set_title('Diff Signal',pad=.1)
        instance.diff_ax.set_xlabel('time',labelpad=.1)
        instance.diff_ax.set_ylabel('wn',labelpad=.1)
        try:
            cbnoise.remove()
            cbmain.remove()
        except:
            print('')

        instance.canvas.draw()

    def TRIR_plot(instance):
        '''This function is called when the plotting button in IRIS is pressed 
        There are a few things that are happing in the last seconds before plotting:
        '''
        print('plotting')
        plt.figure(0,figsize=(10, 10))
        instance.levelnumber = instance.findChild(QtWidgets.QLineEdit,'levelsnumber')
        instance.limitnumber = instance.findChild(QtWidgets.QLineEdit,'limitnumber')
        try:
            levelnum = int(instance.levelnumber.text())
        except ValueError as e:
            print("levelnumber has to be an integer", e)

        TRIR_widgets_defining.clearcanva(instance)
        #raw data pplot plus bg plot
        maxval,minval= pyTRIR_pack.colormapsfor_TRIR.findmaxval(jsondataset['data'],0.8)
        instance.data_ax.contourf(jsondataset['delays'],jsondataset['wn'],jsondataset['data'],levels=levelnum,cmap='RdBu_r',vmin=minval,vmax=maxval,alpha=1)#,vmin=cmin,vmax=cmax)
        maxval,minval= pyTRIR_pack.colormapsfor_TRIR.findmaxval(jsondataset['bgdata'],0.8)
        instance.bg_ax.contourf(jsondataset['delays'],jsondataset['wn'],jsondataset['bgdata'],levels=levelnum,cmap='RdBu_r',vmin=minval,vmax=maxval,alpha=1)
        
        #rmsdata = pyTRIR.modify_arrays.subtract_bg_rms(jsondataset)
        maxval,minval= pyTRIR_pack.colormapsfor_TRIR.findmaxval(jsondataset['std_deviation'],instance.limitnumber.text())
        clipped_stddev = np.clip(jsondataset['std_deviation'],a_min=minval,a_max = maxval)
        instance.rms_bg_ax.pcolormesh(jsondataset['delays'],np.arange(len(jsondataset['wn'])),clipped_stddev,cmap='magma')

        
        #The main plot in the middle to gcheck results
        global mainplot
        databgsub = pyTRIR_pack.modify_arrays.subtract_bg(jsondataset)
        maxval,minval= pyTRIR_pack.colormapsfor_TRIR.findmaxval(databgsub,instance.limitnumber.text())
        clipped_databgsub = np.clip(databgsub,a_min=minval,a_max = maxval)
        mainplot=instance.diff_ax.contourf(jsondataset['delays'],jsondataset['wn'],clipped_databgsub,levels=levelnum,cmap='RdBu_r',vmin=minval,vmax=maxval,alpha=1)#'RdBu_r''seismic'
        print('Max '+str(maxval) + ' & min '+str(minval))

        axins1 = inset_axes(instance.diff_ax,width="30%",  # width: 50% of parent_bbox width
                                    height="5%",  # height: 5%
                                    loc="upper right",borderpad=1)
        global cbmain
        cbmain = instance.fig.colorbar(mainplot, cax=axins1, orientation="horizontal", ticks=[minval,0,maxval],format='%.0e')
        cbmain.set_label(label='OD',weight='bold')
        
        try:
            noisedelayindex = instance.TRIRlistbox.currentRow()
            global mapnoise
            mapnoise=instance.noise_ax.pcolormesh(np.arange(np.shape(jsondataset['noise'])[1]),np.arange(np.shape(jsondataset['noise'])[2]),np.transpose(jsondataset['noise'][noisedelayindex]),cmap='viridis')
            global cbnoise#,cbmain
            cbnoise = instance.fig.colorbar(mapnoise, ax=instance.noise_ax)
            #cbmain = fig.colorbar(mainplot, cax=axins1, orientation="horizontal", ticks=[minval,0,maxval],format='%.0e')
            #cbmain.set_label(label='OD',weight='bold')
        except:
            print('SCANPLOT: Select delay to display scans')
            instance.TRIRlistbox.setCurrentRow(0)
            instance.noise_ax.pcolormesh(np.arange(np.shape(jsondataset['noise'])[1]),np.arange(np.shape(jsondataset['noise'])[2]),np.transpose(jsondataset['noise'][0]),cmap='viridis')

        noisealldata= pyTRIR_pack.modify_arrays.noiseallscans(jsondataset)
        maxval,minval= pyTRIR_pack.colormapsfor_TRIR.findmaxval(jsondataset['noise'],instance.limitnumber.text())
        instance.noiseall_ax.pcolormesh(np.arange(np.shape(noisealldata)[0]),np.arange(np.shape(noisealldata)[1]),noisealldata.transpose(),cmap='viridis')




        instance.logcheck = instance.findChild(QtWidgets.QCheckBox,'logscalecheckbox')
        logstate= instance.logcheck.isChecked()
        if logstate:
            xaxislow,xaxishigh = pyTRIR_pack.modify_arrays.getlogaxis(jsondataset)
            instance.data_ax.set_xlim(xaxislow,xaxishigh)
            instance.data_ax.set_xscale('log')
            instance.rms_bg_ax.set_xscale('log')
            instance.rms_bg_ax.set_xlim(xaxislow,xaxishigh)

        if instance.TRIRweighting.currentText() == 'weights':
            instance.noise_ax.set_title('wheights',pad=.1)
        
        if instance.TRIRweighting.currentText() == 's2s_signal':
            instance.noise_ax.set_title('s2s_signal',pad=.1)

        instance.canvas.draw()
        #print('plotted')

        #plt.close()













    def TRIRimportfunc(instance):
        global jsondataset
        '''Function called when import button is pressed triggers a lot of other functions to load data'''
        print('asking values from GUI as INPUT')
        instance.TRIRpixelnumber = instance.findChild(QtWidgets.QLineEdit,'TRIRpixelnum')
        detectorsize = instance.TRIRpixelnumber.text()
        instance.TRIRscanstring = instance.findChild(QtWidgets.QLineEdit,'scannumberedit')
        scanstring = instance.TRIRscanstring.text()
        instance.TRIRdelaystring = instance.findChild(QtWidgets.QLineEdit,'delaynumberedit')
        delaystring = instance.TRIRdelaystring.text()
        instance.TRIRweighting = instance.findChild(QtWidgets.QComboBox,'TRIRcombobox')
        funcoptstring = instance.TRIRweighting.currentText()
        print('IMPORT METHOD=', funcoptstring,'WITH (detectorsize,scans,delays)=',str(tuple((detectorsize,scanstring,delaystring))))
        instance.file_dialog = QtWidgets.QFileDialog()
        instance.file_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)  # Set the mode to select an existing file
        instance.file_dialog.setNameFilter("All files (*.*)")  # Set filters for file all types
        if instance.file_dialog.exec() == QtWidgets.QFileDialog.DialogCode.Accepted:
            file_path = instance.file_dialog.selectedFiles()
        print('selected path=',file_path)

        if funcoptstring == 's2s_signal':
                    #try function am ende addieren
                    jsondataset = pyTRIR_pack.TRIR.importinitfunction(file_path,scanstring,delaystring,detectorsize)

        if funcoptstring == 'weights':
                    print('still under construction')
                    #jsondataset = pyTRIR_pack.weighted_import.init_weightedimport(scanstring,delaystring,detectorsize)

                
        instance.TRIRwnlowlabel = instance.findChild(QtWidgets.QLabel,'TRIRwnlownumber')
        instance.TRIRwnlowlabel.setText(str(round(float(jsondataset['wn'][0]),2)))
        instance.TRIRwnhighlabel = instance.findChild(QtWidgets.QLabel,'wnhighnumber')
        instance.TRIRwnhighlabel.setText(str(round(float(jsondataset['wn'][-1]),2)))

        instance.TRIRlistbox = instance.findChild(QtWidgets.QListWidget,'delaylistbox')
        instance.TRIRlistbox.clear()
        for delaytime in jsondataset['delays']:
                    instance.TRIRlistbox.addItem(f'{delaytime: .3g}')
        instance.TRIRlistbox.setCurrentRow(0)


        instance.TRIRscannumber = instance.findChild(QtWidgets.QLabel,'scannumber_2')
        instance.TRIRscannumber.setText(str(jsondataset['scannumber']))
        instance.TRIRdelaynumber = instance.findChild(QtWidgets.QLabel,'delaynumber')
        instance.TRIRdelaynumber.setText(str(jsondataset['delaynumber']))
        instance.TRIRdelaystring.setText(str(jsondataset['delayslice']))




        '''After import button activate few other buttons to unlock more actions
        '''
        instance.actionbutton_viewer = instance.findChild(QtGui.QAction,'actionTRIR_viewer')
        instance.actionbutton_viewer.setEnabled(not instance.actionbutton_viewer.isEnabled())

        instance.TRIRplottingbutton.setEnabled(True)
        instance.TRIRscanonebutton.setEnabled(True)
        instance.TRIRreloadbutton.setEnabled(True)
        instance.TRIRgeneratebackground.setEnabled(True)
        instance.TRIRexportdata.setEnabled(True)
        instance.TRIRexportdata_npy.setEnabled(True)







        #toolframe_widget_commands.refresh(jsondataset)

    def import_reloaded(instance):
        '''Function to reimport the last chosenfile again to fastly get data by other scanslices or to cut of delays
        This is needed since the datasets are weighted during import. therefore changing settings for data analysis 
        needs to reload the complete set
        '''
        global jsondataset
        instance.TRIRpixelnumber = instance.findChild(QtWidgets.QLineEdit,'TRIRpixelnum')
        detectorsize = instance.TRIRpixelnumber.text()
        instance.TRIRscanstring = instance.findChild(QtWidgets.QLineEdit,'scannumberedit')
        scanstring = instance.TRIRscanstring.text()
        instance.TRIRdelaystring = instance.findChild(QtWidgets.QLineEdit,'delaynumberedit')
        delaystring = instance.TRIRdelaystring.text()
        instance.TRIRweighting = instance.findChild(QtWidgets.QComboBox,'TRIRcombobox')
        funcoptstring = instance.TRIRweighting.currentText()
        print('IMPORT METHOD=', funcoptstring,'WITH (detectorsize,scans,delays)=',str(tuple((detectorsize,scanstring,delaystring))))

        if funcoptstring == 's2s_signal':
                    #try function am ende addieren
                    jsondataset = pyTRIR_pack.TRIR.reload(scanstring,delaystring,detectorsize)

        if funcoptstring == 'weights':
                    print('still under construction')
                    #jsondataset = pyTRIR_pack.weighted_import.init_weightedimport(scanstring,delaystring,detectorsize)

                
        instance.TRIRwnlowlabel = instance.findChild(QtWidgets.QLabel,'TRIRwnlownumber')
        instance.TRIRwnlowlabel.setText(str(round(float(jsondataset['wn'][0]),2)))
        instance.TRIRwnhighlabel = instance.findChild(QtWidgets.QLabel,'wnhighnumber')
        instance.TRIRwnhighlabel.setText(str(round(float(jsondataset['wn'][-1]),2)))

        instance.TRIRlistbox = instance.findChild(QtWidgets.QListWidget,'delaylistbox')
        instance.TRIRlistbox.clear()
        for delaytime in jsondataset['delays']:
                    instance.TRIRlistbox.addItem(f'{delaytime: .3g}')
        instance.TRIRlistbox.setCurrentRow(0)


        instance.TRIRscannumber = instance.findChild(QtWidgets.QLabel,'scannumber_2')
        instance.TRIRscannumber.setText(str(jsondataset['scannumber']))
        instance.TRIRdelaynumber = instance.findChild(QtWidgets.QLabel,'delaynumber')
        instance.TRIRdelaynumber.setText(str(jsondataset['delaynumber']))
        instance.TRIRdelaystring.setText(str(jsondataset['delayslice']))









    def subtractfirstscan(instance):
        global jsondataset
        print('subtracting scan 1')
        newweighteddat = pyTRIR_pack.modify_arrays.sub_delay1(jsondataset['data'])
        jsondataset['data']= newweighteddat


    def generate_background_fit(instance):
        global jsondataset
        instance.TRIRpolyorder = instance.findChild(QtWidgets.QLineEdit,'polyorderedit')
        polyorder = int(instance.TRIRpolyorder.text())
        instance.TRIRpixelslice = instance.findChild(QtWidgets.QLineEdit,'pixelfitedit')
        pixelslice = str(instance.TRIRpixelslice.text())
        instance.TRIRfitdelay = instance.findChild(QtWidgets.QLineEdit,'fitdelaysedit')
        fitdelayslice = str(instance.TRIRfitdelay.text())
        print('BACKGROUND CORRECTION','Polynomial-order=',polyorder,'Pixelslice=',pixelslice,'Fitting to delays=',fitdelayslice)
        bgarraydata,xdata,ydata,polyfitx,polyfity = pyTRIR_bgcorr.TRIRbgcorr(jsondataset,polyorder,fitdelayslice,pixelslice)
        jsondataset['bgdata'] = bgarraydata


        class newwindow(QtWidgets.QMainWindow):
            def __init__(self,parent=None):
                super().__init__(parent)
                uic.loadUi('pyqtwindowfiles/bggenerating_window.ui', self)
                #Frame for the canvas
                self.BGCORRcanvasframe = self.findChild(QtWidgets.QFrame,'canvasframesc')
                self.BGCORRcanvasframe.setMinimumSize(400, 1000)
                #Canvas
                self.BGCORRhorizontalLayout= QtWidgets.QVBoxLayout(self.BGCORRcanvasframe)
                self.BGCORRhorizontalLayout.setObjectName('canvas scroll layout')
                #the canvas
                self.BGCORRfig = plt.figure(figsize=(10,15))
                plt.rc('font', size=3) #controls default text size
                plt.rcParams['xtick.major.pad']='1'
                plt.rcParams['ytick.major.pad']='1'


                delaynumber = 8
                grid = self.BGCORRfig.add_gridspec(int(delaynumber/2+2), 2, hspace=0.01, wspace=0.01,bottom=0.09,top=0.95,left=0.05,right=0.95)
                for i in range(int(delaynumber/2)):
                    delayplot = self.BGCORRfig.add_subplot(grid[i, 0])
                    delaydist = len(jsondataset['delays'])//delaynumber
                    indexx = delaydist * i
                    print(indexx)
                    delayplot.plot(xdata,jsondataset['data'][:,indexx],label= str(round(jsondataset['delays'][indexx],2)),linewidth=.6)
                    delayplot.legend()
                    delayplot.plot(xdata,bgarraydata[:,indexx],color= 'y',linewidth=.6)
                    delayplot.scatter(pyTRIR_bgcorr.cut1d(pixelslice,xdata),pyTRIR_bgcorr.cut1d(pixelslice,bgarraydata[:,indexx]),marker='.',s=1,color='r')
                    #delayplot.yaxis.set_ticklabels([])

                    delayplot2 = self.BGCORRfig.add_subplot(grid[i, 1])
                    indexx2 = int(delaydist * i + delaynumber/2*delaydist)
                    print(indexx2)
                    delayplot2.plot(xdata,jsondataset['data'][:,indexx2],label= str(round(jsondataset['delays'][indexx2],2)),linewidth=.6)
                    delayplot2.legend()
                    delayplot2.plot(xdata,bgarraydata[:,indexx2],color= 'y',linewidth=.6)
                    delayplot2.scatter(pyTRIR_bgcorr.cut1d(pixelslice,xdata),pyTRIR_bgcorr.cut1d(pixelslice,bgarraydata[:,indexx2]),s=1,marker='.',color='r')
                    delayplot2.yaxis.set_ticklabels([])


                latestdelay = self.BGCORRfig.add_subplot(grid[int(delaynumber/2+1)-1:int(delaynumber/2+2), :])
                latestdelay.plot(xdata,jsondataset['data'][:,-1],label= str(round(jsondataset['delays'][-1],2)))
                latestdelay.legend()
                latestdelay.plot(polyfitx,polyfity)
                latestdelay.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

                self.BGCORRcanvas = FigureCanvasQTAgg(self.BGCORRfig) 
                #add canvas to widget
                self.BGCORRhorizontalLayout.addWidget(self.BGCORRcanvas)
                self.BGCORRcanvas.draw()


        topwindow = newwindow(instance)
        topwindow.show()




    def exportdatacomp(instance):
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName()
        pyTRIR_pack.TRIR.exportdata(file_name,jsondataset)

    def exportdata_npyfile(instance):
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName()
        pyTRIR_pack.TRIR.exportdata_to_npyfile(file_name,jsondataset)

    
