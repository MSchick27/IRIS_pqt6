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
from IRpackages.TRIR import TRIRwidgets


class TRIRviewer_widgets_defining():
    def initer(instance):
        global jsondataset
        jsondataset = TRIRwidgets.getjsonfile()
        print('startup TRIR viewer')
        '''this part is added to build the plotting canvas inside the ui'''

        #Canvas
        if 'CheckerTRIRviewer' in locals() or 'CheckerTRIRviewer' in globals():
            print("no need to regenerate canvas")
        else:
            print("building canvas")
            global CheckerTRIRviewer
            CheckerTRIRviewer=1
            instance.TRIRcanvasframeviewer = instance.findChild(QtWidgets.QFrame,'canvasframeviewer')
            instance.horizontalLayoutviewer= QtWidgets.QVBoxLayout(instance.TRIRcanvasframeviewer)
            instance.horizontalLayoutviewer.setObjectName('canvas_viewer_layout')
            #the canvas

            instance.TRIRviewer_fig = plt.figure(2,figsize=(10, 10))
            plt.rc('font', size=4) #controls default text size
            plt.rcParams['xtick.major.pad']='1'
            plt.rcParams['ytick.major.pad']='1'

            instance.grid = instance.TRIRviewer_fig.add_gridspec(6, 6, hspace=0.01, wspace=0.01,bottom=0.09,top=0.95,left=0.05,right=0.95)
            instance.main_ax_viewer = instance.TRIRviewer_fig.add_subplot(instance.grid[1:, 1:])
            instance.y_hist = instance.TRIRviewer_fig.add_subplot(instance.grid[1:, 0],sharey=instance.main_ax_viewer)
            instance.x_hist = instance.TRIRviewer_fig.add_subplot(instance.grid[0, 1:-1], sharex=instance.main_ax_viewer)
       
            instance.main_ax_viewer.set_xlabel('time [ps]')
            instance.y_hist.set_ylabel(r'wn [$cm^{-1}$]')
            instance.x_hist.xaxis.set_visible(False)
        
            instance.canvasviewer = FigureCanvasQTAgg(instance.TRIRviewer_fig) 
            instance.toolbar= NavigationToolbar2QT(instance.canvasviewer,instance.TRIRcanvasframeviewer)

            #add canvas to widget
            instance.horizontalLayoutviewer.addWidget(instance.canvasviewer)
            instance.horizontalLayoutviewer.addWidget(instance.toolbar)




        #adding relevance to buttons (commands)
        instance.TRIRviewerplotbutton = instance.findChild(QtWidgets.QPushButton,'plotbuttonviewer')
        instance.TRIRviewerplotbutton.clicked.connect(lambda:TRIRviewer_widgets_defining.plotplot(instance))

        instance.TRIRviewersethistogram = instance.findChild(QtWidgets.QPushButton,'sethistbutton')
        instance.TRIRviewersethistogram.clicked.connect(lambda:TRIRviewer_widgets_defining.setyhist(instance))


        instance.TRIRviewerxhistedit = instance.findChild(QtWidgets.QLineEdit,'xhistedit')
        wn_mid = (jsondataset['wn'])[int(len(jsondataset['wn'])//2)]
        instance.TRIRviewerxhistedit.setText(str(round(wn_mid,2)))

        instance.TRIRvieweryhistedit = instance.findChild(QtWidgets.QLineEdit,'yhistedit')
        instance.TRIRvieweryhistedit.setText(str(round(jsondataset['delays'][-1],2)))

        instance.TRIRviewercombobox = instance.findChild(QtWidgets.QComboBox,'colormapcombobox')
        instance.TRIRviewerlevels = instance.findChild(QtWidgets.QLineEdit,'levels')
        instance.TRIRviewercutoffedit = instance.findChild(QtWidgets.QLineEdit,'cutoff')
        instance.TRIRviewermaxfacedit = instance.findChild(QtWidgets.QLineEdit,'maxfacentry')
        
        instance.TRIRviewerlogscale = instance.findChild(QtWidgets.QCheckBox,'Logcheckboxviewer')
        instance.TRIRviewergridcheck = instance.findChild(QtWidgets.QCheckBox,'Gridcheckbox')






    '''####################################################################################
    This part from now will be all about defining functions for the different buttons above
    ####################################################################################'''

    def plotviewer(instance):
        print('plot started')

    def setyhist(instance):
                points = plt.ginput(n=1,timeout=30, show_clicks=True, mouse_add = plt.MouseButton.LEFT,mouse_pop= plt.MouseButton.RIGHT,mouse_stop = plt.MouseButton.MIDDLE)
                points = points[0]
                instance.TRIRviewerxhistedit.setText(str(round(points[1],4)))
                instance.TRIRvieweryhistedit.setText(str(round(points[0],4)))

    
    def clearplot(instance):
        instance.main_ax_viewer.cla()
        instance.y_hist.cla()
        instance.x_hist.cla()
        #plt.clf()
        try:
            instance.cb.remove()
        except:
            print('colorbar not defined')
        
        instance.canvasviewer.draw()

    def plotplot(instance):
        print('plotting')
        TRIRviewer_widgets_defining.clearplot(instance)
        colormap= str(instance.TRIRviewercombobox.currentText())
        levelnum = int(instance.TRIRviewerlevels.text())
        cutoff =str(instance.TRIRviewercutoffedit.text())
        maxfac = str(instance.TRIRviewermaxfacedit.text())
        logscale = instance.TRIRviewerlogscale.isChecked()
        if colormap == 'custom':
            colormap = pyTRIR_pack.colormapsfor_TRIR.create_custom_colormap('#0056AC', '#AC0000')


        databgsub = pyTRIR_pack.modify_arrays.subtract_bg(jsondataset)

        if cutoff=='':
            maxval,minval= pyTRIR_pack.colormapsfor_TRIR.findmaxval(databgsub,maxfac)
        else:
            maxval,minval= float(cutoff),-1*float(cutoff)

        if logscale== False:
            instance.main_ax_viewer.set_xlim(np.min(jsondataset['delays']),np.max(jsondataset['delays']))
        if logscale== True:
            print('plotting with xaxis = LOG10')
            instance.main_ax_viewer.set_xscale('log')
            xaxislow,xaxishigh = pyTRIR_pack.modify_arrays.getlogaxis(jsondataset)
            instance.main_ax_viewer.set_xlim(xaxislow,xaxishigh)

        clipped_databgsub = np.clip(databgsub,a_min=minval,a_max = maxval)
        norm = colors.Normalize(vmin=minval, vmax=maxval)
        levelscontour = np.linspace(minval,maxval,levelnum)
        #mainplot
        map=instance.main_ax_viewer.contourf(jsondataset['delays'],jsondataset['wn'],clipped_databgsub,levels=levelscontour,cmap=colormap)#,vmin=minval,vmax=maxval,alpha=1)
        print('Max '+str(maxval) + ' & min '+str(minval))

        #global cb
        #map=main_ax.contourf(clipped_databgsub,levels=levelnum,cmap=colormap)
        instance.cb = instance.fig.colorbar(map, ax=instance.main_ax_viewer,norm=norm)

        yhistdelay = float(instance.TRIRvieweryhistedit.text())
        yhistdelayindex = np.argmin(abs(jsondataset['delays']-yhistdelay))
        yhistdelay = jsondataset['delays'][yhistdelayindex]
        instance.TRIRvieweryhistedit.setText(str(round(yhistdelay,2)))
        yhisto = clipped_databgsub[:,yhistdelayindex]
                

        xhistwn = float(instance.TRIRviewerxhistedit.text())
        xhistwnindex = np.argmin(abs(jsondataset['wn']-xhistwn))
        xhistwn = jsondataset['wn'][xhistwnindex]
        instance.TRIRviewerxhistedit.setText(str(round(xhistwn,4)))
        
        instance.xhistcombobox = instance.findChild(QtWidgets.QComboBox,'xaxiscombobox')
        if instance.xhistcombobox.currentText() == 'signal':
            xhisto = clipped_databgsub[xhistwnindex,:]
        else:
            xhisto = np.sum(np.abs(clipped_databgsub),axis=0)


        #plot setup axis
        instance.main_ax_viewer.set_xlabel('time [ps]')
        instance.main_ax_viewer.set_ylabel(r'wavenumber $cm^{-1}$')
        instance.main_ax_viewer.set_ylim(np.min(jsondataset['wn']),np.max(jsondataset['wn']))
        

        instance.main_ax_viewer.grid(instance.TRIRviewergridcheck.isChecked())

        instance.cb.set_label(r'∆OD', rotation=270)   

        instance.y_hist.plot(yhisto,jsondataset['wn'],linewidth=.6)
        instance.y_hist.invert_xaxis()
        instance.y_hist.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        instance.y_hist.set_xlabel('∆OD')

        instance.x_hist.plot(jsondataset['delays'],xhisto,linewidth=.6)
        instance.x_hist.ticklabel_format(style='sci', axis='y', scilimits=(2,0))
        instance.x_hist.set_ylabel('∆OD')
        instance.y_hist.set_ylabel(r'wavenumber $cm^{-1}$')

        instance.canvasviewer.draw()
    
