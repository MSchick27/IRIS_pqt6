import sys
import numpy as np
from time import strftime,sleep
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg,NavigationToolbar2QT
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import lmfit
import uncertainties



from PyQt6 import QtCore, QtGui, QtWidgets,uic

#own packages
#from IRpackages.TRIR import pyTRIR_pack,pyTRIR_bgcorr
class readonlydelegate(QtWidgets.QStyledItemDelegate):
     def createEditor(self, parent, option, index):
        global currrow
        currrow= index.row()
        print('selected row for plotting',currrow)
        focus_calc.plot_tableclick()
        return


class focus_calc():
    def initer(instance):

        print('focus analyser startup')
        #Frame for the canvas
        instance.focuscanvasframe = instance.findChild(QtWidgets.QFrame,'focuscanvas')
        #Canvas
        if 'plotchecker' in locals() or 'plotchecker' in globals():
            print("no need to regenerate canvas")
        else:
            print("building canvas")
            global plotchecker
            plotchecker=1
            instance.horizontalLayout= QtWidgets.QVBoxLayout(instance.focuscanvasframe)
            instance.horizontalLayout.setObjectName('focuscanvas_h_layout')
            #the canvas
            instance.focusfig = plt.figure(0,figsize=(10, 10))
            plt.rc('font', size=4) #controls default text size
            plt.rcParams['xtick.major.pad']='1'
            plt.rcParams['ytick.major.pad']='1'

            instance.focusgrid = instance.focusfig.add_gridspec(11, 11, hspace=0.01, wspace=0.01,bottom=0.09,top=0.95,left=0.05,right=0.95)
            instance.rawdata_ax = instance.focusfig.add_subplot(instance.focusgrid[0:3, 0:3])
            instance.rawdata_ax.set_title('raw',pad=.1)
            instance.rawdata_ax.set_xlabel('pixels',labelpad=.1)
            instance.rawdata_ax.set_ylabel('pixels',labelpad=.1)
            instance.fit_ax = instance.focusfig.add_subplot(instance.focusgrid[0:3, 4:7],sharey=instance.rawdata_ax,sharex=instance.rawdata_ax)
            instance.fit_ax.set_title('fit',pad=.1)
            instance.residue_ax = instance.focusfig.add_subplot(instance.focusgrid[0:3, 8:11],sharey=instance.rawdata_ax,sharex=instance.rawdata_ax)
            instance.residue_ax.set_title('residue',pad=.1)

            instance.xhist_ax = instance.focusfig.add_subplot(instance.focusgrid[4:7, 0:3])
            instance.xhist_ax.set_title('xhist',pad=.1)
            instance.yhist_ax = instance.focusfig.add_subplot(instance.focusgrid[8:11, 0:3])
            instance.yhist_ax.set_title('yhist',pad=.1)
            #instance.free = instance.focusfig.add_subplot(instance.focusgrid[10:11, 0:3])
            
            instance.d3plot = instance.focusfig.add_subplot(instance.focusgrid[6:11, 4:11])
        
            instance.focuscanvas = FigureCanvasQTAgg(instance.focusfig) 
            instance.focustoolbar= NavigationToolbar2QT(instance.focuscanvas,instance.focuscanvasframe)

            #add canvas to widget
            instance.horizontalLayout.addWidget(instance.focuscanvas)
            instance.horizontalLayout.addWidget(instance.focustoolbar)


        #adding relevance to buttons (commands)
        instance.focusimportbutton = instance.findChild(QtWidgets.QPushButton,'loadbutton')
        instance.focusimportbutton.clicked.connect(lambda:focus_calc.initialize_import(instance))
        instance.focusimportbutton.setToolTip('import one or multiple files for analysis')

        instance.fitreport = instance.findChild(QtWidgets.QCheckBox,'fitrep')

        instance.progressbar = instance.findChild(QtWidgets.QProgressBar,'progressB')
        instance.progressbar.setValue(0)

        instance.tablewidget = instance.findChild(QtWidgets.QTableWidget,'tableWidget')
        instance.tablewidget.setColumnWidth(0,100)
        instance.tablewidget.setColumnWidth(1,40)
        instance.tablewidget.setColumnWidth(2,60)
        instance.tablewidget.setColumnWidth(3,60)
        instance.tablewidget.setColumnWidth(4,60)
        instance.tablewidget.setColumnWidth(5,60)
        instance.tablewidget.setColumnWidth(6,100)

        instance.plot = instance.findChild(QtWidgets.QPushButton,'analyse')
        instance.plot.clicked.connect(lambda:focus_calc.plot_focus(instance))

        instance.colormap = instance.findChild(QtWidgets.QComboBox,'colormapcomb')
    

    """ functions for buttons  """
    
    def initialize_import(instance):
        #try:
            focus_calc.focusimport(instance)
        #except:
        #    print('ERROR importing #..#')

    def focusimport(instance):
        print('import')
        global currrow
        instance.plotdict = dict()
        currrow = 1

        instance.pixelsize = instance.findChild(QtWidgets.QLineEdit,'pixelsize')
        instance.sizefactor = float(instance.pixelsize.text())

        instance.file_dialog = QtWidgets.QFileDialog()
        instance.file_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFiles)  # Set the mode to select an existing file
        #instance.file_dialog.setNameFilter("All files (*.*)")  # Set filters for file all types
        if instance.file_dialog.exec() == QtWidgets.QFileDialog.DialogCode.Accepted:
            file_path = instance.file_dialog.selectedFiles()

        zvals = np.linspace(0,len(file_path)-1,len(file_path))
        instance.focusdataset= dict()
        for i,file in enumerate(file_path):

            instance.progressbar.setValue(int((i+1)/len(file_path)*100))
            data = open(str(file))
            rows = data.readlines()
            arrayaslist= [item.split('\t') for item in rows]
            data_array = np.array(arrayaslist,dtype=float)
            instance.plotdict[str(i)]={'raw':data_array}
            print(file,np.shape(data_array),'#loaded successfully')


            Xfwhm,Xfwhmerr,Yfwhm,Yfwhmerr,chisqr,fitdata = focus_calc.fitgauss(instance,data_array)

            instance.plotdict[str(i)]['fit']=fitdata
            filedict = dict()
            filename = file.split('/')[-1]
            #print(filename)
            filedict['name']=filename
            filedict['Xfwhm']= Xfwhm
            filedict['Xerr']= Xfwhmerr
            filedict['Yfwhm']= Yfwhm
            filedict['Yerr']= Yfwhmerr
            filedict['chisqr']= chisqr
            #print(filedict)

            instance.focusdataset[str(zvals[i])] = filedict

        #setting row number and make columns noteditable
        instance.tablewidget.setRowCount(len(instance.focusdataset))
        delegate = readonlydelegate(instance.tablewidget)
        instance.tablewidget.setItemDelegateForColumn(0, delegate)
        instance.tablewidget.setItemDelegateForColumn(2, delegate) 
        instance.tablewidget.setItemDelegateForColumn(3, delegate) 
        instance.tablewidget.setItemDelegateForColumn(4, delegate)  
        instance.tablewidget.setItemDelegateForColumn(5, delegate) 
        instance.tablewidget.setItemDelegateForColumn(6, delegate)   

        for i,zvalue in enumerate(instance.focusdataset):
            row = i
            instance.tablewidget.setItem(row,0,QtWidgets.QTableWidgetItem(instance.focusdataset[zvalue]['name']))
            z = f'{float(zvalue):.3g}'
            instance.tablewidget.setItem(row,1,QtWidgets.QTableWidgetItem(z))
            xfwhm = instance.focusdataset[zvalue]['Xfwhm']
            item_xfwhm = QtWidgets.QTableWidgetItem(f'{xfwhm:.3g}')
            instance.tablewidget.setItem(row,2,item_xfwhm)
            xfwhmerr = instance.focusdataset[zvalue]['Xerr']
            instance.tablewidget.setItem(row,3,QtWidgets.QTableWidgetItem(f'{xfwhmerr:.3g}'))
            yfwhm = instance.focusdataset[zvalue]['Yfwhm']
            instance.tablewidget.setItem(row,4,QtWidgets.QTableWidgetItem(f'{yfwhm:.3g}'))
            yfwhmerr = instance.focusdataset[zvalue]['Yerr']
            instance.tablewidget.setItem(row,5,QtWidgets.QTableWidgetItem(f'{yfwhmerr:.3g}'))
            chisqr = instance.focusdataset[zvalue]['chisqr']
            instance.tablewidget.setItem(row,6,QtWidgets.QTableWidgetItem(f'{chisqr:.6g}'))

        xlow,xhigh = 0,np.shape(instance.plotdict[str(currrow)]['raw'])[1]
        ylow,yhigh = 0,np.shape(instance.plotdict[str(currrow)]['raw'])[0]
        print('############',xlow,xhigh)
        instance.rawdata_ax.set_xlim(xlow,xhigh)
        instance.rawdata_ax.set_ylim(ylow,yhigh)


    def fitgauss(instance,dataarray):
        global xpix,ypix
        xpix,ypix = np.meshgrid(np.arange(dataarray.shape[1]),np.arange(dataarray.shape[0]))

        def objectivefunction_residuals(params, data,y0):
                height = params['height']
                x = params['x']
                y = params['y']
                sigmax = params['sigmax']
                sigmay = params['sigmay']
    
                model = height*np.exp(-(((x-xpix)/sigmax)**2+((y-ypix)/sigmay)**2)/2)
                return (data-model)
            
        #guesses
        height = np.max(dataarray)
        d1_index = np.argmax(dataarray)
        d2_index = np.unravel_index(d1_index,dataarray.shape)

        params = lmfit.Parameters()
        params.add('height', value=height)
        params.add('x', value=int(d2_index[1])) #estimated from looking at the amp values
        params.add('y', value=int(d2_index[0]))
        params.add('sigmax', value=10)
        params.add('sigmay', value=10)

        minner = lmfit.Minimizer(objectivefunction_residuals, params, fcn_args=([dataarray,1]))
        result = minner.minimize()
        if instance.fitreport.isChecked() == True:
            lmfit.report_fit(result)

        fit = result.params.valuesdict()
        fiterrors = result.params.create_uvars(covar=None)
        chisqr = result.chisqr
        Xfwhm= np.abs(fit['sigmax'])*2.33* instance.sizefactor
        Xfwhmerr = float(result.params['sigmax'].stderr)*2.33*instance.sizefactor
        Yfwhm= np.abs(fit['sigmay'])*2.33* instance.sizefactor
        Yfwhmerr = float(result.params['sigmay'].stderr)*2.33*instance.sizefactor

        bestfit = fit["height"]*np.exp(-(((fit["x"]-xpix)/fit["sigmax"])**2+((fit["y"]-ypix)/fit["sigmay"])**2)/2)
        global globalplaceholder
        globalplaceholder = instance
        return Xfwhm,Xfwhmerr,Yfwhm,Yfwhmerr,chisqr,bestfit













    def plot_tableclick():
        focus_calc.plot_focus(globalplaceholder)

    def clearplot(instance):
        xlow,xhigh = instance.rawdata_ax.get_xlim()
        ylow,yhigh = instance.rawdata_ax.get_ylim()
        instance.rawdata_ax.clear()
        instance.fit_ax.clear()
        instance.xhist_ax.cla()
        instance.yhist_ax.cla()


        instance.rawdata_ax.set_title('raw',pad=.1)
        instance.rawdata_ax.set_xlabel('pixels',labelpad=.1)
        instance.rawdata_ax.set_ylabel('pixels',labelpad=.1)
        instance.rawdata_ax.set_xlim(xlow,xhigh)
        instance.rawdata_ax.set_ylim(ylow,yhigh)

        instance.fit_ax.set_title('fit',pad=.1)
        instance.residue_ax.set_title('residue',pad=.1)

        instance.xhist_ax.set_xlabel('xaxis',labelpad=.1)
        instance.xhist_ax.set_xlim(xlow,xhigh)
        instance.yhist_ax.set_xlabel('yaxis',labelpad=.1)
        instance.yhist_ax.set_xlim(ylow,yhigh)

        instance.focuscanvas.draw()



    def plot_focus(instance):
        focus_calc.clearplot(instance)
        maximum = np.max(instance.plotdict[str(currrow)]['raw'])
        colormap = str(instance.colormap.currentText())
        instance.rawdata_ax.imshow(instance.plotdict[str(currrow)]['raw'],cmap=colormap)
        instance.fit_ax.imshow(instance.plotdict[str(currrow)]['fit'],cmap=colormap)
        instance.residue_ax.imshow(np.subtract(instance.plotdict[str(currrow)]['fit'],instance.plotdict[str(currrow)]['raw']),cmap='magma')

        d1_index = np.argmax(instance.plotdict[str(currrow)]['fit'])
        d2_index = np.unravel_index(d1_index,np.shape(instance.plotdict[str(currrow)]['fit']))

        instance.xhist_ax.plot(instance.plotdict[str(currrow)]['raw'][d2_index[0],:],label='raw',color='k',linewidth=0.6)
        instance.xhist_ax.plot(instance.plotdict[str(currrow)]['fit'][d2_index[0],:],label='fit',color='r',linewidth=0.6)
        instance.xhist_ax.legend()
        instance.xhist_ax.grid(True)

        instance.yhist_ax.plot(instance.plotdict[str(currrow)]['raw'][:,d2_index[1]],label='raw',color='k',linewidth=0.6)
        instance.yhist_ax.plot(instance.plotdict[str(currrow)]['fit'][:,d2_index[1]],label='fit',color='r',linewidth=0.6)
        instance.yhist_ax.legend()
        instance.yhist_ax.grid(True)


        


        """ instance.focusgrid = instance.focusfig.add_gridspec(11, 11, hspace=0.01, wspace=0.01,bottom=0.09,top=0.95,left=0.05,right=0.95)
        instance.rawdata_ax = instance.focusfig.add_subplot(instance.focusgrid[0:3, 0:3])
        instance.rawdata_ax.set_title('raw',pad=.1)
        instance.rawdata_ax.set_xlabel('pixels',labelpad=.1)
            instance.rawdata_ax.set_ylabel('pixels',labelpad=.1)
            instance.fit_ax = instance.focusfig.add_subplot(instance.focusgrid[0:3, 4:7],sharey=instance.rawdata_ax,sharex=instance.rawdata_ax)
            instance.fit_ax.set_title('fit',pad=.1)
            instance.residue_ax = instance.focusfig.add_subplot(instance.focusgrid[0:3, 8:11],sharey=instance.rawdata_ax,sharex=instance.rawdata_ax)
            instance.residue_ax.set_title('residue',pad=.1)

            instance.xhist_ax = instance.focusfig.add_subplot(instance.focusgrid[4:6, 0:3])
            instance.xhist_ax.set_title('xhist',pad=.1)
            instance.yhist_ax = instance.focusfig.add_subplot(instance.focusgrid[7:9, 0:3])
            instance.yhist_ax.set_title('yhist',pad=.1)
            instance.free = instance.focusfig.add_subplot(instance.focusgrid[10:11, 0:3])
            
            instance.d3plot = instance.focusfig.add_subplot(instance.focusgrid[7:11, 4:11]) """
        instance.focuscanvas.draw()
