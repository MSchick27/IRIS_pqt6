import sys
import os
import numpy as np
import pandas as pd
from time import strftime,sleep
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg,NavigationToolbar2QT
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



from PyQt6 import QtCore, QtGui, QtWidgets,uic
from PyQt6.QtCore import QMimeData


#own packages
from IRpackages.FTIR import FTIR_init_dicts,pyFTIR_pack



class FTIR_widgets_():
    def initer(instance):
        #global jsondict
        instance.jsondict = {}

        #own custom Listwidget to get Drag and drop function to work
        class CustomListWidget(QtWidgets.QListWidget):
            '''Own listwidget class to specify actions on drop event'''
            def __init__(self, parent=None):
                super().__init__(parent)
                self.setAcceptDrops(True)
                self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
                self.setFrameShape(QtWidgets.QListWidget.Shape.NoFrame)
                palette = QtGui.QPalette()
                palette.setColor(palette.ColorGroup.Normal, palette.ColorRole.Highlight, QtGui.QColor("blueviolet"))
                self.setPalette(palette)

            def dragEnterEvent(self, event):
                if True:#event.mimeData().hasUrls():
                    event.acceptProposedAction()

            def dragMoveEvent(self, event):
                if event.mimeData().hasUrls():
                    event.acceptProposedAction()

            def dropEvent(self, event):
                if event.mimeData().hasUrls():
                    urls = event.mimeData().urls()
                    filepaths = []
                    for url in urls:
                        file_path = url.toLocalFile()
                        filepaths.append(file_path)
                        filename = os.path.basename(file_path)
                        self.addItem(filename)

                    FTIR_widgets_.importdptfile(filepaths,instance)

        """ 
            def eventFilter(instance,source,event):
                if event.type() == QtCore.QEvent.ContextMenu and source is instance.listWidget:
                    menu = QtWidgets.QMenu()
                    menu.addAction('delete')
                
                    if menu.exec_(event.globalPos()):
                        item =source.itemAt(event.pos())
                        print(item.text())
                
                    return True
            
                return super().eventFilter(source,event) """
        

        def openItemContextMenu(pos):
            
            item = instance.listWidget.itemAt(pos)
            if item is not None:
                context_menu = QtWidgets.QMenu(instance.listWidget)
                #edit_action = QtGui.QAction("Edit", context_menu)
                delete_action = QtGui.QAction("Delete", context_menu)

                # Connect actions to respective functions
                delete_action.triggered.connect(lambda: FTIR_widgets_.deleteItem(instance,item))
                context_menu.addAction(delete_action)

                context_menu.exec(instance.listWidget.mapToGlobal(pos))












        instance.listboxframe= instance.findChild(QtWidgets.QFrame,'listboxframe')
        instance.hl_listbox= QtWidgets.QVBoxLayout(instance.listboxframe)
        instance.hl_listbox.setObjectName('canvas_h_layout')
        instance.listWidget = CustomListWidget()
        instance.hl_listbox.addWidget(instance.listWidget)
        instance.listboxframe.setFrameStyle(QtWidgets.QFrame.Shape.NoFrame)
        instance.listWidget.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        instance.listWidget.customContextMenuRequested.connect(openItemContextMenu)

        
        
        instance.listWidget.itemClicked.connect(lambda: FTIR_widgets_.selectitem_reload(instance))
        instance.previous_item = None

        instance.init_dict = FTIR_init_dicts.init()
        print('buildingFTIR module with instance as self variable')
        '''this part is added to build the plotting canvas inside the ui'''
        #Frame for the canvas
        instance.FTIRcanvasframe = instance.findChild(QtWidgets.QFrame,'FTIRcanvaframe')
        #Canvas
        if 'CheckerTRIR' in locals() or 'CheckerTRIR' in globals():
            print("no need to regenerate canvas")
        else:
            print("building canvas")
            global CheckerFTIR
            CheckerFTIR=1
            instance.horizontalLayout= QtWidgets.QVBoxLayout(instance.FTIRcanvasframe)
            instance.horizontalLayout.setObjectName('canvas_ftir_layout')

            #the canvas
            instance.FTIRfig = plt.figure(3,figsize=(10, 10))
            plt.rc('font', size=4) #controls default text size
            plt.rcParams['xtick.major.pad']='1'
            plt.rcParams['ytick.major.pad']='1'
            instance.grid = instance.FTIRfig.add_gridspec(3,1, hspace=0.02, wspace=0.03,bottom=0.09,top=0.93,left=0.1,right=0.95)
            instance.main_FTIR = instance.FTIRfig.add_subplot(instance.grid[0:2,0])
            instance.second_FTIR = instance.FTIRfig.add_subplot(instance.grid[2,0],sharex=instance.main_FTIR)
            instance.axes = [instance.main_FTIR,instance.second_FTIR]
            instance.axes[0].set_xlabel(instance.init_dict['xlabel'],labelpad=.1)
            instance.axes[0].set_ylabel(instance.init_dict['ylabel'],labelpad=.1)
            instance.axes[1].set_xlabel(instance.init_dict['xlabel'],labelpad=.1)
            instance.axes[1].set_ylabel(instance.init_dict['ylabel'],labelpad=.1)
            
        
            instance.canvasFTIR = FigureCanvasQTAgg(instance.FTIRfig) 
            instance.toolbar= NavigationToolbar2QT(instance.canvasFTIR,instance.FTIRcanvasframe)

            #add canvas to widget
            instance.horizontalLayout.addWidget(instance.canvasFTIR)
            instance.horizontalLayout.addWidget(instance.toolbar)

            instance.delimiter = instance.findChild(QtWidgets.QComboBox,'delimitercombobox')
            global delimitervar
            delimitervar = str(instance.delimiter.currentText())



        '''part for edits above lowerframe'''
        instance.yfactoredit = instance.findChild(QtWidgets.QLineEdit,'yfactoredit')
        instance.xlowedit = instance.findChild(QtWidgets.QLineEdit,'xlimitlowedit')
        instance.xhighedit = instance.findChild(QtWidgets.QLineEdit,'xlimithighedit')
        instance.xlabeledit = instance.findChild(QtWidgets.QLineEdit,'xlabeledit')
        instance.ylowedit = instance.findChild(QtWidgets.QLineEdit,'ylimitlowedit')
        instance.yhighedit = instance.findChild(QtWidgets.QLineEdit,'ylimithighedit')
        instance.ylabeledit = instance.findChild(QtWidgets.QLineEdit,'ylabeledit')

        instance.legendcheckbox= instance.findChild(QtWidgets.QCheckBox,'legendcheckbox')
        instance.gridcheckbox= instance.findChild(QtWidgets.QCheckBox,'gridcheckbox')
        #adding relevance to buttons (commands)
        '''This part includes all Buttons above the lower frame
        Here they are initiated as variables toconnect functions onclicked and set Enabled statues after import'''
        instance.FTIRplotbutton = instance.findChild(QtWidgets.QPushButton,'FTIRrefreshbutton')
        instance.FTIRplotbutton.clicked.connect(lambda:FTIR_widgets_.plotFTIR(instance))
        instance.FTIRaxisbutton = instance.findChild(QtWidgets.QPushButton,'axisbutton')
        instance.FTIRaxisbutton.clicked.connect(lambda:FTIR_widgets_.setAXIS(instance))
        instance.FTIRaxisbutton.setShortcut('Ctrl+a')
        instance.FTIRaxisbutton.setToolTip('Shortcut Ctr+a')
        



        #defining all controls to variables
        '''This part initiates all control buttons in the lower panel into variables to read out or set later when changing labels'''
        instance.showcheckbox = instance.findChild(QtWidgets.QCheckBox,'showcheckbox')
        instance.labeledit = instance.findChild(QtWidgets.QLineEdit,'filenameedit')
        instance.plotcombobox = instance.findChild(QtWidgets.QComboBox,'plotcomboBox')
        instance.bgcheckbox = instance.findChild(QtWidgets.QCheckBox,'BGcheckbox')
        instance.bgkeyedit = instance.findChild(QtWidgets.QLineEdit,'bgkeyedit')
        instance.colorcombobox = instance.findChild(QtWidgets.QComboBox,'colorcomboBox')
        instance.linewidth = instance.findChild(QtWidgets.QDoubleSpinBox,'linewidth')
        instance.linestylecombobox = instance.findChild(QtWidgets.QComboBox,'linestylecombobox')

        #bgslider needs special care
        instance.bgscaleslider = instance.findChild(QtWidgets.QSlider,'bgslider')
        instance.bgscalelabel = instance.findChild(QtWidgets.QLabel,'bgscalelabel')
        instance.bgscaleslider.valueChanged.connect(lambda: FTIR_widgets_.slideraction.bgslider_label(instance))           #ad function to change label for slider value
        instance.lowerslider = instance.findChild(QtWidgets.QPushButton,'lowerslider')
        instance.lowerslider.clicked.connect(lambda: FTIR_widgets_.slideraction.bglowersliderone(instance))
        instance.lowerslider.setShortcut('Ctrl+u')
        instance.lowerslider.setToolTip('Shortcut Ctr+u')
        instance.higherslider = instance.findChild(QtWidgets.QPushButton,'higherslider')
        instance.higherslider.clicked.connect(lambda: FTIR_widgets_.slideraction.bghighersliderone(instance))
        instance.higherslider.setShortcut('Ctrl+i')
        instance.higherslider.setToolTip('Shortcut Ctr+i')



        instance.lowframe = instance.findChild(QtWidgets.QFrame,'lowerframe')

        '''defining buttons from lowerframe'''
        instance.labelchangebutton = instance.findChild(QtWidgets.QPushButton,'filenamechangebutton')
        instance.labelchangebutton.clicked.connect(lambda: FTIR_widgets_.change_key(instance))
        instance.exportdatabutton = instance.findChild(QtWidgets.QPushButton,'exportdatabutton')
        instance.exportdatabutton.clicked.connect(lambda: FTIR_widgets_.export_data(instance))

        instance.promedit = instance.findChild(QtWidgets.QLineEdit,'promedit')
        instance.peakbutton = instance.findChild(QtWidgets.QPushButton,'peaksbutton')
        instance.peakbutton.clicked.connect(lambda: FTIR_widgets_.findpeaks(instance))
        instance.invertbutton = instance.findChild(QtWidgets.QPushButton,'invertbutton')
        instance.invertbutton.clicked.connect(lambda: FTIR_widgets_.invertdata(instance))
        instance.secdevbutton = instance.findChild(QtWidgets.QPushButton,'secdevbutton')
        instance.secdevbutton.clicked.connect(lambda: FTIR_widgets_.sec_dev(instance))
        instance.secdevbutton.setToolTip('Calculate the second derivative from Spectrum')
        instance.sliceedit = instance.findChild(QtWidgets.QLineEdit,'sliceedit')
        instance.sliceedit.setToolTip('define regions for task, Format  xlow,xhigh')
        instance.cutbutton = instance.findChild(QtWidgets.QPushButton,'cutbutton')
        instance.cutbutton.clicked.connect(lambda: FTIR_widgets_.cut(instance))
        instance.cutbutton.setToolTip('function to cut out a specific region of Spectrum and save in new file')
        instance.trapzbutton = instance.findChild(QtWidgets.QPushButton,'trapzbutton')
        instance.trapzbutton.clicked.connect(lambda: FTIR_widgets_.show_trapz(instance))
        instance.trapzbutton.setToolTip('Calculate integral of spectrum via Trapz-rule')
        instance.polydegedit = instance.findChild(QtWidgets.QLineEdit,'polydegedit')
        instance.polydegedit.setToolTip('define Polynomial degree for baclground coreection')
        instance.polyfitsliceedit = instance.findChild(QtWidgets.QLineEdit,'polyfitsliceedit')
        instance.polyfitsliceedit.setToolTip('define spectral area contributing for bg correction in format:    xlow1,xhigh1,xlow2,xhigh2  ')
        instance.polyfitbutton = instance.findChild(QtWidgets.QPushButton,'polyfitbutton')
        instance.polyfitbutton.clicked.connect(lambda: FTIR_widgets_.polyfit_bgcorrection(instance))
        instance.polyfitbutton.setToolTip('Start polyfit function for bg correction')














    def deleteItem(instance,item):
        instance.listWidget.takeItem(instance.listWidget.row(item))
        instance.jsondict.pop(item.text(), None)
        instance.previous_item = None
        instance.updatestatusbar(str('deleted:'+item.text()),4000,False)
        if instance.listWidget.count()==0:
            instance.FTIRplotbutton.setEnabled(False)
            instance.FTIRaxisbutton.setEnabled(False)
            instance.lowframe.setEnabled(False)
        else:
            instance.listWidget.setCurrentRow(0)

    def importdptfile(filepaths,instance):
        print('import file')
        instance.FTIRplotbutton.setEnabled(True)
        instance.FTIRaxisbutton.setEnabled(True)
        instance.lowframe.setEnabled(True)

        #global instance.jsondict
        try:
            instance.jsondict = pyFTIR_pack.import_and_export_functions.import_files(filepaths,delimitervar,instance.jsondict)
            instance.updatestatusbar(str('Succesfull import for '+str(len(instance.jsondict))+' files'),0,True)
        except:
            print('error while importing, check if right file type / delimiter are choosen correctly')




    class slideraction:
        def bgslider_label(instance):
            instance.value = float(instance.bgscaleslider.value())
            instance.bgscalelabel.setText(f'{float(instance.value*0.001): .3f}')
    
        def bglowersliderone(instance):
            instance.value = int(instance.bgscaleslider.value())
            instance.value = int(instance.value - 1)
            instance.bgscaleslider.setValue(instance.value)
            FTIR_widgets_.plotFTIR(instance)

        def bghighersliderone(instance):
            instance.value = int(instance.bgscaleslider.value())
            instance.value = int(instance.value + 1)
            instance.bgscaleslider.setValue(instance.value)
            FTIR_widgets_.plotFTIR(instance)


    def selectitem_reload(instance):
        """function to save all settings when item in listbox is changed
        this brings convience, since on clickevent chang of Listbox all values are saved
        for the  previous file in jsondict and reset for the clicked one
           """
        #global jsondict
        current_item = instance.listWidget.currentItem()
        if current_item:
            instance.current_text = current_item.text()
            print(f"selected file: {instance.current_text}")

            if instance.previous_item != None:
                if instance.previous_item !=instance.current_text and instance.previous_item in instance.jsondict:
                #get all values set now and save to json then load new
                    instance.jsondict[instance.previous_item]['show']=  instance.showcheckbox.isChecked()  #gets value from checkbutton show and saves it in jsondict
                    instance.jsondict[instance.previous_item]['label'] = str(instance.labeledit.text())  
                    instance.jsondict[instance.previous_item]['subplot'] = str(instance.plotcombobox.currentText())
                    instance.jsondict[instance.previous_item]['bg'] = instance.bgcheckbox.isChecked()
                    instance.jsondict[instance.previous_item]['bgkey'] = str(instance.bgkeyedit.text())
                    instance.jsondict[instance.previous_item]['bgscale'] = float(instance.bgscaleslider.value()) * 0.001
                    instance.jsondict[instance.previous_item]['color'] = str(instance.colorcombobox.currentText())
                    instance.jsondict[instance.previous_item]['linewidth'] = float(instance.linewidth.value())
                    instance.jsondict[instance.previous_item]['linestyle'] = str(instance.linestylecombobox.currentText())
                    print(instance.jsondict[instance.previous_item]['subplot'])
                    print('saved present settings for',instance.previous_item)
            
            instance.previous_item= instance.current_text  #changing previousitem to present for next click

            #now we need to reload the frame with the currenitem settings from json
            instance.showcheckbox.setChecked(instance.jsondict[instance.current_text]['show'])
            instance.labeledit.setText(str(instance.jsondict[instance.current_text]['label']))
            instance.plotcombobox.setCurrentText(str(instance.jsondict[instance.current_text]['subplot']))
            instance.bgcheckbox.setChecked(instance.jsondict[instance.current_text]['bg'])
            instance.bgkeyedit.setText(instance.jsondict[instance.current_text]['bgkey'])
            instance.bgscaleslider.setValue(int(float(instance.jsondict[instance.current_text]['bgscale'])*1000))
            instance.colorcombobox.setCurrentText(instance.jsondict[instance.current_text]['color'])
            instance.linewidth.setValue(instance.jsondict[instance.current_text]['linewidth'])
            instance.linestylecombobox.setCurrentText(instance.jsondict[instance.current_text]['linestyle'])

            FTIR_widgets_.resetlistwidget_items_colorscheme(instance)

        else:
            print("Warning", "No item selected")

        
    def selectitem_reloadplot(instance):
        """function to save all settings beforeplotting
           """
        #global jsondict
        current_item = instance.listWidget.currentItem()
        if current_item:
            instance.current_text = current_item.text()
            #print(f"selected file: {instance.current_text}")
            instance.jsondict[instance.current_text]['show']=  instance.showcheckbox.isChecked()  #gets value from checkbutton show and saves it in jsondict
            instance.jsondict[instance.current_text]['label'] = str(instance.labeledit.text())  
            instance.jsondict[instance.current_text]['subplot'] = str(instance.plotcombobox.currentText())
            instance.jsondict[instance.current_text]['bg'] = instance.bgcheckbox.isChecked()
            instance.jsondict[instance.current_text]['bgkey'] = str(instance.bgkeyedit.text())
            instance.jsondict[instance.current_text]['bgscale'] = float(instance.bgscaleslider.value()) * 0.001
            instance.jsondict[instance.current_text]['color'] = str(instance.colorcombobox.currentText())
            instance.jsondict[instance.current_text]['linewidth'] = float(instance.linewidth.value())
            instance.jsondict[instance.current_text]['linestyle'] = str(instance.linestylecombobox.currentText())


        else:
            print("Warning", "No item selected")

        FTIR_widgets_.resetlistwidget_items_colorscheme(instance)


    def resetlistwidget_items_colorscheme(instance):
        for index in range(instance.listWidget.count()):
            item = instance.listWidget.item(index)
            bgcolor= instance.jsondict[item.text()]['color']
            item.setBackground(QtGui.QColor(bgcolor))
            #item.setForeground(QtGui.QColor(bgcolor))
            if instance.jsondict[item.text()]['show']==False:
                item.setForeground(QtGui.QColor('grey'))
                item.setBackground(QtGui.QColor('black'))
            else:
                item.setForeground(QtGui.QColor('white'))












    def setAXIS(instance):
                points = plt.ginput(n=3,timeout=30, show_clicks=True, mouse_add = plt.MouseButton.LEFT,mouse_pop= plt.MouseButton.RIGHT,mouse_stop = plt.MouseButton.MIDDLE)
                print(points)
                x_margin = [round(float(points[0][0]),6),round(float(points[1][0]),6)]
                x_margin.sort()
                y_margin = [round(float(points[0][1]),6),round(float(points[1][1]),6)]
                y_margin.sort()
                #print(x_margin)
                instance.xlowedit.setText(str(x_margin[0]))
                instance.xhighedit.setText(str(x_margin[1]))
                instance.ylowedit.setText(str(y_margin[0]))
                instance.yhighedit.setText(str(y_margin[1]))
                instance.updatestatusbar('New AXIS set for',0,True)
                FTIR_widgets_.plotFTIR(instance)




    def getxy(instance,datakey):
                x = list(instance.jsondict[datakey]['xdata'])
                y = list(instance.jsondict[datakey]['ydata'])
                bgdatakey = instance.jsondict[datakey]['bgkey']
                if instance.jsondict[datakey]['bg'] == True:
                    xbg= list(instance.jsondict[bgdatakey]['xdata'])
                    ybg= list(instance.jsondict[bgdatakey]['ydata'])
                    scale = float(instance.jsondict[datakey]['bgscale'])
                    x,y = pyFTIR_pack.manipulate_data.subtract_bg(x,y,xbg,ybg,scale)
                return x,y






    '''This part following defines all functions for the buttons in the lowerframe
    '''
    def change_key(instance):
                item = instance.listWidget.currentItem()
                key = item.text()
                print('changed '+str(instance.jsondict[str(key)]['label'])+' to '+str(instance.labeledit.text()))
                new_key = str(instance.labeledit.text())
                instance.jsondict[new_key] = instance.jsondict[key]
                del instance.jsondict[key]
                row = instance.listWidget.row(item)
                instance.listWidget.takeItem(row)
                instance.listWidget.insertItem(row,new_key)
                instance.updatestatusbar(str('key changed from: '+ str(key)+' to '+str(new_key)),0,True)
                

    def export_data(instance):
                key = instance.listWidget.currentItem().text()
                file_name, _ = QtWidgets.QFileDialog.getSaveFileName(instance, "Save File", "", "Text Files (*.txt);;All Files (*)")
        
                x,y = FTIR_widgets_.getxy(instance,key) 
                x,y = [format(item,'.9f') for item in x],[format(item,'.9f') for item in y]
                df = pd.DataFrame({'wavenumber': x,'OD':y})
                df = df.set_index('wavenumber')

                if file_name:
                    df.to_csv(str(file_name), sep ='\t')
                print(str('data exported to: '+str(file_name)))
                instance.updatestatusbar(str('data exported to: '+str(file_name)),0,True)

    def findpeaks(instance):
                key = instance.listWidget.currentItem().text()
                num = int(instance.jsondict[key]['subplot'])
                promentry = float(instance.promedit.text())
                xpeaks,ypeaks = pyFTIR_pack.extract_data.peaker(key,instance.jsondict,promentry)
                for c in range(len(xpeaks)):
                    instance.axes[num].text(xpeaks[c],ypeaks[c],s=str(round(xpeaks[c],4))+'\n',size=3)
                instance.axes[num].scatter(xpeaks,ypeaks,marker='x',color='r',s=1)
                instance.canvasFTIR.draw()

    def invertdata(instance):
                key = instance.listWidget.currentItem().text()
                x,y = FTIR_widgets_.getxy(instance,key)
                nx,ny = x,[item*(-1) for item in y]
                instance.jsondict[key]['xdata']= nx
                instance.jsondict[key]['ydata']= ny
                FTIR_widgets_.plotFTIR(instance)
                print('inverted: '+str(key)+'data *(-1)')
                instance.updatestatusbar(str('inverted: '+str(key)+'data *(-1)'),0,True)
                

    def sec_dev(instance):
                key = instance.listWidget.currentItem().text()
                print('function to calculate the second derivative of:',key)
                x,y = FTIR_widgets_.getxy(instance,key)
                y = np.array(y)
                dy = np.gradient(y)
                ddy = list(np.gradient(dy))
                ddydump = dataconstruct.j_son(x,ddy,False,'',0,True,'green',0.9,'solid',str('Second dev of:'+str(key)),1)

                datadump = str(key+'_sec dev')
                counter=0
                while datadump in instance.jsondict:
                    counter = counter+1
                    datadump = str(datadump + str(counter))
                instance.jsondict[datadump] = ddydump
                instance.listWidget.addItem(datadump)
                instance.updatestatusbar(str('calculated sec dev for: '+str(key)),0,True)

    def cut(instance):
                key = instance.listWidget.currentItem().text()
                print('function to cut out a specific region of:',key)
                x,y = FTIR_widgets_.getxy(instance,key)
                wavecut = instance.sliceedit.text()
                xcut = list(wavecut.split(','))
                if wavecut == '':
                    instance.updatestatusbar('Press 2 times on the plot',5000,False)
                    polypoints = plt.ginput(n=2,timeout=30, show_clicks=True, mouse_add = plt.MouseButton.LEFT,mouse_pop= plt.MouseButton.RIGHT,mouse_stop = plt.MouseButton.MIDDLE)
                    xcut = [polypoints[0][0],polypoints[1][0]]
                
                xcut = [float(item) for item in xcut]
                xcut.sort()
                print('Cutted for:',str(xcut))

                xredcut,yredcut = pyFTIR_pack.manipulate_data.data_reduce(x,y,xcut[0],xcut[1])
                cutteddata =  dataconstruct.j_son(xredcut,yredcut,False,'',0,True,'magenta',0.9,'solid',str('Data cut out'),1)
                datadump = str(key+'_cut')
                counter=0
                while datadump in instance.jsondict:
                        counter = counter+1
                        datadump = str(datadump + str(counter))

                instance.jsondict[datadump] = cutteddata
                instance.listWidget.addItem(datadump)


    def del_ignore(instance):
                print('not sure for now what to do with it')

    def show_trapz(instance):
                key = instance.listWidget.currentItem().text()
                print('integrating:',key)
                x,y = FTIR_widgets_.getxy(instance,key)
                
                wavecut = instance.sliceedit.text()
                xint = list(wavecut.split(','))
                if wavecut == '':
                    instance.updatestatusbar('tap 2 times on the Plot to define the integration area !!!',5000,False)
                    polypoints = plt.ginput(n=2,timeout=30, show_clicks=True, mouse_add = plt.MouseButton.LEFT,mouse_pop= plt.MouseButton.RIGHT,mouse_stop = plt.MouseButton.MIDDLE)
                    xint = [polypoints[0][0],polypoints[1][0]]

                xint = [float(item) for item in xint]
                xint.sort()
                print(str(xint))

                xredint,yredint = pyFTIR_pack.manipulate_data.data_reduce(x,y,xint[0],xint[1])
                trapzvalue = np.abs(np.trapz(yredint,x=xredint))
                print('Integral value: '+str(trapzvalue))
                trapzval_formatted = "%.3g" %trapzvalue

                xint_formatted = [ round(item,2) for item in xint ]
                datadump = str(key+'_trapz :'+str(trapzval_formatted)+' over '+str(xint_formatted))
                integrationarea = dataconstruct.j_son(xredint,yredint,False,'',0,True,'skyblue',0.9,'solid',str(datadump),1)
                counter=0
                while datadump in instance.jsondict:
                        counter = counter+1
                        datadump = str(datadump + str(counter))

                instance.jsondict[datadump] = integrationarea
                instance.listWidget.addItem(datadump)
                instance.updatestatusbar(str('Integrated '+str(key)+' via TRAPZ rule / value of Integral:'+str(trapzvalue)),5000,False)


                

    def polyfit_bgcorrection(instance):
                    key = instance.listWidget.currentItem().text()
                    polydeg = int(instance.polydegedit.text())
                    waveval = instance.polyfitsliceedit.text()
                    xpoly = list(waveval.split(','))
                    if waveval =='':
                        instance.updatestatusbar('Press 4 times on the plot',8000,False)
                        polypoints = plt.ginput(n=5,timeout=30, show_clicks=True, mouse_add = plt.MouseButton.LEFT,mouse_pop= plt.MouseButton.RIGHT,mouse_stop = plt.MouseButton.MIDDLE)
                        xpoly = [polypoints[0][0],polypoints[1][0],polypoints[2][0],polypoints[3][0]]
                    else:
                        instance.updatestatusbar(str('Logged bg correction with manual slice:'+str(xpoly)),5000,False)

                    xpoly = [float(item) for item in xpoly]
                    xpoly.sort()
                    xpolyrounded = [str(f'{item: .5g}') for item in xpoly]
                    instance.polyfitsliceedit.setText(str(''.join(xpolyrounded)))
                    print(xpoly)


                    x = list(instance.jsondict[key]['xdata'])
                    y = list(instance.jsondict[key]['ydata'])
                    bgdatakey = instance.jsondict[key]['bgkey']
                    if instance.jsondict[key]['bg'] == True:
                        xbg= list(instance.jsondict[bgdatakey]['xdata'])
                        ybg= list(instance.jsondict[bgdatakey]['ydata'])
                        scale = float(instance.jsondict[key]['bgscale'])
                        x,y = pyFTIR_pack.manipulate_data.subtract_bg(x,y,xbg,ybg,scale)

                    xred,yred = pyFTIR_pack.manipulate_data.data_reduce(x,y,xpoly[0],xpoly[1])
                    xred2, yred2 = pyFTIR_pack.manipulate_data.data_reduce(x,y,xpoly[2],xpoly[3])

                    for c in range(len(xred2)):
                        xred.append(xred2[c])
                        yred.append(yred2[c])
                    
                    polypar = np.polyfit(xred,yred,polydeg)
                    polyfunk = np.poly1d(polypar)
                    xdatapoly,ydatad = pyFTIR_pack.manipulate_data.data_reduce(x,y,xpoly[0],xpoly[3])
                    ydatapoly = []
                    for i in range(len(xdatapoly)):
                        yval = polyfunk(xdatapoly[i])
                        ydatapoly.append(yval)

                    polyset =  dataconstruct.j_son(xdatapoly,list(polyfunk(xdatapoly)),False,'',0,True,'grey',0.6,'dashed',str('Polyfit'),0)
                    datapol = str(key+'_polyfit')
                    FTIR_widgets_.dont_double_date(instance,name=datapol,dataset=polyset)
                    
                    doubledset =  dataconstruct.j_son(xdatapoly,ydatad,True,str(str(key)+'_polyfit'),1,True,'darkorange',0.6,'solid',str('Data for substraktion'),1)
                    datadub = str(key+'_data')
                    FTIR_widgets_.dont_double_date(instance,name=datadub,dataset=doubledset)
    
                    instance.updatestatusbar(str('BG-CORRECTION with '+str(''.join(xpolyrounded))+' and polynomial: '+str(polyfunk)),0,True)

    def dont_double_date(instance,name,dataset): #nutzfunltiomn um platz zu sparen, name muss häufiger gecheckt werden
                counter=0
                while name in instance.jsondict:
                        counter += 1
                        name = str(name + str(counter))

                instance.jsondict[name] = dataset
                instance.listWidget.addItem(name)












    def plotFTIR(instance):
        FTIR_widgets_.selectitem_reloadplot(instance)
        instance.axes[0].cla()
        instance.axes[1].cla()
        instance.canvasFTIR.draw()

        #print(jsondict)
        for key in instance.jsondict:
            label = str(key)
            x = list(instance.jsondict[key]['xdata'])
            y = list(instance.jsondict[key]['ydata'])
            subnum = int(instance.jsondict[key]['subplot'])
            farb = str(instance.jsondict[key]['color'])
            bgdatakey = instance.jsondict[key]['bgkey']
            linest = instance.jsondict[key]['linestyle']
            linewid = instance.jsondict[key]['linewidth']
            #print(linest)
 
            if instance.jsondict[key]['bg'] == True:
                xbg= list(instance.jsondict[bgdatakey]['xdata'])
                ybg= list(instance.jsondict[bgdatakey]['ydata'])
                scale = float(instance.jsondict[key]['bgscale'])
                x,y = pyFTIR_pack.manipulate_data.subtract_bg(x,y,xbg,ybg,scale)
                #print('bg substracted')

            multi = float(instance.yfactoredit.text())
            y = [(item*multi) for item in y]

            if instance.jsondict[key]['show'] == True:
                if '_trapz' in key:
                    instance.axes[subnum].fill_between(x,y,y2=0,color=farb,linestyle=str(linest),label=label,alpha=.3)
                else:
                    instance.axes[subnum].plot(x,y,color=farb,linestyle=str(linest),linewidth=linewid,label=label)
                

        #self.ax.legend()
        
        #instance.ax[0].set_xlabel(str(xlabel.get()))
        instance.axes[0].set_ylabel(str(instance.ylabeledit.text()))
        instance.axes[1].set_xlabel(str(instance.xlabeledit.text()))
        instance.axes[1].set_ylabel(str(instance.ylabeledit.text()))
        instance.axes[0].grid(instance.gridcheckbox.isChecked())
        instance.axes[0].set_xlim(float(instance.xlowedit.text()),float(instance.xhighedit.text()))
        instance.axes[0].set_ylim(float(instance.ylowedit.text()),float(instance.yhighedit.text()))
        instance.axes[1].set_xlim(float(instance.xlowedit.text()),float(instance.xhighedit.text()))
        if instance.legendcheckbox.isChecked() == True:
            print('legend True')
            instance.axes[0].legend(loc='best', prop={'size': 5})
            

        instance.canvasFTIR.draw()


















class dataconstruct():
    def j_son(x,y,bg,bgkey,bgscale,show,col,lwidth,lstyle,lab,subpl):
        dataset = FTIR_init_dicts.j_son_spectrum(x,y,bg,bgkey,bgscale,show,col,lwidth,lstyle,lab,subpl)
        return dataset
