import sys
from time import strftime,sleep
import matplotlib.pyplot as plt
#from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg,NavigationToolbar2QT
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


from PyQt6 import QtCore, QtGui, QtWidgets,uic


#Part for my peronal written packages
from IRpackages.TRIR import TRIRwidgets,TRIRviewerwidgets
from IRpackages.FTIR import FTIRwidgets


class UI(QtWidgets.QMainWindow):
    def updatestatusbar(self,message,time,addtolist):
        self.statusbar.showMessage(message, time)
        if addtolist == True:
            self.statusbarlist.append(message)
            self.statusbarlist.append("<br>")
        self.statusbar.setToolTip("<html><body style='white-space: nowrap;'>"
                                     +'<p>'+str(''.join(self.statusbarlist[::-1]))+'<\p>'+
                                     "</body></html>")

    def __init__(self):
        super(UI,self).__init__()
        uic.loadUi('pyqtwindowfiles/startupwindow.ui',self)

        self.statusbarlist = []
        self.statusbar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusbar)

        self.updatestatusbar('Root',3000,True)



        self.lcd = self.findChild(QtWidgets.QLCDNumber,'lcdNumber')
        print('clock running')

        def timenow():
            if stoptimer==False:
                self.timer.stop()
            time = strftime("%H:%M:%S")
            print(time)
            self.lcd.display(time)

        self.timer= QtCore.QTimer()
        self.lcd.setDigitCount(8)
        self.timer.timeout.connect(timenow)
        stoptimer=True
        self.timer.start(1000)

        #dropdownmenu die funktionen an buttons
        self.dropdown_FTIR = self.findChild(QtGui.QAction,'actionFTIR_Analysis')
        self.dropdown_FTIR.setStatusTip("This is your button")
        self.dropdown_FTIR.triggered.connect(self.change_to_FTIR)
        self.dropdown_FTIR.setShortcut('Ctrl+F')

        self.dropdown_TRIRimporter = self.findChild(QtGui.QAction,'actionimport_TRIR_data')
        self.dropdown_TRIRimporter.setStatusTip("This is your button")
        self.dropdown_TRIRimporter.triggered.connect(self.change_to_TRIR)
        self.dropdown_TRIRimporter.setShortcut('Ctrl+T')


        self.dropdown_TRIRviewer = self.findChild(QtGui.QAction,'actionTRIR_viewer')
        self.dropdown_TRIRviewer.setStatusTip("This is your button")
        self.dropdown_TRIRviewer.triggered.connect(self.change_to_TRIRviewer)
        self.dropdown_TRIRviewer.setShortcut('Ctrl+V')

        self.show()

    


    def change_to_FTIR(self):
        self.timer.stop()
        self.dropdown_FTIR.setEnabled(False)
        self.dropdown_TRIRimporter.setEnabled(True)
        self.updatestatusbar('changing to FTIR module...',0,False)
        self.stacker =  self.findChild(QtWidgets.QStackedWidget,'stacker')
        self.page_FTIR = self.findChild(QtWidgets.QWidget, 'pageFTIRviewer')
        self.stacker.setCurrentWidget(self.page_FTIR)
        FTIRwidgets.FTIR_widgets_.initer(self)
       

    def change_to_TRIR(self):
        self.timer.stop()
        self.dropdown_TRIRimporter.setEnabled(False)
        self.dropdown_FTIR.setEnabled(True)
        self.updatestatusbar('changing to TRIR import module...',0,False)
        self.stacker =  self.findChild(QtWidgets.QStackedWidget,'stacker')
        self.page_trir_importer = self.findChild(QtWidgets.QWidget, 'pageTRIRimporter')
        self.stacker.setCurrentWidget(self.page_trir_importer)
        TRIRwidgets.TRIR_widgets_defining.initer(self)


    def change_to_TRIRviewer(self):
        self.timer.stop()
        self.dropdown_TRIRviewer.setEnabled(False)
        self.dropdown_TRIRimporter.setEnabled(True)
        self.dropdown_FTIR.setEnabled(True)
        self.updatestatusbar('changing to TRIR viewer...',0,False)
        self.stacker =  self.findChild(QtWidgets.QStackedWidget,'stacker')
        self.page_trir_viewer = self.findChild(QtWidgets.QWidget, 'pageTRIRviewer')
        self.stacker.setCurrentWidget(self.page_trir_viewer)
        TRIRviewerwidgets.TRIRviewer_widgets_defining.initer(self)

        




app = QtWidgets.QApplication(sys.argv)
UIWindow=UI()
app.exec()
