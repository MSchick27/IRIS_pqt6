import numpy as np
from scipy import optimize as opt
from scipy import signal
import os
import json

from IRpackages.FTIR import FTIR_init_dicts


class import_and_export_functions():
    def import_files(filenames,delimitervar,jsondict):
        datatup = filenames
        for datapath in datatup:
            import_and_export_functions.generatejson(datapath,delimitervar,jsondict)

        return jsondict
    
    def generatejson(datapath,delimitervar,jsondict):
        dataname = os.path.basename(datapath)
        print('new data appended to json')
        delimiter = delimitervar
                
        if delimiter == 'tab':
                    x,y = import_and_export_functions.importFTIRdata(datapath,'\t')
        if delimiter == 'space':
                    x,y = import_and_export_functions.importFTIRdata(datapath,' ')
        if delimiter == ';':
                    x,y = import_and_export_functions.importFTIRdata(datapath,';')
        if delimiter == ',':
                    x,y = import_and_export_functions.importFTIRdata(datapath,',')

        dataset = FTIR_init_dicts.j_son_spectrum(x,y,False,'',0,True,'black',0.5,'solid','import',0)
        counter=0
        while dataname in jsondict:
                    counter = counter+1
                    dataname = str(dataname + str(counter))

        jsondict[str(dataname)]= dataset
        



    def importFTIRdata(datapath,seg):
        data = open(datapath,'r',encoding='utf-8-sig')
        data = data.readlines()
        x = []
        y = []
        for item in data:
            it = item.split(str(seg))
            x.append(float(it[0].strip()))
            y.append(float(it[-1].strip()))

        return x,y
    










class extract_data():
     def peaker(key,jsondict,promentry):
                print('function to find the peaks')
                prom = promentry
                x = list(jsondict[key]['xdata'])
                y = list(jsondict[key]['ydata'])
                num = int(jsondict[key]['subplot'])
                bgdatakey = jsondict[key]['bgkey']

                if jsondict[key]['bg'] == True:
                    xbg= list(jsondict[bgdatakey]['xdata'])
                    ybg= list(jsondict[bgdatakey]['ydata'])
                    scale = float(jsondict[key]['bgscale'])
                    x,y = manipulate_data.subtract_bg(x,y,xbg,ybg,scale)
                
                yarr = np.array(y)
                #print(y)
                peaks,_ = signal.find_peaks(yarr,prominence=prom)#height=0.001)
                print(peaks,_)
                xpeaks = []
                ypeaks = []
                for item in peaks:
                    xpeaks.append(x[item])
                    ypeaks.append(y[item]*1000)
                
                return xpeaks,ypeaks

















class manipulate_data():
    def subtract_bg(x,y,xbg,ybg,scale):
        newx = []
        newy = []
        for c in range(len(x)):
            newxval = x[c]
            newx.append(newxval)
            newyval = y[c]- scale* ybg[c]
            newy.append(newyval)
    
        return newx, newy
    
    def data_reduce(x,y,xl,xh):
        xvals = []
        yvals = []
        for i in range(len(x)):
            if x[i] <= xh:
                if x[i] >= xl:
                    xvals.append(x[i])
                    yvals.append(y[i])

        return xvals, yvals