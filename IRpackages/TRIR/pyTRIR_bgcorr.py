import numpy as np

import matplotlib.pyplot as plt

from scipy import optimize as opt


def meanofsliceddelays(d2array,slices):
    if slices == ':':
            new_arr = d2array[:,-1]
    else:
        new_arr_list = []
        slices = slices.split(',')
        for slice in slices:
            slice = slice.split(':')
            arrpart = d2array[:,int(slice[0]):int(slice[1])]
            new_arr_list.append(arrpart)

        
        print(np.shape(new_arr_list)) 
        newslicedarray = np.concatenate(new_arr_list,axis=1)
        print(np.shape(newslicedarray))      
        new_arr = np.mean(newslicedarray,axis=1)
    
    print('Spektrum to fit: ' +str(np.shape(new_arr)))

    return new_arr


def cut1d(listofslices,arr):
            if listofslices == ':':
                new_arr = arr
            else:
                new_arr_list = []
                listofslices = listofslices.split(',')
                for slice in listofslices:
                    slice = slice.split(':')
                    arrpart = arr[int(slice[0]):int(slice[1])]
                    for value in arrpart:
                        new_arr_list.append(value)
                
                new_arr = np.array(new_arr_list)
                

            return new_arr

def fitspec(x,y,polynomialdeg,pixelslice,weights):
    fitpixelsx=cut1d(pixelslice,x)
    fitpixelsy=cut1d(pixelslice,y)
    weights = cut1d(pixelslice,weights)
    polypar = np.polyfit(fitpixelsx,fitpixelsy,polynomialdeg,w=weights)
    polyfunk = np.poly1d(polypar)
    polyfity = polyfunk(x)
    polyfitx = x
    return polyfitx,polyfity,polyfunk






def fitbgtoTRIR(d2dataarray,wavenumbers,d1delays,pixelslice,polyfitfunction):
    print('ask Georg')
    wavenumberscropped = cut1d(pixelslice,wavenumbers)

    def varyfunc(wavenumberscropped,a,b):
        return a + b * polyfitfunction(wavenumberscropped)
    

    d2bgarray = np.zeros(np.shape(d2dataarray))
    for index,delaytime in enumerate(d1delays):
        #print(delaytime)
        spectrumatdelay = d2dataarray[:,index]
        #print(list(spectrumatdelay))
        fitpixelsatdelay = cut1d(pixelslice,spectrumatdelay)
        #print(fitpixelsatdelay)
        
        guess = [0.001,0.001]
        par,cov = opt.curve_fit(varyfunc,wavenumberscropped,fitpixelsatdelay,maxfev=100000,p0=guess)
        #print(par)

        data = varyfunc(wavenumbers,*par)
        #print(np.shape(data))
        d2bgarray[:,index] = data
        #print(np.shape(d2bgarray))

    return d2bgarray
















#starting the fitting process
def TRIRbgcorr(jsondataset,polynomial,avgdelayslice,pixelslice):
    print('i will calculate the TRIR bg correction')
    latest_data = jsondataset['data']
    print('check:'+str(np.shape(latest_data)))
    noisesforfit =np.mean((jsondataset['noise']),axis=1)
    noisesforfit = noisesforfit[-1]
    weightsforfit = [(1/item ) for item in noisesforfit]
    

    
    xdata = jsondataset['wn']
    ydata = meanofsliceddelays(latest_data,avgdelayslice)
    polyfitx,polyfity,polyfunk = fitspec(xdata,ydata,polynomial,pixelslice,weightsforfit)
    
    bgarraydata = fitbgtoTRIR(latest_data,jsondataset['wn'],jsondataset['delays'],pixelslice,polyfunk)
    print('check:'+str(np.shape(bgarraydata)))


    """ 
    fig2= plt.figure(1,figsize=(10, 15))
    delaynumber = 8
    grid = fig2.add_gridspec(int(delaynumber/2+2), 2, hspace=0.01, wspace=0.01,bottom=0.09,top=0.95,left=0.05,right=0.95)
    for i in range(int(delaynumber/2)):
        delayplot = fig2.add_subplot(grid[i, 0])
        delaydist = len(jsondataset['delays'])//delaynumber
        indexx = delaydist * i
        print(indexx)
        delayplot.plot(xdata,jsondataset['data'][:,indexx],label= str(round(jsondataset['delays'][indexx],2)))
        delayplot.legend()
        delayplot.plot(xdata,bgarraydata[:,indexx],color= 'y')
        delayplot.plot(cut1d(pixelslice,xdata),cut1d(pixelslice,bgarraydata[:,indexx]),linestyle='',marker='.',color='r')
        #delayplot.yaxis.set_ticklabels([])

        delayplot2 = fig2.add_subplot(grid[i, 1])
        indexx2 = int(delaydist * i + delaynumber/2*delaydist)
        print(indexx2)
        delayplot2.plot(xdata,jsondataset['data'][:,indexx2],label= str(round(jsondataset['delays'][indexx2],2)))
        delayplot2.legend()
        delayplot2.plot(xdata,bgarraydata[:,indexx2],color= 'y')
        delayplot2.plot(cut1d(pixelslice,xdata),cut1d(pixelslice,bgarraydata[:,indexx2]),linestyle='',marker='.',color='r')
        delayplot2.yaxis.set_ticklabels([])


    latestdelay = fig2.add_subplot(grid[int(delaynumber/2+1)-1:int(delaynumber/2+2), :])
    latestdelay.plot(xdata,jsondataset['data'][:,-1],label= str(round(jsondataset['delays'][-1],2)))
    latestdelay.legend()
    latestdelay.plot(polyfitx,polyfity)
    latestdelay.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    



    plt.show() """

    return bgarraydata,xdata,ydata,polyfitx,polyfity





