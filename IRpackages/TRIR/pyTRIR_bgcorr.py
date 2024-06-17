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
    """ 
    fits a polynomial to the x,y data given to the function
    for fitting the data is reduced to the pixels you want to fit via the slices
    additionally weights are used for polyfit
    """
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
    bgparameters = np.zeros((4,len(d1delays)))
    for index,delaytime in enumerate(d1delays):
        #print(delaytime)
        spectrumatdelay = d2dataarray[:,index]
        #print(list(spectrumatdelay))
        fitpixelsatdelay = cut1d(pixelslice,spectrumatdelay)
        #print(fitpixelsatdelay)
        
        guess = [0.001,0.001]
        par,cov = opt.curve_fit(varyfunc,wavenumberscropped,fitpixelsatdelay,maxfev=100000,p0=guess)
        par_errors = np.diag(cov)
        #print(par)
        bgparameters[0,index] = par[0]
        bgparameters[1,index] = par_errors[0]
        bgparameters[2,index] = par[1]
        bgparameters[3,index] = par_errors[1]

        data = varyfunc(wavenumbers,*par)
        #print(np.shape(data))
        d2bgarray[:,index] = data
        #print(np.shape(d2bgarray))

    return d2bgarray,bgparameters
















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
    
    bgarraydata,bgparamterarray = fitbgtoTRIR(latest_data,jsondataset['wn'],jsondataset['delays'],pixelslice,polyfunk)
    print('check:'+str(np.shape(bgarraydata)))

    return bgarraydata,xdata,ydata,polyfitx,polyfity,bgparamterarray





