import numpy as np
from scipy import optimize as opt
from scipy import signal
import os
import json
import matplotlib.pyplot as plt
import lmfit
import tqdm
import pandas as pd

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
    









class Fitting():
    def fouriersmooth(y,vac):
        rft = np.fft.rfft(y)
        rft[vac:] = 0   # Note, rft.shape = 21
        y_smooth = list(np.fft.irfft(rft))
        return y_smooth
    





    def fitband_allg(x,y,fitfunc,instance):
        if fitfunc == 'lorentz':
            instance.updatestatusbar('press 3 times on plot: LEFT,PEAK,RIGHT',5000,False)
            fitx,fity,parstring,par,fittype,fiterror,fwhm=Fitting.lorentzfit_spec(x,y)
            return fitx,fity,parstring,par,fittype,fiterror,fwhm


    def lorentz(x,a,b,c,g):
            return ((a*c**2)/((x-b)**2+c**2)) + g
    def lorentzFWHM(c):
            return np.abs(c)*2#*np.pi
    def lorentzfit_spec(x,y):
        tpoints = plt.ginput(n=3,timeout=30, show_clicks=True, mouse_add = plt.MouseButton.LEFT,mouse_pop= plt.MouseButton.RIGHT,mouse_stop = plt.MouseButton.MIDDLE)
        print(tpoints)
        xps = [tpoints[0][0],tpoints[1][0],tpoints[2][0]]
        yps = [tpoints[0][1],tpoints[1][1],tpoints[2][1]]
        xh = max(xps)
        xl = min(xps)
        width = (xh-xl)/2.4
        xps.remove(xl)
        xps.remove(xh)
        xpeak = xps[0]
        amp = max(yps)-min(yps)
        height = min(yps)
        x,y = manipulate_data.data_reduce(x,y,xl,xh)

        fittype ='lorentz'

        guess=[amp,xpeak,width,height]
        print('Guess: '+str(guess))
        par,cov = opt.curve_fit(Fitting.lorentz,x,y,guess,maxfev=100000)
        fiterror=  np.sqrt(np.diag(cov))
        print('Err: '+str(fiterror))
        fitx = x
        fity = []
        for i in range(len(x)):
            y_val = Fitting.lorentz(x[i],*par)
            fity.append(y_val)

        print('check')
        fwhm = round(Fitting.lorentzFWHM(par[2]),2)
        parstring = str('Amplitude:' + str(round(par[0],5)) + ', x:' + str(round(par[1],3)) +', c:' + str(round(par[2],3)) +', FWHM:' + str(fwhm))

        return fitx,fity,parstring,par,fittype,fiterror,fwhm












    def fitmulti(x,y,instance,peaknumber):
        instance.updatestatusbar('press on plot: LEFT  ,approx each PEAK,  RIGHT',5000,False)
        tpoints = plt.ginput(n=(peaknumber+2),timeout=30, show_clicks=True, mouse_add = plt.MouseButton.LEFT,mouse_pop= plt.MouseButton.RIGHT,mouse_stop = plt.MouseButton.MIDDLE)
        xps=[]
        yps=[]
        for i in range(len(tpoints)):
              xps.append(tpoints[i][0])
              yps.append(tpoints[i][1])
    
        xps.sort()
        yps.sort()

        xl,xh = xps[0],xps[-1]
        x,y = manipulate_data.data_reduce(x,y,xl,xh)
        approx_amplitudes = yps[1:-1]
        approx_sigmas = np.mean(np.gradient(xps))/4

        print('### FITTING multi Lorentzian')

        def add_peak(prefix, center, amplitude, sigma):
            print('peakadding',prefix, center, amplitude, sigma)
            peak = lmfit.models.LorentzianModel(prefix=prefix)
            pars = peak.make_params()
            pars[prefix + 'center'].set(center)
            pars[prefix + 'amplitude'].set(amplitude)
            pars[prefix + 'sigma'].set(sigma, min=0)
            return peak, pars
        
        model = lmfit.models.QuadraticModel(prefix='bkg_')
        params = model.make_params(a=0, b=0, c=0)

        rough_peak_positions = xps[1:-1]
        for i, cen in enumerate(rough_peak_positions):
            peak, pars = add_peak('lz%d_' % (i+1), cen,amplitude=approx_amplitudes[i],sigma=approx_sigmas)
            model = model + peak
            params.update(pars)

        #init = model.eval(params, x=x)
        result = model.fit(y, params, x=x)
        comps = result.eval_components()
        print(result.fit_report(min_correl=0.5))

        return x,y,result,comps
    










    def superfit_bands(x,y,instance,peaknumber):
        def generate_combinations(n):
            '''function to generate all possible combinations of gaussian and lorentz fits'''
            if n == 0:
                return ['']
    
            sub_combinations = generate_combinations(n - 1)
            combinations = []
    
            for combo in sub_combinations:
                combinations.append(combo + 'g')
                combinations.append(combo + 'l')
            return combinations
            
        instance.updatestatusbar('press on plot: LEFT  ,approx each PEAK with leftFWHM/PEAK/rightFWHM,  RIGHT',10000,False)
        tpoints = plt.ginput(n=((peaknumber*3)+2),timeout=50, show_clicks=True, mouse_add = plt.MouseButton.LEFT,mouse_pop= plt.MouseButton.RIGHT,mouse_stop = plt.MouseButton.MIDDLE)
        xps=[]
        yps=[]
        for i in range(len(tpoints)):
              xps.append(tpoints[i][0])
              yps.append(tpoints[i][1])
        
        xl,xh = xps[0],xps[-1]
        x,y = manipulate_data.data_reduce(x,y,xl,xh)
        print(xps)
        approx_peakcenters = xps[1:-1:3]
        print('peaks:',approx_peakcenters)
        approx_amplitudes = yps[1:-1:3]
        print('amplitudes:',approx_amplitudes)
        approx_FWHMs = []
        for i in range(peaknumber):
              approx_FWHMs.append(xps[i*3+3]-xps[i*3+1])
        print('FWHMs',approx_FWHMs)

        combinations = generate_combinations(peaknumber)
        print(combinations)
        
        resultlist = []
        chisquaredlist=[]
        rsquaredlist = []
        for i in range(len(combinations)):
            comb = combinations[i]
            MODEL = lmfit.models.QuadraticModel(prefix='bkg_')
            params = MODEL.make_params(a=0, b=0, c=0)

            def add_peak(prefix, center, amplitude, sigma,fitstyle):
                #print('peakadding',prefix, center, amplitude, sigma)
                if fitstyle =='g':
                     band = lmfit.models.GaussianModel(prefix=prefix)
                if fitstyle =='l':
                    band = lmfit.models.LorentzianModel(prefix=prefix)

                pars = band.make_params()
                pars[prefix + 'center'].set(center)
                pars[prefix + 'amplitude'].set(amplitude)
                pars[prefix + 'sigma'].set(sigma, min=0)
                pars[prefix + 'height'].set(0)
                pars[prefix + 'height'].vary=False
                return band, pars

            
            for j,fitstyle in enumerate(comb):
                band, pars = add_peak(str(str(fitstyle)+'%d_' % (j+1)), approx_peakcenters[j],amplitude=approx_amplitudes[j],sigma= (approx_FWHMs[j]/2),fitstyle=fitstyle)
                MODEL = MODEL + band
                params.update(pars)
            
            result = MODEL.fit(y, params, x=x)
            chisqr =f'{result.chisqr: .4g}'
            rsqr = result.rsquared
            resultlist.append(result)
            chisquaredlist.append(chisqr)
            rsquaredlist.append(rsqr)

        bestresult_index = np.argmin(chisquaredlist)
        bestresult = resultlist[bestresult_index]
        finishstr = str('#####FINISHED Best result gave combination='+str(str(bestresult_index)+str(combinations[bestresult_index]))+ ' with chisqr='+str(chisquaredlist[bestresult_index]))
        print(finishstr)
        instance.updatestatusbar(finishstr,0,True)
        fitdict={'combinations':combinations, 'Chi sqzared':chisquaredlist, 'R squared':rsquaredlist}
        df = pd.DataFrame.from_dict(fitdict)
        print(df)

        comps = bestresult.eval_components()
        print(bestresult.fit_report(min_correl=0.5))

        return x,y,bestresult,comps


