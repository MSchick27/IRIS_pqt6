import numpy as np
import os
import time
import matplotlib.colors as mcolors


class TRIR:
    def j_son(data_array_tolist,bg_array_tolist,delayarray,wnarray,stdarray,s2s_stdarray,scannumber,scanslice,delaynumber,delayslice):
        dataset = {'data': data_array_tolist,
                    'bgdata': bg_array_tolist,
                    'delays': delayarray,
                    'wn' : wnarray,
                    'std_deviation':stdarray,
                    'noise': s2s_stdarray,
                    'scannumber': scannumber,
                    'scanslice': scanslice,
                    'delaynumber': delaynumber,
                    'delayslice': delayslice,
                            }
        return dataset

    def importinitfunction(file_path,scanslice,delayslice,detektor_size):
        print('\n IMPORT TYPE: \t import data averaged by s2s_signal ##########')
        delays_times,wavnumbers,alphamatrix = TRIR.get_scans(file_path,scanslice,delayslice,detektor_size)
        weighteddata = TRIR.get_s2s_DIFF(alphamatrix)
        weighteddatatr = np.transpose(weighteddata)
        print('Weigthed data format:'+ str(np.shape(weighteddatatr)))
        
        bgdata= np.zeros(np.shape(weighteddatatr))
        stddev_array = np.transpose(np.average(alphamatrix[:,:,:,1],axis=1))
        s2s_stddata= TRIR.getnoise(alphamatrix)
        scannum = scannumber
        delaynum = len(delayfilearray)

        datasetjson = TRIR.j_son(weighteddatatr,bgdata,delays_times,wavnumbers,stddev_array,s2s_stddata,scannum,scanslice,delaynum,delayslice)
        print('import successfull ########## jsonified !')
        return datasetjson



    def get_scans(file_path,scanslice,delayslice,detektor_size):
        '''A function to exfiltrate all different scans either weighted or s2s etc from the different delay directories
            Thereby it is already sorting out the scans to take via the slicing strings given by scanslice,delayslice
            After reading the delayfile and Probeaxis wavenumber file, all the data is combined into a 4d array: alpha with
            the dimensions: Delays, Scans, Pixels, datatype
            -> datatype is referring to the folllowing indexes 0=data and 1= st deviation
            The alpha array is a result of here named:rho arrays stacked on the delay dimension
        '''
        global lastimport
        lastimport= file_path
        datadir = os.path.dirname(os.path.dirname(file_path[0]))
        
        search_files =os.listdir(datadir)


        def cut1darray(listofslices,arr):
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
                #print(new_arr)

            return new_arr


        
        for file in search_files:
            if 'delay_file' in str(file):
                delayfile= str(file)            #define variabel for delayfilepath
                print('DELAYFILE found: \t' + str(delayfile))
                delay = np.load(str(datadir+'/'+delayfile))
                delay = 10**(-3) * (delay[:,0])             #delays femtosekunden zu picosekunden
                delay = cut1darray(delayslice,delay)        #slice gewollte delays 
                print('.........Delayfile loaded successfully')   #delay = 1d array mit den Zeitpunkten in picosekunden

            if 'probe_wn_axis' in str(file):
                wnfile = str(file)              #define variabel for wavenumberfilepath
                print('PROBEAXISFILE found: \t' + str(wnfile))
                wn = np.load(str(datadir+'/'+wnfile))       #wn = 1d array mit wellenzahlen in reihenfolge der pixel
                print('.........Probeaxisfile loaded successfully')

            
        try:
            scdir =str(str(datadir)+'/scans')
            delay_files = os.listdir(scdir)
            for i,item in enumerate(delay_files):
                if 'DS_Store' in item:
                    delay_files.pop(i)
            delay_files = sorted(delay_files)
            delay_files = np.array(delay_files)
            global delayfilearray
            delayfilearray = cut1darray(delayslice,delay_files)
        except:
            print('no directory: /scans opr delayfiles missing')

        
        global scannumber
        if scanslice== ':':
            scannumber =len(os.listdir(str(str(scdir)+'/'+str(delayfilearray[0]))))//5 #-1
            print('Number of scans taken into account:',scannumber)
        else:
            scannumberall = len(os.listdir(str(str(scdir)+'/'+str(delayfilearray[0]))))//5 #-1
            fillinarray = np.zeros(scannumberall)
            fillarray = cut1darray(scanslice,fillinarray)
            scannumber = len(fillarray)
            print('Number of scans taken into account: '+str(scannumber)+'/'+str(scannumberall))

        

        alpha = np.zeros((len(delayfilearray),scannumber,len(wn),2))
        print('generating Data-array with dimensions: Delays, Scans, Pixels, datatype \t '+str(np.shape(alpha)))

        runtimelist_perdelay = []

        for i in range(len(delayfilearray)):
            timestart = time.time()
            delayf = delayfilearray[i]
            objectsperdelaylist = os.listdir(str(str(scdir)+'/'+str(delayf)))
            objectsperdelaylist.sort()
            
            s2ssignal_scanlist = []
            s2s_std_scanlist = []
            counts_scanlist = []
            weights_scanlist = []

            for object in objectsperdelaylist:
                if 's2s_signal' in object:
                    s2ssignal_scanlist.append(object)
                if 's2s_std' in object:
                    s2s_std_scanlist.append(object)
                if 'counts' in object:
                    counts_scanlist.append(object)
                if 'weights' in object:
                    weights_scanlist.append(object)

            s2ssignal_scanarray = cut1darray(scanslice,np.array(s2ssignal_scanlist))
            s2s_std_scanarray = cut1darray(scanslice,np.array(s2s_std_scanlist))
            counts_scanarray = cut1darray(scanslice,np.array(counts_scanlist))
            weights_scanarray = cut1darray(scanslice,np.array(weights_scanlist))

            rho = np.zeros((scannumber,len(wn),2))

            for j in range(scannumber):
                scanf = str(str(scdir)+'/'+str(delayf)+'/'+str(s2ssignal_scanarray[j]))
                s2sf = str(str(scdir)+'/'+str(delayf)+'/'+str(s2s_std_scanarray[j]))
                scandata = np.load(scanf)
                scandata = np.reshape(scandata,(int(detektor_size),1))
                #print(np.shape(scandata))
                s2sdata = np.load(s2sf)
                s2sdata = np.reshape(s2sdata,(int(detektor_size),1))
                #print(np.shape(s2sdata))
                s2sscandata = np.hstack((scandata,s2sdata))
                rho[j,:,:]= s2sscandata

            
                timeend = time.time()
                timeblock = timeend-timestart
                
                hashs = '#'
                points = '-'
                print((int(j/scannumber *40))*hashs + (40-int(j/scannumber *40))*points+':'+ str('{:5.3f}s'.format(timeblock)), end='\r')
                
        
            time_per_delay = time.time()
            runtimelist_perdelay.append(time_per_delay-timestart)
            estimated_runtime = np.mean(runtimelist_perdelay)*(len(delayfilearray)-i)
            alpha[i,:,:,:] = rho
            print((int(j/scannumber *40))*hashs + (39-int(j/scannumber *40))*points+'\t '+ str('{:5.3f}s'.format(timeblock))+'\t loaded Delay: '+str(i) +' \t estimated time to finish: '+str('{:5.3f}s'.format(estimated_runtime)))
        
        

        print('(Delays, Scans, Pixel, P-NP:s2s_std)')
        print('shape of alpha=',np.shape(alpha))
    
        return delay,wn, alpha
    


    def get_s2s_DIFF(data4d):
        '''This function takes the 4d alpha array as an input to calculate the wheighted mean of the scans
            the individual data values are weighted by the inverse square of their standart deviation
        '''
        print('weighting data bei s2s_std')
        signaldata = data4d[:,:,:,0]
        weights = data4d[:,:,:,1]
        weights = (1/weights)**2
        print('## weights: '+str(np.shape(weights)))
        addweights = signaldata*weights
        weightedODsum = np.sum(addweights,axis=1)
        print('## ODs: '+str(np.shape(weightedODsum)))
        weightsum = np.sum(weights,axis=1)

        #function to see the weights as a function of time and wn to see if meas good
        DIFF = weightedODsum/weightsum 
        return DIFF
    
    def getnoise(data4d):
        '''small function to grab the std from the 4d array'''
        weightsdata = data4d[:,:,:,1]
        print(np.shape(weightsdata))

        return weightsdata
    

    def reload(scanslice,delayslice,detektor_size):
        datasetjson =TRIR.importinitfunction(lastimport,scanslice,delayslice,detektor_size)
        return datasetjson
    








    def exportdata(filepath,jsondataset,plotname):
        if plotname == 'data':
            databgsub = modify_arrays.subtract_bg(jsondataset)
            stddeviation = jsondataset['std_deviation']
            print(np.shape(databgsub))
            myFile = open(filepath,'w')
            for i,delay in enumerate(jsondataset['delays']):
                for j,wn in enumerate(jsondataset['wn']):
                    arrayvalue = databgsub[j,i]
                    stddev_value = stddeviation[j,i]
                    myFile.write(str(delay)+'\t'+str(wn)+'\t'+str(arrayvalue)+'\t'+str(stddev_value)+'\n')

            myFile.close()

        if plotname == 'background':
            databg = jsondataset['bgdata']
            print(np.shape(databg))
            myFile = open(filepath,'w')
            for i,delay in enumerate(jsondataset['delays']):
                for j,wn in enumerate(jsondataset['wn']):
                    arrayvalue = databg[j,i]
                    myFile.write(str(delay)+'\t'+str(wn)+'\t'+str(arrayvalue)+'\n')


    def exportdata_to_npyfile(filepath,jsondataset,plotname):
        if plotname == 'data':
            databgsub = modify_arrays.subtract_bg(jsondataset)
            stddeviation = jsondataset['std_deviation']
            print(np.shape(databgsub))
            np.save(str(str(filepath)+'_A_matrix'),databgsub)
            np.save(str(str(filepath)+'_A_matrixerrors'),stddeviation)
            np.save(str(str(filepath)+'_A_wavenumbers'),np.array(jsondataset['wn']))
            np.save(str(str(filepath)+'_A_delays'),np.array(jsondataset['delays']))
            np.savez(str(str(filepath)+'combined'),A=databgsub,Aerror=stddeviation,wn=np.array(jsondataset['wn']),delays=np.array(jsondataset['delays']))
            print('successfully stored at'+str(filepath))

        if plotname == 'raw':
            print('no raw function for now')

        if plotname == 'background':
            databg = jsondataset['bgdata']
            print(np.shape(databg))
            np.save(str(str(filepath)+'_A_matrix'),databg)
            np.save(str(str(filepath)+'_A_wavenumbers'),np.array(jsondataset['wn']))
            np.save(str(str(filepath)+'_A_delays'),np.array(jsondataset['delays']))
            np.savez(str(str(filepath)+'combined'),A=databg,wn=np.array(jsondataset['wn']),delays=np.array(jsondataset['delays']))
            print('successfully stored at'+str(filepath))
        
        


























class modify_arrays():
    def subtract_bg(jsonfile):
        diffdata = np.subtract(jsonfile['data'],jsonfile['bgdata'])
        #print(np.shape(diffdata))
        return diffdata
    
    def subtract_bg_rms(jsonfile):
        diffdata = np.abs(np.subtract(jsonfile['data'],jsonfile['bgdata'])) 
        return diffdata


    def sub_delay1(weighteddata):
        delay1 = weighteddata[:,0]
        subdelay1array = np.zeros(np.shape(weighteddata))
        for i in range(len(weighteddata[0,:])):
            subdelay1array[:,i] = delay1
    
        #subDIFF = np.transpose(subDIFF)
        newweighteddata = np.subtract(weighteddata,subdelay1array)
        print('successfully subtracted first delay as background')
        return newweighteddata



    def noiseallscans(datajson):
        noisearray = datajson['noise']
        shape = np.shape(noisearray)
        noisescan = np.zeros((shape[0]*shape[1],shape[2]))
        #print('zeroes: '+ str(np.shape(noisescan)))
        counter = 0
        for i in range(shape[1]):
            for j in range(shape[0]):
                noisescan[counter] = noisearray[j,i]
                counter = counter + 1

        #print('noisewrap: '+ str(np.shape(noisescan)))
        return noisescan
    



    def getlogaxis(datajson):
        delaays = datajson['delays']
        for i,timestamp in enumerate(delaays):
            if timestamp > 0:
                print(timestamp)
                break
        xlow = delaays[i]
        xhigh = delaays[-1]
        return xlow,xhigh








class colormapsfor_TRIR():
    def findmaxval(dataarray,valfactor):
        absarray = np.abs(dataarray)
        maxval = float(np.nanmax(absarray)) * float(valfactor)
        minval = -maxval

        if type(maxval) != float:
            print('colormap ERROR: nan encountered as maxvalue')
            maxval=0.1
            minval=-0.1

        #print('Maxvalue for colormap:' +str(maxval))
        #print(absarray)
        return maxval,minval
    

    def create_custom_colormap(start_color, end_color):   #, min_value, max_value,levels)
        # Convert hex colors to RGB values
        start_rgb = mcolors.hex2color(start_color)
        mid_rgb = mcolors.hex2color('#FFFFFF')
        end_rgb = mcolors.hex2color(end_color)
    
        # Normalize the data range based on min and max values
        #norm = mcolors.Normalize(vmin=min_value, vmax=max_value)
    
        # Create a linear gradient between the start and end colors
        colormap = mcolors.LinearSegmentedColormap.from_list(
            'custom_colormap',
            [start_rgb, mid_rgb, end_rgb],
            N=256,  # Number of colors in the colormap
        )
    
        return colormap#, norm
    
