""" 
yfactor= [0, -0.24069193184073215, -0.06256564653722935, -0.02781084459692161, 0.02741147802063116, -0.04618092147107985, -0.059513132999214687, 0.0054751897721592325, -0.027044549565807215, -0.04047449387722887, -0.04868857382173544, -0.08046496353804984, -0.03932127648100105, -0.023692062988203453, -0.028402976227040865, -0.022448708970484857, -0.032068792173065575, 0.051052101298567916, 0.10830310432160326, 0.11723724309410646, 0.24534201406171222, 0.275741794451831, 0.3933410835430314, 0.49196051059366364, 0.5435337557312757, 0.6305547452751926, 0.6876536065229668, 0.7529993699496166, 0.8541859217739106, 0.8815753577086847, 0.8706215240160337, 0.9535577987286286, 0.9238127698779571, 0.9458626223382487, 0.9500143166087331, 0.9643101595190983, 1.0004657293589916, 0.973990115268965, 0.9619432230664893, 0.9852860593588351, 0.9599414079435072, 1.0169478054225136, 1.0019955137547059, 0.9981912594522425]
yfactor_error= [0.0, 0.0002370873299191136, 0.00035727835359353617, 0.00044429378102824697, 0.0004121029168939086, 0.0003169616653228706, 0.0003340380660822933, 0.000310921502717231, 0.00015162019602405437, 0.0003464044430112548, 0.0003465483169250903, 0.00034919259538437186, 0.00037928354839949395, 0.0002706162844824322, 0.00040678686060906157, 0.0005137158441381124, 0.00046444562183695586, 0.0006163113928049596, 0.0009715767120699638, 0.0009170737768272695, 0.0011532929554558665, 0.001585850065302819, 0.0018596851382914309, 0.002410881211916123, 0.0019222252221969217, 0.0014001533508545053, 0.0017151344238317387, 0.0008220917767270201, 0.0007531046323584543, 0.0006500917574856955, 0.0007226785127351273, 0.0007079888659921456, 0.0005830866586937918, 0.000740310164191594, 0.0004919773317544468, 0.00041937036490043183, 0.00037664109772233206, 0.0003512010621509475, 0.000578272514350635, 0.0004715612445315427, 0.0003875526619855045, 0.0003790579209402625, 0.0003290996300693489, 0.0002538963609583516]
t=[-40, -0.5, 0, 0.1, 0.13, 0.169, 0.221, 0.28700000000000003, 0.374, 0.488, 0.636, 0.7000000000000001, 0.8280000000000001, 1.079, 1.405, 1.831, 2.3850000000000002, 3.107, 4.047, 5.271, 6.867, 8.944, 11.651, 15.176, 19.767, 25.748, 33.539, 43.687, 56.905, 74.122, 96.548, 125.76, 163.811, 213.373, 277.932, 362.024, 471.558, 614.2330000000001, 800.076, 1042.147, 1357.46, 1768.175, 2303.155, 3000]
 """


import numpy as np
yfactor= [ -0.06256564653722935, -0.02781084459692161, 0.02741147802063116, -0.04618092147107985, -0.059513132999214687, 0.0054751897721592325, -0.027044549565807215, -0.04047449387722887, -0.04868857382173544, -0.08046496353804984, -0.03932127648100105, -0.023692062988203453, -0.028402976227040865, -0.022448708970484857, -0.032068792173065575, 0.051052101298567916, 0.10830310432160326, 0.11723724309410646, 0.24534201406171222, 0.275741794451831, 0.3933410835430314, 0.49196051059366364, 0.5435337557312757, 0.6305547452751926, 0.6876536065229668, 0.7529993699496166, 0.8541859217739106, 0.8815753577086847, 0.8706215240160337, 0.9535577987286286, 0.9238127698779571, 0.9458626223382487, 0.9500143166087331, 0.9643101595190983, 1.0004657293589916, 0.973990115268965, 0.9619432230664893, 0.9852860593588351, 0.9599414079435072, 1.0169478054225136, 1.0019955137547059, 0.9981912594522425]
yfactor_error= [ 0.00035727835359353617, 0.00044429378102824697, 0.0004121029168939086, 0.0003169616653228706, 0.0003340380660822933, 0.000310921502717231, 0.00015162019602405437, 0.0003464044430112548, 0.0003465483169250903, 0.00034919259538437186, 0.00037928354839949395, 0.0002706162844824322, 0.00040678686060906157, 0.0005137158441381124, 0.00046444562183695586, 0.0006163113928049596, 0.0009715767120699638, 0.0009170737768272695, 0.0011532929554558665, 0.001585850065302819, 0.0018596851382914309, 0.002410881211916123, 0.0019222252221969217, 0.0014001533508545053, 0.0017151344238317387, 0.0008220917767270201, 0.0007531046323584543, 0.0006500917574856955, 0.0007226785127351273, 0.0007079888659921456, 0.0005830866586937918, 0.000740310164191594, 0.0004919773317544468, 0.00041937036490043183, 0.00037664109772233206, 0.0003512010621509475, 0.000578272514350635, 0.0004715612445315427, 0.0003875526619855045, 0.0003790579209402625, 0.0003290996300693489, 0.0002538963609583516]
t=np.array([ 0, 0.1, 0.13, 0.169, 0.221, 0.28700000000000003, 0.374, 0.488, 0.636, 0.7000000000000001, 0.8280000000000001, 1.079, 1.405, 1.831, 2.3850000000000002, 3.107, 4.047, 5.271, 6.867, 8.944, 11.651, 15.176, 19.767, 25.748, 33.539, 43.687, 56.905, 74.122, 96.548, 125.76, 163.811, 213.373, 277.932, 362.024, 471.558, 614.2330000000001, 800.076, 1042.147, 1357.46, 1768.175, 2303.155, 3000])


import matplotlib.pyplot as plt

from scipy.optimize import curve_fit,minimize
from scipy.integrate import odeint

def model(t,k):
    return 1-np.exp(-1/k *t)



def IRF(t):
    sigma = 0.1
    gauss = 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-1/2*(t/sigma)**2)
    return gauss

def model_convolution(k,t,y0,a):
    return np.convolve(- 1/k *t*y0,IRF(t)) 

""" def irfplotting(k,t):
    y = 1-odeint(model_convolution,y0,t,args=(k,a))
    return y """
def irfplotting(k,t):
    y = 1-np.exp(np.convolve(- 1/k*t,IRF(t)))
    return y

def objective(k,t,a):
    y0=1
    y = 1-odeint(model_convolution,y0,t,args=(k,a))
    diff= yfactor-y
    residue = np.linalg.norm(diff)
    print(residue)
    return residue


print(len(yfactor),len(t))

parm,opt = curve_fit(model,t,yfactor,p0=[0.1])
error = np.diag(opt)
print(parm,error)
t = np.array(t)
plt.scatter(t,yfactor,color='r',label='yfac')
plt.plot(t,model(t,parm[0]),color='k',label='fit')
plt.xscale('log')


a=1
y0=1
kparameter=26
result = minimize(objective, kparameter ,args=(t,a),tol=1e-3,options={'maxiter':100})
print('finished')
plt.plot(t,irfplotting(result.x,t),color='b',label='fit with IRF')
print(result.x)

plt.show()

