import numpy as np
import time
from tqdm import tqdm
from numba import jit,njit



'''Own Procrastinating Project writing a library to solve differential equation with self modified:

-ODE solvers on Runge Kutta basis (modfifications like adaptive stepsize and error calculation)
-Shooting method with RK4 and newt raphson for boundary value problems
-indirect Numerical solving methods for sparse matrices (or fuller ones (direct solutions))

+ additional rk4 jitted to improve speed
 '''


class ODEsolver():
    def RK1_euler_method(f,y0,t):
        print('Euler calculating')
        y = np.zeros((len(y0),len(t)))
        y[:,0] = np.transpose(y0)
        for i in range(0,len(t)-1):
            tau = t[i+1]-t[i]
            for j in range(len(y0)):
                k1 = tau*f(y[:,i],t[i])[j]
                y[j,i+1] = y[j,i] + k1
        return y
    
    def RK2_method(f,y0,t):
        print('RK2 calculating')
        y = np.zeros((len(y0),len(t)))
        y[:,0] = np.transpose(y0)
        for i in range(0,len(t)-1):
            tau = t[i+1]-t[i]
            for j in range(len(y0)):
                k1 = tau*f(y[:,i],t[i])
                k2 = tau*f((y[:,i]+k1/2),(t[i]+tau/2))
                y[j,i+1] = y[j,i] + k2[j] 
        return y
    
    def RK3_method(f,y0,t):
        print('RK3 calculating')
        y = np.zeros((len(y0),len(t)))
        y[:,0] = np.transpose(y0)
        for i in range(0,len(t)-1):
            tau = t[i+1]-t[i]
            for j in range(len(y0)):
                k1 = tau*f(y[:,i],t[i])
                k2 = tau*f((y[:,i]+k1),(t[i]+tau))
                k3 = tau*f((y[:,i]+1/4*(k1+k2)),(t[i]+tau/2))
                y[j,i+1] = y[j,i] + 1/6*(k1[j] + k2[j] + 4*k3[j])
        return y
    
   
    def RK4_method(f,y0,t):
        print('RK4 calculating for y0= '+str(y0))
        st,cpu_st = time.time(),time.process_time()

        #This is the algorythm for the Runge Kutta 4 Order
        y = np.zeros((len(y0),len(t)))          #y will result in a vector of numeric solutions of the ODE's
        y[:,0] = np.transpose(y0)
        for i in tqdm(range(0,len(t)-1),bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}'):
            tau = t[i+1]-t[i]
            for j in range(len(y0)):
                k1 = tau*f(y[:,i],t[i])
                k2 = tau*f((y[:,i]+k1/2),(t[i]+tau/2))
                k3 = tau*f((y[:,i]+k2/2),(t[i]+tau/2))
                k4 = tau*f((y[:,i]+k3),(t[i]+tau))
                y[j,i+1] = y[j,i] + 1/6*(k1[j] + 2*k2[j] + 2*k3[j] + k4[j])


        ft,cpu_ft = time.time(),time.process_time()
        print('-----------> #EXECUTION-time: '+str(round(ft-st,5))+' seconds '+ '\t'+'#CPU-PROCESS-time: '+str(round(cpu_ft-cpu_st,5))+' seconds '+'\n')
        return y
    
    
    def RK4_jit_method(f,y0,t):
        print('RK4 calculating for y0= '+str(y0))
        st,cpu_st = time.time(),time.process_time()
        
        @jit
        def algorythm(f,y0,t):
            y = np.zeros((len(y0),len(t)))          #y will result in a vector of numeric solutions of the ODE's
            y[:,0] = np.transpose(y0)
            for i in range(0,len(t)-1):
                tau = t[i+1]-t[i]
                for j in range(len(y0)):
                    k1 = tau*f(y[:,i],t[i])
                    k2 = tau*f((y[:,i]+k1/2),(t[i]+tau/2))
                    k3 = tau*f((y[:,i]+k2/2),(t[i]+tau/2))
                    k4 = tau*f((y[:,i]+k3),(t[i]+tau))
                    y[j,i+1] = y[j,i] + 1/6*(k1[j] + 2*k2[j] + 2*k3[j] + k4[j])
            return y

        y= algorythm(f,y0,t)

        ft,cpu_ft = time.time(),time.process_time()
        print('-----------> #EXECUTION-time: '+str(round(ft-st,5))+' seconds '+ '\t'+'#CPU-PROCESS-time: '+str(round(cpu_ft-cpu_st,5))+' seconds '+'\n')
        return y




    def Estimated_errors(f,y0,t,order):                 #t as array of np.arange(t0,tmax, stepsize)
        print('Error estimation calculating')
        RKorders = [ODEsolver.RK1_euler_method,ODEsolver.RK2_method,ODEsolver.RK3_method,ODEsolver.RK4_method]
        
        ttau = t
        tsteps = [(t[i+1]-t[i]) for i in range(len(t)-1)]
        ttwicetauhalf = [ttau[0]]           #[(t[i+1]-t[i])/2 for i in range(len(t)-1)]])     #generate time array with tau/2 steps
        for i in range(len(t)-1):
            ttwicetauhalf.append(t[i] + (t[i+1]-t[i])/2)
            ttwicetauhalf.append(t[i+1])
        #print(len(ttau),len(ttwicetauhalf))

        ytau =  RKorders[order-1](f,y0,ttau)                     #ytau will be the solution of RK n-th order with stepsize Tau
        ytwicetauhalf = RKorders[order-1](f,y0,ttwicetauhalf)    #ytau will be the solution of RK n-th order with stepsize 2*Tau/2  ! twice the stepnumber will be reduce to every second step
        ytwicetauhalf = ytwicetauhalf[:,::2]   
        #print(np.shape(ytau),np.shape(ytwicetauhalf))

        delta_abs = (np.abs(np.subtract(ytwicetauhalf,ytau)))/(2**(order)-1)

        return delta_abs
    


    def RK_with_adaptive_stepsize(f,y0,t,order,error_bound):
        """ 
        Function to calculate the numeric solution of an ODE-system via Runge Kutta method with given order 1-4.
        From the initial timesteps-array [t], the initial stepsize and boundary error [error_bound] will be used, to
        estimate the estimated error for each step with:

        delta abs = |y(2 steps with tau/2)-y(step with tau)| / 2^n-1

        This estimated error for each timestep  helps to calculate tau-optimal with:

        tau-optimal = 0.9 * tau*(error_bound/delta_abs)^(1/n+1)


        INPUTS:

        f = vector function f(y,t) containing the right side of the ODE-system must be array

        y0 = list [] containing the Initial conditions to solve the ODEs

        t = arraytype or list containing all timepoints

        order = integer 1,2,3,4 to choose the nth order of the runge Kutta method

        error_bound = float value
        """ 
        
        first_step_dt = t[1]-t[0]
        tstart, tend = t[0], t[-1]
        list_of_tau = [(t[i+1]-t[i]) for i in range(len(t)-1)]
        np.hstack((list_of_tau,list_of_tau))
        print('calculating: Runge-Kutta'+str(order)+' with adapitve Stepsize'+'\n'+'Boundary condition for error: '+str(error_bound)+'\n'+'Initial stepsize: '+str(first_step_dt))
    
        y_delta_abs = ODEsolver.Estimated_errors(f=f,y0=y0,t=t,order=order)
    
        list_of_tau_opt = np.zeros(np.shape(y_delta_abs))                               #nd array for y vektor with optimizes stepsizes
        for i in range(len(y_delta_abs[:,0])):
            for j in range(len(y_delta_abs[0,:])-1):
                list_of_tau_opt[i,j] = 0.9 * list_of_tau[j]*(y_delta_abs[i,j] and  error_bound/y_delta_abs[i,j]  or 0)**(1/(order+1)) 


        list_of_timepoints = [tstart]
        [list_of_timepoints.append(item) for item in np.cumsum(list_of_tau_opt[0])[1:]]


        RKorders = [ODEsolver.RK1_euler_method,ODEsolver.RK2_method,ODEsolver.RK3_method,ODEsolver.RK4_method]
        yvector = RKorders[order-1](f=f,y0=y0,t=list_of_timepoints)

        return yvector, list_of_timepoints



    def RK_with_adaptive_stepsize_dimensions(f,y0,t,order,error_bound,ODE_dimensions):
        """ 
        Function to calculate the numeric solution of an ODE-system via Runge Kutta method with given order 1-4.
        From the initial timesteps-array [t], the initial stepsize and boundary error [error_bound] will be used, to
        estimate the estimated error for each step with:

        delta abs = |y(2 steps with tau/2)-y(step with tau)| / 2^n-1

        This estimated error for each timestep  helps to calculate tau-optimal with:

        tau-optimal = 0.9 * tau*(error_bound/delta_abs)^(1/n+1)


        INPUTS:

        f = vector function f(y,t) containing the right side of the ODE-system must be array

        y0 = list [] containing the Initial conditions to solve the ODEs

        t = arraytype or list containing all timepoints

        order = integer 1,2,3,4 to choose the nth order of the runge Kutta method

        error_bound = float value
        """ 
        
        first_step_dt = t[1]-t[0]
        tstart, tend = t[0], t[-1]
        list_of_tau = [(t[i+1]-t[i]) for i in range(len(t)-1)]
        np.hstack((list_of_tau,list_of_tau))
        print('calculating: Runge-Kutta'+str(order)+' with adapitve Stepsize'+'\n'+'Boundary condition for error: '+str(error_bound)+'\n'+'Initial stepsize: '+str(first_step_dt))
    
        y_delta_abs = ODEsolver.Estimated_errors(f=f,y0=y0,t=t,order=order)
    
        
        y_delta_dimensionless = np.sum(y_delta_abs[0:ODE_dimensions,:],axis=0)
        list_of_tau_opt = np.zeros(np.shape(y_delta_dimensionless))                             #nd array for y vektor with optimizes stepsizes
        
        for j in range(len(y_delta_dimensionless)-1):
            list_of_tau_opt[j] = 0.9 * list_of_tau[j]*(y_delta_dimensionless[j] and  error_bound/y_delta_dimensionless[j]  or 0)**(1/(order+1)) 

        list_of_timepoints = [tstart]
        [list_of_timepoints.append(item) for item in np.cumsum(list_of_tau_opt)[1:]]
        


        RKorders = [ODEsolver.RK1_euler_method,ODEsolver.RK2_method,ODEsolver.RK3_method,ODEsolver.RK4_method]
        yvector = RKorders[order-1](f=f,y0=y0,t=list_of_timepoints)

        return yvector, list_of_timepoints
    








class root():
    def newton_raphson(f,df,x0):
        print('started newton raphson method to find root')
        delta_x = -f(x0)/df(x0)
        x0 = x0 + delta_x
        return x0

    def numeric_root_newton_raphson(f,x0,x_array):
        index_of_x0 = np.argmin(np.abs(x_array-x0))
        df = np.gradient(f,x_array)
        delta_x = -f[index_of_x0]/df[index_of_x0]
        rootx = x0 + delta_x
        return rootx,delta_x
    
    def iterative_numeric_root_newton_raphson(f,x0,x_array,suff_small_delta_x,maxiterations):
        st,cpu_st = time.time(),time.process_time()
        print('Recursive Newton raphson for: ','|x0:',x0,'|min ∆x:', suff_small_delta_x,'|max iterations:',maxiterations)
        index_of_x0 = np.argmin(np.abs(x_array-x0))
        df = np.gradient(f,x_array)
        delta_x = x_array[-1]-x_array[0]
        c=0
        while np.abs(delta_x) >= suff_small_delta_x:
            index_of_x0 = np.argmin(np.abs(x_array-x0))
            delta_x = -f[index_of_x0]/df[index_of_x0]
            x0 = x0 + delta_x
            c= c+1
            if c == maxiterations:
                break

        rootf = x0
        print('Root found after '+str(c)+' iterations: ','root: ',rootf,' delta_x: ',delta_x)
        ft,cpu_ft = time.time(),time.process_time()
        print('##### Execution-time: '+str(round(ft-st,5))+' seconds '+ '\t'+'#CPU-Process-time: '+str(round(cpu_ft-cpu_st,5))+' seconds '+'\n')
        return rootf,delta_x






















class BoundaryProblemsolver():
    def shooting_method(f,t,nwton_parameter_index,h_dev,delta_nwton_boundary,yRW1,yRW2,yRWindex,tRW,RKorder,maxiterations):  #Schrödinger(,,3,y0,yrw,)
        '''
        Mehtod can use different RK methods to solve a boundary problem via Nwet Raph

        f:  function in form of
                                    def f(y,x):
                                        return np.transpose([f1(y),f2(y),f3(y)])
                                    where y referes to the functions of y vector (y[0],y[1],y[2])
        
        t:  time or dimension of the Problem function. Example r(t) or Psi(x)
                                    form of array -> np.arange(start,stop,stepsize(tau))
        
        nwton_parameter_index:      gives the index of the parameter in y which is adjusted by the newton Raphson root finder
                                    example: QM wavefunction Energy is changed 
                                    -> 3

        h_dev:                      Difference value to calculate the first RK to get the deviation at RW
                                    first RK -> E+h
                                    scond RK -> E-h

        delta_newton_boundary:      given value to ensure the accuracy of the newton raphson calculated root

        yRW1:                       boundary values at beginning of RK

        yRW2:                       boundary values at end of RK

        yRWindex                    

        tRW:                        location of the boundary

        RKorder:                    1-4 Euler to Runge Kutta 4 can be chosen to calculate the trajectory

        '''
        print('#################','START SHOOTING method')
        RKorders = [ODEsolver.RK1_euler_method,ODEsolver.RK2_method,ODEsolver.RK3_method,ODEsolver.RK4_method]

        for i in range(maxiterations):
            print('#ITERATION:',i+1)
            y = RKorders[RKorder-1](f,yRW1,t)
            y_plus_h = list(yRW1)
            y_plus_h[nwton_parameter_index] = y_plus_h[nwton_parameter_index] + h_dev
            y_minus_h = list(yRW1)
            y_minus_h[nwton_parameter_index] = y_minus_h[nwton_parameter_index] - h_dev
            y_plus = RKorders[RKorder-1](f,y_plus_h,t)
            y_minus = RKorders[RKorder-1](f,y_minus_h,t)
        
            tRW_index = np.argmin(np.abs(t-tRW))
            y_RW = y[yRWindex,tRW_index]
            y_plus_RW = y_plus[yRWindex,tRW_index]
            y_minus_RW = y_minus[yRWindex,tRW_index]
            dy = (y_plus_RW - y_minus_RW)/(2*h_dev)

            #Newton-Raphson step
            de =  yRW2[nwton_parameter_index] - y_RW / dy

            if np.abs(de) <= delta_nwton_boundary:
                print('FINISHED after '+str(i+1)+' ITERATIONS'+'\n','\t found ROOT for PARAMETER: ',yRW1,'\n \n')
                break
                

            yRW1[nwton_parameter_index] = list(yRW1)[nwton_parameter_index]+de
            print('### ->new ROOT:',yRW1,'\n')

        return yRW1





            

class lgs_solver():
    def LU_decomposition(A):
        if not A.shape[0]==A.shape[1]:
            raise ValueError("Input matrix must be square")

        n = A.shape[0] 
        L = np.zeros((n,n),dtype='float64') 
        U = np.zeros((n,n),dtype='float64') 
        U[:] = A 
        np.fill_diagonal(L,1) # fill the diagonal of L with 1

        for i in range(n-1):
            for j in range(i+1,n):
                L[j,i] = U[j,i]/U[i,i]
                U[j,i:] = U[j,i:]-L[j,i]*U[i,i:]
                U[j,i] = 0
        return (L,U)
    
    def forward_subs(L,b):
        y=[]
        for i in range(len(b)):
            y.append(b[i])
            for j in range(i):
                y[i]=y[i]-(L[i,j]*y[j])
    
            y[i]=y[i]/L[i,i]
        return y

    def back_subs(U,y):
        x=np.zeros_like(y)
        for i in range(len(x),0,-1):
            x[i-1]=(y[i-1]-np.dot(U[i-1,i:],x[i:]))/U[i-1,i-1]
        return x

    def solve_system_LU(L,U,b):
        y=lgs_solver.forward_subs(L,b)
        x=lgs_solver.back_subs(U,y)
        return x




    def lu_decomposition_partial_pivot(matrix):
        n = len(matrix)
        L = np.eye(n)
        U = np.copy(matrix)
        P = np.eye(n)

        for k in range(n - 1):
            # Partial pivoting: find the row with the largest value in the current column
            pivot_row = np.argmax(np.abs(U[k:, k])) + k

            if pivot_row != k:
                # Swap rows in U matrix
                U[[k, pivot_row]] = U[[pivot_row, k]]
                # Swap rows in L matrix
                L[[k, pivot_row], :k] = L[[pivot_row, k], :k]
                # Keep track of row swaps in permutation matrix P
                P[[k, pivot_row]] = P[[pivot_row, k]]

            for i in range(k + 1, n):
                factor = U[i, k] / U[k, k]
                L[i, k] = factor
                U[i, k:] -= factor * U[k, k:]
    
        return P, L, U
    
    def solve_lu_decomposition(P, L, U, b):
        # Solve Ly = Pb using forward substitution
        n = len(b)
        y = np.zeros(n)
        Pb = np.dot(P, b)

        for i in range(n):
            y[i] = Pb[i] - np.dot(L[i, :i], y[:i])

        # Solve Ux = y using back substitution
        x = np.zeros(n)

        for i in range(n - 1, -1, -1):
            x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

        return x

    
    def sparse_matrix_conjugat_Gradient_method(A,b,guess_x0,tolerance,max_iterations):
        '''Function searches Minimum of n-dimensional Paraboloid defined by:
            f(x) = 1/2 xAx-bx
           The minimum is characterized by Ax-b=0

           A (NxN)= spare matrix of the problem AX=b
           b (N)
           guess x0 can be np.zeros(N)
           tolerance = float        example=e-10
           max iterations = integer    
        '''
        """ def paraboloid(A,x,b):
            return 1/2* x@A@x-b@x """
        
        # Initialize variables
        x = guess_x0
        r = b - np.dot(A,x)
        p = r

        def iterative_domain(x,r,p):
            for iteration in range(max_iterations):
                st = time.time()
                if np.linalg.norm(r) <= tolerance or iteration == max_iterations:
                    break
                alpha = np.float64(np.dot(r, r) / np.dot(p,A@p))
                x = np.float64(x + alpha * p)
                r_new = np.float64(r - alpha * np.dot(A,p))
                beta = np.float64(np.dot(r_new, r_new) / np.dot(r, r))
                p = np.float64(r_new + beta * p)
                r = np.float64(r_new)
                ft= time.time()
                print('Iteration:',iteration,':',np.linalg.norm(r),'Iteration-time:',f'{(ft-st):.3g}')
            
            return x,iteration
        
        x,iteration = iterative_domain(x,r,p)
    
        return x, iteration

    def sparse_matrix_conjugat_Gradient_method_performance(A,b,guess_x0,tolerance,max_iterations):
        
        # Initialize variables
        x = guess_x0
        r = b - A@x
        p = r

        def iterative_domain(x,r,p):
            for iteration in range(max_iterations):
                st = time.time()
                if np.linalg.norm(r) <= tolerance or iteration == max_iterations:
                    break
                alpha = np.float64((r@r)/ (p@(A@p)))
                x = np.float64(x + alpha * p)
                r_new = np.float64(r - alpha * (A@p))
                beta = np.float64((r_new@r_new) / (r@r))
                p = np.float64(r_new + beta * p)
                r = np.float64(r_new)
                ft= time.time()
                print('Iteration:',iteration,':',np.linalg.norm(r),'Iteration-time:',f'{(ft-st):.3g}')
            
            return x,iteration
        
        x,iteration = iterative_domain(x,r,p)
    
        return x, iteration




        

class math():
    def dev(y,x):
        f_dot = np.zeros(np.shape(y))
        for i in range(len(x)):
            f_dot[i]= ()


    def kron_delta(a,b):
        if a ==b:
            return 1
        else:
            return 0
        

class integrate():
    def iterative_trapz(ydata:np.array,xdata:np.array,stepsize:float,boundary=None):
        if not len(ydata)==len(xdata):
                    raise ValueError(str("x and y must have same dimensions, but have dim: "+str(len(ydata))+'x'+str(len(xdata))))
        if boundary == None:
            boundary = [xdata[0],xdata[-1]]

        idxstart = np.argmin(np.abs(xdata-boundary[0]))
        idxend  = np.argmin(np.abs(xdata-boundary[1]))
        print('Iterative TRAPZ for:','boundarys:',boundary,'data indexes:',idxstart,idxend)

        traps_list=[]
        for idx in tqdm(range(idxend-idxstart),bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}'):
            x0 = xdata[idxstart] + idx*stepsize
            idx_x0 = np.argmin(np.abs(xdata-x0))
            x1 = xdata[idxstart] + (idx+1)*stepsize
            idx_x1 = np.argmin(np.abs(xdata-x1))
            datastep = xdata[idx+1] -xdata[idx]
            h = xdata[idx_x1]-xdata[idx_x0]
            #print(x0,idx_x0,x1,idx_x1,'step',h)
            #if stepsize < datastep:
            #            raise ValueError("stepsize can't be shorter than data resolution")
            if x1 > boundary[1]:
                        break

            traps = (1/2*(ydata[idx_x0])+ 1/2*(ydata[idx_x1]) ) *h
            traps_list.append(traps)

        print('Integration Value for iterative TRAPZ calculated:',np.sum(traps_list))
        return sum(traps_list)

        
            
        

        
    

        