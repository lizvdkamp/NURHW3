import numpy as np
import matplotlib.pyplot as plt
import timeit

#Question 1

#1a

#Taking my function from the previous handin, now with the A given
def n2x(x,a=2.4,b=0.25,c=1.6,Nsat=100,A=256/(5*np.pi**(3/2))):
	"""Returns N(x) = 4*pi*n(x)*x**2"""
	return 4*np.pi*A*Nsat*((x**(a-1)/b**(a-3))*np.exp(-(x/b)**c))

#Finding the maximum is the same as finding the minimum from -f(x)
def negn2x(x,a=2.4,b=0.25,c=1.6,Nsat=100,A=256/(5*np.pi**(3/2))):
	"""Returns N(x) = 4*pi*n(x)*x**2"""
	return -1*4*np.pi*A*Nsat*((x**(a-1)/b**(a-3))*np.exp(-(x/b)**c))

#Importing my golden section search algorithm

def goldsecsearch(func,a,b,c,acc,maxit):
    """Finds the minimum of a given function func, within [a,c], with b the first guess for the minimum. Terminates when either the maximum amount of iterations maxit is reached, or when the interval is small than our target accuracy."""

    #Defining the golden ratio
    phi = (1+np.sqrt(5))*0.5
    w = 2-phi
    
    #Iterate a maximum of maxit times
    for k in range(0,maxit):
	#Find the largest interval and set x to the outer edge of that
        if np.abs(b-a) > np.abs(c-b):
            x = a
        else:
            x = c

	#Set the next bracket
        d = b + (x-b)*w
        #print("a,b,c,d",a,b,c,d)

	#If target accuracy is reached, terminate
        if np.abs(c-a) < acc:
            print("Accuracy", np.abs(c-a))
            #Check which value to return
            if func(d) < func(b):
                return d
            else:
                return b

	#Else, keep looping with the new bracket
        else:
            #print("Loop")
            #Check again which interval was bigger by checking x and which values to change
            if x == c:
                if func(d) < func(b):
                    a,b = b,d
                else:
                    c = d

            if x == a:
                if func(d) < func(b):
                    c,b = b,d
                else:
                    a = d

    #Max iterations reached
    #print("Max it. Done",d)
    return d

#We set xmax as a variable
xmax = 5

#Search within [0,5] with first guess x_maximum = 1, because the plot shows that the maximum is somewhere between [0,1]
maximum = goldsecsearch(negn2x, 0,1,5, 10**(-20),100)

print("The maximum found is at x = ", maximum, "with N(x) = ", n2x(maximum))

#xes = np.linspace(0,xmax,1000)
#yes = n2x(xes)
#plt.plot(xes,yes)
#plt.scatter(maximum, n2x(maximum))
#plt.close()

# Save a text file
np.savetxt('Maximumoutput.txt',np.transpose([maximum, n2x(maximum)]))


#1b

#Import functions needed for Levenberg Marquardt routine
#Functions to swap, scale and add rows to each other
def SwapRow(M,i,j):
	"""Takes a matrix M and swaps rows i and j"""
	#Making a copy to make sure that it doesn't overwrite M
	B = np.copy(M).astype('float64')
	#Swaps row i and j
	save = B[i,:].copy()
	B[i,:]= B[j,:]
	B[j,:] = save

	return B

def SwapRowVec(x,i,j):
	"""Takes a vector/array x and swaps rows i and j"""
	#Making a copy to make sure that it doesn't overwrite x
	B = np.copy(x).astype('float64')
	#Swaps row i and j of the vector
	save = B[i].copy()
	B[i]= B[j]
	B[j] = save

	return B

def ScaleRow(M,i,scale):
	"""Takes a matrix M and scales row i with scale = scale"""
	#Making a copy to make sure that it doesn't overwrite M
	B = np.copy(M).astype('float64')
	#Scales row i
	B[i,:] *= scale

	return B

def AddRow(M,i,j,scale):
	"""Takes a matrix M and adds row i to row j scale times"""
	#Making a copy to make sure that it doesn't overwrite M
	B = np.copy(M).astype('float64')
	#Adds row i to row j scale times
	B[j,:] += B[i,:]*scale

	return B

#A function that uses (improved) Crouts algorithm to do LU decomposition
def Crout(A,b):
	"""Takes a Matrix A and vector b and applies Crouts algorithm to do LU decomposition on the Matrix A and using the L and U matrices to calculate a solution to the equation Ax = b."""
	rows = A.shape[0]
	columns = A.shape[1]

	#saving the parity just in case
	parity = 1

	#Making a copy of the matrix and the vector to make sure that it doesn't overwrite them
	LU = np.copy(A).astype('float64')
	b_new = np.copy(b).astype('float64')

	#Making an index array to keep track of swapped indices
	indx = np.arange(0,rows,1, dtype=int)
	#print(indx)

	#First I check if the matrix A is singular by looking if there is an all-zero column
	for j in range(columns):
		if np.all(LU[:,j]==0):
			print("The matrix is singular. Stopping.")
			#Returning a zero solution
			return np.zeros(rows)
        
	#Putting the highest values in a given column on a diagonal by looking through all values below the diagonal and checking if there is a higher value before swapping
	for k in range(columns):
		#looping over columns and saving the diagonal
		mx = np.abs(LU[k,k])
		piv = k
		#print(mx)
        
		for i in range(k+1,rows):
			#checking if the rows below have higher values to put on the diagonal
			xik = np.abs(LU[i,k])
			#print(xik)
            
			if xik > mx:
				mx = xik
				piv = i
				#print(mx)
                
		#If there is a higher value in the column below the diagonal we swap the relevant rows
		if piv != k:
			#swapping row & swapping index array
			#print("Swap", k, piv)
			LU = SwapRow(LU,k,piv)
			indx = SwapRowVec(indx,k,piv).astype(int)
			parity = -parity
			#print(LU, b_new)
            
		#getting the LU matrix
		diag = LU[k,k]
		for i in range(k+1,rows):
			LUik = LU[i,k] / diag
			LU[i,k] = LUik
			for j in range(k+1,rows):
				LU[i,j] -= LUik * LU[k,j]
	#print(LU)

	#Getting the solution
	x_sol = np.zeros(rows)
	y_sol = np.zeros(rows)
    
	#print(indx)
	#Solving the equation Ux = y with forward substitution
	for n in range(rows):
		ay = 0
		for m in range(0,n):
			ay += LU[n,m]*y_sol[m]
		y_sol[n] = b_new[indx[n]]-ay
		#print(b_new[n],y_sol)
    
	#Solving Ly = b with backsubstitution
	for n in range(rows):
		#Making sure that we loop from N-1 --> 0
		backsub = rows-(n+1)
		ax = 0
		for m in range(backsub,rows):
			ax += LU[backsub,m]*x_sol[m]
		x_sol[backsub] = 1/LU[backsub,backsub] * (y_sol[backsub]-ax)
		#print(x_sol)

	return LU, x_sol, indx


#Now defining the new functions for the LM routine
def chisq(func,xs,ys,sigmasq,p0):
    """Returns chi squared for a given model function func, with data points xs, ys, and sigma squared of sigmasq and model values p0"""
    return np.sum((ys-func(xs,p0))**2/sigmasq)

def beta(func,xs,ys,sigmasq,p0,derivs):
    """Returns the beta value needed for Levenberg-Marquardt for a given model function func, with data points xs, ys, and sigma squared of sigmasq and model values p0"""
    return np.sum((ys-func(xs,p0))/sigmasq*derivs(xs,p0), axis=1)

def alpha(xs,sigmasq,p0,derivs,lamb=0):
    """Returns the alpha values needed for Levenberg-Marquardt for given data points xs, with sigma squared sigmasq, with the derivative of the model function derivs and model values p0"""
    #Load in the derivatives
    derivatives = derivs(xs,p0)
    #print(derivatives[0,:])
    #The amount of dimensions is the same as the nr of derivatives
    dim = len(derivatives)
    #We create the regular alpha matrix as well as the modified alpha matrix which takes the stepsize into account
    alphaij = np.zeros((dim,dim))
    alpha2ij = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            alphaij[i,j] = np.sum(1/sigmasq*derivatives[i,:]*derivatives[j,:])

            if i == j:
                alpha2ij[i,j] = (1+lamb)*np.sum(1/sigmasq*derivatives[i,:]*derivatives[j,:])
            else:
                alpha2ij[i,j] = np.sum(1/sigmasq*derivatives[i,:]*derivatives[j,:])
                
    return alphaij, alpha2ij

#Creating a function that doesn't incorporate A so that we can calculate it from the integral (this integral should give 1/A as a result after integrating from 0 to xmax)
def nxA(x,p):
	"""Returns N(x) / (Nsat*A)"""
	return 4*np.pi*((x**(p[0]-1)/p[1]**(p[0]-3))*np.exp(-(x/p[1])**p[2]))

#Function for LM 
def LevMarq(func,xs,ys,sigmasq,p0,derivs,maxit,it=0,lamb=10**(-3),w=10):
    """Performs a Levenberg Marquardt routine to find the best fit using a chi squared. It takes the model function func, the data xs,ys, the variance sigmasq, the model parameters p0, the derivatives of the function with respect to these model parameters, and a maximum number of iterations. This is a specific function for this case."""
    
    #We want to calculate the normalization A inside the function but change the global variable as well
    global Aint

    #Calculate initial parameters: chi squared, alpha matrices and beta 
    chi0 = chisq(func,xs,ys,sigmasq,p0)
    
    alpha0, alpha20 = alpha(xs,sigmasq,p0,derivs,lamb)
    
    beta0 = beta(func,xs,ys,sigmasq,p0,derivs)
    
    #print("beta", beta0)
    #print("Initial params chi0", chi0, "p0", p0)
    
    #Now we use LU decomposition to calculate the stepsize in parameter space by using alpha and beta
    alphanew, dp, indxs = Crout(alpha20, beta0)

    #print("dp", dp)
    
    #Calculate a new guess for the model parameters
    pnew = p0 + dp

    #Calculate the new mean & variance in each bin for these new parameters
    sigmanew = np.zeros(len(sigmasq))
    for i in range(len(sigmasq)):
        sigmanew[i] = Romberg(bins[i],bins[i+1],20,8,func,pnew)
    
    #Calculate new chi squared value
    chinew = chisq(func,xs,ys,sigmanew,pnew)
    #print("New params chinew", chinew, "pnew", pnew)
    
    if chinew >= chi0:
        lamb *= w
        pnew = p0
        
    else:
        lamb = lamb/w
	#Calculate new normalization factor A
        ANsat = Romberg(xs[0],xmax,20,8,nxA,pnew)
        Aint = 1/ANsat
    	#print("Function new Aint", Aint)
        
        #If the change in chi is too small, we return
        if np.abs(chi0-chinew)<10**(-4):
            return pnew,chinew
        
    #Change the number of iterations to keep track of them
    it+= 1
    
    #Return if we reached the max nr of iterations, otherwise call the function again
    if it > maxit:
        return pnew,chinew
    else:
        return LevMarq(func,xs,ys,sigmanew,pnew,derivs,maxit,it,lamb,w)

#Defining the needed functions for integration
def trapezoid(a,b,N,func,p):
	"""Takes a function 'func' and calculates the trapezoidal area underneath the function for on the interval [a,b] with stepsize (b-a)/N"""
	#step size
	h = (b-a)/N

	xs = np.linspace(a,b,N) #x
	fxs = func(xs,p)	#f(x)

	#trapezoidal area
	trpzd = h*(fxs[0]*0.5 + np.sum(fxs[1:N-1]) + fxs[N-1]*0.5)

	return trpzd

#Function for Romberg integration
def Romberg(a,b,N,m,func,p):
	"""Calculates the integral of a function 'func' over the interval [a,b] by iterating m times over a trapezoidal integration with initial stepsize (b-a)/N"""

	h = (b-a)/N	#initial stepsize
	r = np.zeros(m)
	#Initial guess
	r[0] = trapezoid(a,b,N,func,p)
	Np = N

	#First loop where we iterate over different stepsizes
	for i in range(1,m):
		r[i] = 0
		delta = h
		h = h*0.5
		x = a+h
        
		for n in range(Np):
			r[i] += func(x,p)
			x += delta
		    
		#Initial guess
		r[i] = 0.5*(r[i-1]+delta*r[i])
		Np *= 2
    
    
	Np = 1
	#Improving by iterating m times
	for i in range(1,m):
		Np *= 4
		for j in range(0, m-i):
			r[j] = (Np*r[j+1]-r[j])/(Np-1)
            
	return r[0]
	

#Importing functions for part 1c (Quasi Newton minimization)
def QuasiNewt(func, x, maxit, acc, grad, H_0=np.array([[1,0,0],[0,1,0],[0,0,1]]), it=0, xpts=None, minstep=10**(-9)):
    """Uses a Quasi Newton method to minimize a multidimensional function func, with x as the parameters to minimize, and gradient the function gradient. Terminates after either target accuracy acc is reached, or maximum iterations maxit is reached. Returns minimized x, x_new, and the steps it has taken to get to the minimum, xpts. It is possible to give up a minimum stepsize minstep."""
   
    #Matrix vector multiplication to calculate the direction of stepping
    #print(grad(x))
    n_init = -np.sum(H_0 * grad(x), axis=1)
   
    #print("n", n_init)
   
    #Create a function to minimize
    def minim(lambd):
        return func(x + lambd*n_init)
   
    #Find the best stepsize with the golden section search
    lam_i = goldsecsearch(minim, -15,0,15,10**(-10),30)
    #print("lam", lam_i)
   
    #lams = np.linspace(-15,15,100)
    #ys = np.zeros(100)
    #for j in range(100):
    #    ys[j] = minim(lams[j])
        
    #plt.plot(lams,ys)
    #plt.scatter(lam_i, minim(lam_i))
    #plt.show()
   
    delta = lam_i *n_init
    
    #We do not want the stepsize to be zero
    if lam_i == 0:
        lam_i = 10**(-3)
    
    #print(minstep, np.abs(delta)[0] < minstep and np.abs(delta)[1] < minstep)
    #Check if the stepsize is large enough
    while np.abs(delta)[0] < minstep and np.abs(delta)[1] < minstep:
        #print("times ten")
        delta *= 10
    
    #Take the step
    x_new = x + delta
    #print("x", x, "delta", delta, "xnew", x_new, "x+delta",x+delta)
    
    #Save the steps taken in case we want to see the road to the minimum
    if it == 0:
        xpts = x_new.copy()
        #print("triangles", triangles)    
    else:
        #print("xpts, x_new", xpts, x_new)
        xpts = np.vstack((xpts,x_new)).copy()
        #print("triangles new", triangles)
   
    #print("xnew", x_new)
   
    #Function values
    f0, f_new = func(x), func(x_new)
   
    #Check if we've reached the accuracy return
    if np.abs(f_new - f0)/(0.5*np.abs(f_new - f0)) < acc:
        #print("acc return")
        return x_new, xpts
   
    D_i = grad(x_new) - grad(x)
    #print("D", D_i)
   
    #Check if the gradient is smaller than the target accuracy
    if np.abs(np.amax(grad(x_new), axis=0)) < acc:
        print("grad conv", np.amax(grad(x_new), axis=0))
        return x_new, xpts
   
   
    HD =np.sum(H_0*D_i, axis=1)
   
    u = delta/np.sum(delta*D_i) - HD/np.sum(D_i * HD)
    #print("u", u)
   
    H_i = H_0 + np.outer(delta, delta)/np.sum(delta*D_i) - np.outer(HD, HD)/np.sum(D_i * HD) + np.sum(D_i * HD)*np.outer(u,u)
    #print("H", H_i)
   
    it+= 1
    #print("next it", it)
    if it >= maxit:
        #print("max its reached")
        return x_new, xpts
    else:
        #print("x now", x_new)
        return QuasiNewt(func, x_new, maxit, acc, grad, H_i, it, xpts, minstep)
    


#Taking the function from the example
def readfile(filename):
    f = open(filename, 'r')
    data = f.readlines()[3:] #Skip first 3 lines 
    nhalo = int(data[0]) #number of halos
    radius = []
    
    for line in data[1:]:
        if line[:-1]!='#':
            radius.append(float(line.split()[0]))
    
    radius = np.array(radius, dtype=float)    
    f.close()
    return radius, nhalo #Return the virial radius for all the satellites in the file, and the number of halos

#Call this function as:
for nr in range(11,16):
	radius, nhalo = readfile(f'satgals_m{nr}.txt')
	#After plotting the histograms of the files, I see that they go to about a maximum radius
	#of x = 2.4, where the first goes out to a bit more than that. I take real space, since it takes on a Poisson curve with real axes already, and I take range [0,2.4]
	#In the plot of N(x) we could already see that the function is close to zero after x = 1.5, but with many halos we want to go out further, since
	#Twenty four bins looks good
	bins = np.linspace(0,2.4,24)
	
	#The mean number of satellites is the total nr of satellites divided by the nr of halos
	Nsat = len(radius)/(nhalo)
	#print(Nsat, np.amax(radius))
	
	#plt.hist(radius, bins=bins, density=False)
	#plt.show()
        #Make the histogram
	Ns, bins = np.histogram(radius, bins=bins)
	binwidth = (bins[3]-bins[2])
	#The mean number per halo in bins
	Nis = Ns/(nhalo*binwidth)
	#Make an array of the middle of the bins
	binmids = np.array(bins - binwidth*0.5)[1:]
	#print(binwidth, "bin mids", binmids)

	#Initial guess
	p_init = np.array([2,0.5,1.6])

	#Initial A
	ANsat = Romberg(0,xmax,20,8,nxA,p_init)
	Aint = 1/ANsat
	#print("First Aint", Aint)

	#Now import the function and its derivatives
	def nxfit(x,p, A=Aint, Nsat=Nsat):
		"""Returns N(x) = 4*pi*n(x)*x**2 """
		#p = [a,b,c]
		#print("nxfit Aint, Nsat", Aint, Nsat)
		return 4*np.pi*A*Nsat*((x**(p[0]-1)/p[1]**(p[0]-3))*np.exp(-(x/p[1])**p[2]))

	def derivats(x,p,A=Aint, Nsat=Nsat):
		"""Returns the derivatives of the above functions, to p=[a,b,c]."""
		#print("derivats Aint", Aint)
		#With respect to a (p[0])
		der0 = 4*np.pi*A*Nsat*((x**(p[0]-1)/p[1]**(p[0]-3))*np.exp(-(x/p[1])**p[2])) * np.log(x/p[1])
		#With respect to b (p[1])
		der1 = -4*np.pi*A*Nsat*((x**(p[0]-1)/p[1]**(p[0]-2))*np.exp(-(x/p[1])**p[2]))*(-p[2]*(x/p[1])**p[2]+p[0]-3)
		#With respect to c (p[2])
		der2 = -4*np.pi*A*Nsat*((x**(p[0]-1)/p[1]**(p[0]-3))*np.exp(-(x/p[1])**p[2])) * (x/p[1])**p[2] * np.log(x/p[1])
		return np.array([der0,der1,der2])
	
        #Calculate the initial sigma values
	sigmasqs = np.zeros(len(Nis))
	for i in range(len(Nis)):
		sigmasqs[i] = Romberg(bins[i],bins[i+1],20,8,nxfit,p_init)
		
        #Use LM to find the best fit 
	psol, chisol = LevMarq(nxfit,binmids,Nis,sigmasq=sigmasqs,p0=p_init,derivs=derivats,maxit=50)
	
	#final calculation of Aint
	Ainverse = Romberg(0,xmax,20,8,nxA,psol)
	Aint = 1/Ainverse

        #Calculate the predicted bin values
	Nis_fitchi_unscaled = nxfit(binmids,p=psol,Nsat=Nsat,A=Aint)
	#scale so that the sums are equal
	Nis_fitchi = Nis_fitchi_unscaled * np.sum(Nis) / np.sum(Nis_fitchi_unscaled)
	#Intchi = Romberg(0,xmax,20,8,nxfit,psol)

	#Plotting
	plt.bar(binmids, Nis, binwidth, label='binned data')
	plt.plot(binmids, Nis_fitchi, color='k', label=r'best-fit, $<N_{sat}>$ = '+str(np.round(Nsat,3))+' [a,b,c] = '+str(np.round(psol,2))+r' $\chi^2 = $'+str(np.round(chisol,3)))
	plt.xlim(0.04,2.5)
	plt.yscale('log')
	plt.xscale('log')
	plt.ylabel('Mean number of satellites per halo')
	plt.xlabel(r'Radius $x = r/r_{vir}$')
	plt.title(r"$\chi^2$ best fit with Levenberg Marquardt routine.")
	plt.legend(loc='lower left')
	plt.savefig(f'LMplot{nr}.png')
	plt.close()
	#plt.show()

	print("Nsat", Nsat, "psol", psol, "chi", chisol, "Aint", Aint)
	#print("Integral",Intchi)

	
	#1c

	#Poisson likelihood
	def lnLikelihood(p0, func=nxfit,ys=Nis, xs=binmids):
		"""Returns minus the natural log of the Poisson likelihood, for a given model function func, data ys,xs, and model parameters p0."""
		#Calculating A & mu needed for nxfit
		global Aint
		ANsat = Romberg(0,xmax,20,6,nxA,p0)
		Aint = 1/ANsat
		return -np.sum(ys*np.log(func(xs,p0))-func(xs,p0))
		
	#Gradient of the likelihood
	def Likelihoodgrad(p, func=nxfit,ys=Nis, xs=binmids, derivs=derivats):
		"""Returns the gradient minus the natural log of the Poisson likelihood, for a given model function func, data ys,xs, model parameters p, and model derivatives derivs."""
		#Calculating A & mu needed for nxfit
		global Aint
		ANsat = Romberg(0,xmax,20,6,nxA,p)
		Aint = 1/ANsat
		#mus = np.zeros(len(Nis))
		#for i in range(len(Nis)):
		#	mus[i] = Romberg(bins[i],bins[i+1],20,4,func,p)
		#print(mus)
		return np.sum((ys/func(xs,p)-1)*derivs(xs,p), axis=1)
		
	
	#Initialize a first guess and calculate the fit value
	p_firstguess = np.array([2,0.5,1.6])
	#p_firstguess = psol
	p_guess, ps = QuasiNewt(lnLikelihood, p_firstguess, 50, 10**(-10), grad=Likelihoodgrad, minstep=10**(-5))

	#Minimum likelihood
	lnLmin = lnLikelihood(p_guess)

	#final calculation of Aint
	Ainverse = Romberg(0,xmax,20,8,nxA,p_guess)
	Aint = 1/Ainverse
	Nis_fitPoiss_unscaled = nxfit(binmids,p=p_guess,Nsat=Nsat,A=Aint)
	#Scale so that the sum is equal to nhalo
	Nis_fitPoiss = Nis_fitPoiss_unscaled * np.sum(Nis) / np.sum(Nis_fitPoiss_unscaled)

	
	print("Nsat", Nsat, "psol", p_guess, "lnL", lnLmin, "Aint", Aint)
	#print("Integral", IntPoiss)
	
	#Plotting
	plt.bar(binmids, Nis, binwidth, label='binned data')
	plt.plot(binmids, Nis_fitPoiss, color='k', label=r'best-fit, $<N_{sat}>$ = '+str(np.round(Nsat,3))+' [a,b,c] = '+str(np.round(p_guess,2))+r' ln(L) = '+str(np.round(lnLmin,3)))
	plt.xlim(0.04,2.5)
	plt.yscale('log')
	plt.xscale('log')
	plt.ylabel('Mean number of satellites per halo')
	plt.xlabel(r'Radius $x = r/r_{vir}$')
	plt.title(r"Poisson likelihood best fit with Quasi Newtonroutine.")
	plt.legend(loc='lower left')
	plt.savefig(f'QNplot{nr}.png')
	plt.close()
	#plt.show()
	
	
	#1d
	#Make sure these are comparable with Ns
	Ns_fitchi = Nis_fitchi*nhalo*binwidth
	Ns_fitPoiss = Nis_fitPoiss*nhalo*binwidth
	
	#print(Ns_fitchi, Ns_fitPoiss, Ns)
	#We have zero values in the bin, so we artificially sum
	G_chi = 0
	G_Poiss = 0
	for i in range(len(Ns)):
		if Ns[i] != 0:
			G_chi += 2*Ns[i]*np.log(Ns[i]/Ns_fitchi[i])
			G_Poiss += 2*Ns[i]*np.log(Ns[i]/Ns_fitPoiss[i])
		else:
			G_chi += 0
			G_Poiss += 0
	print("Gs",G_chi, G_Poiss)
	print("sums",np.sum(Ns),np.sum(Ns_fitchi),np.sum(Ns_fitPoiss))
	
	#Now calculate the Q 
	import scipy.special as scp
	def Qval(x,k):
		#print(scp.gammainc(k*0.5,x*0.5),scp.gamma(k*0.5))
		return 1-(scp.gammainc(k*0.5,x*0.5)/scp.gamma(k*0.5))
		
	#The number of independent degrees of freedom is the nr of data pts len(Ns) minus the parameters = 3 [a,b,c] minus 1 (because the last one is determined by the rest
	k = len(Ns)-4
	Q_chi = Qval(G_chi,k)
	Q_Poiss = Qval(G_Poiss,k)
	print("Qs",Q_chi,Q_Poiss)

	

	#Saving everything
	names1  = np.array(['Nsat', 'a, chi sq', 'b, chi sq', 'c, chi sq', 'chi squared', 'a, Poisson', 'b, Poisson', 'c, Poisson','-ln Likelihood'])
	fitresults = np.transpose([Nsat, psol[0], psol[1], psol[2], chisol, p_guess[0], p_guess[1], p_guess[2], lnLmin])

	fitres = np.zeros(names1.size, dtype=[('var1', 'U6'), ('var2', float)])
	fitres['var1'] = names1
	fitres['var2'] = fitresults

	np.savetxt(f'Fitresultsoutput{nr}.txt',fitres, fmt="%10s %10.16f")


	names2  = np.array(['G chi sq', 'G Poisson', 'Q chi sq', 'Q Poisson'])
	results = np.transpose([G_chi, G_Poiss, Q_chi, Q_Poiss])

	res = np.zeros(names2.size, dtype=[('var1', 'U6'), ('var2', float)])
	res['var1'] = names2
	res['var2'] = results
	np.savetxt(f'Resultsoutput{nr}.txt',res, fmt="%10s %10.16f")
	
#It seems like the fits are basically equivalent between Poisson and Chi squared. All of them give Q values of very close to one, but the G values are pretty high for the 11-13, 14 & 15 give better G values and also are a better fit to the data











