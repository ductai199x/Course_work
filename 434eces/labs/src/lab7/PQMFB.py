#Pseudo Quadrature Mirror Filter Bank implementation
#Gerald Schuller, gerald.schuller@tu-ilmenau.de, Sep. 2017

from __future__ import print_function
import numpy as np
# from polmatmult import polmatmult 
# from DCT4 import *
# from x2polyphase import *

import scipy.fftpack as spfft
#The DCT4 transform:
def DCT4(samples):
   #Argument: 3-D array of samples, shape (y,x,# of blocks), each row correspond to 1 row 
   #to apply the DCT to.
   #Output: 3-D array where each row ist DCT4 transformed, orthonormal.
   import numpy as np
   #use a DCT3 to implement a DCT4:
   r,N,blocks=samples.shape
   samplesup=np.zeros((1,2*N,blocks))
   #upsample signal:
   samplesup[0,1::2,:]=samples
   y=spfft.dct(samplesup,type=3,axis=1,norm='ortho')*np.sqrt(2)
   return y[:,0:N,:]

def polyphase2x(xp):
   """Converts polyphase input signal xp (a row vector) into a contiguos row vector
   For block length N, for 3D polyphase representation (exponents of z in the third 
   matrix/tensor dimension)"""

   x=np.reshape(xp,(1,1,-1),order='F') #order=F: first index changes fastest
   x=x[0,0,:]
   return x

def x2polyphase(x,N):
    """Converts input signal x (a 1D array) into a polyphase row vector 
    xp for blocks of length N, with shape: (1,N,#of blocks)"""     
    import numpy as np
    #Convert stream x into a 2d array where each row is a block:
    #xp.shape : (y,x, #num blocks):
    x=x[:int(len(x)/N)*N] #limit signal to integer multiples of N
    xp=np.reshape(x,(N,-1),order='F') #order=F: first index changes fastest
    #add 0'th dimension for function polmatmult:
    xp=np.expand_dims(xp,axis=0)
    return xp


def polmatmult( A,B ):
   """polmatmult(A,B)
   multiplies two polynomial matrices (arrays) A and B, where each matrix entry is a polynomial.
   Those polynomial entries are in the 3rd dimension
   The thirs dimension can also be interpreted as containing the (2D) coefficient
   exponent of z^-1.
   Result is C=A*B;"""  
   [NAx, NAy, NAz] = np.shape(A);
   [NBx, NBy, NBz] = np.shape(B);
   #Degree +1 of resulting polynomial, with NAz-1 and NBz-1 being the degree of the...
   Deg = NAz + NBz -1;
   C = np.zeros((NAx,NBy,Deg));
   #Convolution of matrices:
   for n in range(0,(Deg)):
       for m in range(0,n+1):
           if ((n-m)<NAz and m<NBz):
               C[:,:,n] = C[:,:,n]+ np.dot(A[:,:,(n-m)],B[:,:,m]);
   return C


def ha2Fa3d(qmfwin,N):
        #usage: Fa=ha2Fa3d_fast(ha,N);
        #produces the analysis polyphase folding matrix Fa with all polyphase components
        #in 3D matrix representation
        #from a basband filter ha with
        #a cosine modulation
        #N: Blocklength
        #Matrix Fa according to
        #chapter about "Filter Banks", cosine modulated filter banks.
       
        overlap=int(len(qmfwin)/N)
#         print("overlap=", overlap)
        Fa=np.zeros((N,N,overlap))
        for m in range(int(overlap/2)):
           Fa[:,:,2*m]+=np.fliplr(np.diag(np.flipud(
           -qmfwin[m*2*N:int(m*2*N+N/2)]*((-1)**m)),k=int(-N/2)))
           Fa[:,:,2*m]+=(np.diag(np.flipud(
           qmfwin[m*2*N+int(N/2):(m*2*N+N)]*((-1)**m)),k=int(N/2)))
           Fa[:,:,2*m+1]+=(np.diag(np.flipud(
           qmfwin[m*2*N+N:(m*2*N+int(1.5*N))]*((-1)**m)),k=-int(N/2)))
           Fa[:,:,2*m+1]+=np.fliplr(np.diag(np.flipud(
           qmfwin[m*2*N+int(1.5*N):(m*2*N+2*N)]*((-1)**m)),k=int(N/2)))
           #print -qmfwin[m*2*N:(m*2*N+N/2)]*((-1)**m)
        return Fa
        
        
        
def hs2Fs3d(qmfwin,N):
        #usage: Fs=hs2Fs3d_fast(hs,N);
        #produces the synthesis polyphase folding matrix Fs with all polyphase components
        #in 3D matrix representation
        #from a basband filter ha with
        #a cosine modulation
        #N: Blocklength
        #Fast implementation

        #Fa=ha2Fa3d_fast(hs,N)
        #print "Fa.shape in hs2Fs : ", Fa.shape
        #Transpose first two dimensions to obtain synthesis folding matrix:
        #Fs=np.transpose(Fa, (1, 0, 2))
        overlap=int(len(qmfwin)/N)
#         print("overlap=", overlap)
        Fs=np.zeros((N,N,overlap))
        for m in range(int(overlap/2)):
           Fs[:,:,2*m]+=np.fliplr(np.diag(np.flipud(
           qmfwin[m*2*N:int(m*2*N+N/2)]*((-1)**m)),k=int(N/2)))
           Fs[:,:,2*m]+=(np.diag((
           qmfwin[int(m*2*N+N/2):(m*2*N+N)]*((-1)**m)),k=int(N/2)))
           Fs[:,:,2*m+1]+=(np.diag((
           qmfwin[m*2*N+N:(m*2*N+int(1.5*N))]*((-1)**m)),k=int(-N/2)))
           Fs[:,:,2*m+1]+=np.fliplr(np.diag(np.flipud(
           -qmfwin[m*2*N+int(1.5*N):(m*2*N+2*N)]*((-1)**m)),k=int(-N/2)))
        #print "Fs.shape in hs2Fs : ", Fs.shape
        #avoid sign change after reconstruction:
        return -Fs
                
                
                
                
def PQMFBana(x,N,fb):
   #Pseudo Quadrature Mirror analysis filter bank.
   #Arguments: x: input signal, e.g. audio signal, a 1-dim. array
   #N: number of subbands
   #fb: coefficients for the Quadrature filter bank.
   #returns y, consisting of blocks of subband in in a 2-d array of shape (N,# of blocks)
   
   Fa=ha2Fa3d(fb,N)
#    print("Fa.shape=",Fa.shape)
   y=x2polyphase(x,N)
#    print("y[:,:,0]=", y[:,:,0])
   y=polmatmult(y,Fa)   
   y=DCT4(y)
   #strip first dimension:
   y=y[0,:,:]
   return y
   
# from polyphase2x import *   
def PQMFBsyn(y,fb):
   #Pseudo Quadrature Mirror synthesis filter bank.
   #Arguments: y: 2-d array of blocks of subbands, of shape (N, # of blokcs)
   #fb: prototype impulse response
   #returns xr, the reconstructed signal, a 1-d array.   
   N, m=y.shape
#    print("N=",N)
   Fs=hs2Fs3d(fb,N)
   #print np.transpose(Fs, axes=(2,0,1))
   #add first dimension to y for polmatmult:
   y=np.expand_dims(y,axis=0)
   xp=DCT4(y)
   xp=polmatmult(xp,Fs)
   xr=polyphase2x(xp)
   return xr
   
#Testing:
if __name__ == '__main__':
   import numpy as np
   import matplotlib.pyplot as plt

   #Number of subbands:
   N=32

   #D=Dmatrix(N)
   #Dinv=Dinvmatrix(N)
   #Filter bank coefficients, 1.5*N of sine window:
   fb=np.sin(np.pi/(2*N)*(np.arange(int(1.5*N))+0.5))
   # fb=np.loadtxt("QMFcoeff.txt")
   #compute the resulting baseband prototype function:
   fb = np.concatenate((fb,np.flipud(fb)))
   print("fb=", fb)
   plt.plot(fb)
   plt.title('The PQMF Protoype Impulse Response')
   plt.xlabel('Sample')
   plt.ylabel('Value')
   plt.xlim((0,31))
   plt.show()
   #input test signal, ramp:
   x=np.arange(16*N)
   plt.plot(x)
   plt.title('Input Signal')
   plt.xlabel('Sample')
   plt.ylabel('Value')
   plt.show()
   y=PQMFBana(x,N,fb)
   plt.imshow(np.abs(y))
   plt.title('PQMF Subbands')
   plt.xlabel('Block No.')
   plt.ylabel('Subband No.')
   plt.show()
   xr=PQMFBsyn(y,fb)
   plt.plot(xr)
   plt.title('Reconstructed Signal')
   plt.xlabel('Sample')
   plt.ylabel('Value')
   plt.show()
   #Input to the synthesis filter bank: unit pulse in lowest subband
   #to see its impulse response:
   y=np.zeros((N,2))
   y[0,0]=1
   xr=PQMFBsyn(y,fb)
   plt.plot(xr)
   plt.title('Impulse Response of Modulated Synthesis Subband 0')
   plt.xlabel('Sample')
   plt.ylabel('Value')
   plt.show()
   #Check for reconstruction error:
   #we can compute the residual matrix R from Fa and Fs:
   Fa=ha2Fa3d(fb,N)
   Fs=hs2Fs3d(fb,N)
   R=polmatmult(Fa,Fs)
   print( np.transpose(R, axes=(2,0,1)))
   print("R[:,:,7]=", R[:,:,7])
   R[:,:,7] = np.eye(N)-R[:,:,7]
   #This is our residual matrix, from which we can compute the   
   #reconstruction error as the sum of all its squared diagonal 
   #elements,
   re = 0;
   for m in range(15):
      re = re + sum(np.square(np.diag(R[:,:,m])))
   print("Reconstruction error=",re)
   print("The resulting SNR for the reconstruction error=",10*np.log10(N/re), "dB")
