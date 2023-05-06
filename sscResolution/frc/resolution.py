import matplotlib.pyplot as plt
import random

from scipy.interpolate import interp1d

import ctypes
from ctypes import POINTER
from ctypes import c_void_p  as void_p

import numpy
import pkg_resources
import time
import SharedArray as sa
import uuid

import multiprocessing as mp

import functools
from functools import partial

from ..resolutiontypes import *
from ..statistics import *

#less or equal than 512x512 pixels  
SERIAL = 'le512'

#greather than 512x512 pixels 
PARALLEL = 'gt512'

#
#
##############################################################

def computeLocal_python( image, x, y, pad, eps, L , delta):

    def getWindow(pad,eps,L):
        t   = numpy.linspace(-1,1,2*(eps + pad))
        
        def sigmoidal(t,L):
            return 1.0/(1 + numpy.exp(-L[0]*(t+L[1]))) - 1.0/(1 + numpy.exp(-L[0]*(t-L[1])))
        
        u,v  = numpy.meshgrid(t,t)    
        mask = sigmoidal(u,L) * sigmoidal(v,L)
    
        return mask
   
    window = getWindow(pad, eps, L)

    local = image[ y - (eps+pad): y + (eps+pad), x - (eps+pad):x + (eps+pad) ] * window

    if  (local**2).sum() <  1e-5:
        return None, None
    else:
        return compute( local, delta), local


def _frc_( f1, f2, delta):

    alpha = f1.shape[0]/2
    
    def inprod(x, y):
        return (x * numpy.conjugate(y)).sum()
    
    F1 = numpy.fft.fftshift( numpy.fft.fft2(f1) )
    F2 = numpy.fft.fftshift( numpy.fft.fft2(f2) )
    
    dx  = 2.0/(f1.shape[1]-1)
    dwx = 1.0/(2*dx)
    dy  = 2.0/(f1.shape[0]-1)
    dwy = 1.0/(2*dy)
    
    ty = numpy.linspace(-dwy,dwy,f1.shape[0])
    tx = numpy.linspace(-dwx,dwx,f1.shape[1])
    wx,wy = numpy.meshgrid(tx,ty)
    wy = numpy.flipud(wy)
    
    r = numpy.linspace(0, min(dwx,dwy), f1.shape[1]//2)
    rc = alpha * dx
    
    curve = numpy.ones(r.shape) + 0*1j
    T  = numpy.ones(r.shape)
    Ts = numpy.ones(r.shape)
    B  = numpy.ones(r.shape) 
    I  = numpy.zeros(r.shape) 
    U  = numpy.zeros(r.shape) 
    N  = numpy.zeros([len(r),9])
    
    consistency_Ts = True
    consistency_T  = True
    consistency_B  = True
    
    for k in range(1,len(r)):
    
        mask = (wx**2 + wy**2 <= (r[k]+rc)**2) & (wx**2 + wy**2 >= (r[k]-rc)**2 )
        F1at = F1[mask]
        F2at = F2[mask]
        
        nF1at = numpy.linalg.norm(F1at)
        nF2at = numpy.linalg.norm(F2at)
        
        curve[k] = inprod(F1at, F2at) / (nF1at * nF2at)

        npixels = (mask.flatten()*1.0 > 0).sum()
        T[k] = ( 0.2071 + 1.9102/numpy.sqrt(npixels) ) / (1.2071 + 0.9102/numpy.sqrt(npixels))        
        Ts[k] = 3.0 / numpy.sqrt(npixels/2.0)

        M1 = numpy.max( abs( F1at ) )
        M2 = numpy.max( abs( F2at ) ) 
        M = max(M1, M2)
    
        #m = 1
        m1 = numpy.min( abs( F1at ) )
        m2 = numpy.min( abs( F2at ) ) 
        m  = max(delta, min( m1, m2) ) 
        
        a = M/m

        N[k,0] = M
        N[k,1] = m
        N[k,2] = a
        N[k,3] = npixels
        N[k,4] = nF1at
        N[k,5] = nF2at
        N[k,6] = m1/M2
        N[k,7] = m2/M1
        N[k,8] = N[k,6] * (1 / (N[k,0] + N[k,1])) + ((N[k,0]*N[k,1])/(N[k,0] + N[k,1])) * N[k,7]
        
        B[k] = 2.0 * numpy.sqrt(a) / (1 + a) 

        I[k] = 1.0/(2*(1+( curve[k].real ) )) 
        
        U[k] = 1.0/(2*(1+( B[k]) )) 
        
        if curve[k] < B[k]:
            consistency_B = False
            
        if curve[k] < T[k]:
            consistency_T = False

        if curve[k] < Ts[k]:
            consistency_Ts = False
        
    B[0] = 0
    I[0] = I[1]
    U[0] = U[1]

    curves = {}
    curves['C'] = curve
    curves['H'] = T
    curves['S'] = Ts
    curves['B'] = B
    curves['I'] = I
    curves['U'] = U
    curves['A'] = N
    curves['cons'] = {'H':consistency_T, 'S': consistency_Ts, 'B': consistency_B}
    
    return curves


def getRoi(img,x,y,pad,eps,L):
    roi = img[y-eps:y+eps, x-eps:x+eps]
    roi = numpy.pad( roi, (pad,pad))
    t   = numpy.linspace(-1,1,2*(eps + pad))
    
    def sigmoidal(t,L):
        return 1.0/(1 + numpy.exp(-L[0]*(t+L[1]))) - 1.0/(1 + numpy.exp(-L[0]*(t-L[1])))
    
    u,v  = numpy.meshgrid(t,t)    
    mask = sigmoidal(u,L) * sigmoidal(v,L)
    
    return roi * mask


def window(img,pad,L):
    roi = numpy.pad(img, (pad,pad))
    t   = numpy.linspace(-1,1,roi.shape[0])
    
    def sigmoidal(t,L):
        return 1.0/(1 + numpy.exp(-L[0]*(t+L[1]))) - 1.0/(1 + numpy.exp(-L[0]*(t-L[1])))

    u,v  = numpy.meshgrid(t,t)    
    mask = sigmoidal(u,L) * sigmoidal(v,L)

    return roi * mask


def computep(image, nthreads, *args):

    if not args:
        delta = 1e-5
    else:
        delta = args[0]
            
    ######### FRC ###########
    
    #even split
    eimg1 = image[0:image.shape[0]:2,0:image.shape[1]:2]
    eimg2 = image[1:image.shape[0]:2,1:image.shape[1]:2]
   
    #odd split
    oimg1 = image[1:image.shape[0]:2,0:image.shape[1]:2]
    oimg2 = image[0:image.shape[0]:2,1:image.shape[1]:2]
   
    ecurves = _frc_parallel_(eimg1, eimg2, nthreads, delta)
    ocurves = _frc_parallel_(oimg1, oimg2, nthreads, delta)

    ########################
    
    size = len(ecurves['C'])

    xaxis = numpy.linspace(0, 1, size)

    C = (ocurves['C'].real + ecurves['C'].real )/2.0
    H = (ocurves['H'] + ecurves['H'])/2.0
    S = (ocurves['S'] + ecurves['S'])/2.0
    B = (ocurves['B'] + ecurves['B'])/2.0
    
    xth,_, IdxTh = interpolated_intercept( xaxis, C, H )
    xts,_, IdxTs = interpolated_intercept( xaxis, C, S )
    xrev,_,IdxB  = interpolated_intercept( xaxis, C, B )

    if ocurves['cons']['B'] and ecurves['cons']['B']:
        xrev = 1
        IdxB = size-1

    if ocurves['cons']['S'] and ecurves['cons']['S']:
        xts = 1
        IdxTs = size-1

    if ocurves['cons']['H'] and ecurves['cons']['H']:
        xth = 1
        IdxTh = size-1
    
        
    dic = {}
    dic['curves'] = {
        'C': (ecurves['C'].real + ocurves['C'].real)/2.0,
        'H': (ecurves['H']      + ocurves['H'])/2.0,
        'S': (ecurves['S']      + ocurves['S'])/2.0,
        'B': (ecurves['B']      + ocurves['B'])/2.0,
        'U': (ecurves['U']      + ocurves['U'])/2.0,
        'I': (ecurves['I']      + ocurves['I'])/2.0,
        'A': (ecurves['A']      + ocurves['A'])/2.0,
    }
    
    dic['splitting'] = True
    
    dic['x'] = {
        'axis': xaxis,
        'H':    xth,
        'S':    xts,
        'B':    xrev,
        'idx':  [IdxTh, IdxTs, IdxB] 
    }
    
    return dic

def compute(image, *args):

    if not args:
        delta = 1e-5
    else:
        delta = args[0]
                
    ######### FRC ###########
   
    #even split
    eimg1 = image[0:image.shape[0]:2,0:image.shape[1]:2]
    eimg2 = image[1:image.shape[0]:2,1:image.shape[1]:2]
   
    #odd split
    oimg1 = image[1:image.shape[0]:2,0:image.shape[1]:2]
    oimg2 = image[0:image.shape[0]:2,1:image.shape[1]:2]
       
    ecurves = _frc_(eimg1, eimg2, delta)
    ocurves = _frc_(oimg1, oimg2, delta)

    ########################
   
    size = len(ecurves['C'])
        
    xaxis = numpy.linspace(0, 1, size)    

    C = (ocurves['C'].real + ecurves['C'].real)/2.0
    H = (ocurves['H']      + ecurves['H'])/2.0
    S = (ocurves['S']      + ecurves['S'])/2.0
    B = (ocurves['B']      + ecurves['B'])/2.0
    
    xth,_, IdxTh = interpolated_intercept( xaxis, C, H )
    xts,_, IdxTs = interpolated_intercept( xaxis, C, S )
    xrev,_,IdxB  = interpolated_intercept( xaxis, C, B )

    
    if ocurves['cons']['B'] and ecurves['cons']['B']:
        xrev = 1
        IdxB = size-1

    if ocurves['cons']['S'] and ecurves['cons']['S']:
        xts = 1
        IdxTs = size-1

    if ocurves['cons']['H'] and ecurves['cons']['H']:
        xth = 1
        IdxTh = size-1

        
    dic = {}
    dic['curves'] = {
        'C': (ecurves['C'].real + ocurves['C'].real)/2.0,
        'H': (ecurves['H']      + ocurves['H'])/2.0,
        'S': (ecurves['S']      + ocurves['S'])/2.0,
        'B': (ecurves['B']      + ocurves['B'])/2.0,
        'U': (ecurves['U']      + ocurves['U'])/2.0,
        'I': (ecurves['I']      + ocurves['I'])/2.0,
        'A': (ecurves['A']      + ocurves['A'])/2.0,
    }
    
    dic['splitting'] = True
    
    dic['x'] = {
        'axis': xaxis,
        'H':    xth,
        'S':    xts,
        'B':    xrev,
        'idx':  [IdxTh, IdxTs, IdxB] 
    }
    
    return dic


def compute2i(image1, image2, *args):

    if not args:
        delta = 1e-5
    else:
        delta = args[0]
            
    ######### FRC ###########
   
    curves = _frc_(image1, image2, delta)

    ########################
    
    size = len(curves['C'])
    
    xaxis = numpy.linspace(0, 1, size)

    C = curves['C'].real
    H = curves['H']
    S = curves['S']
    B = curves['B']
    
    xth,_, IdxTh = interpolated_intercept( xaxis, C, H )
    xts,_, IdxTs = interpolated_intercept( xaxis, C, S )
    xrev,_,IdxB  = interpolated_intercept( xaxis, C, B )
    
    if curves['cons']['B']:
        xrev = 1
        IdxB = size-1

    if curves['cons']['S']:
        xts = 1
        IdxTs = size-1

    if curves['cons']['H']:
        xth = 1
        IdxTh = size-1
    
    dic = {}
    dic['curves'] = {
        'C': curves['C'],
        'H': curves['H'],
        'S': curves['S'],
        'B': curves['B'],
        'U': curves['U'],
        'I': curves['I'],
        'A': curves['A']
    }

    dic['splitting'] = False
  
    dic['x'] = {
        'axis': xaxis,
        'H':    xth,
        'S':    xts,
        'B':    xrev,
        'idx':  [IdxTh, IdxTs, IdxB] 
    }
    
    return dic


def plot( dic ,usr ):

    if dic is None:
        print('ssc-resolution error! Empty FRC (probably due to an empty image)')
        return None

    else:
        label = usr['label']
        dx    = usr['pxlsize']
        un    = usr['unit']
        
        if 'figsize' in usr.keys():
            figsize = usr['figsize']
        else:
            figsize = (10,10)

        if 'fontsize' in usr.keys():
            fontsize = usr['fontsize']
        else:
            fontsize = [20,20]
            
        if 'full' in usr.keys():
            full = usr['full']
        else:
            full = False

        size  = len(dic['curves']['C'])
        xaxis = dic['x']['axis']

        C    = dic['curves']['C'].real
        Th   = dic['curves']['H']
        Ts   = dic['curves']['S']
        B    = dic['curves']['B']
        I    = dic['curves']['I']
        U    = dic['curves']['U']
        A    = dic['curves']['A']
        
        xth  = dic['x']['H']
        xts  = dic['x']['S']
        xrev = dic['x']['B']
        
        xvals = numpy.linspace(0, xaxis.max(), 4*size)
        
        plt.figure(figsize=figsize)
        plt.plot(xaxis, C, 'k.-')
        plt.plot(xaxis, Th, 'ys-')
        plt.plot(xaxis, Ts, 'gd-')
        
        if full:
            alphaColor=0.3
            
            OneOver7 = numpy.ones(len(xaxis)) * (1/7)
            
            plt.plot(xaxis, B, 'ro-')
            plt.plot(xaxis, abs(I), 'm-')
            plt.plot(xaxis, abs(U), 'b-')
            plt.plot(xaxis, A[:,8], '-.',color='0.3')
            plt.plot(xaxis, OneOver7, '--',color='orange')
            
            plt.fill_between(xaxis, abs(I), 0, color='magenta', alpha=alphaColor)
            
            legend = ['FRC: {}'.format(label), 0, 0, 0, 0, 0, 0, 0]
        else:

            legend = ['FRC: {}'.format(label), 0, 0]

        if xth > 0:

            dxth = 0.02 
            xth_ = dic['x']['H'] 
            legend[1] = '{}: {} - res: {} {}'.format(r'$T_{1/2}$', round(xth,4), round(dx/xth,3), un)
            
            plt.axvspan(xth_ - dxth/2, xth_ + dxth/2, color='yellow', alpha=0.6, lw=0)

        if xts > 0:
 
            dxts = 0.02 
            xts_ = dic['x']['S'] 

            legend[2] = '{}: {} - res: {} {}'.format(r'$T_{3\sigma}$',round(xts,4), round(dx/xts,3), un)

            plt.axvspan(xts_ - dxts/2, xts_ + dxts/2, color='green', alpha=0.4, lw=0)

        if xrev > 0 and full:
 
            dxrev = 0.02 
            xrev_ = dic['x']['B'] 
            
            legend[3] = '{}: {} - res: {} {}'.format(r'$B$', round(xrev,4), round(dx/xrev,3), un)
            legend[4] = '{}'.format( r'$I$' )
            legend[5] = '{}'.format( r'$U$')
            legend[6] = '{}'.format( r'$N$')
            legend[7] = '{}'.format( r'$\frac{1}{7}$' )
            
            plt.axvspan(xrev_ - dxrev/2, xrev_ + dxrev/2, color='red', alpha=0.2, lw=0)

        plt.legend(legend,prop={'size': fontsize[1]}, bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.xticks(fontsize=fontsize[0])
        plt.yticks(fontsize=fontsize[0])
        plt.ylim([ B.min(), C.max() ])

        
############################
# Shared arrays with Python
# Reference:

#https://stackoverflow.com/questions/46811709/large-numpy-arrays-in-shared-memory-for-multiprocessing-is-something-wrong-with

def _get_size_from_shape(shape):
    return functools.reduce(lambda x, y: x * y, shape)
 
def _create_np_shared_array(shape, dtype, ctype):
    # Feel free to create a map from usual dtypes to ctypes. Or suggest a more elegant way
    size = _get_size_from_shape(shape)
    shared_mem_chunck = mp.sharedctypes.RawArray(ctype, size)
    numpy_array_view = numpy.frombuffer(shared_mem_chunck, dtype).reshape(shape)
    return numpy_array_view

############################

def _worker_batch_frc_(params, idx_start,idx_end, elapsed):
   
    #params = (nr, threads, F1, F2, wx, wy, r, rc, curve, T, Ts)
    
    def _inprod(x, y):
        return (x * numpy.conjugate(y)).sum()

    nr  =  params[0]
    F1  =  params[2]
    F2  =  params[3]
    wx  =  params[4]
    wy  =  params[5]
    r   =  params[6]
    rc  =  params[7]
    curve = params[8]
    T     = params[9]
    Ts    = params[10]
    B     = params[11]
    I     = params[12]
    U     = params[13]
    N     = params[14]
    delta = params[15]

    for k in range(idx_start, idx_end):
        
        start = time.time()

        mask = (wx**2 + wy**2 <= (r[k]+rc)**2) & (wx**2 + wy**2 >= (r[k]-rc)**2 )
        F1at = F1[mask]
        F2at = F2[mask]

        nF1at = numpy.linalg.norm(F1at)
        nF2at = numpy.linalg.norm(F2at)

        if k > 0:
            curve[k] = ( _inprod(F1at, F2at) / (nF1at * nF2at) ).real
        
            npixels = (mask.flatten()*1.0 > 0).sum()

            T[k] = ( 0.2071 + 1.9102/numpy.sqrt(npixels) ) / (1.2071 + 0.9102/numpy.sqrt(npixels))
            Ts[k] = 3.0 / numpy.sqrt(npixels/2.0)
            
            M1 = numpy.max( abs(F1at) )
            M2 = numpy.max( abs(F2at) ) 
            M  = max(M1, M2)

            m1 = numpy.min( abs( F1at ) )
            m2 = numpy.min( abs( F2at ) ) 
            m  = max(delta, min( m1, m2) ) 

            a = M/m

            N[k,0] = M
            N[k,1] = m
            N[k,2] = a
            N[k,3] = npixels
            N[k,4] = nF1at
            N[k,5] = nF2at
            N[k,6] = m1/M2
            N[k,7] = m2/M1
            N[k,8] = N[k,6] * (1 / (N[k,0] + N[k,1])) + ((N[k,0]*N[k,1])/(N[k,0] + N[k,1])) * N[k,7]
            
            B[k] = 2.0 * numpy.sqrt(a) / (1 + a) 
            
            I[k] = 1.0/(2*(1+( curve[k]) )) 
            
            U[k] = 1.0/(2*(1+( B[k]))) 
     
        else:
            curve[k] = 1
            T[k]     = 1
            Ts[k]    = 1
            B[k]     = 0

        elapsed0 = time.time() - start
        
        elapsed[k] = elapsed0
        

def _batch_frc_(params):

    V = params[0]
    t = params[1]
    b = int( numpy.ceil(V/t) ) 

    elapsed = _create_np_shared_array([V,], numpy.float32, ctypes.c_float)
    
    processes = []
    for k in range(t):
        begin_ = k*b
        end_   = min( (k+1)*b, V)
        p = multiprocessing.Process(target=_worker_batch_frc_, args=(params, begin_, end_, elapsed))
        processes.append(p)
    
    for p in processes:
        p.start()

    for p in processes:
        p.join()
    
    return elapsed


####

def _frc_parallel_( f1, f2, threads, delta ):

    alpha = f1.shape[0]/2
    
    start = time.time()

    F1 = numpy.fft.fftshift( numpy.fft.fft2(f1) )
    F2 = numpy.fft.fftshift( numpy.fft.fft2(f2) )

    elapsed = time.time() - start

    dx  = 2.0/(f1.shape[1]-1)
    dwx = 1.0/(2*dx)
    dy  = 2.0/(f1.shape[0]-1)
    dwy = 1.0/(2*dy)

    ty = numpy.linspace(-dwy,dwy,f1.shape[0])
    tx = numpy.linspace(-dwx,dwx,f1.shape[1])
    wx,wy = numpy.meshgrid(tx,ty)
    wy = numpy.flipud(wy)

    r = numpy.linspace(0, min(dwx,dwy), f1.shape[1]//2)
    rc = alpha * dx

    #--- shared arrays ----
    
    name1 = str( uuid.uuid4() )[0:8]    
    try:
        sa.delete(name1)
    except:
        pass;
    curve = sa.create(name1,r.shape, dtype=numpy.float32)

    name2 = str( uuid.uuid4() )[0:8]
    try:
        sa.delete(name2)
    except:
        pass;
    T = sa.create(name2,r.shape, dtype=numpy.float32)

    name3 = str( uuid.uuid4() )[0:8]
    try:
        sa.delete(name3)
    except:
        pass;
    Ts = sa.create(name3,r.shape, dtype=numpy.float32)

    name4 = str( uuid.uuid4() )[0:8]
    try:
        sa.delete(name4)
    except:
        pass;
    B = sa.create(name4,r.shape, dtype=numpy.float32)

    name5 = str( uuid.uuid4() )[0:8]
    try:
        sa.delete(name5)
    except:
        pass;
    I = sa.create(name5,r.shape, dtype=numpy.float32)

    name6 = str( uuid.uuid4() )[0:8]
    try:
        sa.delete(name6)
    except:
        pass;
    U = sa.create(name6,r.shape, dtype=numpy.float32)

    name7 = str( uuid.uuid4() )[0:8]
    try:
        sa.delete(name7)
    except:
        pass;
    A = sa.create(name7,[len(r), 9], dtype=numpy.float32)

    #-----------------------

    # number of rings
    nr = f1.shape[1]//2
    
    params = (nr, threads, F1, F2, wx, wy, r, rc, curve, T, Ts, B, I, U, A, delta)
    
    etimes = _batch_frc_(params)
    
    elapsed = time.time() - start

    sa.delete(name1)
    sa.delete(name2)
    sa.delete(name3)
    sa.delete(name4)
    sa.delete(name5)
    sa.delete(name6)
    sa.delete(name7)

    consistency_T = True
    consistency_Ts= True
    consistency_B = True
    
    for k in range(len(curve)):
        if curve[k] < B[k]:
            consistency_B = False
            
        if curve[k] < T[k]:
            consistency_T = False

        if curve[k] < Ts[k]:
            consistency_Ts = False   
    
    curves = {}
    curves['C'] = curve
    curves['H'] = T
    curves['S'] = Ts
    curves['B'] = B
    curves['I'] = I
    curves['U'] = U
    curves['A'] = A
        
    curves['cons'] = {'H':consistency_T, 'S': consistency_Ts, 'B': consistency_B}
    
    return curves


############

def _worker_batch_frcMap_(params, idx_start,idx_end):

    #check = lambda x : 0 if x > 1 else ( 0 if x < 0 else x )
    check = lambda x: x
    
    image    = params[0]
    N        = params[2]
    output   = params[3]
    
    L     = params[4]
    ngrid = params[5]
    eps   = params[6]
    flag  = params[7]
    bound = params[8]
    power = params[9]
    
    n      = image.shape[0]
    delta  = float( n/ngrid )
    rad    = int( numpy.ceil( delta/2.0 ) )

    for k in range(idx_start, idx_end):

        row = k // ngrid
        col = k % ngrid

        y = int( delta * (0.5 + col) )
        x = int( delta * (0.5 + row) )
        
        start    = time.time()
        
        if ( x - eps > 0)  and (x + eps < n) and (y - eps > 0) and (y + eps < n):
            
            dic, _ = computeLocal_python( image, x, y, 0, eps, L, power)
            
            if dic is None:

                output[ max(0,y - rad): min( y + rad+1,n) , max(0,x-rad): min(x+rad+1,n)] = 0 
                
            else:
                # halfbit
                if bound == "halfbit":
                    output[ max(0,y - rad): min( y + rad+1,n) , max(0,x-rad): min(x+rad+1,n)] = check( dic['x']['H'])
                
                # sigma
                if bound == "sigma":
                    output[ max(0,y - rad): min( y + rad+1,n) , max(0,x-rad): min(x+rad+1,n)] = check( dic['x']['S'])
                
                # reverse
                if bound == "fisher":
                    output[ max(0,y - rad): min( y + rad+1,n) , max(0,x-rad): min(x+rad+1,n)] = check( dic['x']['B'])
            
        elapsed  = time.time() - start


def _worker_serial_frcMap_( params ):

    #check = lambda x : 0 if x > 1 else ( 0 if x < 0 else x )
    check = lambda x: x
    
    image    = params[0]
    N        = params[1]
    L        = params[2]
    ngrid    = params[3]
    eps      = params[4]
    flag     = params[5]
    bound    = params[6]
    power    = params[7]
    
    n      = image.shape[0]
    delta  = float( n/ngrid )
    rad    = int( numpy.ceil( delta/2.0 ) )
    
    output = numpy.zeros( image.shape, dtype=numpy.float32)
    
    for k in range( ngrid * ngrid ):

        row = k // ngrid
        col = k % ngrid

        y = int( delta * (0.5 + col) )
        x = int( delta * (0.5 + row) )
        
        start    = time.time()
        
        if ( x - eps > 0)  and (x + eps < n) and (y - eps > 0) and (y + eps < n):
            
            dic, _ = computeLocal_python( image, x, y, 0, eps, L, power)
            
            if dic is None:
                
                output[ max(0,y - rad): min( y + rad+1,n) , max(0,x-rad): min(x+rad+1,n)] = 0 
                
            else:
                # halfbit
                if bound == "halfbit":
                    output[ max(0,y - rad): min( y + rad+1,n) , max(0,x-rad): min(x+rad+1,n)] = check( dic['x']['H'])
                
                # sigma
                if bound == "sigma":
                    output[ max(0,y - rad): min( y + rad+1,n) , max(0,x-rad): min(x+rad+1,n)] = check( dic['x']['S'])
                
                # reverse
                if bound == "fisher":
                    output[ max(0,y - rad): min( y + rad+1,n) , max(0,x-rad): min(x+rad+1,n)] = check( dic['x']['B'])

                    
        elapsed  = time.time() - start

    return output
        
        
def _frcMap_batch_(params):

    N = params[2]
    t = params[1]
    b = int( numpy.ceil(N/t) ) 
    
    processes = []
    for k in range(t):
        begin_ = k*b
        end_   = min( (k+1)*b, N)
        p = mp.Process(target=_worker_batch_frcMap_, args=(params, begin_, end_))
        processes.append(p)
    
    for p in processes:
        p.start()

    for p in processes:
        p.join()


def _frcMap_serial_(params):
   
    return _worker_serial_frcMap_ ( params,  )
        
        
def map(image, dic):

    if 'running' not in dic.keys():
        running = SERIAL
    else:
        running = dic['running']

    if 'threshold' not in dic.keys():
        bound = 'halfbit'
    else:
        bound = dic['threshold']

    if 'delta' not in dic.keys():
        delta = 1e-5
    else:
        delta= dic['delta']

        
    gridsize = dic['ngrid']
    eps      = dic['eps']
    L        = dic['L']
        
    #
    #

    bextension = 1
    
    if running == SERIAL:

        start = time.time()

        # 4 times extension on borders
        _eps_ = bextension * eps
        
        imgpadded = numpy.pad( image, (_eps_, _eps_ ) )
        
        N = gridsize * gridsize 
        
        params = (imgpadded, N, L, gridsize, eps, None, bound, delta )
        
        output = _frcMap_serial_(params)
        
        elapsed = time.time() - start
                
        print('ssc-resolution: 2D resolution map processed within {} sec'.format(elapsed))
        
        output = output[_eps_:_eps_+image.shape[0],_eps_:_eps_+image.shape[1]]
       
    else:
        
        if 'nproc' not in dic.keys():
            threads = nthreads
        else:
            threads = int( dic['nproc'] )

        _eps_ = bextension * eps

        imgpadded = numpy.pad(image, (_eps_, _eps_))
        
        name = str( uuid.uuid4() )[0:8]
        
        try:
            sa.delete(name)
        except:
            pass;
        
        output = sa.create(name, imgpadded.shape, dtype=numpy.float32)
        
        start = time.time()

        N = gridsize * gridsize 
        
        params = (imgpadded, threads, N, output, L, gridsize, eps, None, bound, delta)
        
        _frcMap_batch_(params)
        
        elapsed = time.time() - start
        
        sa.delete(name)
        
        print('ssc-resolution: 2D resolution map processed within {} sec'.format(elapsed))
        
        output = output[_eps_:_eps_+image.shape[0],_eps_:_eps_+image.shape[1]]

    
    return output

