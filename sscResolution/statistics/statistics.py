import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy
import time

from ..resolutiontypes import *

def interpolated_intercept(x, y1, y2):

    #
    # find point where y1 >= y2
    #
    
    k = numpy.argmin( y1 >= y2 )

    Dx  =  1.0 / (len(y1)-1)
    x   =  k * Dx
    Dy  = ( y1[k] - y2[k ] ) - ( y1[k-1] - y2[k-1] )
    x   = x - Dx * ( y1[k]  - y2[k] ) / Dy
    
    return x, None, k


def histmap( rmap , dic, usr ):

    if len( dic['shape']) == 3:
        N    = dic['shape'][0]
        nvxl = N * N * N
        label = 'FSC/Map: ' + usr['label']
    else: 
        N    = dic['shape'][0]
        nvxl = N * N
        label = 'FRC/Map: ' + usr['label']

    print(label)
        
    eps = dic['eps']          
    color = 'r'

    #-----------

    if 'figsize' in usr.keys():
        figsize = usr['figsize']
    else:
        figsize = (10,10)

    if 'fontsize' in usr.keys():
        fontsize = usr['fontsize']
    else:
        fontsize = [20,20]

    #
        
    binmin = []
    binmax = []
    H = []
    histo = []
    bine = []
    
    filtered = rmap[ rmap > 0].flatten()
    bins = numpy.linspace(filtered.min(),filtered.max(), 50, endpoint=True)
    hist, bin_edges = numpy.histogram(filtered, bins=bins)
    bin_edges = bin_edges[:-1]
        
    H = interp1d( bin_edges, hist)
    histo.append( hist )
    bine.append( bin_edges )
    
    binmin.append(bin_edges.min())
    binmax.append(bin_edges.max())
    
    baxis = numpy.linspace(max(binmin), min(binmax),400)

    plt.figure(figsize=figsize)
    V =  H(baxis)/(nvxl)
    plt.plot(baxis,V,'-{}'.format(color),linewidth=5)    
    
    plt.xticks(numpy.round(numpy.linspace(baxis.min(), baxis.max(), 10),1), fontsize=fontsize[1])
    plt.yticks(numpy.round(numpy.linspace(V.min(), V.max(), 10),1), fontsize=fontsize[1])
    plt.legend( [label] , prop={'size': fontsize[0]})
    
