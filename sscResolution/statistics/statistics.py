import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy
import time

from ..resolutiontypes import *

def interpolated_intercept(x, y1, y2):
   
    def _intercept(point1, point2, point3, point4):
        def _line(p1, p2):
            A = (p1[1] - p2[1])
            B = (p2[0] - p1[0])
            C = (p1[0] * p2[1] - p2[0] * p1[1])
            return A, B, -C

        def _intersection(L1, L2):
            D = L1[0] * L2[1] - L1[1] * L2[0]
            Dx = L1[2] * L2[1] - L1[1] * L2[2]
            Dy = L1[0] * L2[2] - L1[2] * L2[0]

            x = Dx / D
            y = Dy / D
            return x, y

        L1 = _line([point1[0], point1[1]], [point2[0], point2[1]])
        L2 = _line([point3[0], point3[1]], [point4[0], point4[1]])

        R = _intersection(L1, L2)

        return R

    found = -1
    for k in range(len(y1)):
        if y1[k] < y2[k] and found < 0 and k > 0:
            idx = k
            found = 1
   
    if found > 0 and idx < len(y1): 
        if idx == len(y1) - 1:
            xc = x[idx]
            yc = y1[idx]
        else:
            xc, yc = _intercept((x[idx], y1[idx]), ((x[idx + 1], y1[idx + 1])),
                        ((x[idx], y2[idx])), ((x[idx + 1], y2[idx + 1])))
    else:
        idx = -1
        xc = 0
        yc = 0

    return xc, yc, idx


def interpolated_intercept_2(x, y1, y2):
   
    def _intercept(point1, point2, point3, point4):
        def _line(p1, p2):
            A = (p1[1] - p2[1])
            B = (p2[0] - p1[0])
            C = (p1[0] * p2[1] - p2[0] * p1[1])
            return A, B, -C

        def _intersection(L1, L2):
            D = L1[0] * L2[1] - L1[1] * L2[0]
            Dx = L1[2] * L2[1] - L1[1] * L2[2]
            Dy = L1[0] * L2[2] - L1[2] * L2[0]

            x = Dx / D
            y = Dy / D
            return x, y

        L1 = _line([point1[0], point1[1]], [point2[0], point2[1]])
        L2 = _line([point3[0], point3[1]], [point4[0], point4[1]])

        R = _intersection(L1, L2)

        return R

    #'''
    idx = numpy.argwhere(numpy.diff(numpy.sign(y1 - y2)) != 0)
    
    idx = numpy.argwhere(numpy.diff(numpy.sign(y1 - y2)) != 0)
    xc, yc = _intercept((x[idx], y1[idx]), ((x[idx + 1], y1[idx + 1])),
                        ((x[idx], y2[idx])), ((x[idx + 1], y2[idx + 1])))


    xc = xc.flatten()
    yc = yc.flatten()

    if idx.size > 0:

        if len(xc) > 1:
            xc = xc[1]
            yc = yc[1]
        else:
            xc = xc[0]
            yc = yc[0]

        if xc == 0:
            idx = -1
            xc = 0
            yc = 0

    else:
        idx = -1
        xc = 0
        yc = 0
    #'''
    
    '''
    found = -1
    for k in range(len(y1)):
        if abs(y1[k] - y2[k]) < 1e-1 and found < 0 and k > 0:
            idx = k
            found = 1
   
    if found > 0:    
        xc, yc = _intercept((x[idx], y1[idx]), ((x[idx + 1], y1[idx + 1])),
                        ((x[idx], y2[idx])), ((x[idx + 1], y2[idx + 1])))
    else:
        xc = -1 
        yc = -1
    '''

    return xc, yc,idx


def histmap( rmap , dic, usr ):

    if len( dic['shape']) == 3:

        N    = dic['shape'][0]
        nvxl = N * N * N
        label = 'FSC/Map: ' + usr['label']
        
    elif len( dic['shape']) == 2:

        N    = dic['shape'][0]
        nvxl = N * N
        label = 'FRC/Map: ' + usr['label']

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
    plt.legend( label , prop={'size': fontsize[0]})
    
