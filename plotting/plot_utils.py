import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle

default_colors = {
    'QCD': 'midnightblue',
    'SUEP-m1000-darkPho': 'red',
    'SUEP-m1000-darkPhoHad': 'red',
    'SUEP-m1000-generic': 'red',
    'SUEP-m750-darkPho': 'orange',
    'SUEP-m750-darkPhoHad': 'orange',
    'SUEP-m750-generic': 'orange',
    'SUEP-m400-darkPho': 'green',
    'SUEP-m400-darkPhoHad': 'green',
    'SUEP-m400-generic': 'green',
    'SUEP-m125-darkPho': 'cyan',
    'SUEP-m125-darkPhoHad': 'cyan',
    'SUEP-m125-generic': 'cyan',
        
}

lumis = {
    '2016_apv': 19.5*1000,
    '2016': 16.8*1000,
    '2017': 41.5*1000,
    '2018': 61000
}

# function to load files from pickle
def openpkl(infile_name):
    plots={}
    with (open(infile_name, "rb")) as openfile:
        while True:
            try:
                plots.update(pickle.load(openfile))
            except EOFError:
                break
    return plots

def plot1d(h, ax, label, rebin=-1, color='default', lw=1):
    
    if color == 'default': color = default_colors[label]
    if label == 'QCD' and lw==1: lw=3
    
    y, x = h.to_numpy()
    e = np.sqrt(h.variances())
    x = x[:-1]
    if rebin!=-1: x, y, e = combine_bins(x, y, e, n=rebin)
    
    #ax.step(x[:-1],values, label=label, color=color, lw=lw)
    ax.errorbar(x, y, yerr=e, label=label, lw=lw, color=color, fmt="", drawstyle='steps-mid')
    ax.set_xlabel(h.axes[0].label)
    ax.set_ylabel("Events")
    
def plot1d_stacked(hlist, ax, labels, rebin=-1, color='midnightblue', lw=1):
    
    cmap = plt.cm.rainbow(np.linspace(0, 1, len(labels)))

    ylist, elist = [], []
    for l, h, c in zip(labels, hlist, cmap): 
        y, x = h.to_numpy()
        e = np.sqrt(h.variances())
        x = x[:-1]
        if rebin!=-1: x, y, e = combine_bins(x, y, e, n=rebin)

        if len(ylist) > 0: y = y + ylist[len(ylist)-1]
        ylist.append(y)
    
        #ax.step(x[:-1],values, label=label, color=color, lw=lw)
        ax.errorbar(x, y, yerr=e, label=l, lw=lw, color=c, fmt="", drawstyle='steps-mid')
    ax.set_xlabel(hlist[0].axes[0].label)
    ax.set_ylabel("Events")
    
def plot2d(h, ax, log=False, cmap='RdYlBu'):
    w, x, y = h.to_numpy()
    if log: mesh = ax.pcolormesh(x, y, w.T, cmap=cmap, norm=matplotlib.colors.LogNorm())
    else: mesh = ax.pcolormesh(x, y, w.T, cmap=cmap)
    ax.set_xlabel(h.axes[0].label)
    ax.set_ylabel(h.axes[1].label)
    fig.colorbar(mesh)
    
# def plot_ratio(h1, h2, 
#                plot_label, label1, label2, 
#                rebin=-1, 
#                lumi1=1, lumi2=1, 
#                xlim='default', 
#                log=True):

#     #Set up variables for the stacked histogram
#     plt.figure(figsize=(12,10))
#     plt.gcf().subplots_adjust(bottom=0.15, left=0.17)
#     ax1 = plt.subplot2grid((4,1), (0,0),rowspan=2)

#     y1, x1 = h1.to_numpy()
#     y1 = y1*lumi1
#     x1 = x1[:-1]
#     y1_errs = np.sqrt(h1.variances())*lumi1
#     if rebin!=-1: x1, y1, y1_errs = combine_bins(x1, y1, y1_errs, n=rebin)
#     ax1.step(x1, y1, color='maroon',label=label1, where='mid')
#     ax1.errorbar(x1, y1, yerr=y1_errs, color="maroon".upper(), fmt="", drawstyle='steps-mid')

#     y2, x2 = h2.to_numpy()
#     y2 = y2*lumi2
#     x2 = x2[:-1]
#     y2_errs = np.sqrt(h2.variances())*lumi2
#     if rebin!=-1: x2, y2, y2_errs = combine_bins(x2, y2, y2_errs, n=rebin)
#     ax1.step(x2, y2, color='blue',label=label2, where= 'mid')
#     ax1.errorbar(x2, y2, yerr=y2_errs, color="blue".upper(), fmt="", drawstyle='steps-mid')
    
#     #Set parameters that will be used to make the plots prettier
#     if log: ax1.set_yscale("log")
#     ymax = max([max(y1), max(y2)])*1.5
#     ymin = min([min(y1), min(y2)])*0.5
#     ax1.set_ylim([ymin, ymax])
#     if type(xlim) is not str:
#         xmin = xlim[0]
#         xmax = xlim[1]
#         ax1.set_xlim([xmin,xmax])
#     else:
#         xmin1 = min(x1[y1>0]) if len(x1[y1>0]) else 0
#         xmin2 = min(x2[y2>0]) if len(x2[y2>0]) else 0
#         xmax1 = max(x1[y1>0]) if len(x1[y1>0]) else 0
#         xmax2 = max(x2[y2>0]) if len(x2[y2>0]) else 0
#         xmin = max([xmin1, xmin2])
#         xmax = max([xmax1, xmax2])
#         x_range = xmax - xmin
#         ax1.set_xlim([xmin - x_range*0.25, xmax + x_range*0.25])
 
#     ax1.set_ylabel("Events", y=1, ha='right')
#     ax1.legend()

#     ax2 = plt.subplot2grid((4,1), (2,0), sharex=ax1)
#     plt.setp(ax1.get_xticklabels(), visible=False)
    
#     # calculate the upper and lower errors
#     # suppress errors where the denonminator is 0
#     y1 = np.where(y1>0, y1, -1)
#     yerrors_up = np.where(y1>0, y2/y1 - (y2-y2_errs)/(y1+y1_errs), np.nan)
#     yerrors_low = np.where(y1>0, (y2+y2_errs)/(y1-y1_errs) - y2/y1, np.nan)
#     yerrors = [yerrors_up, yerrors_low]

#     ax2.errorbar(x1,np.where((y2>0) & (y1>0),y2/y1,1),yerr=yerrors, color="black", fmt="", drawstyle='steps-mid')
#     ax2.axhline(1, ls="--", color='gray')
#     ax2.set_ylim(0.4,1.6)
#     ax2.set_ylabel("Ratio", y=1, ha='right')
#     ax2.set_xlabel(plot_label, y=1)
    
#     return ax1, ax2

def plot_ratio(h1, h2, 
               plot_label, label1, label2, 
               rebin=-1, 
               lumi1=1, lumi2=1, 
               xlim='default', 
               log=True):

    #Set up variables for the stacked histogram
    # plt.figure(figsize=(12,10))
    # plt.gcf().subplots_adjust(bottom=0.15, left=0.17)
    # ax1 = plt.subplot2grid((4,1), (0,0),rowspan=2)
    fig = plt.figure(figsize=(12,10))
    plt.subplots_adjust(bottom=0.15, left=0.17)
    ax1 = plt.subplot2grid((4,1), (0,0),rowspan=2)

    y1, x1 = h1.to_numpy()
    y1 = y1*lumi1
    x1 = x1[:-1]
    y1_errs = np.sqrt(h1.variances())*lumi1
    if rebin!=-1: x1, y1, y1_errs = combine_bins(x1, y1, y1_errs, n=rebin)
    ax1.step(x1, y1, color='maroon',label=label1, where='mid')
    ax1.errorbar(x1, y1, yerr=y1_errs, color="maroon".upper(), fmt="", drawstyle='steps-mid')

    y2, x2 = h2.to_numpy()
    y2 = y2*lumi2
    x2 = x2[:-1]
    y2_errs = np.sqrt(h2.variances())*lumi2
    if rebin!=-1: x2, y2, y2_errs = combine_bins(x2, y2, y2_errs, n=rebin)
    ax1.step(x2, y2, color='blue',label=label2, where= 'mid')
    ax1.errorbar(x2, y2, yerr=y2_errs, color="blue".upper(), fmt="", drawstyle='steps-mid')
    
    #Set parameters that will be used to make the plots prettier
    if log: ax1.set_yscale("log")
    ymax = max([max(y1), max(y2)])*1.5
    ymin = min([min(y1), min(y2)])*0.5
    ax1.set_ylim([ymin, ymax])
    if type(xlim) is not str:
        xmin = xlim[0]
        xmax = xlim[1]
        ax1.set_xlim([xmin,xmax])
    else:
        xmin1 = min(x1[y1>0]) if len(x1[y1>0]) else 0
        xmin2 = min(x2[y2>0]) if len(x2[y2>0]) else 0
        xmax1 = max(x1[y1>0]) if len(x1[y1>0]) else 0
        xmax2 = max(x2[y2>0]) if len(x2[y2>0]) else 0
        xmin = max([xmin1, xmin2])
        xmax = max([xmax1, xmax2])
        x_range = xmax - xmin
        ax1.set_xlim([xmin - x_range*0.25, xmax + x_range*0.25])
 
    ax1.set_ylabel("Events", y=1, ha='right')
    ax1.legend()

    # ax2 = fig.subplot2grid((4,1), (2,0), sharex=ax1)
    # fig.setp(ax1.get_xticklabels(), visible=False)
    ax2 = plt.subplot2grid((4,1), (2,0), sharex=ax1)
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    # calculate the upper and lower errors
    # suppress errors where the denonminator is 0
    y1 = np.where(y1>0, y1, -1)
    yerrors_up = np.where(y1>0, y2/y1 - (y2-y2_errs)/(y1+y1_errs), np.nan)
    yerrors_low = np.where(y1>0, (y2+y2_errs)/(y1-y1_errs) - y2/y1, np.nan)
    ratio_errs = [yerrors_up, yerrors_low]
    ratios = np.where((y2>0) & (y1>0), y2/y1, 1)

    ax2.errorbar(x1, ratios, yerr=ratio_errs, color="black", fmt="", drawstyle='steps-mid')
    ax2.axhline(1, ls="--", color='gray')
    ax2.set_ylim(0.4,1.6)
    ax2.set_ylabel("Ratio", y=1, ha='right')
    ax2.set_xlabel(plot_label, y=1)
    
    residuals = ratios - 1
    residuals_errs = ratio_errs
    
    return fig, (ax1, ax2), (residuals, residuals_errs)
    
def integrate(h, lower, upper):
    i = h[lower:upper].sum()
    return i.value, np.sqrt(i.variance)
    
def find_nth(string, substring, n):
    if (n == 1):
        return string.find(substring)
    else:
        return string.find(substring, find_nth(string, substring, n - 1) + 1)
    
def combine_bins(x, y, e, n=4):
    old_e = e
    old_y = y
    old_x = x
    
    y = []
    x = []
    e = []
   
    for j in list(range(0,len(old_y), n)):
        
        y.append(np.sum(old_y[j:j+n]))
        x.append(np.sum(old_x[j:j+n])/n)
        e.append(np.sqrt(np.sum(old_e[j:j+n]**2)))
        
    return np.array(x), np.array(y), np.array(e)

def nested_dict(n, type):
    if n == 1:
        return defaultdict(type)
    else:
        return defaultdict(lambda: nested_dict(n-1, type))
    
def D_expect(plots, selection):
    sizeC = plots['C_' + selection].sum()
    sizeA = plots['A_' + selection].sum()
    Bhist = plots['B_' + selection]
    if sizeA.value > 0:
        D_exp = Bhist * sizeC.value / sizeA.value
    else: #Cannot properly predict D without any events in A
        D_exp = Bhist * 0.0
    return D_exp

def D_expect_var2(plots, selection):
    sizeB = plots['B_var2_' + selection].sum()
    sizeA = plots['A_var2_' + selection].sum()
    Chist = plots['C_var2_' + selection]
    if sizeA.value > 0:
        D_exp = Chist * sizeB.value / sizeA.value
    else: #Cannot properly predict D without any events in A
        D_exp = Chist * 0.0
    return D_exp