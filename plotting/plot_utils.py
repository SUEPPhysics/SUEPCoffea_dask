import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import boost_histogram as bh
import mplhep as hep
import pandas as pd
import logging
import shutil
from sympy import symbols, diff, sqrt
import sympy
import json
import hist
from copy import deepcopy

default_colors = {
    'QCD': 'midnightblue',
    'QCD_HT': 'midnightblue',
    'QCD_HT_2018': 'midnightblue',
    'QCD_HT_2017': 'midnightblue',
    'QCD_HT_2016': 'midnightblue',
    'QCD_HT_allyears': 'midnightblue',
    'data': 'maroon',
    'data': 'maroon',
    'data_2018': 'maroon',
    'data_2017': 'maroon',
    'data_2016': 'maroon',
    'data_allyears': 'maroon',
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
    'SUEP-m125-generic-htcut' : 'magenta'
}

lumis = {
    '2016_apv': 19.5*1000,
    '2016': 16.8*1000,
    '2017': 41.5*1000,
    '2018': 61000
}
    
# load file(s)
def loader(infile_names, 
           apply_lumis=True,
           exclude_low_bins=False):
    plots = {}
    for infile_name in infile_names:
        if not os.path.isfile(infile_name): 
            print("WARNING:",infile_name,"doesn't exist")
            continue
        elif ".pkl" not in infile_name: continue
        elif "QCD_Pt_3200toInf_TuneCP5_13TeV_pythia8+RunIISummer20UL18MiniAODv2-pilot" in infile_name: continue

        # sets the lumi based on year
        lumi = 1
        if apply_lumis:
            if ('20UL16MiniAODv2' in infile_name):
                lumi = lumis['2016']
            if ('20UL17MiniAODv2' in infile_name):
                lumi = lumis['2017']
            if ('20UL16MiniAODAPVv2' in infile_name):
                lumi = lumis['2016_apv']
            if ('20UL18' in infile_name):
                lumi = lumis['2018']
            if 'SUEP-m' in infile_name:
                lumi = lumis['2018']
            if 'JetHT+Run' in infile_name:
                lumi = 1
                
        # exclude low bins
        if exclude_low_bins:
            if '50to100' in infile_name: continue
            if '100to200' in infile_name: continue
            # if '200to300' in infile_name: continue
            # if '300to500' in infile_name: continue
            # if '500to700' in infile_name: continue
            # if '700to1000' in infile_name: continue

            # if '15to30' in infile_name: continue
            # if '30to50' in infile_name: continue
            # if '50to80' in infile_name: continue
            # if '80to120' in infile_name: continue
            # if '120to170' in infile_name: continue
            # if '170to300' in infile_name: continue

        # plots[sample] sample is filled here
        if 'QCD_Pt' in infile_name:
            sample = 'QCD_Pt'

            # include this block to import the QCD bins individually
            temp_sample = infile_name.split('/')[-1].split('.pkl')[0]
            plots[temp_sample] = openpkl(infile_name)
            for plot in list(plots[temp_sample].keys()):
                plots[temp_sample][plot] = plots[temp_sample][plot]*lumi

        elif 'QCD_HT' in infile_name:
            sample = 'QCD_HT'

            # include this block to import the QCD bins individually
            temp_sample = infile_name.split('/')[-1].split('.pkl')[0]    
            temp_sample =  temp_sample.split('QCD_HT')[1].split('_Tune')[0]
            plots[temp_sample] = openpkl(infile_name)
            for plot in list(plots[temp_sample].keys()):
                plots[temp_sample][plot] = plots[temp_sample][plot]*lumi

        elif 'JetHT+Run' in infile_name or 'ScoutingPFHT' in infile_name: 
            sample = 'data'

        elif 'SUEP-m' in infile_name:
            sample = infile_name.split('/')[-1].split('+')[0]
        else:
            sample = infile_name
            
        if sample not in list(plots.keys()): 
            plots[sample] = openpkl(infile_name)
            for plot in list(plots[sample].keys()):
                plots[sample][plot] = plots[sample][plot]*lumi
        else:
            plotsToAdd = openpkl(infile_name) 
            for plot in list(plotsToAdd.keys()):
                plots[sample][plot]  = plots[sample][plot] + plotsToAdd[plot]*lumi
            
    return plots

def combineYears(inplots, tag='QCD_HT', years=['2018','2017','2016']):
    """
    Combines all samples in plots with a certain tag and with certain
    years. Returns combined plots.
    """
    outPlots = {}
    yearsAdded = []
    initialize=True
    for sample in inplots.keys():
        if tag not in sample: continue
        if not any([y in sample for y in years]): continue
        
        # keep track of which years we've added already
        for year in years:
            if year in sample: 
                if year in yearsAdded: raise Exception("Already loaded this year: "+year)
                yearsAdded.append(year)
                
        # combine samples
        if initialize:
            outPlots = inplots[sample].copy()
        else:            
            for plot in list(inplots[sample].keys()):
                outPlots[plot] = outPlots[plot] + inplots[sample][plot].copy()
                    
        initialize = False
    
    return outPlots
 
def check_proxy(time_min=100):
    """
    Checks for existance of proxy with at least time_min
    left on it.
    If it's inactive or below time_min, it will regenerate
    it with 140 hours.
    """
    home_base  = os.environ['HOME']
    proxy_base = 'x509up_u{}'.format(os.getuid())
    proxy_copy = os.path.join(home_base,proxy_base)
    regenerate_proxy = False
    if not os.path.isfile(proxy_copy):
        logging.warning('--- proxy file does not exist')
        regenerate_proxy = True
    else:
        lifetime = subprocess.check_output(
            ['voms-proxy-info', '--file', proxy_copy, '--timeleft']
        )
        lifetime = float(lifetime)
        lifetime = lifetime / (60*60)
        if lifetime < time_min:
            logging.warning("--- proxy has expired !")
            regenerate_proxy = True

    if regenerate_proxy:
        redone_proxy = False
        while not redone_proxy:
            status = os.system('voms-proxy-init -voms cms --hours=140')
            lifetime = 140
            if os.WEXITSTATUS(status) == 0:
                redone_proxy = True
        shutil.copyfile('/tmp/'+proxy_base,  proxy_copy)
        
    return lifetime

def apply_binwise_scaling(h_in, bins, scales, dim='x'):
    """
    Apply scales to bins of a particular histogram.
    """
    h_npy = h_in.to_numpy()
    if len(h_npy) == 2:
        h_fragments = []
        for iBin in range(len(bins)-1): h_fragments.append(h_in[bins[iBin]:bins[iBin+1]] * scales[iBin])
        h = hist.Hist(hist.axis.Variable(h_npy[1]), storage=bh.storage.Weight())
        new_z = np.concatenate([f.to_numpy()[0] for f in h_fragments])
        h[:] = np.stack([new_z, new_z], axis=-1)
    elif len(h_npy) == 3:
        h_fragments = []
        for iBin in range(len(bins)-1):
            if dim == 'x': h_fragments.append(h_in[bins[iBin]:bins[iBin+1], :] * scales[iBin])
            elif dim == 'y': h_fragments.append(h_in[:, bins[iBin]:bins[iBin+1]] * scales[iBin])
        h = hist.Hist(hist.axis.Variable(h_npy[1]), hist.axis.Variable(h_npy[2]), storage=bh.storage.Weight())
        if dim == 'x': new_z = np.concatenate([f.to_numpy()[0] for f in h_fragments])
        if dim == 'y': new_z = np.hstack([f.to_numpy()[0] for f in h_fragments])
        h[:,:] = np.stack([new_z, new_z], axis=-1)
    return h

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

def plot1d(h, ax, label, color='default', lw=1):
    
    if color == 'default': color = default_colors[label]
    if label == 'QCD' and lw==1: lw=3
    
    y, x = h.to_numpy()
    e = np.sqrt(h.variances())
    x = x[:-1]
    
    #ax.step(x[:-1],values, label=label, color=color, lw=lw)
    ax.errorbar(x, y, yerr=e, label=label, lw=lw, color=color, fmt="", drawstyle='steps-mid')
    ax.set_xlabel(h.axes[0].label)
    ax.set_ylabel("Events")
    
def plot1d_stacked(hlist, ax, labels, color='midnightblue', lw=1):
    
    cmap = plt.cm.rainbow(np.linspace(0, 1, len(labels)))

    ylist, elist = [], []
    for l, h, c in zip(labels, hlist, cmap): 
        y, x = h.to_numpy()
        e = np.sqrt(h.variances())
        x = x[:-1]

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

def bin_midpoints(bins):
    midpoints = []
    for i in range(len(bins)-1):
        midpoints.append((bins[i]+bins[i+1])/2)
    return np.array(midpoints)

def plot_ratio(h1, h2, 
               plot_label=None, 
               label1=None, label2=None, 
               xlim='default', 
               log=True):

    #Set up variables for the stacked histogram
    fig = plt.figure(figsize=(12,10))
    plt.subplots_adjust(bottom=0.15, left=0.17)
    ax1 = plt.subplot2grid((4,1), (0,0),rowspan=2)

    y1, x1 = h1.to_numpy()
    y1_errs = np.sqrt(h1.variances())
    ax1.stairs(y1, x1, color='maroon',label=label1)
    x1_mid = bin_midpoints(x1)
    ax1.errorbar(x1_mid, y1, yerr=y1_errs, color="maroon".upper(), fmt="", drawstyle='default', linestyle='')

    y2, x2 = h2.to_numpy()
    y2_errs = np.sqrt(h2.variances())
    ax1.stairs(y2, x2, color='blue',label=label2)
    x2_mid = bin_midpoints(x2)
    ax1.errorbar(x2_mid, y2, yerr=y2_errs, color="blue".upper(), fmt="", drawstyle='default', linestyle='')
    
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
        xmin1 = np.min([i for i, x in enumerate((y1>0)) if x]) if len(x1_mid[y1>0]) else x1[0]
        xmin2 = np.min([i for i, x in enumerate((y1>0)) if x]) if len(x2_mid[y2>0]) else x2[0]
        xmax1 = np.max([i for i, x in enumerate((y1>0)) if x]) if len(x1_mid[y1>0]) else x1[-1]
        xmax2 = np.max([i for i, x in enumerate((y2>0)) if x]) if len(x2_mid[y2>0]) else x2[-1]
        xmin = min([x1[xmin1], x2[xmin2]])
        xmax = max([x1[xmax1+1], x2[xmax2+1]])
        x_range = xmax - xmin
        ax1.set_xlim([xmin, xmax])
        
    ax1.set_ylabel("Events", y=1, ha='right')

    ax2 = plt.subplot2grid((4,1), (2,0), sharex=ax1)
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    # calculate the upper and lower errors
    # suppress errors where the denonminator is 0
    y1 = np.where(y1>0, y1, -1)
    yerrors_up = np.where(y1>0, y2/y1 - (y2-y2_errs)/(y1+y1_errs), np.nan)
    yerrors_low = np.where(y1>0, (y2+y2_errs)/(y1-y1_errs) - y2/y1, np.nan)
    ratio_errs = [yerrors_up, yerrors_low]
    ratios = np.where((y2>0) & (y1>0), y2/y1, 1)

    ax2.errorbar(x1_mid, ratios, yerr=ratio_errs, color="black", fmt='o', linestyle='none')
    ax2.axhline(1, ls="--", color='gray')
    ax2.set_ylim(0.4,1.6)
    ax2.set_ylabel("Ratio", y=1, ha='right')
    
    if plot_label is None: plot_label = h1.axes[0].label
    ax2.set_xlabel(plot_label, y=1)
    
    residuals = ratios - 1
    residuals_errs = ratio_errs
    
    return fig, (ax1, ax2), (residuals, residuals_errs)

def plot_ratio_regions(plots, plot_label, 
               sample1, sample2, 
               regions,
               density=False):

    fig = plt.figure(figsize=(20,7))
    ax1 = plt.subplot2grid((4,1),(0,0), rowspan=2)
    ax2 = plt.subplot2grid((4,1),(2,0), sharex=ax1)
    _ = plt.setp(ax1.get_xticklabels(), visible=False)
    ax2 = plt.subplot2grid((4,1), (2,0), sharex=ax1)
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    offset = 0
    mids = []
    for i,r in enumerate(regions):
        h1 = plots[sample1][plot_label.replace("A_", r+"_")]
        h2 = plots[sample2][plot_label.replace("A_", r+"_")]
        
        if density:
            h1 = h1/h1.sum().value
            h2 = h2/h2.sum().value
        
        y1, x1 = h1.to_numpy()
        x1 = x1[:-1]
        y2, x2 = h2.to_numpy()
        x2 = x2[:-1]
                
        xmin1 = np.argwhere(y1>0)[0] if any(y1>0) else [len(x1)]
        xmin2 = np.argwhere(y2>0)[0] if any(y2>0) else [len(x1)]
        xmax1 = np.argwhere(y1>0)[-1] if any(y1>0) else [0]
        xmax2 = np.argwhere(y2>0)[-1] if any(y2>0) else [0]
        xmin = min(np.concatenate((xmin1, xmin2)))
        xmax = max(np.concatenate((xmax2, xmax2)))
        x1 = x1[xmin:xmax+1]
        x2 = x2[xmin:xmax+1]
        y1 = y1[xmin:xmax+1]
        y2 = y2[xmin:xmax+1]
        
        x1 = x1 - x1[0]
        x2 = x2 - x2[0]
        
        this_offset = x1[-1]-x1[0]
        x1 = x1 + offset
        x2 = x2 + offset
        offset += this_offset
        
        mids.append((x1[-1]+x1[0])/2)
        
        y1_errs = np.sqrt(h1.variances())
        y1_errs = y1_errs[xmin:xmax+1]
        if i == 0: ax1.step(x1, y1, color='midnightblue',label=sample1, where='mid')
        else: ax1.step(x1, y1, color='midnightblue', where='mid')
        ax1.errorbar(x1, y1, yerr=y1_errs, color="maroon".upper(), fmt="", drawstyle='steps-mid')

        y2_errs = np.sqrt(h2.variances())
        y2_errs = y2_errs[xmin:xmax+1]
        if i == 0: ax1.step(x2, y2, color='maroon',label=sample2, where= 'mid')
        else: ax1.step(x2, y2, color='maroon', where= 'mid')
        ax1.errorbar(x2, y2, yerr=y2_errs, color="blue".upper(), fmt="", drawstyle='steps-mid')
        
        ax1.axvline(x2[0], ls="--", color='black')
        ax2.axvline(x2[0], ls="--", color='black')
        
        # calculate the upper and lower errors
        # suppress errors where the denonminator is 0
        y1 = np.where(y1>0, y1, -1)
        yerrors_up = np.where(y1>0, y2/y1 - (y2-y2_errs)/(y1+y1_errs), np.nan)
        yerrors_low = np.where(y1>0, (y2+y2_errs)/(y1-y1_errs) - y2/y1, np.nan)
        ratio_errs = [yerrors_up, yerrors_low]
        ratios = np.where((y2>0) & (y1>0), y2/y1, 1)
        ax2.errorbar(x1, ratios, yerr=ratio_errs, color="black", fmt="", drawstyle='steps-mid')
    
    ax1.set_yscale("log")
 
    ax1.set_xticks(mids)
    ax1.set_xticklabels(list(regions))
    
    ax1.set_ylabel("Events", y=1, ha='right')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    
    ax2.axhline(1, ls="--", color='gray')
    ax2.set_ylim(0.4,1.6)
    ax2.set_ylabel("Ratio", y=1, ha='right')
    ax2.set_xlabel(h1.axes[0].label, y=1)
        
    return fig, (ax1, ax2)
    
def plot_all_regions(plots, plot_label, samples, labels,
               regions='ABCDEFGH',
               density=False,
               xlim='default', 
               log=True):

    fig = plt.figure(figsize=(20,7))
    ax = fig.subplots()
    
    offset = 0
    mids = []
    for i,r in enumerate(regions):
        
        # get (x, y) for each sample in rhig region
        hists, ys, xs = [], [], []
        for sample in samples:
            h = plots[sample][plot_label.replace("A_", r+"_")]
            if density: h = h/h.sum().value
            y, x = h.to_numpy()
            x = x[:-1]
            hists.append(h)
            ys.append(y)
            xs.append(x)
        
        # get args for min and max
        xmins, xmaxs = [], []
        for x, y in zip(xs, ys):
            xmin = np.argwhere(y>0)[0] if any(y>0) else [1e6]
            xmax = np.argwhere(y>0)[-1] if any(y>0) else [1e-6]
            xmins.append(xmin)
            xmaxs.append(xmax)
        xmin = min(xmins)[0]
        xmax = max(xmaxs)[0]
                        
        # get only range that matters
        Xs, Ys = [], []
        for x, y in zip(xs, ys):
            x = x[xmin:xmax+1]
            y = y[xmin:xmax+1]
            x = x - x[0]
            this_offset = x[-1]-x[0]
            x = x + offset
            Xs.append(x)
            Ys.append(y)
                
        # total offset
        offset += this_offset
        
        mids.append((Xs[0][-1]+Xs[0][0])/2)
        
        for h, x, y, sample, label in zip(hists, Xs, Ys, samples, labels):
            y_errs = np.sqrt(h.variances())
            y_errs = y_errs[xmin:xmax+1]
            if i == 0: ax.step(x, y, color=default_colors[sample], label=label, where='mid')
            else: ax.step(x, y, color=default_colors[sample], where='mid')
     
        ax.axvline(Xs[0][0], ls="--", color='black')
        
    if log: ax.set_yscale("log")
 
    ax.set_xticks(mids)
    ax.set_xticklabels(list(regions))
    
    ax.set_ylabel("Events", y=1, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
           
    return fig, ax

def slice_hist2d(hist, regions_list, slice_var='y'):
    """
    Inputs:
        hist: 2d Hist histogram.
        regions_list: list of regions using Hist slicing. e.g. [[10j,20j],[20j,30j],...]
        slice_var: 'x' or 'y', which dimensions to slice in
    Returns:
        A list of Hist histograms.
    """
    hist_list = []
    for regions in regions_list:
        if slice_var == 'y': h = hist[:,regions[0]:regions[1]:sum]
        elif slice_var == 'x': h = hist[regions[0]:regions[1]:sum,:]
        hist_list.append(h)
    return hist_list

def plot_sliced_hist2d(hist, regions_list, slice_var='y', labels=None):
    """
    Takes a 2d histogram, slices it in different regions, and plots the
    regions stacked.
    Inputs:
        hist: 2d Hist histogram.
        regions_list: list of regions using Hist slicing. e.g. [[10j,20j],[20j,30j],...]
        bin_var: 'x' or 'y', which dimensions to slice in
        labels: list of strings to use as labels in plot.
    Returns:
        matplotlib fig and ax
    """
    if labels: assert len(labels) == len(regions_list)
    hist_list = slice_hist2d(hist, regions_list, slice_var)
    cmap = plt.cm.jet(np.linspace(0, 1, len(hist_list)))
    
    fig = plt.figure()
    ax = fig.subplots()
    hep.histplot(hist_list, yerr=True, stack=True, histtype ='fill',
                 label=labels, color=cmap, ax=ax)
    ax.legend(fontsize=14, framealpha=1, facecolor='white', shadow=True, bbox_to_anchor=(1.04,1), loc="upper left")
    ax.set_yscale("log")
    
    return fig, ax

def ABCD_4regions(hist_abcd, xregions, yregions, sum_var='x',):
    
    if sum_var == 'x':
        A = hist_abcd[xregions[0]:xregions[1]:sum,yregions[0]:yregions[1]]
        B = hist_abcd[xregions[0]:xregions[1]:sum,yregions[1]:yregions[2]]
        C = hist_abcd[xregions[1]:xregions[2]:sum,yregions[0]:yregions[1]]
        SR = hist_abcd[xregions[1]:xregions[2]:sum,yregions[1]:yregions[2]]
        SR_exp = B * C.sum().value/A.sum().value
    elif sum_var == 'y':
        A = hist_abcd[xregions[0]:xregions[1],yregions[0]:yregions[1]:sum]
        B = hist_abcd[xregions[0]:xregions[1],yregions[1]:yregions[2]:sum]
        C = hist_abcd[xregions[1]:xregions[2],yregions[0]:yregions[1]:sum]
        SR = hist_abcd[xregions[1]:xregions[2],yregions[1]:yregions[2]:sum]
        SR_exp = C * B.sum().value/A.sum().value
        
    return SR, SR_exp

def ABCD_6regions(hist_abcd, xregions, yregions, sum_var='x'):
    
    if sum_var == 'x':
        if len(xregions) == 3:
            A = hist_abcd[xregions[0]:xregions[1]:sum,yregions[0]:yregions[1]]
            B = hist_abcd[xregions[0]:xregions[1]:sum,yregions[1]:yregions[2]]
            C = hist_abcd[xregions[0]:xregions[1]:sum,yregions[2]:yregions[3]]
            D = hist_abcd[xregions[1]:xregions[2]:sum,yregions[0]:yregions[1]]
            E = hist_abcd[xregions[1]:xregions[2]:sum,yregions[1]:yregions[2]]
            SR = hist_abcd[xregions[1]:xregions[2]:sum,yregions[2]:yregions[3]]
        elif len(xregions) == 4:
            A = hist_abcd[xregions[0]:xregions[1]:sum,yregions[0]:yregions[1]]
            B = hist_abcd[xregions[1]:xregions[2]:sum,yregions[0]:yregions[1]]
            C = hist_abcd[xregions[2]:xregions[3]:sum,yregions[0]:yregions[1]]
            D = hist_abcd[xregions[0]:xregions[1]:sum,yregions[1]:yregions[2]]
            E = hist_abcd[xregions[1]:xregions[2]:sum,yregions[1]:yregions[2]]
            SR = hist_abcd[xregions[2]:xregions[3]:sum,yregions[1]:yregions[2]]
        SR_exp = E * E.sum().value * C.sum().value * A.sum().value / (B.sum().value**2 * D.sum().value)
    elif sum_var == 'y':
        if len(xregions) == 3:
            A = hist_abcd[xregions[0]:xregions[1],yregions[0]:yregions[1]:sum]
            B = hist_abcd[xregions[0]:xregions[1],yregions[1]:yregions[2]:sum]
            C = hist_abcd[xregions[0]:xregions[1],yregions[2]:yregions[3]:sum]
            D = hist_abcd[xregions[1]:xregions[2],yregions[0]:yregions[1]:sum]
            E = hist_abcd[xregions[1]:xregions[2],yregions[1]:yregions[2]:sum]
            SR = hist_abcd[xregions[1]:xregions[2],yregions[2]:yregions[3]:sum]
        elif len(xregions) == 4:
            A = hist_abcd[xregions[0]:xregions[1],yregions[0]:yregions[1]:sum]
            B = hist_abcd[xregions[1]:xregions[2],yregions[0]:yregions[1]:sum]
            C = hist_abcd[xregions[2]:xregions[3],yregions[0]:yregions[1]:sum]
            D = hist_abcd[xregions[0]:xregions[1],yregions[1]:yregions[2]:sum]
            E = hist_abcd[xregions[1]:xregions[2],yregions[1]:yregions[2]:sum]
            SR = hist_abcd[xregions[2]:xregions[3],yregions[1]:yregions[2]:sum]
        SR_exp = C * E.sum().value**2 * A.sum().value / (B.sum().value**2 * D.sum().value)
    
    return SR, SR_exp

def ABCD_9regions(hist_abcd, xregions, yregions, sum_var='x', return_all=False):
    
    if sum_var == 'x':
        A = hist_abcd[xregions[0]:xregions[1]:sum,yregions[0]:yregions[1]]
        B = hist_abcd[xregions[0]:xregions[1]:sum,yregions[1]:yregions[2]]
        C = hist_abcd[xregions[0]:xregions[1]:sum,yregions[2]:yregions[3]]
        D = hist_abcd[xregions[1]:xregions[2]:sum,yregions[0]:yregions[1]]
        E = hist_abcd[xregions[1]:xregions[2]:sum,yregions[1]:yregions[2]]
        F = hist_abcd[xregions[1]:xregions[2]:sum,yregions[2]:yregions[3]]
        G = hist_abcd[xregions[2]:xregions[3]:sum,yregions[0]:yregions[1]]
        H = hist_abcd[xregions[2]:xregions[3]:sum,yregions[1]:yregions[2]]
        SR = hist_abcd[xregions[2]:xregions[3]:sum,yregions[2]:yregions[3]]
        SR_exp = F * F.sum().value**3 * (G.sum().value * C.sum().value / A.sum().value) * \
                ((H.sum().value / E.sum().value)**4) \
                * (G.sum().value * F.sum().value / D.sum().value)**-2 \
                * (H.sum().value * C.sum().value / B.sum().value)**-2
    elif sum_var == 'y':
        A = hist_abcd[xregions[0]:xregions[1],yregions[0]:yregions[1]:sum]
        B = hist_abcd[xregions[0]:xregions[1],yregions[1]:yregions[2]:sum]
        C = hist_abcd[xregions[0]:xregions[1],yregions[2]:yregions[3]:sum]
        D = hist_abcd[xregions[1]:xregions[2],yregions[0]:yregions[1]:sum]
        E = hist_abcd[xregions[1]:xregions[2],yregions[1]:yregions[2]:sum]
        F = hist_abcd[xregions[1]:xregions[2],yregions[2]:yregions[3]:sum]
        G = hist_abcd[xregions[2]:xregions[3],yregions[0]:yregions[1]:sum]
        H = hist_abcd[xregions[2]:xregions[3],yregions[1]:yregions[2]:sum]
        SR = hist_abcd[xregions[2]:xregions[3],yregions[2]:yregions[3]:sum]
        SR_exp = H * H.sum().value **3 * (G.sum().value * C.sum().value / A.sum().value) * \
            ((F.sum().value / E.sum().value)**4) \
            * (G.sum().value * F.sum().value / D.sum().value)**-2 \
            * (H.sum().value * C.sum().value / B.sum().value)**-2
        
    if return_all: return A, B, C, D, E, F, G, H, SR, SR_exp
    else: return SR, SR_exp

def ABCD_9regions_errorProp(abcd, xregions, yregions, sum_var='x'):
    """
    Does 9 region ABCD using error propagation of the statistical
    uncerantities of the regions. We need to scale histogram F or H
    by some factor 'alpha' (defined in exp). For example, for F,
    the new variance is:
    variance = F_value**2 * sigma_alpha**2 + alpha**2 * F_variance**2
    """
    
    A, B, C, D, E, F, G, H, SR, SR_exp = ABCD_9regions(abcd, xregions, yregions, sum_var=sum_var, return_all=True)

    # define the scaling factor function
    a, b, c, d, e, f, g, h = symbols('A B C D E F G H')
    if sum_var == 'x': exp = f * h**2 * d**2 * b**2 * g**-1 * c**-1 * a**-1 * e**-4
    elif sum_var == 'y': exp = h * d**2 * b**2 * f**2 * g**-1 * c**-1 * a**-1 * e**-4

    # defines lists of variables (sympy symbols) and accumulators (hist.sum())
    variables = [a, b, c, d, e, f, g, h]
    accs = [A.sum(), B.sum(), C.sum(), D.sum(), 
            E.sum(), F.sum(), G.sum(), H.sum()]

    # calculate scaling factor by substituting values of the histograms' sums for the sympy symbols
    alpha = exp.copy()
    for var, acc in zip(variables, accs):
        alpha = alpha.subs(var, acc.value)

    # calculate the error on the scaling factor
    variance = 0
    for var, acc in zip(variables, accs):
        der = diff(exp, var)
        var = abs(acc.variance)
        variance += der**2 * var
    for var, acc in zip(variables, accs):
        variance = variance.subs(var, acc.value)
    sigma_alpha = sqrt(variance)

    # define SR_exp and propagate the error on the scaling factor to the bin variances
    if sum_var == 'x': SR_exp = F.copy()
    elif sum_var == 'y': SR_exp = H.copy()
    new_var = SR_exp.values()**2 * float(sigma_alpha)**2 + float(alpha)**2 * abs(SR_exp.variances())
    SR_exp.view().variance = new_var
    SR_exp.view().value = SR_exp.view().value * float(alpha)
    
    return SR, SR_exp, alpha, sigma_alpha

def integrate(h, lower, upper):
    i = h[lower:upper].sum()
    return i.value, np.sqrt(i.variance)
    
def find_nth(string, substring, n):
    if (n == 1):
        return string.find(substring)
    else:
        return string.find(substring, find_nth(string, substring, n - 1) + 1)

def rebin_piecewise(h_in, bins, histtype='hist'):
    """
    Inputs:
        h : histogram
        bins: list of bins as real numbers
        histtype: one of allowed_histtypes to return

    Returns:
        h_out: a histogram of type 'histtype', rebinned according to desired bins
    """

    # only 1D hists supported for now
    if len(h_in.shape) != 1:
        raise Exception("Only 1D hists supported for now")

    # only hist and bh supported
    allowed_histtypes = ['hist', 'bh']
    if histtype not in allowed_histtypes:
        raise Exception("histtype in not in allowed_histtypes")

    # check that the bins are real numbers
    if any([x.imag != 0 for x in bins]):
        raise Exception("Only pass real-valued bins")

    # split the histogram by the bins
    # and for each bin, calculate total amount of events and variance
    z_vals, z_vars = [], []
    for iBin in range(len(bins)-1): 

        if histtype == 'hist':
            bin_lo = bins[iBin]*1.0j
            bin_hi = bins[iBin+1]*1.0j            
        elif histtype == 'bh':
            bin_lo = bh.loc(bins[iBin])
            bin_hi = bh.loc(bins[iBin+1])

        h_fragment = h_in[bin_lo:bin_hi]    
        z_vals.append(h_fragment.sum().value)
        z_vars.append(h_fragment.sum().variance)

    # fill the histograms
    if histtype == 'hist':
        h_out = hist.Hist(hist.axis.Variable(bins), storage=hist.storage.Weight())
        h_out[:] = np.stack([z_vals, z_vars], axis=-1)

    elif histtype == 'bh':
        h_out = bh.Histogram(bh.axis.Variable(bins), storage=bh.storage.Weight())
        h_out[:] = np.stack([z_vals, z_vars], axis=-1)

    return h_out

def nested_dict(n, type):
    if n == 1:
        return defaultdict(type)
    else:
        return defaultdict(lambda: nested_dict(n-1, type))