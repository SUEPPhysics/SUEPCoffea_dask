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
def loader(infile_names, apply_lumis=True, exclude_low_bins=False):
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

# load hdf5 with pandas
def h5load(ifile, label):
    try:
        with pd.HDFStore(ifile, 'r') as store:
            try:
                data = store[label] 
                metadata = store.get_storer(label).attrs.metadata
                return data, metadata

            except KeyError:
                print("No key",label,ifile)
                return 0, 0
    except:
        print("Some error occurred", ifile)
        return 0, 0
 
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

def getXSection(dataset, year):
    xsection = 1
    with open('../data/xsections_{}.json'.format(year)) as file:
        MC_xsecs = json.load(file)
        try:
            xsection *= MC_xsecs[dataset]["xsec"]
            xsection *= MC_xsecs[dataset]["kr"]
            xsection *= MC_xsecs[dataset]["br"]
        except:
            print("WARNING: I did not find the xsection for that MC sample. Check the dataset name and the relevant yaml file")
            return 1
    return xsection

def get_tracks_up(nom, down):
    """
    Use envelope method to define the
    """
    nom_out = nom.to_numpy()
    down_out = down.to_numpy()
    if len(nom_out) == 2:
        variation = nom_out[0] - down_out[0]
        h = hist.Hist(hist.axis.Variable(nom_out[1]), storage=bh.storage.Weight())
        new_z = np.where(nom_out[0] + variation > 0, nom_out[0] + variation, 0)
        h[:] = np.stack([new_z, np.sqrt(new_z)], axis=-1)
    elif len(nom_out) == 3:
        variation = nom_out[0] - down_out[0]
        h = hist.Hist(hist.axis.Variable(nom_out[1]), hist.axis.Variable(nom_out[2]), storage=bh.storage.Weight())
        new_z = np.where(nom_out[0] + variation > 0, nom_out[0] + variation, 0)
        h[:,:] = np.stack([new_z, np.sqrt(new_z)], axis=-1)
    return h

def make_selection(df, variable, operator, value, apply=True):
    """
    Apply a selection on DataFrame df based on on the df column'variable'
    using the 'operator' and 'value' passed as arguments to the function.
    Returns the resulting DataFrame after the operation is applied.
    
    df: input dataframe.
    variable: df column.
    operator: see code below.
    value: value to cut variable on using operator.
    apply: toggles whether the selection is applied to the dataframe, or
    whether a list of booleans is returned matching the indices that
    passed and failed the selection.
    """
    if operator in ["greater than","gt",">"]:
        if apply: return df.loc[(df[variable] > value)]
        else: return (df[variable] > value)
    if operator in ["greater than or equal to", ">="]:
        if apply: return df.loc[(df[variable] >= value)]
        else: return (df[variable] >= value)
    elif operator in ["less than", "lt", "<"]:
        if apply: return df.loc[(df[variable] < value)]
        else: return (df[variable] < value)
    elif operator in ["less than or equal to", "<="]:
        if apply: return df.loc[(df[variable] <= value)]
        else: return (df[variable] <= value)
    elif operator in ["equal to", "eq", "=="]:
        if apply: return df.loc[(df[variable] == value)]
        else: return (df[variable] == value)
    else:
        sys.exit("Couldn't find operator requested " + operator)

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

def apply_scaling_weights(df, scaling_weights,
    x_var_regions, y_var_regions,
    regions = "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    x_var = 'SUEP_S1_CL',
    y_var = 'SUEP_nconst_CL',
    z_var = 'ht'):
    """
    df: input DataFrame to scale
    *_var_regions: x/y ABCD regions
    scaling_weights: nested dictionary, region x (bins or ratios)
    regions: string of ordered regions, used to apply corrections
    *_var: x/y are of the ABCD plane, z of the scaling histogram
    """
    
    x_var_regions = abcd['x_var_regions']
    y_var_regions = abcd['y_var_regions']
    iRegion = 0

    # S1 regions
    for i in range(len(x_var_regions)-1):
        x_val_lo = x_var_regions[i]
        x_val_hi = x_var_regions[i+1]

        # nconst regions
        for j in range(len(y_var_regions)-1):
            y_val_lo = y_var_regions[j]
            y_val_hi = y_var_regions[j+1]

            r = regions[iRegion]

            # from the weights
            bins = weights[r]['bins']
            ratios = weights[r]['ratios']

            # ht bins
            for k in range(len(bins)-1):
                z_val_lo = bins[k]
                z_val_hi = bins[k+1]
                ratio = ratios[k]

                zslice = (df[z_var] >= z_val_lo) & (df[z_var] < z_val_hi)
                yslice = (df[y_var] >= y_val_lo) & (df[y_var] < y_val_hi)
                xslice = (df[x_var] >= x_val_lo) & (df[x_var] < x_val_hi)

                df.loc[xslice & yslice & zslice, 'event_weight'] *= ratio

            iRegion += 1
    return df

def auto_fill(df, output, abcd, label_out, isMC=False, do_abcd=False):
    
    input_method = abcd['input_method']

    #####################################################################################
    # ---- Fill Histograms
    # Automatically fills all histograms that are declared in the output dict.
    #####################################################################################
    
    # 1. fill the distributions as they are saved in the dataframes
    # 1a. Plot event wide variables
    plot_labels = [key for key in df.keys() if key+"_"+label_out in list(output.keys())]  
    for plot in plot_labels: output[plot+"_"+label_out].fill(df[plot], weight=df['event_weight']) 
    # 1b. Plot method variables
    plot_labels = [key for key in df.keys() if key.replace(input_method, label_out) in list(output.keys()) and key.endswith(input_method)]
    for plot in plot_labels: output[plot.replace(input_method, label_out)].fill(df[plot], weight=df['event_weight'])  
    # FIXME: plot ABCD 2d
    
    # 2. fill some 2D distributions  
    keys = list(output.keys())
    keys_2Dhists = [k for k in keys if '2D' in k]
    for key in keys_2Dhists:
        if not key.endswith(label_out): continue
        string = key[len("2D")+1:-(len(label_out)+1)] # cut out "2D_" and output label
        var1 = string.split("_vs_")[0]
        var2 = string.split("_vs_")[1]
        if var1 not in list(df.keys()): var1 += "_" + input_method
        if var2 not in list(df.keys()): var2 += "_" + input_method
        output[key].fill(df[var1], df[var2], weight=df['event_weight'])

    # 3. divide the dfs by region
    if do_abcd:
        regions = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        xvar = abcd['xvar']
        yvar = abcd['yvar']
        xvar_regions = abcd['xvar_regions']
        yvar_regions = abcd['yvar_regions']
        iRegion = 0
        for i in range(len(xvar_regions)-1):
            x_val_lo = xvar_regions[i]
            x_val_hi = xvar_regions[i+1]

            for j in range(len(yvar_regions)-1):
                y_val_lo = yvar_regions[j]
                y_val_hi = yvar_regions[j+1]

                x_cut = (make_selection(df, xvar, '>=', x_val_lo, False) & make_selection(df, xvar, '<', x_val_hi, False))
                y_cut = (make_selection(df, yvar, '>=', y_val_lo, False) & make_selection(df, yvar, '<', y_val_hi, False))
                df_r = df.loc[(x_cut & y_cut)]

                r = regions[iRegion] + "_"
                iRegion += 1

                # double check blinding
                if iRegion == (len(xvar_regions)-1)*(len(yvar_regions)-1) and not isMC:
                    if df_r.shape[0] > 0: 
                        sys.exit(label_out+": You are not blinding correctly! Exiting.")

                # 3a. Plot event wide variables
                plot_labels = [key for key in df_r.keys() if r+key+"_"+label_out in list(output.keys())]   # event wide variables
                for plot in plot_labels: output[r+plot+"_"+label_out].fill(df_r[plot], weight=df_r['event_weight']) 
                # 3b. Plot method variables
                plot_labels = [key for key in df_r.keys() if r+key.replace(input_method, label_out) in list(output.keys())]  # method vars
                for plot in plot_labels: output[r+plot.replace(input_method, label_out)].fill(df_r[plot], weight=df_r['event_weight'])  
                
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

def plot_ratio(h1, h2, 
               plot_label=None, 
               label1=None, label2=None, 
               rebin=-1, 
               xlim='default', 
               log=True):

    #Set up variables for the stacked histogram
    fig = plt.figure(figsize=(12,10))
    plt.subplots_adjust(bottom=0.15, left=0.17)
    ax1 = plt.subplot2grid((4,1), (0,0),rowspan=2)

    y1, x1 = h1.to_numpy()
    x1 = x1[:-1]
    y1_errs = np.sqrt(h1.variances())
    if rebin!=-1: x1, y1, y1_errs = combine_bins(x1, y1, y1_errs, n=rebin)
    ax1.step(x1, y1, color='maroon',label=label1, where='mid')
    ax1.errorbar(x1, y1, yerr=y1_errs, color="maroon".upper(), fmt="", drawstyle='steps-mid')

    y2, x2 = h2.to_numpy()
    y2 = y2
    x2 = x2[:-1]
    y2_errs = np.sqrt(h2.variances())
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
    
    if plot_label is None: plot_label = h1.axes[0].label
    ax2.set_xlabel(plot_label, y=1)
    
    residuals = ratios - 1
    residuals_errs = ratio_errs
    
    return fig, (ax1, ax2), (residuals, residuals_errs)

def plot_ratio_regions(plots, plot_label, 
               sample1, sample2, 
               regions,
               rebin=-1, 
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
                
        xmin1 = np.argwhere(y1>0)[0] if any(y1>0) else [1e6]
        xmin2 = np.argwhere(y2>0)[0] if any(y2>0) else [1e6]
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
        if rebin!=-1: x1, y1, y1_errs = combine_bins(x1, y1, y1_errs, n=rebin)
        if i == 0: ax1.step(x1, y1, color='midnightblue',label=sample1, where='mid')
        else: ax1.step(x1, y1, color='midnightblue', where='mid')
        ax1.errorbar(x1, y1, yerr=y1_errs, color="maroon".upper(), fmt="", drawstyle='steps-mid')

        y2_errs = np.sqrt(h2.variances())
        y2_errs = y2_errs[xmin:xmax+1]
        if rebin!=-1: x2, y2, y2_errs = combine_bins(x2, y2, y2_errs, n=rebin)
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
               rebin=-1, 
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
            if rebin!=-1: x, y, y_errs = combine_bins(x, y, y_errs, n=rebin)
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
    
    A, B, C, D, E, F, G, H, SR, SR_exp = ABCD_9regions(abcd, xregions, yregions, sum_var='x', return_all=True)

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
    SR_exp = F.copy()
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

def rebin_piecewise_constant(x1, y1, x2):
    x1 = np.asarray(x1)
    y1 = np.asarray(y1)
    x2 = np.asarray(x2)

    # the fractional bin locations of the new bins in the old bins
    i_place = np.interp(x2, x1, np.arange(len(x1)))
    cum_sum = np.r_[[0], np.cumsum(y1)]

    # calculate bins where lower and upper bin edges span
    # greater than or equal to one original bin.
    # This is the contribution from the 'intact' bins (not including the
    # fractional start and end parts.
    whole_bins = np.floor(i_place[1:]) - np.ceil(i_place[:-1]) >= 1.
    start = cum_sum[np.ceil(i_place[:-1]).astype(int)]
    finish = cum_sum[np.floor(i_place[1:]).astype(int)]
    y2 = np.where(whole_bins, finish - start, 0.)
    bin_loc = np.clip(np.floor(i_place).astype(int), 0, len(y1) - 1)

    # fractional contribution for bins where the new bin edges are in the same
    # original bin.
    same_cell = np.floor(i_place[1:]) == np.floor(i_place[:-1])
    frac = i_place[1:] - i_place[:-1]
    contrib = (frac * y1[bin_loc[:-1]])
    y2 += np.where(same_cell, contrib, 0.)

    # fractional contribution for bins where the left and right bin edges are in
    # different original bins.
    different_cell = np.floor(i_place[1:]) > np.floor(i_place[:-1])
    frac_left = np.ceil(i_place[:-1]) - i_place[:-1]
    contrib = (frac_left * y1[bin_loc[:-1]])
    frac_right = i_place[1:] - np.floor(i_place[1:])
    contrib += (frac_right * y1[bin_loc[1:]])
    y2 += np.where(different_cell, contrib, 0.)

    return y2

def rebin(h, bins):

    current_bins, current_edges = h.to_numpy()
    new_freq = rebin_piecewise_constant(current_edges, h.values(), bins)
    new_variances = rebin_piecewise_constant(current_bins, h.variances(), bins)
    
    h = hist.Hist(hist.axis.Variable(bins),storage=hist.storage.Weight())

    h[:] = np.stack([new_freq, new_variances], axis=-1)
    
    return h

def nested_dict(n, type):
    if n == 1:
        return defaultdict(type)
    else:
        return defaultdict(lambda: nested_dict(n-1, type))