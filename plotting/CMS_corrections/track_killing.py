import hist

def generate_up_histograms(output_labels, plots):
    """
    Given a output labels corresponding to different methods,
    and a plots dictionary (plots), loops through all the plots to find the
    track_down and nominal variations, and adds a new histogram with an 'up'
    symmetric variation.
    
    Inputs:
        output_labels: list of strings, output labels.
        plots: dictionary of Hist histograms
    Outputs:
        updated plots dictionary
    """
    sys = 'track_up'
    for label_out in output_labels:
        if 'track_down' in label_out: continue

        new_output = {}
        for hist_name in output.keys():
            if not hist_name.endswith('_track_down'): continue
            hDown = plots[hist_name].copy()
            hNom = plots[hist_name.replace('_track_down','')].copy()
            hUp = make_up_symmetric_variation(hNom, hDown)
            new_output.update({hist_name.replace('_track_down','_track_up'): hUp})
        output = new_output | plots
        
    return output

def make_up_symmetric_variation(h_nom, h_down):
    """
    Given input Hist histograms h_nom and h_down, get the symmetric up variation.
    
    Inputs: Hist 2D or 3D histograms h_nom, h_down
    Outputs: Hist hiostgram h
    """
    h_nom_npy = h_nom.to_numpy()
    h_down_npy = h_down.to_numpy()
    if len(h_nom_npy) == 2:
        variation = h_nom_npy[0] - h_down_npy[0]
        h_out = hist.Hist(hist.axis.Variable(h_nom_npy[1]), storage=bh.storage.Weight())
        new_z = np.where(nom_out[0] + variation > 0, nom_out[0] + variation, 0)
        h_out[:] = np.stack([new_z, new_z], axis=-1)
    elif len(nom_out) == 3:
        variation = nom_out[0] - h_down_npy[0]
        h_out = hist.Hist(hist.axis.Variable(nom_out[1]), hist.axis.Variable(nom_out[2]), storage=bh.storage.Weight())
        new_z = np.where(nom_out[0] + variation > 0, nom_out[0] + variation, 0)
        h_out[:,:] = np.stack([new_z, new_z], axis=-1)
    return h