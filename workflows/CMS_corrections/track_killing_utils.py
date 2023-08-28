import awkward as ak
import numpy as np


def track_killing(self, tracks):
    """
    Drop 2.7%, 2.2%, and 2.1% of the tracks randomly at reco-level
    for charged-particles with 1 < pT < 20 GeV in simulation for 2016, 2017, and
    2018, respectively when reclustering the constituents.
     For charged-particles with pT > 20 GeV, 1% of the tracks are dropped randomly
    """

    if self.scouting == 1:
        block1_percent = 0.05
        block2_percent = 0.01
    else:
        year_percent = {"2018": 0.021, "2017": 0.022, "2016": 0.027, "2016apv": 0.027}
        block1_percent = year_percent[str(self.era)]
        block2_percent = 0.01

    block0_indices = tracks.pt <= 1
    block1_indices = (tracks.pt > 1) & (tracks.pt < 20)
    block2_indices = tracks.pt >= 20

    new_indices = []

    for i in range(len(tracks)):
        event_indices = np.arange(len(tracks[i]))
        event_bool = np.array([True] * len(tracks[i]))

        block1_event_indices = event_indices[block1_indices[i]]
        block1_event_indices_drop = np.random.choice(
            block1_event_indices, int((block1_percent) * len(block1_event_indices))
        )
        event_bool[block1_event_indices_drop] = False

        block2_event_indices = event_indices[block2_indices[i]]
        block2_event_indices_drop = np.random.choice(
            block2_event_indices, int((block2_percent) * len(block2_event_indices))
        )
        event_bool[block2_event_indices_drop] = False

        new_indices.append(list(event_bool))

    new_indices = ak.Array(new_indices)
    tracks = tracks[new_indices]
    return tracks

def scout_track_killing(self, tracks):
    """
    Drop 2.5% of the tracks randomly at reco-level
    for charged-particles with 1 < pT < 20 GeV in simulation when reclustering the constituents.
    For charged-particles with pT > 20 GeV, 1% of the tracks are dropped randomly.
    As these studies were only done for offline data alone, we additionally scale these 
    probabilities by another scale factor to estimate the reconstruction efficiencies for scouting data. 
    This is done by looking at the ratio of the offline to scouting tracks in
    both data and Monte Carlo as a function of pT. If there is a 2.5% chance to not reconstruct MC track in real data, 
    then MC_off/Data_off = 1.05 -> more MC than DATA. 
    Then ratio in scouting = (MC_off/Data_off)* (MC_s/MC_off)*(Data_off/Data_s) = 1.05 * QCDscale/datascale
    """

    #Read in the scaling files
    np.random.seed(2022) #random seed to get repeatable results
    pt_bins = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.75,1,1.25,1.5,2.0,3,10,20,50])
    datascale =  np.loadtxt("data/tracks/track_datascaling_2018_pt.txt", delimiter=",")
    qcdscale =  np.loadtxt("data/tracks/track_offlinescaling_2018_pt.txt", delimiter=",")

    #Create the scaling and apply it to the random killing
    scaling = np.divide(qcdscale,datascale)
    scaling = np.append(scaling,scaling[-1])
    trackbin = np.digitize(ak.flatten(tracks.pt),pt_bins)-1
    scale = np.take(scaling,trackbin)
    probs = ak.where(ak.flatten(tracks["pt"]) < 20,0.025*scale,0.01*scale) #5% if pt < 1, otherwise 2%.
    rands = np.random.rand(len(probs)) > probs
    trk_killer = ak.Array(ak.layout.ListOffsetArray64(tracks.layout.offsets, ak.layout.NumpyArray(rands)))

    #Create the new track collection with killed tracks
    tracks = tracks[trk_killer]
    return tracks

def scout_track_killingOffline(self, tracks):
    #Read in the scaling files
    np.random.seed(2022) #random seed to get repeatable results
    pt_bins = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.75,1,1.25,1.5,2.0,3,10,20,50])
    eta_bins = np.array(range(-250,275,25))/100.
    trackwgts = np.loadtxt("data/tracks/track_drop_2018.txt", delimiter=",")
    trackwgts = np.vstack((trackwgts,trackwgts[-1])) #repeat last bin for overflow
    trackwgts = trackwgts.flatten()

    #Create the scaling and apply it to the random killing
    ptbin = np.digitize(ak.flatten(tracks["pt"]),pt_bins)-1
    etabin = np.digitize(ak.flatten(tracks["eta"]),eta_bins)-1
    bins = np.add((ptbin)*(len(eta_bins)-1),etabin)
    probs = np.take(trackwgts,bins)
    rands = np.random.rand(len(probs)) < probs # keep track is less than probability
    trk_killer = ak.Array(ak.layout.ListOffsetArray64(tracks.layout.offsets, ak.layout.NumpyArray(rands)))

    #Create the new track collection with killed tracks
    tracks = tracks[trk_killer]
    return tracks

def scaleTracksOffline(self, spherex):
    track_bins= np.array([  0.,   6.,  12.,  18.,  24.,  30.,  36.,  42.,  48.,  54.,  60.,
        66.,  72.,  78.,  84.,  90.,  96., 102., 108., 114., 120., 126.,
       132., 138., 144., 150., 156., 162., 168., 174., 180., 186., 192.,
       198., 204., 210., 216., 222., 228., 234., 240., 246., 252., 258.,
       264., 270., 276., 282., 288., 294., 300.])
    trackwgts = np.loadtxt("data/tracks/track_multiplicity_ratio_2018.txt", delimiter=",")
    trackwgts[trackwgts == 0] = 0.12907325
    trackbin = np.digitize(spherex.FatJet.nconst,track_bins)-1
    probs = np.take(trackwgts,trackbin)
    spherex["wgt"] = spherex["wgt"] * 1/probs
    return spherex

