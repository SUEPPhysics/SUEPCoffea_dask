import fastjet
import awkward as ak
import vector
vector.register_awkward()

def FastJetReclustering(self, tracks, r, minPt):
        
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, r)        
    cluster = fastjet.ClusterSequence(tracks, jetdef)

    # have to set min_pt = 0 and cut later to avoid some memory issues
    # FIXME: should try to understand this failure
    ak_inclusive_jets = cluster.inclusive_jets()[:] 
    ak_inclusive_cluster = cluster.constituents()[:]

    # apply minimum pT cut
    minPtCut = ak_inclusive_jets.pt > minPt
    ak_inclusive_jets = ak_inclusive_jets[minPtCut]
    ak_inclusive_cluster = ak_inclusive_cluster[minPtCut]

    return ak_inclusive_jets, ak_inclusive_cluster
        
def getTopTwoJets(self, tracks, indices, ak_inclusive_jets, ak_inclusive_cluster):
    # order the reclustered jets by pT (will take top 2 for ISR removal method)
    highpt_jet = ak.argsort(ak_inclusive_jets.pt, axis=1, ascending=False, stable=True)
    jets_pTsorted = ak_inclusive_jets[highpt_jet]
    clusters_pTsorted = ak_inclusive_cluster[highpt_jet]     

    # at least 2 tracks in highest pt jet
    highpt_cands = clusters_pTsorted[:,0]                    # tracks for highest pt jet         
    singletrackCut = (ak.num(highpt_cands)>1)             
    jets_pTsorted = jets_pTsorted[singletrackCut]          
    clusters_pTsorted = clusters_pTsorted[singletrackCut]
    tracks = tracks[singletrackCut]
    indices = indices[singletrackCut]

    # number of constituents per jet, sorted by pT
    nconst_pTsorted = ak.num(clusters_pTsorted, axis=-1)

    # Top 2 pT jets. If jet1 has fewer tracks than jet2 then swap
    SUEP_cand = ak.where(nconst_pTsorted[:,1]<=nconst_pTsorted[:,0],jets_pTsorted[:,0],jets_pTsorted[:,1])
    ISR_cand = ak.where(nconst_pTsorted[:,1]>nconst_pTsorted[:,0],jets_pTsorted[:,0],jets_pTsorted[:,1])
    SUEP_cluster_tracks = ak.where(nconst_pTsorted[:,1]<=nconst_pTsorted[:,0], clusters_pTsorted[:,0], clusters_pTsorted[:,1])
    ISR_cluster_tracks = ak.where(nconst_pTsorted[:,1]>nconst_pTsorted[:,0], clusters_pTsorted[:,0], clusters_pTsorted[:,1])

    return tracks, indices, (SUEP_cand, ISR_cand, SUEP_cluster_tracks, ISR_cluster_tracks)