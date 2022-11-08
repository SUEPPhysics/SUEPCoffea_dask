import numpy as np
import pandas as pd
import awkward as ak
import fastjet
import vector
vector.register_awkward()

def ClusterMethod(self, indices, tracks, 
                  SUEP_cand, ISR_cand, 
                  SUEP_cluster_tracks, ISR_cluster_tracks,
                  do_inverted=False,
                  out_label=None):
    #####################################################################################
    # ---- Cluster Method (CL)
    # In this method, we use the tracks that were already clustered into the SUEP jet
    # to be the SUEP jet. Variables such as sphericity are calculated using these.
    #####################################################################################

    # boost into frame of SUEP
    boost_SUEP = ak.zip({
        "px": SUEP_cand.px*-1,
        "py": SUEP_cand.py*-1,
        "pz": SUEP_cand.pz*-1,
        "mass": SUEP_cand.mass
    }, with_name="Momentum4D")        

    # SUEP tracks for this method are defined to be the ones from the cluster
    # that was picked to be the SUEP jet
    SUEP_tracks_b = SUEP_cluster_tracks.boost_p4(boost_SUEP)        

    # SUEP jet variables
    eigs = sphericity(SUEP_tracks_b,1.0) #Set r=1.0 for IRC safe
    self.out_vars.loc[indices, "SUEP_nconst_CL"+out_label] = ak.num(SUEP_tracks_b)
    self.out_vars.loc[indices, "SUEP_pt_avg_b_CL"+out_label] = ak.mean(SUEP_tracks_b.pt, axis=-1)
    self.out_vars.loc[indices, "SUEP_S1_CL"+out_label] = 1.5 * (eigs[:,1]+eigs[:,0])

    # unboost for these
    SUEP_tracks = SUEP_tracks_b.boost_p4(SUEP_cand)
    self.out_vars.loc[indices, "SUEP_pt_avg_CL"+out_label] = ak.mean(SUEP_tracks.pt, axis=-1)
    deltaR = SUEP_tracks.deltaR(SUEP_cand)
    #self.out_vars.loc[indices, "SUEP_rho0_CL"+out_label] = rho(0, SUEP_cand, SUEP_tracks, deltaR)
    #self.out_vars.loc[indices, "SUEP_rho1_CL"+out_label] = rho(1, SUEP_cand, SUEP_tracks, deltaR)

    self.out_vars.loc[indices, "SUEP_pt_CL"+out_label] = SUEP_cand.pt
    self.out_vars.loc[indices, "SUEP_eta_CL"+out_label] = SUEP_cand.eta
    self.out_vars.loc[indices, "SUEP_phi_CL"+out_label] = SUEP_cand.phi
    self.out_vars.loc[indices, "SUEP_mass_CL"+out_label] = SUEP_cand.mass

    self.out_vars.loc[indices, "SUEP_delta_mass_genMass_CL"+out_label] = SUEP_cand.mass - self.out_vars['SUEP_genMass'+out_label][indices]
    self.out_vars.loc[indices, "SUEP_delta_pt_genPt_CL"+out_label] = SUEP_cand.pt - self.out_vars['SUEP_genPt'+out_label][indices]

    # inverted selection
    if do_inverted:

        boost_ISR = ak.zip({
            "px": ISR_cand.px*-1,
            "py": ISR_cand.py*-1,
            "pz": ISR_cand.pz*-1,
            "mass": ISR_cand.mass
        }, with_name="Momentum4D")    
        ISR_tracks_b = ISR_cluster_tracks.boost_p4(boost_ISR)

        # consistency check: we required already that ISR and SUEP have each at least 2 tracks
        assert all(ak.num(ISR_tracks_b) > 1)

        # ISR jet variables
        eigs = sphericity(ISR_tracks_b,1.0) #Set r=1.0 for IRC safe
        self.out_vars.loc[indices, "ISR_nconst_CL"+out_label] = ak.num(ISR_tracks_b)
        self.out_vars.loc[indices, "ISR_pt_avg_b_CL"+out_label] = ak.mean(ISR_tracks_b.pt, axis=-1)
        self.out_vars.loc[indices, "ISR_S1_CL"+out_label] = 1.5 * (eigs[:,1]+eigs[:,0])

        # unboost for these
        ISR_tracks = ISR_tracks_b.boost_p4(ISR_cand)
        self.out_vars.loc[indices, "ISR_pt_avg_CL"+out_label] = ak.mean(ISR_tracks.pt, axis=-1)
        deltaR = ISR_tracks.deltaR(ISR_cand)
        #self.out_vars.loc[indices, "ISR_rho0_CL"+out_label] = rho(0, ISR_cand, ISR_tracks, deltaR)
        #self.out_vars.loc[indices, "ISR_rho1_CL"+out_label] = rho(1, ISR_cand, ISR_tracks, deltaR)

        self.out_vars.loc[indices, "ISR_pt_CL"+out_label] = ISR_cand.pt
        self.out_vars.loc[indices, "ISR_eta_CL"+out_label] = ISR_cand.eta
        self.out_vars.loc[indices, "ISR_phi_CL"+out_label] = ISR_cand.phi
        self.out_vars.loc[indices, "ISR_mass_CL"+out_label] = ISR_cand.mass

def ISRRemovalMethod(self, indices, tracks, 
                     SUEP_cand, ISR_cand):
    #####################################################################################
    # ---- ISR Removal Method (IRM)
    # In this method, we boost into the frame of the SUEP jet as selected previously
    # and select all tracks that are dphi > 1.6 from the ISR jet in this frame
    # to be the SUEP tracks. Variables such as sphericity are calculated using these.
    #####################################################################################

    # boost into frame of SUEP
    boost_SUEP = ak.zip({
        "px": SUEP_cand.px*-1,
        "py": SUEP_cand.py*-1,
        "pz": SUEP_cand.pz*-1,
        "mass": SUEP_cand.mass
    }, with_name="Momentum4D")        
    ISR_cand_b = ISR_cand.boost_p4(boost_SUEP)
    tracks_b = tracks.boost_p4(boost_SUEP)

    # SUEP and IRM tracks as defined by IRS Removal Method (IRM):
    # all tracks outside/inside dphi 1.6 from ISR jet
    SUEP_tracks_b = tracks_b[abs(tracks_b.deltaphi(ISR_cand_b)) > 1.6]
    ISR_tracks_b = tracks_b[abs(tracks_b.deltaphi(ISR_cand_b)) <= 1.6]
    oneIRMtrackCut = (ak.num(SUEP_tracks_b)>1)

    # output file if no events pass selections for ISR
    # avoids leaving this chunk without these columns
    if not any(oneIRMtrackCut):
        print("No events in ISR Removal Method, oneIRMtrackCut.")
        for c in self.columns_IRM: self.out_vars[c] = np.nan
    else:
        #remove the events left with one track
        SUEP_tracks_b = SUEP_tracks_b[oneIRMtrackCut]
        ISR_tracks_b = ISR_tracks_b[oneIRMtrackCut]
        SUEP_cand = SUEP_cand[oneIRMtrackCut]
        ISR_cand_IRM = ISR_cand[oneIRMtrackCut]        
        tracks = tracks[oneIRMtrackCut]
        indices = indices[oneIRMtrackCut]

        self.out_vars.loc[indices, "SUEP_dphi_SUEP_ISR_IRM"] = ak.mean(abs(SUEP_cand.deltaphi(ISR_cand_IRM)), axis=-1)

        # SUEP jet variables
        eigs = sphericity(SUEP_tracks_b,1.0) #Set r=1.0 for IRC safe
        self.out_vars.loc[indices, "SUEP_nconst_IRM"] = ak.num(SUEP_tracks_b)
        self.out_vars.loc[indices, "SUEP_pt_avg_b_IRM"] = ak.mean(SUEP_tracks_b.pt, axis=-1)
        self.out_vars.loc[indices, "SUEP_S1_IRM"] = 1.5 * (eigs[:,1]+eigs[:,0])

        # unboost for these
        SUEP_tracks = SUEP_tracks_b.boost_p4(SUEP_cand)
        self.out_vars.loc[indices, "SUEP_pt_avg_IRM"] = ak.mean(SUEP_tracks.pt, axis=-1)
        deltaR = SUEP_tracks.deltaR(SUEP_cand)
        #self.out_vars.loc[indices, "SUEP_rho0_IRM"] = rho(0, SUEP_cand, SUEP_tracks, deltaR)
        #self.out_vars.loc[indices, "SUEP_rho1_IRM"] = rho(1, SUEP_cand, SUEP_tracks, deltaR)

        # redefine the jets using the tracks as selected by IRM
        SUEP = ak.zip({
            "px": ak.sum(SUEP_tracks.px, axis=-1),
            "py": ak.sum(SUEP_tracks.py, axis=-1),
            "pz": ak.sum(SUEP_tracks.pz, axis=-1),
            "energy": ak.sum(SUEP_tracks.energy, axis=-1),
        }, with_name="Momentum4D")
        self.out_vars.loc[indices, "SUEP_pt_IRM"] = SUEP.pt
        self.out_vars.loc[indices, "SUEP_eta_IRM"] = SUEP.eta
        self.out_vars.loc[indices, "SUEP_phi_IRM"] = SUEP.phi
        self.out_vars.loc[indices, "SUEP_mass_IRM"] = SUEP.mass

def ConeMethod(self, indices, tracks, 
               SUEP_cand, ISR_cand,
               do_inverted=False):
    #####################################################################################
    # ---- Cone Method (CO)
    # In this method, all tracks outside a cone of abs(deltaR) of 1.6 (in lab frame)
    # are the SUEP tracks, those inside the cone are ISR tracks.
    #####################################################################################

    # SUEP tracks are all tracks outside a deltaR cone around ISR
    SUEP_tracks = tracks[abs(tracks.deltaR(ISR_cand)) > 1.6]
    ISR_tracks = tracks[abs(tracks.deltaR(ISR_cand)) <= 1.6]
    oneCOtrackCut = (ak.num(SUEP_tracks)>1) 

    # output file if no events pass selections for CO
    # avoids leaving this chunk without these columns
    if not any(oneCOtrackCut):
        print("No events in Cone Method, oneCOtrackCut.")
        for c in self.columns_CO: self.out_vars[c] = np.nan
        if do_inverted: 
            for c in self.columns_CO_ISR: self.out_vars[c] = np.nan
    else:
        #remove the events left with one track
        SUEP_tracks = SUEP_tracks[oneCOtrackCut]
        ISR_tracks = ISR_tracks[oneCOtrackCut]        
        tracks = tracks[oneCOtrackCut]
        indices = indices[oneCOtrackCut]

        SUEP_cand = ak.zip({
            "px": ak.sum(SUEP_tracks.px, axis=-1),
            "py": ak.sum(SUEP_tracks.py, axis=-1),
            "pz": ak.sum(SUEP_tracks.pz, axis=-1),
            "energy": ak.sum(SUEP_tracks.energy, axis=-1),
        }, with_name="Momentum4D")

        # boost into frame of SUEP
        boost_SUEP = ak.zip({
            "px": SUEP_cand.px*-1,
            "py": SUEP_cand.py*-1,
            "pz": SUEP_cand.pz*-1,
            "mass": SUEP_cand.mass
        }, with_name="Momentum4D")        

        SUEP_tracks_b = SUEP_tracks.boost_p4(boost_SUEP)

        # SUEP jet variables
        eigs = sphericity(SUEP_tracks_b, 1.0) #Set r=1.0 for IRC safe
        self.out_vars.loc[indices, "SUEP_nconst_CO"] = ak.num(SUEP_tracks_b)
        self.out_vars.loc[indices, "SUEP_pt_avg_b_CO"] = ak.mean(SUEP_tracks_b.pt, axis=-1)
        self.out_vars.loc[indices, "SUEP_S1_CO"] = 1.5 * (eigs[:,1]+eigs[:,0])

        # unboost for these
        SUEP_tracks = SUEP_tracks_b.boost_p4(SUEP_cand)
        self.out_vars.loc[indices, "SUEP_pt_avg_CO"] = ak.mean(SUEP_tracks.pt, axis=-1)
        deltaR = SUEP_tracks.deltaR(SUEP_cand)
        #self.out_vars.loc[indices, "SUEP_rho0_CO"] = rho(0, SUEP_cand, SUEP_tracks, deltaR)
        #self.out_vars.loc[indices, "SUEP_rho1_CO"] = rho(1, SUEP_cand, SUEP_tracks, deltaR)               

        self.out_vars.loc[indices, "SUEP_pt_CO"] = SUEP_cand.pt
        self.out_vars.loc[indices, "SUEP_eta_CO"] = SUEP_cand.eta
        self.out_vars.loc[indices, "SUEP_phi_CO"] = SUEP_cand.phi
        self.out_vars.loc[indices, "SUEP_mass_CO"] = SUEP_cand.mass

        # inverted selection
        if do_inverted: 

            oneCOISRtrackCut = (ak.num(ISR_tracks)>1) 

            # output file if no events pass selections for ISR
            # avoids leaving this chunk without these columns
            if not any(oneCOISRtrackCut):
                print("No events in Inverted CO Removal Method, oneCOISRtrackCut.")
                for c in self.columns_CO_ISR: self.out_vars[c] = np.nan
            else:

                # remove events with one ISR track
                ISR_tracks = ISR_tracks[oneCOISRtrackCut]        
                indices = indices[oneCOISRtrackCut]

                ISR_cand = ak.zip({
                    "px": ak.sum(ISR_tracks.px, axis=-1),
                    "py": ak.sum(ISR_tracks.py, axis=-1),
                    "pz": ak.sum(ISR_tracks.pz, axis=-1),
                    "energy": ak.sum(ISR_tracks.energy, axis=-1),
                }, with_name="Momentum4D")

                boost_ISR = ak.zip({
                    "px": ISR_cand.px*-1,
                    "py": ISR_cand.py*-1,
                    "pz": ISR_cand.pz*-1,
                    "mass": ISR_cand.mass
                }, with_name="Momentum4D")   

                ISR_tracks_b = ISR_tracks.boost_p4(boost_ISR)

                # ISR jet variables
                eigs = sphericity(ISR_tracks_b,1.0) #Set r=1.0 for IRC safe
                self.out_vars.loc[indices, "ISR_nconst_CO"] = ak.num(ISR_tracks_b)
                self.out_vars.loc[indices, "ISR_pt_avg_b_CO"] = ak.mean(ISR_tracks_b.pt, axis=-1)
                self.out_vars.loc[indices, "ISR_pt_mean_scaled_CO"] = ak.mean(ISR_tracks_b.pt, axis=-1)/ak.max(ISR_tracks_b.pt, axis=-1)
                self.out_vars.loc[indices, "ISR_S1_CO"] = 1.5 * (eigs[:,1]+eigs[:,0])

                # unboost for these
                ISR_tracks = ISR_tracks_b.boost_p4(ISR_cand)
                self.out_vars.loc[indices, "ISR_pt_avg_CO"] = ak.mean(ISR_tracks.pt, axis=-1)
                deltaR = ISR_tracks.deltaR(ISR_cand)
                self.out_vars.loc[indices, "ISR_rho0_CO"] = rho(0, ISR_cand, ISR_tracks, deltaR)
                self.out_vars.loc[indices, "ISR_rho1_CO"] = rho(1, ISR_cand, ISR_tracks, deltaR)

                self.out_vars.loc[indices, "ISR_pt_CO"] = ISR_cand.pt
                self.out_vars.loc[indices, "ISR_eta_CO"] = ISR_cand.eta
                self.out_vars.loc[indices, "ISR_phi_CO"] = ISR_cand.phi
                self.out_vars.loc[indices, "ISR_mass_CO"] = ISR_cand.mass

def sphericity(particles, r):
    norm = ak.sum(particles.p ** r, axis=1, keepdims=True)
    s = np.array([[
                   ak.sum(particles.px * particles.px * particles.p ** (r-2.0), axis=1 ,keepdims=True)/norm,
                   ak.sum(particles.px * particles.py * particles.p ** (r-2.0), axis=1 ,keepdims=True)/norm,
                   ak.sum(particles.px * particles.pz * particles.p ** (r-2.0), axis=1 ,keepdims=True)/norm
                  ],
                  [
                   ak.sum(particles.py * particles.px * particles.p ** (r-2.0), axis=1 ,keepdims=True)/norm,
                   ak.sum(particles.py * particles.py * particles.p ** (r-2.0), axis=1 ,keepdims=True)/norm,
                   ak.sum(particles.py * particles.pz * particles.p ** (r-2.0), axis=1 ,keepdims=True)/norm
                  ],
                  [
                   ak.sum(particles.pz * particles.px * particles.p ** (r-2.0), axis=1 ,keepdims=True)/norm,
                   ak.sum(particles.pz * particles.py * particles.p ** (r-2.0), axis=1 ,keepdims=True)/norm,
                   ak.sum(particles.pz * particles.pz * particles.p ** (r-2.0), axis=1 ,keepdims=True)/norm
                  ]])
    s = np.squeeze(np.moveaxis(s, 2, 0),axis=3)
    evals = np.sort(np.linalg.eigvalsh(s))
    return evals

def rho(number, jet, tracks, deltaR, dr=0.05):
    r_start = number*dr
    r_end = (number+1)*dr
    ring = (deltaR > r_start) & (deltaR < r_end)
    rho_values = ak.sum(tracks[ring].pt, axis=1)/(dr*jet.pt)
    return rho_values

def FastJetReclustering(tracks, r, minPt):
        
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, r)        
    cluster = fastjet.ClusterSequence(tracks, jetdef)

    # have to set min_pt = 0 and cut later to avoid some memory issues
    # FIXME: should try to understand this failure
    ak_inclusive_jets = cluster.inclusive_jets()
    ak_inclusive_cluster = cluster.constituents()
    
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

    # at least 2 tracks in SUEP and ISR
    singletrackCut = (ak.num(clusters_pTsorted[:,0])>1) & (ak.num(clusters_pTsorted[:,1])>1)   
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