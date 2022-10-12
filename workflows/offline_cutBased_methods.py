import numpy as np
import pandas as pd
import awkward as ak
import vector
vector.register_awkward()

from workflows.math_utils import *

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

    # inverted selection
    if do_inverted:

        boost_ISR = ak.zip({
            "px": ISR_cand.px*-1,
            "py": ISR_cand.py*-1,
            "pz": ISR_cand.pz*-1,
            "mass": ISR_cand.mass
        }, with_name="Momentum4D")    
        ISR_tracks_b = ISR_cluster_tracks.boost_p4(boost_ISR)

        oneISRtrackCut = (ak.num(ISR_tracks_b) > 1)

        # output file if no events pass selections for ISR
        # avoids leaving this chunk without these columns
        if not any(oneISRtrackCut):
            print("No events in Inverted CL Removal Method, oneISRtrackCut.")
            for c in self.columns_CL_ISR: self.out_vars[c] = np.nan
        else:
            # remove events with only one track in ISR
            indices = indices[oneISRtrackCut]
            ISR_tracks_b = ISR_tracks_b[oneISRtrackCut]
            ISR_cand = ISR_cand[oneISRtrackCut]

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
