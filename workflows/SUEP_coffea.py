"""
SUEP_coffea.py
Coffea producer for SUEP analysis. Uses fastjet package to recluster large jets:
https://github.com/scikit-hep/fastjet
Chad Freer and Luca Lavezzo, 2021
"""
from coffea import hist, processor, lumi_tools
from typing import List, Optional
import awkward as ak
import pandas as pd
import numpy as np
import fastjet
import vector
vector.register_awkward()

#Importing SUEP specific functions
from workflows.inference_utils import *
from workflows.pandas_utils import *
from workflows.math_utils import *

#Importing SUEP specific corrections/systematics
from workflows.jetmet_utils import *

class SUEP_cluster(processor.ProcessorABC):
    def __init__(self, isMC: int, era: int, scouting: int, sample: str,  do_syst: bool, syst_var: str, weight_syst: bool, flag: bool, do_inf: bool, output_location: Optional[str]) -> None:
        self._flag = flag
        self.output_location = output_location
        self.do_syst = do_syst
        self.gensumweight = 1.0
        self.scouting = scouting
        self.era = era
        self.isMC = isMC
        self.sample = sample
        self.syst_var, self.syst_suffix = (syst_var, f'_sys_{syst_var}') if do_syst and syst_var else ('', '')
        self.weight_syst = weight_syst
        self.do_inf = do_inf
        self.prefixes = {"SUEP": "SUEP"}
        
        self.out_vars = pd.DataFrame()

        #Set up the image size and pixels
        self.eta_pix = 280
        self.phi_pix = 360
        self.eta_span = (-2.5, 2.5)
        self.phi_span = (-np.pi, np.pi)
        self.eta_scale = self.eta_pix/(self.eta_span[1]-self.eta_span[0])
        self.phi_scale = self.phi_pix/(self.phi_span[1]-self.phi_span[0])
        self.models = ['model125']#Add to this list. There will be an output for each prediction in this list

        #Set up for the histograms
        self._accumulator = processor.dict_accumulator({})
        
    @property
    def accumulator(self):
        return self._accumulator
    
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
  
    def MLAnalysis(self, indices, inf_cands):
        #####################################################################################
        # ---- ML Analysis
        # Each event is converted into an input for the ML models. Using ONNX, we run
        # inference on each event to obtain a prediction of the class (SUEP or QCD).
        # N.B.: Conversion is done elsewhere.
        # N.B.: The inference skips the lost tracks for now.
        #####################################################################################
         
        if self.do_inf:    
            
            pred_dict = {}
            ort_infs = {}
            options = ort.SessionOptions() 
            options.inter_op_num_threads = 1 # number of threads used to parallelize the execution of the graph (across nodes). Default is 0 to let onnxruntime choose.
            for model in self.models:
                  ort_infs.update({model: ort.InferenceSession('data/onnx_models/resnet_{}_{}.onnx'.format(model,self.era))})
            # In order to avoid memory issues convert events to images and run inference in batches
            # also exploits the numba-compiled convert_to_images function
            batch_size = 100
            for i in range(0, len(inf_cands), batch_size):
                if i + batch_size > len(inf_cands): batch_size = len(inf_cands) - i
                batch = inf_cands[i:i+batch_size]
                imgs = convert_to_images(self, batch)
                for model in self.models:
                    batch_resnet_jets = run_inference(self, imgs, ort_infs[model])
                    if i == 0: resnet_jets = batch_resnet_jets
                    else: resnet_jets = np.concatenate((resnet_jets, batch_resnet_jets))    
                    pred_dict.update({model: resnet_jets[:,1]}) #highest SUEP prediction per event

        else:
            for c in self.columns_ML: self.out_vars[c] = np.nan
          
    def jet_awkward(self,Jets):
        """"
        Create awkward array of jets. Applies basic selections.
        Returns: awkward array of dimensions (events x jets x 4 momentum)
        """
        Jets_awk = ak.zip({
            "pt": Jets.pt,
            "eta": Jets.eta,
            "phi": Jets.phi,
            "mass": Jets.mass,
        })
        jet_awk_Cut = (Jets.pt > 30) & (abs(Jets.eta)<2.4)
        Jets_correct = Jets_awk[jet_awk_Cut]
        return Jets_correct
  
    def eventSelection(self, events):
        """
        Applies trigger, returns events.
        """
        if self.scouting != 1:
            if self.era == 2016:
                trigger = (events.HLT.PFHT900 == 1)
            else:
                trigger = (events.HLT.PFHT1050 == 1)
            events = events[trigger]

        return events
    
    def getGenTracks(self, events):
        genParts = events.GenPart
        genParts = ak.zip({
            "pt": genParts.pt,
            "eta": genParts.eta,
            "phi": genParts.phi,
            "mass": genParts.mass,
            "pdgID": genParts.pdgId
        }, with_name="Momentum4D")
        return genParts
    
    def getTracks(self, events):
        Cands = ak.zip({
            "pt": events.PFCands.trkPt,
            "eta": events.PFCands.trkEta,
            "phi": events.PFCands.trkPhi,
            "mass": events.PFCands.mass
        }, with_name="Momentum4D")
        cut = (events.PFCands.fromPV > 1) & \
                 (events.PFCands.trkPt >= 0.75) & \
                 (abs(events.PFCands.trkEta) <= 2.5) & \
                 (abs(events.PFCands.dz) < 10) & \
                 (events.PFCands.dzErr < 0.05)
        Cleaned_cands = Cands[cut]
        Cleaned_cands = ak.packed(Cleaned_cands)

        #Prepare the Lost Track collection
        LostTracks = ak.zip({
            "pt": events.lostTracks.pt,
            "eta": events.lostTracks.eta,
            "phi": events.lostTracks.phi,
            "mass": 0.0
        }, with_name="Momentum4D")
        cut = (events.lostTracks.fromPV > 1) & \
            (events.lostTracks.pt >= 0.75) & \
            (abs(events.lostTracks.eta) <= 1.0) & \
            (abs(events.lostTracks.dz) < 10) & \
            (events.lostTracks.dzErr < 0.05)
        Lost_Tracks_cands = LostTracks[cut]
        Lost_Tracks_cands = ak.packed(Lost_Tracks_cands)

        # select which tracks to use in the script
        # dimensions of tracks = events x tracks in event x 4 momenta
        tracks = ak.concatenate([Cleaned_cands, Lost_Tracks_cands], axis=1)
        
        return tracks, Cleaned_cands
            
    def tracksSystematics(self, tracks):
        """
        Drop 2.7%, 2.2%, and 2.1% of the tracks randomly at reco-level
        for charged-particles with 1 < pT < 20 GeV in simulation for 2016, 2017, and
        2018, respectively when reclustering the constituents.
         For charged-particles with pT > 20 GeV, 1% of the tracks are dropped randomly
        """
        
        if self.scouting:
            block1_percent = 0.05
            block2_percent = 0.01
        else:
            year_percent = {
                "2018": 0.021,
                "2017": 0.022,
                "2016": 0.027
            }
            block1_percent = year_percent[str(self.era)]
            block2_percent = 0.01
        
        block0_indices = (tracks.pt <= 1)
        block1_indices = (tracks.pt > 1) & (tracks.pt < 20)
        block2_indices = (tracks.pt >= 20)

        new_indices = []
        for i in range(len(tracks)):
            event_indices = np.arange(len(tracks[i]))
            event_bool = np.array([True]*len(tracks[i]))

            block1_event_indices = event_indices[block1_indices[i]]
            block1_event_indices_drop = np.random.choice(block1_event_indices, int((block1_percent) * len(block1_event_indices)))

            event_bool[block1_event_indices_drop] = False

            block2_event_indices = event_indices[block2_indices[i]]
            block2_event_indices_drop = np.random.choice(block2_event_indices, int((block2_percent) * len(block2_event_indices)))
            event_bool[block2_event_indices_drop] = False
    
            new_indices.append(list(event_bool))

        new_indices = ak.Array(new_indices)
        tracks = tracks[new_indices]
        return tracks
        
    def getScoutingTracks(self, events):
        Cands = ak.zip({
            "pt": events.PFcand.pt,
            "eta": events.PFcand.eta,
            "phi": events.PFcand.phi,
            "mass": events.PFcand.mass
        }, with_name="Momentum4D")
        cut = (events.PFcand.pt >= 0.75) & \
                (abs(events.PFcand.eta) <= 2.5) & \
                (events.PFcand.vertex == 0) & \
                (events.PFcand.q != 0)
        Cleaned_cands = Cands[cut]
        tracks =  ak.packed(Cleaned_cands)
        return tracks, Cleaned_cands
    
    def storeEventVars(self, events, tracks, 
                       ak_inclusive_jets, ak_inclusive_cluster,
                       out_label=""):
        
        # select out ak4jets
        ak4jets = self.jet_awkward(events.Jet)        
        
        # work on JECs and systematics
        jets_c = apply_jecs(isMC=self.isMC, Sample=self.sample, era=self.era, events=events)
        jets_jec = self.jet_awkward(jets_c)
        if self.isMC:
            jets_jec_JERUp   = self.jet_awkward(jets_c["JER"].up)
            jets_jec_JERDown = self.jet_awkward(jets_c["JER"].down)
            jets_jec_JESUp   = self.jet_awkward(jets_c["JES_jes"].up)
            jets_jec_JESDown = self.jet_awkward(jets_c["JES_jes"].down)
        # For data set these all to nominal so we can plot without switching all of the names
        else: 
            jets_jec_JERUp   = jets_jec
            jets_jec_JERDown = jets_jec
            jets_jec_JESUp   = jets_jec
            jets_jec_JESDown = jets_jec
            
        # save per event variables to a dataframe
        self.out_vars["ntracks"+out_label] = ak.num(tracks).to_list()
        self.out_vars["ngood_fastjets"+out_label] = ak.num(ak_inclusive_jets).to_list()
        if out_label == "":
            self.out_vars["ht"+out_label] = ak.sum(ak4jets.pt,axis=-1).to_list()
            self.out_vars["ht_JEC"+out_label] = ak.sum(jets_jec.pt,axis=-1).to_list()
            self.out_vars["ht_JEC"+out_label+"_JER_up"] = ak.sum(jets_jec_JERUp.pt,axis=-1).to_list()
            self.out_vars["ht_JEC"+out_label+"_JER_down"] = ak.sum(jets_jec_JERDown.pt,axis=-1).to_list()
            self.out_vars["ht_JEC"+out_label+"_JES_up"] = ak.sum(jets_jec_JESUp.pt,axis=-1).to_list()
            self.out_vars["ht_JEC"+out_label+"_JES_down"] = ak.sum(jets_jec.pt,axis=-1).to_list()

            if self.era == 2016 and self.scouting == 0:
                self.out_vars["HLT_PFHT900"+out_label] = events.HLT.PFHT900
            elif self.scouting == 0:
                self.out_vars["HLT_PFHT1050"+out_label] = events.HLT.PFHT1050
            self.out_vars["ngood_ak4jets"+out_label] = ak.num(ak4jets).to_list()
            if self.scouting == 1:
                self.out_vars["PV_npvs"+out_label] = ak.num(events.Vertex.x)
            else:
                if self.isMC:
                    self.out_vars["Pileup_nTrueInt"+out_label] = events.Pileup.nTrueInt
                    if len(events.PSWeight[0])==4:
                        self.out_vars["PSWeight"+out_label+"_ISR_up"] = events.PSWeight[:,0]
                        self.out_vars["PSWeight"+out_label+"_ISR_down"] = events.PSWeight[:,2]
                        self.out_vars["PSWeight"+out_label+"_FSR_up"] = events.PSWeight[:,1]
                        self.out_vars["PSWeight"+out_label+"_FSR_down"] = events.PSWeight[:,3]
                    else:
                        self.out_vars["PSWeight"+out_label] = events.PSWeight[:,0]
                self.out_vars["PV_npvs"+out_label] = events.PV.npvs
                self.out_vars["PV_npvsGood"+out_label] = events.PV.npvsGood
                
        # get gen SUEP mass
        SUEP_genMass = len(events)*[0]
        if self.isMC and not self.scouting:
            genParts = self.getGenTracks(events)
            genSUEP = genParts[(abs(genParts.pdgID) == 25)]
            # we need to grab the last SUEP in the chain for each event
            SUEP_genMass = [g[-1].mass if len(g) > 0 else 0 for g in genSUEP]
        self.out_vars["SUEP_genMass"+out_label] = SUEP_genMass
    
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
    
    def initializeColumns(self, label=""):
        # need to add these to dataframe when no events pass to make the merging work
        # for some reason, initializing these as empty and then trying to fill them doesn't work
        self.columns_IRM = [
                "SUEP_nconst_IRM", "SUEP_ntracks_IRM", 
                "SUEP_pt_avg_IRM", "SUEP_pt_avg_b_IRM",
                "SUEP_S1_IRM", "SUEP_rho0_IRM", "SUEP_rho1_IRM", 
                "SUEP_pt_IRM", "SUEP_eta_IRM", "SUEP_phi_IRM", "SUEP_mass_IRM",
                "dphi_SUEP_ISR_IRM"
        ]
        self.columns_CL = [c.replace("IRM", "CL") for c in self.columns_IRM]
        self.columns_CL_ISR = [c.replace("IRM", "CL".replace("SUEP", "ISR")) for c in self.columns_IRM]
        self.columns_ML = ["resnet_pred_{}".format(m) for m in self.models] 
        self.columns = self.columns_IRM + self.columns_CL + self.columns_CL_ISR
        
        # add a specific label to all columns
        for iCol in range(len(self.columns)): self.columns[iCol] = self.columns[iCol] + label
            
    def applyGoldenJSON(self, events):
        if self.era == 2016:
                LumiJSON = lumi_tools.LumiMask('data/GoldenJSON/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt')
        elif self.era == 2017:
            LumiJSON = lumi_tools.LumiMask('data/GoldenJSON/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt')
        elif self.era == 2018:
            LumiJSON = lumi_tools.LumiMask('data/GoldenJSON/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt')
        else:
            print('No era is defined. Please specify the year')

        events = events[LumiJSON(events.run, events.luminosityBlock)]
        
        return events
    
    def analysis(self, events, do_syst=False, col_label=""):
        #####################################################################################
        # ---- Trigger event selection
        # Cut based on ak4 jets to replicate the trigger
        #####################################################################################
        
        events = self.eventSelection(events)
        
        # output empty dataframe if no events pass trigger
        if len(events) == 0:
            print("No events passed trigger. Saving empty outputs.")
            self.out_vars = pd.DataFrame(['empty'], columns=['empty'])
            return
        
        #####################################################################################
        # ---- Track selection
        # Prepare the clean PFCand matched to tracks collection     
        #####################################################################################

        if self.scouting == 1: tracks, Cleaned_cands = self.getScoutingTracks(events)
        else: tracks, Cleaned_cands = self.getTracks(events)
            
        if self.isMC and do_syst:
            tracks = self.tracksSystematics(tracks)
            Cleaned_cands = self.tracksSystematics(Cleaned_cands)
        
        #####################################################################################
        # ---- FastJet reclustering
        # The jet clustering part
        #####################################################################################
        
        ak_inclusive_jets, ak_inclusive_cluster = self.FastJetReclustering(tracks, r=1.5, minPt=150)
        
        #####################################################################################
        # ---- Event level information
        #####################################################################################
                
        self.storeEventVars(events, tracks,
                            ak_inclusive_jets, ak_inclusive_cluster,
                            out_label=col_label)

        # indices of events in tracks, used to keep track which events pass selections
        indices = np.arange(0,len(tracks))
        
        # initialize the columns with all the variables that you want to fill
        self.initializeColumns(col_label)
 
        #####################################################################################
        # ---- ML Analysis
        #####################################################################################
        
        indices_ML = indices
        self.MLAnalysis(indices_ML, Cleaned_cands)
                
        #####################################################################################
        # ---- Cut Based Analysis
        #####################################################################################
        
        # remove events with at least 2 clusters (i.e. need at least SUEP and ISR jets for IRM)
        clusterCut = (ak.num(ak_inclusive_jets, axis=1)>1)
        ak_inclusive_cluster = ak_inclusive_cluster[clusterCut]
        ak_inclusive_jets = ak_inclusive_jets[clusterCut]
        tracks = tracks[clusterCut]
        indices = indices[clusterCut]
        
        # output file if no events pass selections, avoids errors later on
        if len(tracks) == 0:
            print("No events pass clusterCut.")
            for c in self.columns: self.out_vars[c] = np.nan
            return
        
        tracks, indices, topTwoJets = self.getTopTwoJets(tracks, indices, ak_inclusive_jets, ak_inclusive_cluster)
        SUEP_cand, ISR_cand, SUEP_cluster_tracks, ISR_cluster_tracks = topTwoJets
        
        #self.ISRRemovalMethod(indices, tracks, 
        #                      SUEP_cand, ISR_cand)
        
        self.ClusterMethod(indices, tracks, 
                           SUEP_cand, ISR_cand, 
                           SUEP_cluster_tracks, ISR_cluster_tracks, 
                           do_inverted=True,
                           out_label=col_label)
                
        # self.ConeMethod(indices, tracks, 
        #                 SUEP_cand, ISR_cand)

    def process(self, events):
        output = self.accumulator.identity()
        dataset = events.metadata['dataset']

        # gen weights
        if self.isMC and self.scouting==1: self.gensumweight = ak.num(events.PFcand.pt,axis=0)
        elif self.isMC: self.gensumweight = ak.sum(events.genWeight)
        
        # golden jsons for offline data
        if not self.isMC and self.scouting!=1: events = self.applyGoldenJSON(events)

        # run the anlaysis with the track systematics applied
        if self.isMC and self.do_syst:
            self.analysis(events, do_syst=True, col_label='_trackDOWN')
        
        # run the analysis
        self.analysis(events)
        
        # save the out_vars object as a Pandas DataFrame
        save_dfs(self, [self.out_vars],["vars"], events.behavior["__events_factory__"]._partition_key.replace("/", "_")+".hdf5")
        return output

    def postprocess(self, accumulator):
        return accumulator
