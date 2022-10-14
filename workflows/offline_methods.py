import numpy as np
import pandas as pd
import awkward as ak
import vector
vector.register_awkward()

from workflows.math_utils import *
from workflows.inference_utils import *

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

def SSDMethod(self, indices, events, out_label=''):
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
        for model in self.ssd_models:
              ort_infs.update({model: ort.InferenceSession('data/onnx_models/resnet_{}_{}.onnx'.format(model,self.era))})
        # In order to avoid memory issues convert events to images and run inference in batches
        # also exploits the numba-compiled convert_to_images function
        for i in range(0, len(inf_cands), self.batch_size):
            if i + self.batch_size > len(inf_cands): self.batch_size = len(inf_cands) - i
            batch = inf_cands[i:i+self.batch_size]
            imgs = convert_to_images(self, batch)
            for model in self.ssd_models:
                batch_resnet_jets = run_inference(self, imgs, ort_infs[model])
                if i == 0: resnet_jets = batch_resnet_jets
                else: resnet_jets = np.concatenate((resnet_jets, batch_resnet_jets))    
                pred_dict.update({model: resnet_jets[:,1]}) #highest SUEP prediction per event

        for model in self.ssd_models:
            self.out_vars.loc[indices, model+"_ssd"+out_label] = pred_dict[model]
    else:
        for c in self.ssd_models: self.out_vars.loc[indices, c+"_ssd"+out_label] = np.nan

def DGNNMethod(self, indices, events, SUEP_tracks, SUEP_cand, out_label=''):
    #####################################################################################
    # ---- ML Analysis
    # Each event is converted into an input for the ML models. Using ONNX, we run
    # inference on each event to obtain a prediction of the class (SUEP or QCD).
    # N.B.: Conversion is done elsewhere.
    #####################################################################################

    if self.do_inf:    
        
        import yaml
        import torch
        from torch import nn
        from torch_geometric.data import Data

        from workflows.SUEPNet import Net        

        device = torch.device('cpu')

        # consistency check
        assert len(self.dgnn_model_names) == len(self.configs)
        
        for model_name, config in zip(self.dgnn_model_names, self.configs):
            
            model_path = 'data/' + model_name + '.pt'
            # initialize model with original configurations and import the weights
            config = yaml.safe_load(open(config))
            suep = Net(out_dim=config['model_pref']['out_dim'], 
                    hidden_dim=config['model_pref']['hidden_dim']).to(device)
            suep.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model'])
            suep = suep.float()
            suep.eval()
            sigmoid = torch.nn.Sigmoid()
            
            results = np.array([])
            for i in range(0, len(events), self.batch_size):
                
                # define batch and convert objects in a coordinate frame
                batch = events[i:i+self.batch_size]
                this_batch_size = len(batch)
                batch_SUEP_cand = SUEP_cand[i:i+self.batch_size]
                batch = GNN_convertEvents(self, batch, batch_SUEP_cand)
                
                with torch.no_grad():
                    
                    # count how many tracks per event are nonzero: dimensions of (events, tracks)
                    Nlc = np.count_nonzero(batch[:,:,0], axis=1)
                    
                    # reshape the batch for correct DGNN format
                    x_pf = None  # dims: (events times tracks, 4)
                    x_pf_batch = None   #dim: (events times tracks)
                    for idx in range(this_batch_size):
                        # flatten the batch
                        if idx == 0: x_pf = batch[idx, :Nlc[idx], :]
                        else: x_pf = np.vstack((x_pf, batch[idx, :Nlc[idx], :]))
                        # event indices
                        if idx == 0: x_pf_batch = np.ones(Nlc[idx])*idx
                        else: x_pf_batch = np.concatenate((x_pf_batch, np.ones(Nlc[idx])*idx))

                    # convert to torch
                    x_pf = torch.from_numpy(x_pf).float()
                    x_pf_batch = torch.from_numpy(x_pf_batch).float()

                    # batch predictions
                    out = suep(x_pf, x_pf_batch)

                    # normalize the outputs
                    nn1 = out[0][:,0]
                    nn1 = sigmoid(nn1)
                    results = np.concatenate((results, nn1.cpu().numpy()))
            
            self.out_vars.loc[indices, model_name + "_GNN"+out_label] = results
            
            eigs = sphericity(SUEP_tracks, 1.0) #Set r=1.0 for IRC safe
            self.out_vars.loc[indices, "SUEP_nconst_GNN"+out_label] = ak.num(SUEP_tracks)
            self.out_vars.loc[indices, "SUEP_S1_GNN"+out_label] = 1.5 * (eigs[:,1]+eigs[:,0])

    else:
        for c in self.dgnn_model_names: self.out_vars.loc[indices, c+"_GNN"+out_label] = np.nan
