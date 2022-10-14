import numpy as np
import pandas as pd
import awkward as ak
import vector
vector.register_awkward()

from workflows.math_utils import *
from workflows.inference_utils import *

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
