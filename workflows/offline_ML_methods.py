import numpy as np
import pandas as pd
import awkward as ak
import vector
vector.register_awkward()

from workflows.inference_utils import *

def SSDMethod(self, indices, events):
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

    else:
        for c in self.columns_ML: self.out_vars[c] = np.nan

def DGNNMethod(self, indices, events, SUEP_cand):
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
        assert len(self.dgnn_models) == len(self.configs)
        
        for model_path, config in zip(self.dgnn_models, self.configs):
            
            # initialize model with original configurations and import the weights
            config = yaml.safe_load(open(config))
            suep = Net(out_dim=config['model_pref']['out_dim'], 
                    hidden_dim=config['model_pref']['hidden_dim']).to(device)
            suep.load_state_dict(torch.load('data/epoch-49.pt', map_location=torch.device('cpu'))['model'])
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
            
            self.out_vars.loc[indices, model_path] = results

    else:
        for c in self.columns_ML: self.out_vars[c] = np.nan
