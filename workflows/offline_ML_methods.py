import numpy as np
import pandas as pd
import awkward as ak
import vector
vector.register_awkward()

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
        for model in self.models:
              ort_infs.update({model: ort.InferenceSession('data/onnx_models/resnet_{}_{}.onnx'.format(model,self.era))})
        # In order to avoid memory issues convert events to images and run inference in batches
        # also exploits the numba-compiled convert_to_images function
        for i in range(0, len(inf_cands), self.batch_size):
            if i + self.batch_size > len(inf_cands): self.batch_size = len(inf_cands) - i
            batch = inf_cands[i:i+self.batch_size]
            imgs = convert_to_images(self, batch)
            for model in self.models:
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
        from SUEPNet import Net

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for model_path, config in zip(self.models, self.configs):

            config = yaml.safe_load(open(out_dir+files[0]))

            suep = Net(out_dim=config['model_pref']['out_dim'], 
                    hidden_dim=config['model_pref']['hidden_dim']).to(device)
            suep.load_state_dict(torch.load(model_path)['model'])

            suep.eval()
            sigmoid = torch.nn.Sigmoid()
            
            results = np.array([])
            for i in range(0, len(events), self.batch_size):
                
                batch = events[i:i+self.batch_size]
                batch_SUEP_cand = SUEP_cands[i:i+self.batch_size]
                batch = GNN_convertEvents(self, batch, batch_SUEP_cand)
                
                Nlc = np.count_nonzero(batch[:,:,0])
                
                # convert to torch
                x = torch.from_numpy(batch[:,:][None])
                x_pf = torch.from_numpy(batch[:,:Nlc,:]).float()
                
                with torch.no_grad():
                    out = suep(x_pf, x_pf_batch)

                    # normalize the outputs
                    nn1 = out[0][:,0]
                    nn1 = sigmoid(nn1)
                    results = np.concatenate((results, nn1.cpu().numpy()))
                    
            self.out_vars[indices, model_path.replace(".pt","")] = results

    else:
        for c in self.columns_ML: self.out_vars[c] = np.nan
