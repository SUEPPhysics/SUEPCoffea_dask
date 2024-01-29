import awkward as ak
import numpy as np
import onnxruntime as ort
import vector
from numba import jit

vector.register_awkward()

import torch
from torch import nn
from torch_geometric.nn.conv import DynamicEdgeConv
from torch_geometric.nn.pool import avg_pool_x

import workflows.SUEP_utils as SUEP_utils


def SSDMethod(self, indices, events, out_label=""):
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
        options.inter_op_num_threads = 1  # number of threads used to parallelize the execution of the graph (across nodes). Default is 0 to let onnxruntime choose.
        for model in self.ssd_models:
            ort_infs.update(
                {
                    model: ort.InferenceSession(
                        f"data/onnx_models/resnet_{model}_{self.era}.onnx"
                    )
                }
            )
        # In order to avoid memory issues convert events to images and run inference in batches
        # also exploits the numba-compiled convert_to_images function
        for i in range(0, len(inf_cands), self.batch_size):
            if i + self.batch_size > len(inf_cands):
                self.batch_size = len(inf_cands) - i
            batch = inf_cands[i : i + self.batch_size]
            imgs = convert_to_images(self, batch)
            for model in self.ssd_models:
                batch_resnet_jets = run_inference_SSD(self, imgs, ort_infs[model])
                if i == 0:
                    resnet_jets = batch_resnet_jets
                else:
                    resnet_jets = np.concatenate((resnet_jets, batch_resnet_jets))
                pred_dict.update(
                    {model: resnet_jets[:, 1]}
                )  # highest SUEP prediction per event

        for model in self.ssd_models:
            self.out_vars.loc[indices, model + "_ssd" + out_label] = pred_dict[model]


def DGNNMethod(
    self,
    indices,
    SUEP_tracks,
    SUEP_cand,
    ISR_tracks=None,
    ISR_cand=None,
    out_label="",
    do_inverted=False,
):
    #####################################################################################
    # ---- ML Analysis
    # Each event is converted into an input for the ML models. Using ONNX, we run
    # inference on each event to obtain a prediction of the class (SUEP or QCD).
    # N.B.: Conversion is done elsewhere.
    #####################################################################################

    if self.do_inf:
        import yaml

        device = torch.device("cpu")

        modelDir = "data/GNN/"

        # consistency check
        assert len(self.dgnn_model_names) == len(self.configs)

        for model_name, config in zip(self.dgnn_model_names, self.configs):
            model_path = modelDir + model_name + ".pt"

            # initialize model with original configurations and import the weights
            config = yaml.safe_load(open(modelDir + config))
            suep = SUEPNet(
                out_dim=config["model_pref"]["out_dim"],
                hidden_dim=config["model_pref"]["hidden_dim"],
            ).to(device)
            suep.load_state_dict(
                torch.load(model_path, map_location=torch.device("cpu"))["model"]
            )
            suep = suep.float()
            suep.eval()

            # run GNN inference on the SUEP tracks
            results = run_inference_GNN(self, suep, SUEP_tracks, SUEP_cand)
            self.out_vars.loc[indices, "SUEP_" + model_name + "_GNN" + out_label] = (
                results
            )

            # calculate other obserables to store
            boost_SUEP = ak.zip(
                {
                    "px": SUEP_cand.px * -1,
                    "py": SUEP_cand.py * -1,
                    "pz": SUEP_cand.pz * -1,
                    "mass": SUEP_cand.mass,
                },
                with_name="Momentum4D",
            )
            SUEP_tracks_b = SUEP_tracks.boost_p4(boost_SUEP)
            eigs = SUEP_utils.sphericity(SUEP_tracks_b, 1.0)  # Set r=1.0 for IRC safe
            self.out_vars.loc[indices, "SUEP_nconst_GNN" + out_label] = ak.num(
                SUEP_tracks
            )
            self.out_vars.loc[indices, "SUEP_S1_GNN" + out_label] = 1.5 * (
                eigs[:, 1] + eigs[:, 0]
            )

            if do_inverted:
                # run GNN inference on the SUEP tracks
                results = run_inference_GNN(self, suep, ISR_tracks, ISR_cand)
                self.out_vars.loc[indices, "ISR_" + model_name + "_GNN" + out_label] = (
                    results
                )

                # calculate other obserables to store
                boost_ISR = ak.zip(
                    {
                        "px": ISR_cand.px * -1,
                        "py": ISR_cand.py * -1,
                        "pz": ISR_cand.pz * -1,
                        "mass": ISR_cand.mass,
                    },
                    with_name="Momentum4D",
                )
                ISR_tracks_b = ISR_tracks.boost_p4(boost_ISR)
                eigs = SUEP_utils.sphericity(
                    ISR_tracks_b, 1.0
                )  # Set r=1.0 for IRC safe
                self.out_vars.loc[indices, "ISR_nconst_GNN" + out_label] = ak.num(
                    ISR_tracks
                )
                self.out_vars.loc[indices, "ISR_S1_GNN" + out_label] = 1.5 * (
                    eigs[:, 1] + eigs[:, 0]
                )


def run_inference_GNN(self, model, tracks, SUEP_cand):
    results = np.array([])
    for i in range(0, len(tracks), self.batch_size):
        # define batch and convert objects in a coordinate frame
        batch = tracks[i : i + self.batch_size]
        this_batch_size = len(batch)
        batch_SUEP_cand = SUEP_cand[i : i + self.batch_size]
        batch = GNN_convertEvents(self, batch, batch_SUEP_cand)

        sigmoid = torch.nn.Sigmoid()
        with torch.no_grad():
            # count how many tracks per event are nonzero: dimensions of (events, tracks)
            Nlc = np.count_nonzero(batch[:, :, 0], axis=1)

            # reshape the batch for correct DGNN format
            x_pf = None  # dims: (events times tracks, 4)
            x_pf_batch = None  # dim: (events times tracks)
            for idx in range(this_batch_size):
                # flatten the batch
                if idx == 0:
                    x_pf = batch[idx, : Nlc[idx], :]
                else:
                    x_pf = np.vstack((x_pf, batch[idx, : Nlc[idx], :]))
                # event indices
                if idx == 0:
                    x_pf_batch = np.ones(Nlc[idx]) * idx
                else:
                    x_pf_batch = np.concatenate((x_pf_batch, np.ones(Nlc[idx]) * idx))

            # convert to torch
            x_pf = torch.from_numpy(x_pf).float()
            x_pf_batch = torch.from_numpy(x_pf_batch).float()

            # batch predictions
            out = model(x_pf, x_pf_batch)

            # normalize the outputs
            nn1 = out[0][:, 0]
            nn1 = sigmoid(nn1)
            results = np.concatenate((results, nn1.cpu().numpy()))

    return results


class SUEPNet(nn.Module):
    def __init__(self, out_dim=1, hidden_dim=16):
        super().__init__()

        # hidden_dim = 16
        # out_dim = 2

        self.pf_encode = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )

        self.conv = DynamicEdgeConv(
            nn=nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ELU()), k=8
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 8),
            nn.ELU(),
            nn.Linear(8, 4),
            nn.ELU(),
            nn.Linear(4, out_dim),
            # nn.Sigmoid()
        )

    def forward(self, x_pf, batch_pf):
        x_pf_enc = self.pf_encode(x_pf)

        # create a representation of PFs to PFs
        feats1 = self.conv(x=(x_pf_enc, x_pf_enc), batch=(batch_pf, batch_pf))
        feats2 = self.conv(x=(feats1, feats1), batch=(batch_pf, batch_pf))

        batch = batch_pf
        out, batch = avg_pool_x(batch, feats2, batch)

        out = self.output(out)

        return out, batch


@jit(forceobj=True)
def convert_to_images(self, events):
    # Turn the PFcand info into indexes on the image map
    idx_eta = ak.values_astype(
        np.floor((events.eta - self.eta_span[0]) * self.eta_scale), "int64"
    )
    idx_phi = ak.values_astype(
        np.floor((events.phi - self.phi_span[0]) * self.phi_scale), "int64"
    )
    idx_eta = ak.where(idx_eta == self.eta_pix, self.eta_pix - 1, idx_eta)
    idx_phi = ak.where(idx_phi == self.phi_pix, self.phi_pix - 1, idx_phi)
    pt = events.pt

    to_infer = np.zeros((len(events), 1, self.eta_pix, self.phi_pix))
    for event_i in range(len(events)):
        # form image
        to_infer[event_i, 0, idx_eta[event_i], idx_phi[event_i]] = pt[event_i]

        # normalize pt
        m = np.mean(to_infer[event_i, 0, :, :])
        s = np.std(to_infer[event_i, 0, :, :])
        if s != 0:
            to_infer[event_i, 0, :, :] = (to_infer[event_i, 0, :, :] - m) / s

    return to_infer


def run_inference_SSD(self, imgs, ort_sess):
    # Running the inference in batch mode
    input_name = ort_sess.get_inputs()[0].name
    cl_outputs = np.array([])

    for i, img in enumerate(imgs):
        # SSD: grab classification outputs (0 - loc, 1 - classifation, 2 - regression)
        # resnet: only classification as output
        cl_output = ort_sess.run(None, {input_name: np.array([img.astype(np.float32)])})
        cl_output_softmax = softmax(cl_output)[0]
        if i == 0:
            cl_outputs = cl_output_softmax
        else:
            cl_outputs = np.concatenate((cl_outputs, cl_output_softmax))

    return cl_outputs


def softmax(data):
    # some numpy magic
    return np.exp(data) / (np.exp(data).sum(axis=-1)[:, :, None])


def GNN_convertEvents(self, events, SUEP_cand, max_objects=1000):
    allowed_objects = ["pfcand", "bpfcand"]
    if self.obj.lower() not in allowed_objects:
        raise Exception(self.obj + " is not supported in GNN_convertEvents.")

    if self.obj.lower() == "pfcand":
        events = events

    elif self.obj.lower() == "bpfcand":
        boost_SUEP = ak.zip(
            {
                "px": SUEP_cand.px * -1,
                "py": SUEP_cand.py * -1,
                "pz": SUEP_cand.pz * -1,
                "mass": SUEP_cand.mass,
            },
            with_name="Momentum4D",
        )
        events = events.boost_p4(boost_SUEP)

    else:
        raise Exception()

    new_events = SUEP_utils.convert_coords(self.coords, events, max_objects)

    return new_events
