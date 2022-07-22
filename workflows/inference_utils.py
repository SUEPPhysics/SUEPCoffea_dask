import onnxruntime as ort
from numba import jit
import awkward as ak
import numpy as np


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


def run_inference(self, imgs, ort_sess):

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
