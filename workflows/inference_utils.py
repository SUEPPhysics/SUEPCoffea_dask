import onnxruntime as ort
from numba import jit
import awkward as ak
import numpy as np

@jit(forceobj=True)
def convert_to_images(self, events):
    
     #Turn the PFcand info into indexes on the image map
    idx_eta = ak.values_astype(np.floor((events.eta-self.eta_span[0])*self.eta_scale),"int64")
    idx_phi = ak.values_astype(np.floor((events.phi-self.phi_span[0])*self.phi_scale),"int64")
    idx_eta = ak.where(idx_eta == self.eta_pix, self.eta_pix-1, idx_eta)
    idx_phi = ak.where(idx_phi == self.phi_pix, self.phi_pix-1, idx_phi)
    pt = events.pt
    
    to_infer = np.zeros((len(events), 1, self.eta_pix, self.phi_pix))    
    for event_i in range(len(events)):
        
        # form image
        to_infer[event_i, 0, idx_eta[event_i],idx_phi[event_i]] = pt[event_i]  
        
        # normalize pt
        m = np.mean(to_infer[event_i, 0,:,:])
        s = np.std(to_infer[event_i, 0,:,:])
        if s != 0: 
            to_infer[event_i, 0,:,:] = (to_infer[event_i, 0,:,:]-m)/s
            
    return to_infer

def run_inference(self, imgs, ort_sess):

    #Running the inference in batch mode
    input_name = ort_sess.get_inputs()[0].name
    cl_outputs = np.array([])
               
    for i, img in enumerate(imgs):
        # SSD: grab classification outputs (0 - loc, 1 - classifation, 2 - regression)
        # resnet: only classification as output
        cl_output =  ort_sess.run(None, {input_name: np.array([img.astype(np.float32)])})
        cl_output_softmax = softmax(cl_output)[0]
        if i == 0: 
            cl_outputs = cl_output_softmax
        else: 
            cl_outputs = np.concatenate((cl_outputs, cl_output_softmax))

    return  cl_outputs

def softmax(data):
    # some numpy magic
    return np.exp(data)/(np.exp(data).sum(axis=-1)[:,:,None])

def GNN_convertEvents(self, events, SUEP_cand, max_objects=1000):
    
    allowed_objects = ['pfcand', 'bpfcand']
    if self.obj.lower() not in allowed_objects: raise Exception(self.obj + " is not supported in GNN_convertEvents.")
    
    if self.obj.lower() == 'pfcand':
        events = events
        
    elif self.obj.lower() == 'bpfcand':
        boost_SUEP = ak.zip({
            "px": SUEP_cand.px*-1,
            "py": SUEP_cand.py*-1,
            "pz": SUEP_cand.pz*-1,
            "mass": SUEP_cand.mass
        }, with_name="Momentum4D")    
        events = events.boost_p4(boost_SUEP)    
        
    new_events = convert_coords(self.coords, events, max_objects)
    
    return new_events

def convert_coords(coords, tracks, nobj):
    allowed_coords = ['cyl', 'cart', 'p4']
    if coords.lower() not in allowed_coords: raise Exception(self.coords + " is not supported in GNN_convertEvents.")
    
    if coords.lower() == 'p4': new_tracks = convert_p4(tracks, nobj)
    elif coords.lower() == 'cart': new_tracks = convert_cart(tracks, nobj)
    elif coords.lower() == 'cyl': new_tracks = convert_cyl(tracks, nobj)
    
    return new_tracks
    
def convert_p4(tracks, nobj=10):
    '''store objects in zero-padded numpy arrays'''

    # bad naming, tracks dimensions is (events x tracks)
    nentries = len(tracks)

    l1Obj_p4 = np.zeros((nentries,nobj,4))

    px = to_np_array(tracks.px,maxN=nobj)
    py = to_np_array(tracks.py,maxN=nobj)
    pz = to_np_array(tracks.pz,maxN=nobj)
    m = to_np_array(tracks.mass,maxN=nobj)

    l1Obj_p4[:,:,0] = px
    l1Obj_p4[:,:,1] = py
    l1Obj_p4[:,:,2] = pz
    l1Obj_p4[:,:,3] = m

    return l1Obj_p4

def convert_cyl(tracks, nobj=10):
    '''store objects in zero-padded numpy arrays'''

    # bad naming, tracks dimensions is (events x tracks)
    nentries = len(tracks)

    l1Obj_cyl = np.zeros((nentries,nobj,4))

    pt = to_np_array(tracks.pt,maxN=nobj)
    eta = to_np_array(tracks.eta,maxN=nobj)
    phi = to_np_array(tracks.phi,maxN=nobj)
    m = to_np_array(tracks.mass,maxN=nobj)

    l1Obj_cyl[:,:,0] = pt
    l1Obj_cyl[:,:,1] = eta
    l1Obj_cyl[:,:,2] = phi
    l1Obj_cyl[:,:,3] = m

    return l1Obj_cyl

def convert_cart(tracks, nobj=10):
    '''store objects in zero-padded numpy arrays'''

    # bad naming, tracks dimensions is (events x tracks)
    nentries = len(tracks)

    l1Obj_cart = np.zeros((nentries,nobj,4))

    pt = to_np_array(tracks.pt,maxN=nobj)
    eta = to_np_array(tracks.eta,maxN=nobj)
    phi = to_np_array(tracks.phi,maxN=nobj)
    m = to_np_array(tracks.mass,maxN=nobj)

    l1Obj_cart[:,:,0] = pt*np.cos(phi)
    l1Obj_cart[:,:,1] = pt*np.sin(phi)
    l1Obj_cart[:,:,2] = pt*np.sinh(eta)
    l1Obj_cart[:,:,3] = m

    return l1Obj_cart

def to_np_array(ak_array, maxN=100, pad=0):
    '''convert awkward array to regular numpy array'''
    return ak.to_numpy(ak.fill_none(ak.pad_none(ak_array,maxN,clip=True,axis=-1),pad))