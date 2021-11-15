import h5py
import torch
import torch.cuda as tcuda
import numpy as np

from ssd import qutils


class CalorimeterJetDataset(torch.utils.data.Dataset):

    def __init__(self,
                 rank,
                 hdf5_source_path,
                 input_dimensions,
                 jet_size,
                 cpu=False,
                 flip_prob=None,
                 raw=False,
                 return_baseline=False,
                 return_pt=False):
        """Generator for calorimeter and jet data"""

        self.rank = rank
        self.source = hdf5_source_path
        self.channels = input_dimensions[0]  # Number of channels
        self.width = input_dimensions[1]  # Width of input
        self.height = input_dimensions[2]  # Height of input
        self.size = jet_size / 2
        self.flip_prob = flip_prob
        self.cpu = cpu
        self.raw = raw
        self.return_baseline = return_baseline
        self.return_pt = return_pt
        
    def __getitem__(self, index):

        if not hasattr(self, 'hdf5_dataset'):
            self.open_hdf5()

        idx_PFCands_Eta = tcuda.FloatTensor(np.array([self.cands_eta[index]]),
                                           device=self.rank)
        idx_PFCands_Phi = tcuda.FloatTensor(np.array([self.cands_phi[index]]),
                                           device=self.rank)
        val_PFCands_PT = tcuda.FloatTensor(np.array([self.cands_pt[index]]),
                                           device=self.rank)

        images, scaler = self.process_images(idx_PFCands_Eta,
                                                  idx_PFCands_Phi,
                                                  val_PFCands_PT)
        # Set labels
        labels = tcuda.FloatTensor(np.array([self.labels[index]]), device=self.rank)
        labels = self.process_labels(labels, scaler)

        if self.flip_prob:
            if torch.rand(1) < self.flip_prob:
                images, labels = self.flip_image(images,
                                                      labels,
                                                      vertical=True)
            if torch.rand(1) < self.flip_prob:
                images, labels = self.flip_image(images,
                                                      labels,
                                                      vertical=False)

        if self.cpu:
            images = images.cpu()
            labels = labels.cpu()
            scaler = scaler.cpu()

        if self.return_baseline:
            base = tcuda.FloatTensor(self.base[index], device=self.rank)
            base = self.process_baseline(base)
            return images, labels, base, scaler

        return images, labels

    def __len__(self):

        if not hasattr(self, 'hdf5_dataset'):
            self.open_hdf5()

        return self.dataset_size

    ### FIXME: this is broken for labels
    def flip_image(self, image, labels, vertical=True):
        if vertical:
            axis = 1
            labels[:, [0, 2]] = 1 - labels[:, [0, 2]]
            labels = labels[:, [2, 1, 0, 3, 4, 5]]
        else:
            axis = 2
            labels[:, [1, 3]] = 1 - labels[:, [1, 3]]
            labels = labels[:, [0, 3, 2, 1, 4, 5]]
        image = torch.flip(image, [axis])
        return image, labels

    def normalize(self, tensor):
        m = torch.mean(tensor)
        s = torch.std(tensor)
        return tensor.sub(m).div(s)

    def open_hdf5(self):
        self.hdf5_dataset = h5py.File(self.source, 'r')

        self.cands_phi = self.hdf5_dataset['phi']
        self.cands_eta = self.hdf5_dataset['eta']
        self.cands_pt = self.hdf5_dataset['pt']
        self.labels = self.hdf5_dataset['labels']
        
        ### FIXME: what is this
        ### guess: baseline predictions (i.e. for us the ISR removal method predictions)
        # self.base = self.hdf5_dataset['baseline']

        self.dataset_size = len(self.labels)
        
    def pixelize(self, idx_Eta, idx_Phi):
        """"
        This method is used to convert eta and phi values to a pixelation
        of dimensions (n_eta, n_phi): these dimensions are obtained from the
        ssd-config.yml file.
        We assume eta in [-2.5,2.5], phi in [-pi, pi]
        """
        
        n_eta = self.height
        n_phi = self.width
        min_eta = -2.51
        max_eta = 2.51
        min_phi = -np.pi
        max_phi = np.pi
        
        idx_Eta_discrete = (n_eta/(max_eta - min_eta)) * (idx_Eta - min_eta)
        idx_Phi_discrete = (n_phi/(max_phi - min_phi)) * (idx_Phi - min_phi)
        
        idx_Eta_discrete = idx_Eta_discrete.floor().int()
        idx_Phi_discrete = idx_Phi_discrete.floor().int()
        
        ##### debug
        if torch.max(idx_Eta_discrete) >= 100:
            print("MAX ETA", torch.max(idx_Eta))
            sys.exit()
        if torch.max(idx_Phi_discrete) >= 100:
            print("MAX PHI", torch.max(idx_Phi))
            sys.exit()
        
        return idx_Eta_discrete, idx_Phi_discrete
        
    def process_images(self, idx_Eta, idx_Phi, idx_Pt):
        
        idx_Eta, idx_Phi = self.pixelize(idx_Eta, idx_Phi)
        
        v0 = 0*torch.ones(idx_Eta.size(1),
                          dtype=torch.long).cuda(self.rank)
        idx_channels = v0.unsqueeze(0)
        i = torch.cat((idx_channels, idx_Eta, idx_Phi), 0)
        v = idx_Pt

        scaler = torch.max(v)
        
        pixels = torch.sparse.FloatTensor(i, v[0,:], torch.Size([self.channels,
                                                            self.width,
                                                            self.height]))
        pixels = pixels.to_dense()
        pixels = self.normalize(pixels)
        return pixels, scaler

    def process_baseline(self, base_raw):
        base_reshaped = base_raw.reshape(-1, 5)
        base = torch.empty_like(base_reshaped)

        # Set fractional coordinates
        base[:, 0] = base_reshaped[:, 1] / float(self.width)
        base[:, 1] = base_reshaped[:, 2] / float(self.height)

        # Set class label
        base[:, 2] = base_reshaped[:, 0] + 1

        # Add score
        base[:, 3] = 1 - base_reshaped[:, 4]

        # Add truth
        base[:, 4] = 0

        # Add pT
        base = torch.cat((base, base_reshaped[:, 3].unsqueeze(1)), 1)

        return base

    def process_labels(self, labels_raw, scaler):
                        
        # returns [[isSUEP, phi,eta,pt], [isSUEP2, phi2,eta2,pt2], ...]
        labels_reshaped = labels_raw.reshape(-1,4)
        labels_classes = labels_reshaped[:,0].unsqueeze(1)
        
        eta, phi = self.pixelize(labels_reshaped[:,2],labels_reshaped[:,1])
        labels_reshaped[:,1] = phi
        labels_reshaped[:,2] = eta
                
        # (Number of jets, 4)
        labels = torch.empty((labels_reshaped.size(0), 4), dtype=labels_reshaped.dtype, device=labels_reshaped.device)
        
        ### FIXME: no idea why we need to divide by dimensions????
        # Set fractional coordinates
        # labels[:, 0] = (labels_reshaped[:, 0] - self.size) / float(self.width)
        # labels[:, 1] = (labels_reshaped[:, 1] - self.size) / float(self.height)
        # labels[:, 2] = (labels_reshaped[:, 0] + self.size) / float(self.width)
        # labels[:, 3] = (labels_reshaped[:, 1] + self.size) / float(self.height)
        labels[:, 0] = (labels_reshaped[:, 1] - self.size) 
        labels[:, 1] = (labels_reshaped[:, 2] - self.size)
        labels[:, 2] = (labels_reshaped[:, 1] + self.size) 
        labels[:, 3] = (labels_reshaped[:, 2] + self.size)

        # Set class label
        labels = torch.cat((labels, labels_classes), 1)

        if self.return_pt:
            pts = labels_reshaped[:, 3].unsqueeze(1)
            if not self.raw:
                pts = pts / scaler
            labels = torch.cat((labels, pts), 1)

        return labels
