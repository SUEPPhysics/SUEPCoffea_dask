import h5py
import torch
import torch.cuda as tcuda

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

        idx_PFCands_Eta = tcuda.LongTensor([self.PFCands_Eta[index]],
                                           device=self.rank)
        idx_PFCands_Phi = tcuda.LongTensor([self.PFCands_Phi[index]],
                                           device=self.rank)
        val_PFCands_PT = tcuda.FloatTensor(self.ePFCands_PT[index],
                                           device=self.rank)

        images, scaler = self.process_images(idx_PFCands_Eta,
                                                  idx_PFCands_Phi,
                                                  val_PFCands_PT)
        # Set labels
        labels = tcuda.FloatTensor(self.labels[index], device=self.rank)
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

        self.eFTrack_Eta = self.hdf5_dataset['EFlowTrack_Eta']
        self.eFTrack_Phi = self.hdf5_dataset['EFlowTrack_Phi']
        self.eFTrack_PT = self.hdf5_dataset['EFlowTrack_PT']

        self.eFPhoton_Eta = self.hdf5_dataset['EFlowPhoton_Eta']
        self.eFPhoton_Phi = self.hdf5_dataset['EFlowPhoton_Phi']
        self.eFPhoton_ET = self.hdf5_dataset['EFlowPhoton_ET']

        self.eFNHadron_Eta = self.hdf5_dataset['EFlowNeutralHadron_Eta']
        self.eFNHadron_Phi = self.hdf5_dataset['EFlowNeutralHadron_Phi']
        self.eFNHadron_ET = self.hdf5_dataset['EFlowNeutralHadron_ET']

        self.labels = self.hdf5_dataset['labels']
        self.base = self.hdf5_dataset['baseline']

        self.dataset_size = len(self.labels)

    def process_images(self,
                       idx_Phi, idx_Eta, idx_Pt):

        i = torch.cat((idx_Eta, idx_Phi), 0)
        v = torch.cat((idx_Pt))

        scaler = torch.max(v)
        pixels = torch.sparse.FloatTensor(i, v, torch.Size([self.channels,
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
        labels_reshaped = labels_raw.reshape(-1, 4)
        labels = torch.empty_like(labels_reshaped)

        # Set fractional coordinates
        labels[:, 0] = (labels_reshaped[:, 1] - self.size) / float(self.width)
        labels[:, 1] = (labels_reshaped[:, 2] - self.size) / float(self.height)
        labels[:, 2] = (labels_reshaped[:, 1] + self.size) / float(self.width)
        labels[:, 3] = (labels_reshaped[:, 2] + self.size) / float(self.height)

        # Set class label
        labels = torch.cat((labels, labels_reshaped[:, 0].unsqueeze(1) + 1), 1)

        if self.return_pt:
            pts = labels_reshaped[:, 3].unsqueeze(1)
            if not self.raw:
                pts = pts / scaler
            labels = torch.cat((labels, pts), 1)

        return labels
