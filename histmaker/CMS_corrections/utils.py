import numpy as np
import hist
import boost_histogram as bh

def apply_binwise_scaling(h_in, bins, scales, dim="x"):
    """
    Apply scales to bins of a particular histogram.
    """
    h_npy = h_in.to_numpy()
    if len(h_npy) == 2:
        h_fragments = []
        for iBin in range(len(bins) - 1):
            h_fragments.append(h_in[bins[iBin] : bins[iBin + 1]] * scales[iBin])
        h = hist.Hist(hist.axis.Variable(h_npy[1]), storage=bh.storage.Weight())
        new_z = np.concatenate([f.to_numpy()[0] for f in h_fragments])
        h[:] = np.stack([new_z, new_z], axis=-1)
    elif len(h_npy) == 3:
        h_fragments = []
        for iBin in range(len(bins) - 1):
            if dim == "x":
                h_fragments.append(h_in[bins[iBin] : bins[iBin + 1], :] * scales[iBin])
            elif dim == "y":
                h_fragments.append(h_in[:, bins[iBin] : bins[iBin + 1]] * scales[iBin])
        h = hist.Hist(
            hist.axis.Variable(h_npy[1]),
            hist.axis.Variable(h_npy[2]),
            storage=bh.storage.Weight(),
        )
        if dim == "x":
            new_z = np.concatenate([f.to_numpy()[0] for f in h_fragments])
        if dim == "y":
            new_z = np.hstack([f.to_numpy()[0] for f in h_fragments])
        h[:, :] = np.stack([new_z, new_z], axis=-1)
    return h