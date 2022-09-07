import numpy as np
import awkward as ak

def sphericity(particles, r):
    norm = ak.sum(particles.p ** r, axis=1, keepdims=True)
    s = np.array([[
                   ak.sum(particles.px * particles.px * particles.p ** (r-2.0), axis=1 ,keepdims=True)/norm,
                   ak.sum(particles.px * particles.py * particles.p ** (r-2.0), axis=1 ,keepdims=True)/norm,
                   ak.sum(particles.px * particles.pz * particles.p ** (r-2.0), axis=1 ,keepdims=True)/norm
                  ],
                  [
                   ak.sum(particles.py * particles.px * particles.p ** (r-2.0), axis=1 ,keepdims=True)/norm,
                   ak.sum(particles.py * particles.py * particles.p ** (r-2.0), axis=1 ,keepdims=True)/norm,
                   ak.sum(particles.py * particles.pz * particles.p ** (r-2.0), axis=1 ,keepdims=True)/norm
                  ],
                  [
                   ak.sum(particles.pz * particles.px * particles.p ** (r-2.0), axis=1 ,keepdims=True)/norm,
                   ak.sum(particles.pz * particles.py * particles.p ** (r-2.0), axis=1 ,keepdims=True)/norm,
                   ak.sum(particles.pz * particles.pz * particles.p ** (r-2.0), axis=1 ,keepdims=True)/norm
                  ]])
    s = np.squeeze(np.moveaxis(s, 2, 0),axis=3)
    evals = np.sort(np.linalg.eigvalsh(s))
    return evals

def rho(number, jet, tracks, deltaR, dr=0.05):
    r_start = number*dr
    r_end = (number+1)*dr
    ring = (deltaR > r_start) & (deltaR < r_end)
    rho_values = ak.sum(tracks[ring].pt, axis=1)/(dr*jet.pt)
    return rho_values
