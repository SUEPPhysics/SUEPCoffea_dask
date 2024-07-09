"""
Make event displays for SUEP events.
This script gets objects from a ROOT file and displays the ones that you tell it to, via the arguments.
Event selection is NOT done here, you should pass exactly the (event, run, lumi block) that you want to plot.

e.g.
python additional_tools/eventDisplay.py
        --file events.txt
        --input ../WHleptonicpythia_leptonic_M125.0_MD8.00_T8.00_HT-1_UL17_NANOAOD.root
        --output ~/public_html/SUEP/event_displays/
        --leptons --MET --pfcands --WHCandidate
        --channel WH

Author: Luca Lavezzo, based on an older version by Karri di Petrillo
Date: January 2024
"""

import argparse
import os
import re
import sys
from functools import reduce
from math import pi

import awkward as ak
import fastjet
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot
import vector

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from workflows import SUEP_utils

vector.register_awkward()


decaysLabels = {
    "hadronic": r"$A^' \rightarrow e^{+}e^{-}$ ($15\%$), $\mu^{+}\mu^{-}$ ($15\%$), $\pi^{+}\pi^{-}$ ($70\%$)",
    "leptonic": r"$A^' \rightarrow e^{+}e^{-}$ ($40\%$), $\mu^{+}\mu^{-}$ ($40\%$), $\pi^{+}\pi^{-}$ ($20\%$)",
    "generic": r"$A^' \rightarrow \pi^{+}\pi^{-}$ ($100\%$) ",
}


def get_dr_ring(dr, phi_c=0, eta_c=0, n_points=600):
    deta = np.linspace(-dr, +dr, n_points)
    dphi = np.sqrt(dr**2 - np.square(deta))
    deta = eta_c + np.concatenate((deta, deta[::-1]))
    dphi = phi_c + np.concatenate((dphi, -dphi[::-1]))
    return dphi, deta


def get_branch_mask(tree, eventsToPlot):
    cuts = []
    _events = tree["event"].array()
    _run = tree["run"].array()
    _lumiBlock = tree["luminosityBlock"].array()
    for i in range(len(eventsToPlot)):
        cut_i = (
            (eventsToPlot[i, 0] == _events)
            & (eventsToPlot[i, 1] == _run)
            & (eventsToPlot[i, 2] == _lumiBlock)
        )
        cuts.append(cut_i)
    combined_cut = reduce(lambda x, y: x | y, cuts)
    return combined_cut


def get_branch(tree, branchname):
    array = tree[branchname].array()
    if branch_mask is not None:
        array = array[branch_mask]
    return array


def getParamsFromSampleName(sample):
    pattern = r"_T(\d+p?\d*)_mS(\d+\.\d+)_mPhi(\d+\.\d+)_T(\d+\.\d+)_mode(\w+)_TuneCP5"

    # Use re.search to find the first occurrence of the pattern in the sample name
    match = re.search(pattern, sample)

    if match:
        # Extract the matched groups and convert them to the appropriate data types
        temp = float(match.group(1).replace("p", "."))
        mS = float(match.group(2))
        mPhi = float(match.group(3))
        decay = match.group(5)

        # Return the extracted parameters as a tuple
        return mS, mPhi, temp, decay
    else:
        # Return None if no match is found
        return None


def getTracks(tree, channel, lepton=None):
    tracks = ak.zip(
        {
            "pt": get_branch(tree, "PFCands_pt"),
            "phi": get_branch(tree, "PFCands_phi"),
            "eta": get_branch(tree, "PFCands_eta"),
            "mass": get_branch(tree, "PFCands_mass"),
        },
        with_name="Momentum4D",
    )
    # Apply track selection
    if channel == "ggF":
        trackSelection = (
            (get_branch(tree, "PFCands_fromPV") > 1)
            & (tracks.pt >= 0.75)
            & (abs(tracks.eta) <= 2.5)
            & (abs(get_branch(tree, "PFCands_dz")) < 10)
            & (get_branch(tree, "PFCands_dzErr") < 0.05)
        )
    elif channel == "WH":
        trackSelection = (
            (get_branch(tree, "PFCands_fromPV") > 1)
            & (tracks.pt >= 1.0)
            & (abs(tracks.eta) <= 2.5)
            & (abs(get_branch(tree, "PFCands_dz")) < 10)
            & (get_branch(tree, "PFCands_dzErr") < 0.05)
            & (get_branch(tree, "PFCands_puppiWeight") > 0.1)
            & (tracks.deltaR(lepton[:, 0]) >= 0.4)
        )
    tracks = tracks[trackSelection]
    return tracks


def getGenParticles(tree):
    genParticles = ak.zip(
        {
            "pt": get_branch(tree, "GenPart_pt"),
            "phi": get_branch(tree, "GenPart_phi"),
            "eta": get_branch(tree, "GenPart_eta"),
            "mass": get_branch(tree, "GenPart_mass"),
            "genPartIdxMother": get_branch(tree, "GenPart_genPartIdxMother"),
            "pdgId": get_branch(tree, "GenPart_pdgId"),
            "status": get_branch(tree, "GenPart_status"),
        },
        with_name="Momentum4D",
    )
    genParticles = genParticles[genParticles.pt > 1]
    return genParticles


def getLeptons(tree):
    muons = ak.zip(
        {
            "pt": get_branch(tree, "Muon_pt"),
            "phi": get_branch(tree, "Muon_phi"),
            "eta": get_branch(tree, "Muon_eta"),
            "mass": get_branch(tree, "Muon_mass"),
        },
        with_name="Momentum4D",
    )
    electrons = ak.zip(
        {
            "pt": get_branch(tree, "Electron_pt"),
            "phi": get_branch(tree, "Electron_phi"),
            "eta": get_branch(tree, "Electron_eta"),
            "mass": get_branch(tree, "Electron_mass"),
        },
        with_name="Momentum4D",
    )
    cutMuons = (
        (get_branch(tree, "Muon_looseId"))
        & (get_branch(tree, "Muon_pt") >= 10)
        & (abs(get_branch(tree, "Muon_dxy")) <= 0.02)
        & (abs(get_branch(tree, "Muon_dz")) <= 0.1)
        & (get_branch(tree, "Muon_pfIsoId") >= 2)
        & (abs(get_branch(tree, "Muon_eta")) < 2.4)
    )
    cutElectrons = (
        (get_branch(tree, "Electron_cutBased") >= 2)
        & (get_branch(tree, "Electron_pt") >= 15)
        & (get_branch(tree, "Electron_mvaFall17V2Iso_WP90"))
        & (
            abs(get_branch(tree, "Electron_dxy"))
            < 0.05 + 0.05 * (abs(get_branch(tree, "Electron_eta")) > 1.479)
        )
        & (
            abs(get_branch(tree, "Electron_dz"))
            < 0.10 + 0.10 * (abs(get_branch(tree, "Electron_eta")) > 1.479)
        )
        & (
            (abs(get_branch(tree, "Electron_eta")) < 1.444)
            | (abs(get_branch(tree, "Electron_eta")) > 1.566)
        )
        & (abs(get_branch(tree, "Electron_eta")) < 2.5)
    )
    selMuons = muons[cutMuons]
    selElectrons = electrons[cutElectrons]
    selLeptons = ak.concatenate([selElectrons, selMuons], axis=1)
    return selLeptons


def getMET(tree):
    MET = ak.zip(
        {
            "pt": get_branch(tree, "MET_pt"),
            "phi": get_branch(tree, "MET_phi"),
            "eta": 0,
            "mass": 0,
        },
        with_name="Momentum4D",
    )
    return MET

def getJets(tree, lepton=None):
    Jets_awk = ak.zip(
        {
            "pt": get_branch(tree, "Jet_pt"),
            "eta": get_branch(tree, "Jet_eta"),
            "phi": get_branch(tree, "Jet_phi"),
            "mass": get_branch(tree, "Jet_mass"),
            "btag": get_branch(tree, "Jet_btagDeepFlavB"),
            "jetId": get_branch(tree, "Jet_jetId"),
            "qgl": get_branch(tree, "Jet_qgl"),
        },
        with_name="Momentum4D",
    )
    # jet pt cut, eta cut, and minimum separation from lepton
    jet_awk_Cut = (Jets_awk.pt > 30) & (abs(Jets_awk.eta) < 2.4)
    # and minimum separation from lepton
    if lepton is not None:
        jet_awk_Cut = jet_awk_Cut & (Jets_awk.deltaR(lepton[:, 0]) >= 0.4)
    Jets_correct = Jets_awk[jet_awk_Cut]

    return Jets_correct

def getScalarParticle(genParticles):
    genParticles_PdgId = genParticles.pdgId
    genParticles_Status = genParticles.status
    scalarParticle = genParticles[
        (genParticles_PdgId == 25) & (genParticles_Status == 62)
    ]
    return scalarParticle

def classifyGenParticles_ggF(genParticles):
    genParticles_ParentId = genParticles.genPartIdxMother
    genParticles_PdgId = genParticles.pdgId
    genParticles_Status = genParticles.status

    # The last copy of the scalar mediator
    scalarParticle = genParticles[
        (genParticles_PdgId == 25) & (genParticles_Status == 62)
    ]

    # Define mask arrays to select the desired particles
    finalParticles = (genParticles_Status == 1) & (genParticles.pt > 1)
    fromScalarParticles = genParticles_ParentId == 999998
    isrParticles = genParticles_ParentId != 999998

    # Apply the selection criteria to get the final particle arrays
    # 10 arrays of final particles in total
    # Dividing to e, mu, gamma, pi, all other hadrons
    # for particles that come from the scalar mediator or not
    fromScalarParticles_e = genParticles[
        finalParticles & fromScalarParticles & (abs(genParticles_PdgId) == 11)
    ]
    fromScalarParticles_mu = genParticles[
        finalParticles & fromScalarParticles & (abs(genParticles_PdgId) == 13)
    ]
    fromScalarParticles_gamma = genParticles[
        finalParticles & fromScalarParticles & (abs(genParticles_PdgId) == 22)
    ]
    fromScalarParticles_pi = genParticles[
        finalParticles & fromScalarParticles & (abs(genParticles_PdgId) == 211)
    ]
    fromScalarParticles_hadron = genParticles[
        finalParticles & fromScalarParticles & (abs(genParticles_PdgId) > 100)
    ]

    isrParticles_e = genParticles[
        finalParticles & isrParticles & (abs(genParticles_PdgId) == 11)
    ]
    isrParticles_mu = genParticles[
        finalParticles & isrParticles & (abs(genParticles_PdgId) == 13)
    ]
    isrParticles_gamma = genParticles[
        finalParticles & isrParticles & (abs(genParticles_PdgId) == 22)
    ]
    isrParticles_pi = genParticles[
        finalParticles & isrParticles & (abs(genParticles_PdgId) == 211)
    ]
    isrParticles_hadron = genParticles[
        finalParticles & isrParticles & (abs(genParticles_PdgId) > 100)
    ]

    return (
        fromScalarParticles_e,
        fromScalarParticles_mu,
        fromScalarParticles_gamma,
        fromScalarParticles_pi,
        fromScalarParticles_hadron,
        isrParticles_e,
        isrParticles_mu,
        isrParticles_gamma,
        isrParticles_pi,
        isrParticles_hadron,
        scalarParticle,
    )

def classifyGenParticles_WH(genParticles):
    genParticles_ParentId = genParticles.genPartIdxMother
    genParticles_PdgId = genParticles.pdgId
    genParticles_Status = genParticles.status

    # The last copy of the scalar mediator
    scalarParticle = genParticles[
        (genParticles_PdgId == 25) & (genParticles_Status == 62)
    ]

    bQuarkIds = np.arange(len(genParticles_PdgId))[(abs(genParticles_PdgId) == 5)]
    fromBQuark = [id in bQuarkIds for id in genParticles_ParentId]

    # Define mask arrays to select the desired particles
    finalParticles = (genParticles.pt > 1) # & (genParticles_Status == 1) 
 
    genParticles_e = genParticles[
        finalParticles & (abs(genParticles_PdgId) == 11)
    ]
    genParticles_mu = genParticles[
        finalParticles & (abs(genParticles_PdgId) == 13)
    ]
    genParticles_nu = genParticles[
        finalParticles & ((abs(genParticles_PdgId) == 14) | (abs(genParticles_PdgId) == 16) | (abs(genParticles_PdgId) == 18))
    ]
    genParticles_gamma = genParticles[
        finalParticles & (abs(genParticles_PdgId) == 22)
    ]
    genParticles_pi = genParticles[
        finalParticles & (abs(genParticles_PdgId) == 211)
    ]
    genParticles_hadron = genParticles[
        finalParticles & (abs(genParticles_PdgId) > 100)
    ]
    genParticles_W = genParticles[
        finalParticles & (abs(genParticles_PdgId) == 24)
    ]
    genParticles_top = genParticles[
        finalParticles & ((abs(genParticles_PdgId) == 6) | (abs(genParticles_PdgId) == 8))
    ]
    other_genParticles = genParticles[
       finalParticles & ((abs(genParticles_PdgId) != 11) & (abs(genParticles_PdgId) != 13) & (abs(genParticles_PdgId) != 14) & (abs(genParticles_PdgId) != 16) & 
       (abs(genParticles_PdgId) != 18) & (abs(genParticles_PdgId) != 22) & (abs(genParticles_PdgId) != 211) & (abs(genParticles_PdgId) <= 100) &
         (abs(genParticles_PdgId) != 24) & (abs(genParticles_PdgId) != 6) & (abs(genParticles_PdgId) != 8))
    ]
    return (
        genParticles_e,
        genParticles_mu,
        genParticles_nu,
        genParticles_gamma,
        genParticles_pi,
        genParticles_hadron,
        genParticles_top,
        genParticles_W,
        other_genParticles,
        scalarParticle,
    )


def drawJetCone(ax, jet_eta, jet_phi, R=1.5, color="red"):
    phis, etas = get_dr_ring(R, jet_phi, jet_eta)
    phis = phis[1:]
    etas = etas[1:]
    ax.plot(
        phis[phis > pi] - 2 * pi,
        etas[phis > pi],
        color=color,
        linestyle="--",
    )
    ax.plot(
        phis[phis < -pi] + 2 * pi,
        etas[phis < -pi],
        color=color,
        linestyle="--",
    )
    ax.plot(phis[phis < pi], etas[phis < pi], color=color, linestyle="--")
    return ax


# Function that sets the scaling for markers
# Two methods for the moment: use tanh or square.
# Scaling using just the energy is also an option
def scale(particles, scalar, method=2):
    """Just to scale to a reasonable dot size"""
    energies = particles.energy
    e_max = scalar.energy
    if len(energies) == 0:
        return []
    if method == 1:
        e_normed = 500000.0 * np.square(energies / e_max)
    elif method == 2:
        e_normed = 1000.0 * np.tanh(energies / e_max)
        return e_normed
    else:
        e_normed = 2500.0 * energies / e_max
        return e_normed

def getNormParticle(genParticles):
    scalarParticle = getScalarParticle(genParticles)
    if len(scalarParticle) == 0:
        finalParticles = genParticles[(genParticles.status == 1) & (genParticles.pt > 1)]
        return finalParticles[ak.argmax(finalParticles.pt)]
    else:
        return scalarParticle[0]

# Main plotting function
def plot(
    ievt,
    tracks,
    genParticles,
    leptons,
    MET,
    jetsAK15,
    jetsAK15_tracks,
    this_jetsAK4,
    channel,
    boost=False,
    ax=None,
    params=None,
    showGen=False,
    showPFCands=True,
    showRingOfFire=True,
    showSUEPCandidate=False,
    showLeptons=True,
    showAK4=True,
    showMET=True,
    showLegend=True
):
    if ax is None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.addsubplot()

    normParticle = getNormParticle(genParticles)

    if showGen:
        if channel == 'ggF':
            (
                fromScalarParticles_e,
                fromScalarParticles_mu,
                fromScalarParticles_gamma,
                fromScalarParticles_pi,
                fromScalarParticles_hadron,
                isrParticles_e,
                isrParticles_mu,
                isrParticles_gamma,
                isrParticles_pi,
                isrParticles_hadron,
                scalarParticle,
            ) = classifyGenParticles_ggF(genParticles)
        elif channel == 'WH':
            (
                fromScalarParticles_e,
                fromScalarParticles_mu,
                fromScalarParticles_nu,
                fromScalarParticles_gamma,
                fromScalarParticles_pi,
                fromScalarParticles_hadron,
                fromScalarParticles_top,
                fromScalarParticles_W,
                other_genParticles,
                scalarParticle,
            ) = classifyGenParticles_WH(genParticles)

    if not boost:
        # Add jet info to the plot
        for jet in jetsAK15:
            ax = drawJetCone(ax, jet.eta, jet.phi, R=1.5, color="xkcd:green")

    if showSUEPCandidate and channel=='WH':
        highpt_jet = ak.argsort(jetsAK15.pt, axis=0, ascending=False, stable=True)
        jets_pTsorted = jetsAK15[highpt_jet]
        clusters_pTsorted = jetsAK15_tracks[highpt_jet]
        SUEP_cand = jets_pTsorted[0]
        SUEP_cluster_tracks = clusters_pTsorted[0]
        SUEP_pt = SUEP_cand.pt

        if boost:  # highlight tracks from SUEP candidate
            ax.scatter(
                SUEP_cluster_tracks.phi,
                SUEP_cluster_tracks.eta,
                s=scale(SUEP_cluster_tracks, normParticle),
                c="xkcd:red",
                marker="o",
            )

        else:  # draw SUEP and ISR ak15 candidates
            ax = drawJetCone(ax, SUEP_cand.eta, SUEP_cand.phi, color="xkcd:red")

    if showSUEPCandidate and channel=='ggF':
        _, _, topTwoJets = SUEP_utils.getTopTwoJets(
            None,
            ak.ones_like(jetsAK15),
            ak.ones_like(
                jetsAK15
            ),  # some moot arrays because this function needs rewriting
            ak.Array([jetsAK15], with_name="Momentum4D"),
            ak.Array([jetsAK15_tracks], with_name="Momentum4D"),
        )
        SUEP_cand, ISR_cand, SUEP_cluster_tracks, ISR_cluster_tracks = topTwoJets
        SUEP_pt = SUEP_cand.pt[0]
        ISR_pt = ISR_cand.pt[0]

        if boost:  # highlight tracks from SUEP and ISR candidates
            ax.scatter(
                SUEP_cluster_tracks.phi,
                SUEP_cluster_tracks.eta,
                s=scale(SUEP_cluster_tracks, normParticle),
                c="xkcd:red",
                marker="o",
            )
            ax.scatter(
                ISR_cluster_tracks.phi,
                ISR_cluster_tracks.eta,
                s=scale(ISR_cluster_tracks, normParticle),
                c="xkcd:blue",
                marker="o",
            )

        else:  # draw SUEP and ISR ak15 candidates
            ax = drawJetCone(ax, SUEP_cand.eta, SUEP_cand.phi, color="xkcd:red")
            ax = drawJetCone(ax, ISR_cand.eta, ISR_cand.phi, color="xkcd:blue")

    # show AK4jets
    if showAK4:
        for jet in this_jetsAK4:
            ax = drawJetCone(ax, jet.eta, jet.phi, R=0.4, color="xkcd:orange")

    # Boost everything to scalar's rest frame
    if boost:
        boost_SUEP = ak.zip(
            {
                "px": SUEP_cand.px * -1,
                "py": SUEP_cand.py * -1,
                "pz": SUEP_cand.pz * -1,
                "mass": SUEP_cand.mass,
            },
            with_name="Momentum4D",
        )

        tracks = tracks.boost_p4(boost_SUEP)
        if showGen:
            fromScalarParticles_e = fromScalarParticles_e.boost_p4(boost_SUEP)
            fromScalarParticles_mu = fromScalarParticles_mu.boost_p4(boost_SUEP)
            fromScalarParticles_gamma = fromScalarParticles_gamma.boost_p4(boost_SUEP)
            fromScalarParticles_pi = fromScalarParticles_pi.boost_p4(boost_SUEP)
            fromScalarParticles_hadron = fromScalarParticles_hadron.boost_p4(boost_SUEP)
            if channel == 'ggF':
                isrParticles_e = isrParticles_e.boost_p4(boost_SUEP)
                isrParticles_mu = isrParticles_mu.boost_p4(boost_SUEP)
                isrParticles_gamma = isrParticles_gamma.boost_p4(boost_SUEP)
                isrParticles_pi = isrParticles_pi.boost_p4(boost_SUEP)
                isrParticles_hadron = isrParticles_hadron.boost_p4(boost_SUEP)
        jetsAK15 = jetsAK15.boost_p4(boost_SUEP)
        jetsAK15_tracks = jetsAK15_tracks.boost_p4(boost_SUEP)
        scalarParticle = scalarParticle.boost_p4(boost_SUEP)
        leptons = leptons.boost_p4(boost_SUEP)
        MET = MET.boost_p4(boost_SUEP)

    # Plot parameters
    ax.set_xlim(-pi, pi)
    ax.set_ylim(-4, 4)
    ax.set_xlabel(r"$\phi$", fontsize=18)
    ax.set_ylabel(r"$\eta$", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=12)

    # Add scatters to figure
    if showGen:  # gen particles
        ax.scatter(
            fromScalarParticles_e.phi,
            fromScalarParticles_e.eta,
            s=scale(fromScalarParticles_e, normParticle),
            c="xkcd:light blue",
            marker="^",
        )
        ax.scatter(
            fromScalarParticles_mu.phi,
            fromScalarParticles_mu.eta,
            s=scale(fromScalarParticles_mu, normParticle),
            c="xkcd:light blue",
            marker="v",
        )
        ax.scatter(
            fromScalarParticles_nu.phi,
            fromScalarParticles_nu.eta,
            s=scale(fromScalarParticles_nu, normParticle),
            c="xkcd:light blue",
            marker="1",
        )
        ax.scatter(
            fromScalarParticles_gamma.phi,
            fromScalarParticles_gamma.eta,
            s=scale(fromScalarParticles_gamma, normParticle),
            c="xkcd:light blue",
            marker="s",
        )
        ax.scatter(
            fromScalarParticles_pi.phi,
            fromScalarParticles_pi.eta,
            s=scale(fromScalarParticles_pi, normParticle),
            c="xkcd:light blue",
            marker="P",
        )
        ax.scatter(
            fromScalarParticles_hadron.phi,
            fromScalarParticles_hadron.eta,
            s=scale(fromScalarParticles_hadron, normParticle),
            c="xkcd:light blue",
            marker="*",
        )
        ax.scatter(
            fromScalarParticles_top.phi,
            fromScalarParticles_top.eta,
            s=scale(fromScalarParticles_top, normParticle),
            c="xkcd:light blue",
            marker='$t$'
        )
        ax.scatter(
            fromScalarParticles_W.phi,
            fromScalarParticles_W.eta,
            s=scale(fromScalarParticles_W, normParticle),
            c="xkcd:light blue",
            marker='$W$'
        )
        ax.scatter(
            other_genParticles.phi,
            other_genParticles.eta,
            s=scale(other_genParticles, normParticle),
            c="xkcd:light blue",
            marker='o',
        )
        if channel == 'ggF':
            ax.scatter(
                isrParticles_e.phi,
                isrParticles_e.eta,
                s=scale(isrParticles_e, normParticle),
                c="xkcd:magenta",
                marker="o",
            )
            ax.scatter(
                isrParticles_mu.phi,
                isrParticles_mu.eta,
                s=scale(isrParticles_mu, normParticle),
                c="xkcd:magenta",
                marker="v",
            )
            ax.scatter(
                isrParticles_gamma.phi,
                isrParticles_gamma.eta,
                s=scale(isrParticles_gamma, normParticle),
                c="xkcd:magenta",
                marker="s",
            )
            ax.scatter(
                isrParticles_pi.phi,
                isrParticles_pi.eta,
                s=scale(isrParticles_pi, normParticle),
                c="xkcd:magenta",
                marker="P",
            )
            ax.scatter(
                isrParticles_hadron.phi,
                isrParticles_hadron.eta,
                s=scale(isrParticles_hadron, normParticle),
                c="xkcd:magenta",
                marker="*",
            )

    if showPFCands:  # plot PFCands
        ax.scatter(
            tracks.phi,
            tracks.eta,
            s=scale(tracks, normParticle),
            c="xkcd:gray",
            marker="o",
        )

    if showMET:
        phi_center = MET.phi
        width = MET.pt / normParticle.energy
        rectangle = plt.Rectangle(
            (phi_center - width / 2, -4),
            width=width,
            height=8,
            alpha=0.5,
            color="darkorange",
        )
        ax.add_patch(rectangle)

    if showLeptons:
        ax.scatter(
            leptons.phi,
            leptons.eta,
            s=scale(leptons, normParticle),
            marker="o",
            color="darkcyan",
        )

    if not boost:
        # Add the scalar mediator to the plot
        ax.scatter(
            scalarParticle.phi,
            scalarParticle.eta,
            s=scale(scalarParticle, normParticle),
            marker="x",
            color="xkcd:red",
        )


    if showRingOfFire:
        # draw two straight dashed lines, adjacent to the edges of the ring, that wrap around in phi
        ax.hlines(
            scalarParticle.eta + 1.0,
            xmin=-pi,
            xmax=pi,
            color="xkcd:red",
            alpha=0.7,
            linestyle="dotted",
        )
        ax.hlines(
            scalarParticle.eta - 1.0,
            xmin=-pi,
            xmax=pi,
            color="xkcd:red",
            alpha=0.7,
            linestyle="dotted",
        )

    # Legend 1 is particle type
    line1 = ax.scatter([-100], [-100], label="$e$", marker="^", c="xkcd:black")
    line2 = ax.scatter([-100], [-100], label=r"$\mu$", marker="v", c="xkcd:black")
    line3 = ax.scatter([-100], [-100], label=r"$\gamma$", marker="s", c="xkcd:black")
    line4 = ax.scatter([-100], [-100], label=r"$\pi$", marker="P", c="xkcd:black")
    line5 = ax.scatter([-100], [-100], label="Hadrons", marker="*", c="xkcd:black")
    line11 = ax.scatter([-100], [-100], label=r"$\nu$", marker="1", c="xkcd:black")
    line12 = ax.scatter([-100], [-100], label="W", marker="$W$", c="xkcd:black")
    line13 = ax.scatter([-100], [-100], label="top", marker="$t$", c="xkcd:black")
    line14 = ax.scatter([-100], [-100], label="other gen particle", marker="o", c="xkcd:black")
    line6 = ax.scatter(
        [-100],
        [-100],
        label="Scalar mediator",
        marker="x",
        color="xkcd:red",
    )
    line7 = ax.plot(
        [-100], [-100], label='"Ring of fire"', linestyle="dotted", c="xkcd:red"
    )[0]
    line8 = ax.scatter(
        [-100],
        [-100],
        label="Other AK15 jets" if showSUEPCandidate and channel=='ggF'else "AK15 Jets",
        marker="o",
        facecolors="none",
        edgecolors="xkcd:green",
    )
    line15 = ax.scatter(
        [-100],
        [-100],
        label="AK4 jets",
        marker="o",
        facecolors="none",
        edgecolors="xkcd:orange",
    )
    if showSUEPCandidate:
        line9 = ax.scatter(
            [-100],
            [-100],
            label="AK15 SUEP Candidate\n($p_T$ = " + str(round(SUEP_pt)) + " GeV)",
            facecolor="none",
            linestyle="--",
            edgecolors="xkcd:red",
        )
    if showSUEPCandidate and channel == 'ggF':
        line10 = ax.scatter(
            [-100],
            [-100],
            label="AK15 ISR Candidate\n($p_T$ = " + str(round(ISR_pt)) + " GeV)",
            facecolor="none",
            linestyle="--",
            edgecolors="xkcd:blue",
        )

    if showLegend:
        light_blue_patch = mpatches.Patch(color="xkcd:light blue", label="from scalar")
        magenta_patch = mpatches.Patch(color="xkcd:magenta", label="not from scalar")
        gray_patch = mpatches.Patch(color="xkcd:gray", label="Tracks")
        red_patch = mpatches.Patch(color="xkcd:red", label="SUEP Candidate tracks")
        blue_patch = mpatches.Patch(color="xkcd:blue", label="ISR Candidate tracks")
        bloodorange_patch = mpatches.Patch(
            color="darkorange", label="MET\n($p_T$ = " + str(round(MET.pt)) + " GeV)"
        )
        if showLeptons and len(leptons) == 1:
            darkcyan_patch = mpatches.Patch(
                color="darkcyan",
                label="Lepton\n($p_T$ = " + str(round(leptons[0].pt)) + " GeV)",
            )
        elif showLeptons and len(leptons) > 1:
            darkcyan_patch = mpatches.Patch(color="darkcyan", label="Lepton")
        if showGen:
            handles = [line1, line2, line11, line3, line4, line5, line12, line13, line14, light_blue_patch, magenta_patch]
        else:
            handles = [gray_patch]
        if not boost and len(scalarParticle.phi) > 0:  # add AK15, scalar mediator
            handles.append(line6)
        if showSUEPCandidate and channel == 'ggF' and len(jetsAK15) > 2 and not boost:
            handles.append(line8)
        if showSUEPCandidate and channel == 'WH' and len(jetsAK15) > 1 and not boost:
            handles.append(line8)
        if showRingOfFire and boost:
            handles.append(line7)
        if showSUEPCandidate and channel == 'WH':
            if not boost:
                handles.append(line9)
            else:
                handles.append(red_patch)
        if showSUEPCandidate and channel == 'ggF':
            if not boost:
                handles.append(line9)
                handles.append(line10)
            else:
                handles.append(red_patch)
                handles.append(blue_patch)
        if showAK4:
            handles.append(line15)
        if showMET:
            handles.append(bloodorange_patch)
        if showLeptons:
            handles.append(darkcyan_patch)

        ax.legend(handles=handles, loc=(1.01,0), fontsize=10)

    # build a rectangle in axes coords
    left, width = 0.0, 1.0
    bottom, height = 0.0, 1.0
    center = left + width / 2.0
    right = left + width
    top = bottom + height

    # axes coordinates are 0,0 is bottom left and 1,1 is upper right
    p = mpatches.Rectangle(
        (left, bottom), width, height, fill=False, transform=ax.transAxes, clip_on=False
    )

    ax.add_patch(p)

    # Print details about cuts
    ax.text(
        left + 0.02,
        bottom + 0.01,
        "SUEP Candidate Frame" if boost else "Lab Frame",
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=ax.transAxes,
        fontsize=12,
    )
    if params:
        mS, mPhi, temp, decay = params
        _ = ax.text(
            -3,
            3.9,
            r"$m_S = {}$ GeV"
            "\n"
            "$T$ = {} GeV"
            "\n"
            r"$m_{{\phi}}$ = {} GeV"
            "\n"
            "{}".format(mS, temp, mPhi, decaysLabels[decay]),
            fontsize=10,
            horizontalalignment="left",
            verticalalignment="top",
            # transform=ax.transAxes,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Input file")
    parser.add_argument(
        "-o", "--output", type=str, default=".", help="Output directory"
    )
    parser.add_argument(
        "-m", "--max", default=None, type=int, help="Max number of events to process"
    )
    parser.add_argument(
        "-b",
        "--boost",
        action="store_true",
        help="Show boosted to scalar mediator frame alongside lab frame",
    )
    parser.add_argument(
        "-bo",
        "--boostOnly",
        action="store_true",
        help="Show only boosted to scalar mediator (no lab frame)",
    )
    parser.add_argument("-g", "--gen", action="store_true", help="Plot gen particles")
    parser.add_argument(
        "-r",
        "--ring",
        action="store_true",
        help="Show ring of fire (only in boosted frame)",
    )
    parser.add_argument("-l", "--leptons", action="store_true", help="Show leptons")
    parser.add_argument("-MET", "--MET", action="store_true", help="Show MET")
    parser.add_argument("-p", "--pfcands", action="store_true", help="Show PFCands")
    parser.add_argument(
        "--showSUEPCandidate",
        action="store_true",
        help="Show SUEP candidate (defined via the --channel).",
    )
    parser.add_argument(
        "-c",
        "--channel",
        type=str,
        choices=["ggF", "WH"],
        required=True,
        help="Channel to plot, used only for object selection.",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        required=False,
        default=None,
        help="File of event numbers, run number, luminosity blocks to plot.",
    )
    args = parser.parse_args()

    # get input file
    rootfile = args.input
    
    fin = uproot.open(rootfile)
    tree = fin["Events"]

    # make a mask to only select the evnets we want to plot
    global branch_mask
    if args.file:
        eventsToPlot = np.loadtxt(args.file, delimiter=",", dtype=int)
        branch_mask = get_branch_mask(tree, eventsToPlot)
    else:
        branch_mask = None

    # get parameters (if signal) (i.e. mS, mPhi, T, decay)
    params = getParamsFromSampleName(args.input)

    # get relevant objects from the tree
    eventNumbers = get_branch(tree, "event")
    runNumbers = get_branch(tree, "run")
    luminosityBlocks = get_branch(tree, "luminosityBlock")
    genParticles = getGenParticles(tree)
    leptons = getLeptons(tree)
    tracks = getTracks(tree, args.channel, leptons)
    MET = getMET(tree)
    # and make AK15 jets
    if args.channel == "ggF":
        jetsAK15_pTmin = 200
    elif args.channel == "WH":
        jetsAK15_pTmin = 60
    jetsAK15, jetsAK15_tracks = SUEP_utils.FastJetReclustering(
        tracks, 1.5, jetsAK15_pTmin
    )
    jetsAK4 = getJets(tree)

    # call the plotting function for each event
    for i in range(0, len(tracks)):
        if args.max and i > args.max:
            break

        print("Plotting event", i)

        # get the event objects
        this_tracks = tracks[i]
        this_genParticles = genParticles[i]
        this_leptons = leptons[i]
        this_MET = MET[i]
        this_jetsAK15 = jetsAK15[i]
        this_jetsAK15_tracks = jetsAK15_tracks[i]
        this_jetsAK4 = jetsAK4[i]

        # show boosted and unboosted
        if args.boost:
            fig = plt.figure(figsize=(15, 7))
            ax1, ax2 = fig.subplots(1, 2)
            hep.cms.label(llabel="Simulation Preliminary", data=False, ax=ax1)
            hep.cms.label(llabel="Simulation Preliminary", data=False, ax=ax2)
            plot(
                i,
                this_tracks,
                this_genParticles,
                this_leptons,
                this_MET,
                this_jetsAK15,
                this_jetsAK15_tracks,
                this_jetsAK4,
                channel=args.channel,
                boost=False,
                ax=ax1,
                showSUEPCandidate=args.showSUEPCandidate,
                params=params,
                showRingOfFire=args.ring,
                showGen=args.gen,
                showPFCands=args.pfcands,
                showLeptons=args.leptons,
                showMET=args.MET,
                showLegend=False
            )
            plot(
                i,
                this_tracks,
                this_genParticles,
                this_leptons,
                this_MET,
                this_jetsAK15,
                this_jetsAK15_tracks,
                this_jetsAK4,
                channel=args.channel,
                boost=True,
                ax=ax2,
                showSUEPCandidate=args.showSUEPCandidate,
                params=params,
                showRingOfFire=args.ring,
                showGen=args.gen,
                showPFCands=args.pfcands,
                showLeptons=args.leptons,
                showMET=args.MET,
            )

        # show boosted only
        elif args.boostOnly:
            fig = plt.figure(figsize=(9, 7))
            ax1 = fig.subplots(1)
            hep.cms.label(llabel="Preliminary", data=False, ax=ax1)
            plot(
                i,
                this_tracks,
                this_genParticles,
                this_leptons,
                this_MET,
                this_jetsAK15,
                this_jetsAK15_tracks,
                this_jetsAK4,
                channel=args.channel,
                boost=True,
                showSUEPCandidate=args.offlineCandidates,
                ax=ax1,
                params=params,
                showRingOfFire=args.ring,
                showGen=args.gen,
                showPFCands=args.pfcands,
                showLeptons=args.leptons,
                showMET=args.MET,
            )

        # show unboosted only
        else:
            fig = plt.figure(figsize=(9, 7))
            ax1 = fig.subplots(1)
            hep.cms.label(llabel="Preliminary", data=False, ax=ax1)
            plot(
                i,
                this_tracks,
                this_genParticles,
                this_leptons,
                this_MET,
                this_jetsAK15,
                this_jetsAK15_tracks,
                this_jetsAK4,
                channel=args.channel,
                boost=False,
                showSUEPCandidate=args.showSUEPCandidate,
                ax=ax1,
                params=params,
                showRingOfFire=args.ring,
                showGen=args.gen,
                showPFCands=args.pfcands,
                showLeptons=args.leptons,
                showMET=args.MET,
            )

        if not os.path.exists(args.output):
            os.makedirs(args.output)

        fig.tight_layout()

        # save the figure
        if params:
            fig.savefig(
                args.output
                + "/mS-{:.2f}_mPhi-{:.2f}_T-{:.2f}_decay-{:s}_Event{:d}_Run{:d}_Lumi{:d}.pdf".format(
                    *params, eventNumbers[i], runNumbers[i], luminosityBlocks[i]
                ),
                bbox_inches="tight",
            )
            fig.savefig(
                args.output
                + "/mS-{:.2f}_mPhi-{:.2f}_T-{:.2f}_decay-{:s}_Event{:d}_Run{:d}_Lumi{:d}.png".format(
                    *params, eventNumbers[i], runNumbers[i], luminosityBlocks[i]
                ),
                bbox_inches="tight",
            )
        else:
            fig.savefig(
                args.output
                + f"/Event{eventNumbers[i]:d}_Run{runNumbers[i]:d}_Lumi{luminosityBlocks[i]:d}.pdf",
                bbox_inches="tight",
            )
            fig.savefig(
                args.output
                + f"/Event{eventNumbers[i]:d}_Run{runNumbers[i]:d}_Lumi{luminosityBlocks[i]:d}.png",
                bbox_inches="tight",
            )

        plt.close(fig)


if __name__ == "__main__":
    main()
