import math

import awkward as ak
import fastjet
import numpy as np
import vector
from numba import njit

vector.register_awkward()
ak.numba.register()


def ClusterMethod(
    self,
    output,
    dataset,
    indices,
    tracks,
    SUEP_cand,  # SUEP jet candidate
    ISR_cand,
    SUEP_cluster_tracks,  # The tracks within the Jet
    ISR_cluster_tracks,
    do_inverted=False,
    out_label=None,
):
    #####################################################################################
    # ---- Cluster Method (CL)
    # In this method, we use the tracks that were already clustered into the SUEP jet
    # to be the SUEP jet. Variables such as sphericity are calculated using these.
    #####################################################################################

    # boost into frame of SUEP
    boost_SUEP = ak.zip(
        {
            "px": SUEP_cand.px * -1,
            "py": SUEP_cand.py * -1,
            "pz": SUEP_cand.pz * -1,
            "mass": SUEP_cand.mass,
        },
        with_name="Momentum4D",
    )

    # SUEP tracks for this method are defined to be the ones from the cluster
    # that was picked to be the SUEP jet
    SUEP_tracks_b = SUEP_cluster_tracks.boost_p4(
        boost_SUEP
    )  # boost the SUEP tracks to their restframe

    # SUEP jet variables
    eigs = sphericity(SUEP_tracks_b, 1.0)  # Set r=1.0 for IRC safe

    # debug
    output[dataset]["vars"].loc(
        indices, "SUEP_nconst_CL" + out_label, ak.num(SUEP_tracks_b)
    )
    output[dataset]["vars"].loc(
        indices, "SUEP_nconst_CL" + out_label, ak.num(SUEP_tracks_b)
    )
    output[dataset]["vars"].loc(
        indices, "SUEP_pt_avg_b_CL" + out_label, ak.mean(SUEP_tracks_b.pt, axis=-1)
    )
    output[dataset]["vars"].loc(
        indices, "SUEP_S1_CL" + out_label, 1.5 * (eigs[:, 1] + eigs[:, 0])
    )

    # unboost for these
    SUEP_tracks = SUEP_tracks_b.boost_p4(SUEP_cand)
    output[dataset]["vars"].loc(
        indices, "SUEP_pt_avg_CL" + out_label, ak.mean(SUEP_tracks.pt, axis=-1)
    )

    # deltaR = SUEP_tracks.deltaR(SUEP_cand)
    # output[dataset]['vars'].loc(indices, "SUEP_rho0_CL"+out_label, rho(0, SUEP_cand, SUEP_tracks, deltaR)
    # output[dataset]['vars'].loc(indices, "SUEP_rho1_CL"+out_label, rho(1, SUEP_cand, SUEP_tracks, deltaR)

    output[dataset]["vars"].loc(indices, "SUEP_pt_CL" + out_label, SUEP_cand.pt)
    output[dataset]["vars"].loc(indices, "SUEP_eta_CL" + out_label, SUEP_cand.eta)
    output[dataset]["vars"].loc(indices, "SUEP_phi_CL" + out_label, SUEP_cand.phi)
    output[dataset]["vars"].loc(indices, "SUEP_mass_CL" + out_label, SUEP_cand.mass)

    output[dataset]["vars"].loc(
        indices,
        "SUEP_delta_mass_genMass_CL" + out_label,
        (SUEP_cand.mass - output[dataset]["vars"]["SUEP_genMass" + out_label][indices]),
    )
    output[dataset]["vars"].loc(
        indices,
        "SUEP_delta_pt_genPt_CL" + out_label,
        (SUEP_cand.pt - output[dataset]["vars"]["SUEP_genPt" + out_label][indices]),
    )

    # Calculate orientation difference between candidate and actual SUEP

    SUEP_genEta_diff_CL = (
        output[dataset]["vars"]["SUEP_eta_CL" + out_label]
        - output[dataset]["vars"]["SUEP_genEta" + out_label]
    )
    SUEP_genPhi_diff_CL = (
        output[dataset]["vars"]["SUEP_phi_CL" + out_label]
        - output[dataset]["vars"]["SUEP_genPhi" + out_label]
    )
    SUEP_genR_diff_CL = (SUEP_genEta_diff_CL**2 + SUEP_genPhi_diff_CL**2) ** 0.5

    output[dataset]["vars"]["SUEP_genEta_diff_CL" + out_label] = SUEP_genEta_diff_CL
    output[dataset]["vars"]["SUEP_genPhi_diff_CL" + out_label] = SUEP_genPhi_diff_CL
    output[dataset]["vars"]["SUEP_genR_diff_CL" + out_label] = SUEP_genR_diff_CL

    # inverted selection - will keep it off for now...
    if do_inverted:
        boost_ISR = ak.zip(
            {
                "px": ISR_cand.px * -1,
                "py": ISR_cand.py * -1,
                "pz": ISR_cand.pz * -1,
                "mass": ISR_cand.mass,
            },
            with_name="Momentum4D",
        )
        ISR_tracks_b = ISR_cluster_tracks.boost_p4(boost_ISR)

        # consistency check: we required already that ISR and SUEP have each at least 2 tracks
        assert all(ak.num(ISR_tracks_b) > 1)

        # ISR jet variables
        eigs = sphericity(ISR_tracks_b, 1.0)  # Set r=1.0 for IRC safe
        output[dataset]["vars"].loc(
            indices, "ISR_nconst_CL" + out_label, ak.num(ISR_tracks_b)
        )
        output[dataset]["vars"].loc(
            indices, "ISR_pt_avg_b_CL" + out_label, ak.mean(ISR_tracks_b.pt, axis=-1)
        )
        output[dataset]["vars"].loc(
            indices, "ISR_S1_CL" + out_label, 1.5 * (eigs[:, 1] + eigs[:, 0])
        )

        # unboost for these
        ISR_tracks = ISR_tracks_b.boost_p4(ISR_cand)
        output[dataset]["vars"].loc(
            indices, "ISR_pt_avg_CL" + out_label, ak.mean(ISR_tracks.pt, axis=-1)
        )

        # deltaR = ISR_tracks.deltaR(ISR_cand)
        # output[dataset]['vars'].loc(indices, "ISR_rho0_CL"+out_label, rho(0, ISR_cand, ISR_tracks, deltaR)
        # output[dataset]['vars'].loc(indices, "ISR_rho1_CL"+out_label, rho(1, ISR_cand, ISR_tracks, deltaR)

        output[dataset]["vars"].loc(indices, "ISR_pt_CL" + out_label, ISR_cand.pt)
        output[dataset]["vars"].loc(indices, "ISR_eta_CL" + out_label, ISR_cand.eta)
        output[dataset]["vars"].loc(indices, "ISR_phi_CL" + out_label, ISR_cand.phi)
        output[dataset]["vars"].loc(indices, "ISR_mass_CL" + out_label, ISR_cand.mass)


def ISRRemovalMethod(self, output, dataset, indices, tracks, SUEP_cand, ISR_cand):
    #####################################################################################
    # ---- ISR Removal Method (IRM)
    # In this method, we boost into the frame of the SUEP jet as selected previously
    # and select all tracks that are dphi > 1.6 from the ISR jet in this frame
    # to be the SUEP tracks. Variables such as sphericity are calculated using these.
    #####################################################################################

    # boost into frame of SUEP
    boost_SUEP = ak.zip(
        {
            "px": SUEP_cand.px * -1,
            "py": SUEP_cand.py * -1,
            "pz": SUEP_cand.pz * -1,
            "mass": SUEP_cand.mass,
        },
        with_name="Momentum4D",
    )
    ISR_cand_b = ISR_cand.boost_p4(boost_SUEP)
    tracks_b = tracks.boost_p4(boost_SUEP)

    # SUEP and IRM tracks as defined by IRS Removal Method (IRM):
    # all tracks outside/inside dphi 1.6 from ISR jet
    SUEP_tracks_b = tracks_b[abs(tracks_b.deltaphi(ISR_cand_b)) > 1.6]
    ISR_tracks_b = tracks_b[abs(tracks_b.deltaphi(ISR_cand_b)) <= 1.6]
    oneIRMtrackCut = ak.num(SUEP_tracks_b) > 1

    # output file if no events pass selections for ISR
    # avoids leaving this chunk without these columns
    if not any(oneIRMtrackCut):
        print("No events in ISR Removal Method, oneIRMtrackCut.")
        for c in self.columns_IRM:
            output[dataset]["vars"][c] = np.nan
    else:
        # remove the events left with one track
        SUEP_tracks_b = SUEP_tracks_b[oneIRMtrackCut]
        ISR_tracks_b = ISR_tracks_b[oneIRMtrackCut]
        SUEP_cand = SUEP_cand[oneIRMtrackCut]
        ISR_cand_IRM = ISR_cand[oneIRMtrackCut]
        tracks = tracks[oneIRMtrackCut]
        indices = indices[oneIRMtrackCut]

        output[dataset]["vars"].loc[indices, "SUEP_dphi_SUEP_ISR_IRM"] = ak.mean(
            abs(SUEP_cand.deltaphi(ISR_cand_IRM)), axis=-1
        )

        # SUEP jet variables
        eigs = sphericity(SUEP_tracks_b, 1.0)  # Set r=1.0 for IRC safe
        output[dataset]["vars"].loc[indices, "SUEP_nconst_IRM"] = ak.num(SUEP_tracks_b)
        output[dataset]["vars"].loc[indices, "SUEP_pt_avg_b_IRM"] = ak.mean(
            SUEP_tracks_b.pt, axis=-1
        )
        output[dataset]["vars"].loc[indices, "SUEP_S1_IRM"] = 1.5 * (
            eigs[:, 1] + eigs[:, 0]
        )

        # unboost for these
        SUEP_tracks = SUEP_tracks_b.boost_p4(SUEP_cand)
        output[dataset]["vars"].loc[indices, "SUEP_pt_avg_IRM"] = ak.mean(
            SUEP_tracks.pt, axis=-1
        )
        # deltaR = SUEP_tracks.deltaR(SUEP_cand)
        # output[dataset]["vars"].loc[indices, "SUEP_rho0_IRM"] = rho(0, SUEP_cand, SUEP_tracks, deltaR)
        # output[dataset]["vars"].loc[indices, "SUEP_rho1_IRM"] = rho(1, SUEP_cand, SUEP_tracks, deltaR)

        # redefine the jets using the tracks as selected by IRM
        SUEP = ak.zip(
            {
                "px": ak.sum(SUEP_tracks.px, axis=-1),
                "py": ak.sum(SUEP_tracks.py, axis=-1),
                "pz": ak.sum(SUEP_tracks.pz, axis=-1),
                "energy": ak.sum(SUEP_tracks.energy, axis=-1),
            },
            with_name="Momentum4D",
        )
        output[dataset]["vars"].loc[indices, "SUEP_pt_IRM"] = SUEP.pt
        output[dataset]["vars"].loc[indices, "SUEP_eta_IRM"] = SUEP.eta
        output[dataset]["vars"].loc[indices, "SUEP_phi_IRM"] = SUEP.phi
        output[dataset]["vars"].loc[indices, "SUEP_mass_IRM"] = SUEP.mass


def ConeMethod(
    self, output, dataset, indices, tracks, SUEP_cand, ISR_cand, do_inverted=False
):
    #####################################################################################
    # ---- Cone Method (CO)
    # In this method, all tracks outside a cone of abs(deltaR) of 1.6 (in lab frame)
    # are the SUEP tracks, those inside the cone are ISR tracks.
    #####################################################################################

    # SUEP tracks are all tracks outside a deltaR cone around ISR
    SUEP_tracks = tracks[abs(tracks.deltaR(ISR_cand)) > 1.6]
    ISR_tracks = tracks[abs(tracks.deltaR(ISR_cand)) <= 1.6]
    oneCOtrackCut = ak.num(SUEP_tracks) > 1

    # output file if no events pass selections for CO
    # avoids leaving this chunk without these columns
    if not any(oneCOtrackCut):
        print("No events in Cone Method, oneCOtrackCut.")
        for c in self.columns_CO:
            output[dataset]["vars"][c] = np.nan
        if do_inverted:
            for c in self.columns_CO_ISR:
                output[dataset]["vars"][c] = np.nan
    else:
        # remove the events left with one track
        SUEP_tracks = SUEP_tracks[oneCOtrackCut]
        ISR_tracks = ISR_tracks[oneCOtrackCut]
        tracks = tracks[oneCOtrackCut]
        indices = indices[oneCOtrackCut]

        SUEP_cand = ak.zip(
            {
                "px": ak.sum(SUEP_tracks.px, axis=-1),
                "py": ak.sum(SUEP_tracks.py, axis=-1),
                "pz": ak.sum(SUEP_tracks.pz, axis=-1),
                "energy": ak.sum(SUEP_tracks.energy, axis=-1),
            },
            with_name="Momentum4D",
        )

        # boost into frame of SUEP
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

        # SUEP jet variables
        eigs = sphericity(SUEP_tracks_b, 1.0)  # Set r=1.0 for IRC safe
        output[dataset]["vars"].loc[indices, "SUEP_nconst_CO"] = ak.num(SUEP_tracks_b)
        output[dataset]["vars"].loc[indices, "SUEP_pt_avg_b_CO"] = ak.mean(
            SUEP_tracks_b.pt, axis=-1
        )
        output[dataset]["vars"].loc[indices, "SUEP_S1_CO"] = 1.5 * (
            eigs[:, 1] + eigs[:, 0]
        )

        # unboost for these
        SUEP_tracks = SUEP_tracks_b.boost_p4(SUEP_cand)
        output[dataset]["vars"].loc[indices, "SUEP_pt_avg_CO"] = ak.mean(
            SUEP_tracks.pt, axis=-1
        )
        deltaR = SUEP_tracks.deltaR(SUEP_cand)
        # output[dataset]["vars"].loc[indices, "SUEP_rho0_CO"] = rho(0, SUEP_cand, SUEP_tracks, deltaR)
        # output[dataset]["vars"].loc[indices, "SUEP_rho1_CO"] = rho(1, SUEP_cand, SUEP_tracks, deltaR)

        output[dataset]["vars"].loc[indices, "SUEP_pt_CO"] = SUEP_cand.pt
        output[dataset]["vars"].loc[indices, "SUEP_eta_CO"] = SUEP_cand.eta
        output[dataset]["vars"].loc[indices, "SUEP_phi_CO"] = SUEP_cand.phi
        output[dataset]["vars"].loc[indices, "SUEP_mass_CO"] = SUEP_cand.mass

        # inverted selection
        if do_inverted:
            oneCOISRtrackCut = ak.num(ISR_tracks) > 1

            # output file if no events pass selections for ISR
            # avoids leaving this chunk without these columns
            if not any(oneCOISRtrackCut):
                print("No events in Inverted CO Removal Method, oneCOISRtrackCut.")
                for c in self.columns_CO_ISR:
                    output[dataset]["vars"][c] = np.nan
            else:
                # remove events with one ISR track
                ISR_tracks = ISR_tracks[oneCOISRtrackCut]
                indices = indices[oneCOISRtrackCut]

                ISR_cand = ak.zip(
                    {
                        "px": ak.sum(ISR_tracks.px, axis=-1),
                        "py": ak.sum(ISR_tracks.py, axis=-1),
                        "pz": ak.sum(ISR_tracks.pz, axis=-1),
                        "energy": ak.sum(ISR_tracks.energy, axis=-1),
                    },
                    with_name="Momentum4D",
                )

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

                # ISR jet variables
                eigs = sphericity(ISR_tracks_b, 1.0)  # Set r=1.0 for IRC safe
                output[dataset]["vars"].loc[indices, "ISR_nconst_CO"] = ak.num(
                    ISR_tracks_b
                )
                output[dataset]["vars"].loc[indices, "ISR_pt_avg_b_CO"] = ak.mean(
                    ISR_tracks_b.pt, axis=-1
                )
                output[dataset]["vars"].loc[indices, "ISR_pt_mean_scaled_CO"] = ak.mean(
                    ISR_tracks_b.pt, axis=-1
                ) / ak.max(ISR_tracks_b.pt, axis=-1)
                output[dataset]["vars"].loc[indices, "ISR_S1_CO"] = 1.5 * (
                    eigs[:, 1] + eigs[:, 0]
                )

                # unboost for these
                ISR_tracks = ISR_tracks_b.boost_p4(ISR_cand)
                output[dataset]["vars"].loc[indices, "ISR_pt_avg_CO"] = ak.mean(
                    ISR_tracks.pt, axis=-1
                )
                deltaR = ISR_tracks.deltaR(ISR_cand)
                output[dataset]["vars"].loc[indices, "ISR_rho0_CO"] = rho(
                    0, ISR_cand, ISR_tracks, deltaR
                )
                output[dataset]["vars"].loc[indices, "ISR_rho1_CO"] = rho(
                    1, ISR_cand, ISR_tracks, deltaR
                )

                output[dataset]["vars"].loc[indices, "ISR_pt_CO"] = ISR_cand.pt
                output[dataset]["vars"].loc[indices, "ISR_eta_CO"] = ISR_cand.eta
                output[dataset]["vars"].loc[indices, "ISR_phi_CO"] = ISR_cand.phi
                output[dataset]["vars"].loc[indices, "ISR_mass_CO"] = ISR_cand.mass


def sphericity(particles, r):
    norm = ak.sum(particles.p**r, axis=1, keepdims=True)
    s = np.array(
        [
            [
                ak.sum(
                    particles.px * particles.px * particles.p ** (r - 2.0),
                    axis=1,
                    keepdims=True,
                )
                / norm,
                ak.sum(
                    particles.px * particles.py * particles.p ** (r - 2.0),
                    axis=1,
                    keepdims=True,
                )
                / norm,
                ak.sum(
                    particles.px * particles.pz * particles.p ** (r - 2.0),
                    axis=1,
                    keepdims=True,
                )
                / norm,
            ],
            [
                ak.sum(
                    particles.py * particles.px * particles.p ** (r - 2.0),
                    axis=1,
                    keepdims=True,
                )
                / norm,
                ak.sum(
                    particles.py * particles.py * particles.p ** (r - 2.0),
                    axis=1,
                    keepdims=True,
                )
                / norm,
                ak.sum(
                    particles.py * particles.pz * particles.p ** (r - 2.0),
                    axis=1,
                    keepdims=True,
                )
                / norm,
            ],
            [
                ak.sum(
                    particles.pz * particles.px * particles.p ** (r - 2.0),
                    axis=1,
                    keepdims=True,
                )
                / norm,
                ak.sum(
                    particles.pz * particles.py * particles.p ** (r - 2.0),
                    axis=1,
                    keepdims=True,
                )
                / norm,
                ak.sum(
                    particles.pz * particles.pz * particles.p ** (r - 2.0),
                    axis=1,
                    keepdims=True,
                )
                / norm,
            ],
        ]
    )
    s = np.squeeze(np.moveaxis(s, 2, 0), axis=3)
    evals = np.sort(np.linalg.eigvalsh(s))
    return evals


def rho(number, jet, tracks, deltaR, dr=0.05):
    r_start = number * dr
    r_end = (number + 1) * dr
    ring = (deltaR > r_start) & (deltaR < r_end)
    rho_values = ak.sum(tracks[ring].pt, axis=1) / (dr * jet.pt)
    return rho_values


def FastJetReclustering(tracks, r, min_pt):
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, r)
    cluster = fastjet.ClusterSequence(tracks, jetdef)
    ak_inclusive_jets = cluster.inclusive_jets(min_pt=min_pt)
    ak_inclusive_cluster = cluster.constituents(min_pt=min_pt)
    return ak_inclusive_jets, ak_inclusive_cluster


def getTopTwoJets(self, tracks, indices, ak_inclusive_jets, ak_inclusive_cluster):
    # order the reclustered jets by pT (will take top 2 for ISR removal method)
    # make sure there are at least 2 entries in the array
    ak_inclusive_jets = ak.pad_none(ak_inclusive_jets, 2, axis=1)
    ak_inclusive_cluster = ak.pad_none(ak_inclusive_cluster, 2, axis=1)
    highpt_jet = ak.argsort(ak_inclusive_jets.pt, axis=1, ascending=False, stable=True)
    jets_pTsorted = ak_inclusive_jets[highpt_jet]
    clusters_pTsorted = ak_inclusive_cluster[highpt_jet]

    # at least 2 tracks in SUEP
    singletrackCut = ak.num(clusters_pTsorted[:, 0]) > 1
    jets_pTsorted = jets_pTsorted[singletrackCut]
    clusters_pTsorted = clusters_pTsorted[singletrackCut]
    tracks = tracks[singletrackCut]
    indices = indices[singletrackCut]

    # number of constituents per jet, sorted by pT
    nconst_pTsorted = ak.fill_none(ak.num(clusters_pTsorted, axis=-1), 0)

    # Top 2 pT jets. If jet1 has fewer tracks than jet2 then swap
    SUEP_cand = ak.where(
        nconst_pTsorted[:, 1] <= nconst_pTsorted[:, 0],
        jets_pTsorted[:, 0],
        jets_pTsorted[:, 1],
    )
    SUEP_cand = ak.where(ak.is_none(SUEP_cand), jets_pTsorted[:, 0], SUEP_cand)
    ISR_cand = ak.where(
        nconst_pTsorted[:, 1] > nconst_pTsorted[:, 0],
        jets_pTsorted[:, 0],
        jets_pTsorted[:, 1],
    )
    SUEP_cluster_tracks = ak.where(
        nconst_pTsorted[:, 1] <= nconst_pTsorted[:, 0],
        clusters_pTsorted[:, 0],
        clusters_pTsorted[:, 1],
    )
    SUEP_cluster_tracks = ak.where(
        ak.is_none(SUEP_cluster_tracks), clusters_pTsorted[:, 0], SUEP_cluster_tracks
    )
    ISR_cluster_tracks = ak.where(
        nconst_pTsorted[:, 1] > nconst_pTsorted[:, 0],
        clusters_pTsorted[:, 0],
        clusters_pTsorted[:, 1],
    )

    return (
        tracks,
        indices,
        (SUEP_cand, ISR_cand, SUEP_cluster_tracks, ISR_cluster_tracks),
    )


def convert_coords(self, coords, tracks, nobj):
    allowed_coords = ["cyl", "cart", "p4"]
    if coords.lower() not in allowed_coords:
        raise Exception(self.coords + " is not supported in GNN_convertEvents.")

    if coords.lower() == "p4":
        new_tracks = convert_p4(tracks, nobj)
    elif coords.lower() == "cart":
        new_tracks = convert_cart(tracks, nobj)
    elif coords.lower() == "cyl":
        new_tracks = convert_cyl(tracks, nobj)
    else:
        raise Exception()

    return new_tracks


def convert_p4(tracks, nobj=10):
    """store objects in zero-padded numpy arrays"""

    # bad naming, tracks dimensions is (events x tracks)
    nentries = len(tracks)

    l1Obj_p4 = np.zeros((nentries, nobj, 4))

    px = to_np_array(tracks.px, maxN=nobj)
    py = to_np_array(tracks.py, maxN=nobj)
    pz = to_np_array(tracks.pz, maxN=nobj)
    m = to_np_array(tracks.mass, maxN=nobj)

    l1Obj_p4[:, :, 0] = px
    l1Obj_p4[:, :, 1] = py
    l1Obj_p4[:, :, 2] = pz
    l1Obj_p4[:, :, 3] = m

    return l1Obj_p4


def convert_cyl(tracks, nobj=10):
    """store objects in zero-padded numpy arrays"""

    # bad naming, tracks dimensions is (events x tracks)
    nentries = len(tracks)

    l1Obj_cyl = np.zeros((nentries, nobj, 4))

    pt = to_np_array(tracks.pt, maxN=nobj)
    eta = to_np_array(tracks.eta, maxN=nobj)
    phi = to_np_array(tracks.phi, maxN=nobj)
    m = to_np_array(tracks.mass, maxN=nobj)

    l1Obj_cyl[:, :, 0] = pt
    l1Obj_cyl[:, :, 1] = eta
    l1Obj_cyl[:, :, 2] = phi
    l1Obj_cyl[:, :, 3] = m

    return l1Obj_cyl


def convert_cart(tracks, nobj=10):
    """store objects in zero-padded numpy arrays"""

    # bad naming, tracks dimensions is (events x tracks)
    nentries = len(tracks)

    l1Obj_cart = np.zeros((nentries, nobj, 4))

    pt = to_np_array(tracks.pt, maxN=nobj)
    eta = to_np_array(tracks.eta, maxN=nobj)
    phi = to_np_array(tracks.phi, maxN=nobj)
    m = to_np_array(tracks.mass, maxN=nobj)

    l1Obj_cart[:, :, 0] = pt * np.cos(phi)
    l1Obj_cart[:, :, 1] = pt * np.sin(phi)
    l1Obj_cart[:, :, 2] = pt * np.sinh(eta)
    l1Obj_cart[:, :, 3] = m

    return l1Obj_cart


def to_np_array(ak_array, maxN=100, pad=0):
    """convert awkward array to regular numpy array"""
    return ak.to_numpy(
        ak.fill_none(ak.pad_none(ak_array, maxN, clip=True, axis=-1), pad)
    )


def inter_isolation(leptons_1, leptons_2, dR=1.6):
    """
    Compute the inter-isolation of each particle. It is supposed to work for one particle per event. The input is:
    - leptons_1: array of leptons for isolation calculation
    - leptons_2: array of all leptons in the events
    - dR: deltaR cut for isolation calculation
    """
    a, b = ak.unzip(ak.cartesian([leptons_1, leptons_2]))
    deltar_mask = a.deltaR(b) < dR
    return (ak.sum(b[deltar_mask].pt, axis=-1) - leptons_1.pt) / leptons_1.pt


@njit
def interIsolation_old(particles, v_electrons, v_muons, cone_size=0.4):
    """
    Compute the inter-isolation of each particle. It is supposed to work for one particle per event. The input is:
    - particles: array of particles for isolation calculation
    - v_electrons: array of arrays of electrons for each event
    - v_muons: array of arrays of muons for each event
    """
    n_particles = len(particles)
    out = np.zeros(n_particles)
    for i in range(n_particles):
        particle = particles[i]
        muons = v_muons[i]
        electrons = v_electrons[i]
        for j in range(len(muons)):
            dEta = particle.eta - muons[j].eta
            dPhi = particle.phi - muons[j].phi
            if abs(dPhi) > math.pi:
                dPhi = 2 * math.pi - abs(dPhi)
            dR = math.sqrt((dEta) ** 2 + (dPhi) ** 2)
            if dR < 1.6:
                out[i] += muons[j].pt
        for j in range(len(electrons)):
            dEta = particle.eta - electrons[j].eta
            dPhi = particle.phi - electrons[j].phi
            if abs(dPhi) > math.pi:
                dPhi = 2 * math.pi - abs(dPhi)
            dR = math.sqrt((dEta) ** 2 + (dPhi) ** 2)
            if dR < cone_size:
                out[i] += electrons[j].pt
        out[i] -= particle.pt
        out[i] /= particle.pt
    return out


def n_eta_ring(muonsCollection, eta_cutoff):
    """Return the number of muons in the eta ring around the leading muon"""
    leading_muon_eta = muonsCollection[:, 0].eta
    return ak.num(
        muonsCollection[abs(leading_muon_eta - muonsCollection.eta) < eta_cutoff]
    )


def transverse_mass(particles, met):
    """Return the transverse mass of an array of particles and the missing transverse energy"""
    return np.sqrt(2 * particles.pt * met.pt * (1 - np.cos(particles.delta_phi(met))))


def get_last_parents(genParts):
    # Begin with the matched gen muons
    temp_parents = genParts
    # This mask keeps track of which non-muon parents have not appeared already. Initialize to True.
    has_not_appeared = ak.full_like(temp_parents.pt, True, dtype=bool)
    # The useful parents are the last non-muon parents. It should be empty initially.
    last_parents = ak.mask(temp_parents, ~has_not_appeared)
    # Terminate when none of the particles is a muon
    while ak.any(abs(temp_parents.pdgId) == abs(genParts.pdgId)):
        # Create mask of the particles that are not muons and have not already been kept
        mask = (abs(temp_parents.pdgId) != abs(genParts.pdgId)) & has_not_appeared
        # Create the parents collection that includes all non-muon parents
        parents = ak.zip(
            {
                "pt": ak.mask(temp_parents, mask).pt,
                "eta": ak.mask(temp_parents, mask).eta,
                "phi": ak.mask(temp_parents, mask).phi,
                "M": ak.mask(temp_parents, mask).mass,
                "pdgId": ak.mask(temp_parents, mask).pdgId,
                "status": ak.mask(temp_parents, mask).status,
            },
            with_name="Momentum4D",
        )
        # Remove the parents that were found from the mask
        has_not_appeared = has_not_appeared & ~mask
        # Add the parents that are not None
        last_parents = ak.where(ak.fill_none(parents.pt, 0) > 0, parents, last_parents)
        # Get the parents of the particles
        temp_parents = temp_parents.parent
    return last_parents


def probabilistic_removal(muons_genPartFlav):
    """Will return a mask that will remove 7.2% of muons with flavor 0"""
    is_matched = muons_genPartFlav != 0
    rng = np.random.default_rng(12345)
    counts = ak.num(muons_genPartFlav)
    numbers = rng.random(len(ak.flatten(muons_genPartFlav)))
    probs = ak.unflatten(numbers, counts)
    unmatched_muons_passing = probs > 0.072
    return is_matched | unmatched_muons_passing


def LLP_free_muons(events, muons):
    """Will return a mask that will remove the non-0 muons that have an LLP in their gen history"""
    genParts = events.GenPart[muons.genPartIdx]
    is_unmatched = muons.genPartFlav == 0
    temp_parents = genParts
    has_matched = ak.full_like(temp_parents.pt, False, dtype=bool)
    while not ak.all(ak.is_none(temp_parents.pt, axis=-1)):
        mask = ak.fill_none(
            (abs(temp_parents.pdgId) == 130)
            | (abs(temp_parents.pdgId) == 211)
            | (abs(temp_parents.pdgId) == 321),
            False,
        )
        has_matched = ak.where(
            mask, ak.full_like(temp_parents.pt, True, dtype=bool), has_matched
        )
        temp_parents = temp_parents.parent
    return ~has_matched | is_unmatched


def discritize_pdg_codes(pdf_codes, extended=False):
    discritized_codes = ak.where(pdf_codes == 0, 0, -1)
    if extended:
        discritized_codes = ak.where(pdf_codes == 1, 1, discritized_codes)
        discritized_codes = ak.where(pdf_codes == 2, 2, discritized_codes)
        discritized_codes = ak.where(pdf_codes == 3, 3, discritized_codes)
        discritized_codes = ak.where(pdf_codes == 4, 4, discritized_codes)
        discritized_codes = ak.where(pdf_codes == 5, 5, discritized_codes)
    discritized_codes = ak.where(pdf_codes == 11, 11, discritized_codes)
    discritized_codes = ak.where(pdf_codes == 13, 13, discritized_codes)
    discritized_codes = ak.where(pdf_codes == 15, 15, discritized_codes)
    discritized_codes = ak.where(pdf_codes == 22, 22, discritized_codes)
    discritized_codes = ak.where(
        (100 <= pdf_codes) & (pdf_codes < 200), 100, discritized_codes
    )
    discritized_codes = ak.where(
        (200 <= pdf_codes) & (pdf_codes < 300), 200, discritized_codes
    )
    discritized_codes = ak.where(
        (300 <= pdf_codes) & (pdf_codes < 400), 300, discritized_codes
    )
    discritized_codes = ak.where(
        (400 <= pdf_codes) & (pdf_codes < 500), 400, discritized_codes
    )
    discritized_codes = ak.where(
        (500 <= pdf_codes) & (pdf_codes < 600), 500, discritized_codes
    )
    discritized_codes = ak.where(1000 <= pdf_codes, 1000, discritized_codes)
    return discritized_codes


@njit
def loop_over_arr(arr1, arr2, builder):
    """
    Loop over two arrays and append the elements of arr1 that are not in arr2 to the builder.
    Used to filter the dimuon pairs after some selection (e.g., mass cut).
    arr1: array with indices
    arr2: array with all the indices of the pairs failing the selection
    """
    for i in range(len(arr1)):
        arr1_i = arr1[i]
        builder.begin_list()
        for j in range(len(arr1_i)):
            if arr1[i][j] not in arr2[i]:
                builder.integer(arr1[i][j])
        builder.end_list()
    return builder
