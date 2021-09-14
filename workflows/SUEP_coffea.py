import awkward as ak
import numpy as np
import fastjet
import math
from coffea import hist, processor
import vector
import scipy.linalg as la
import sys, os
from vector import _methods
vector.register_awkward()

class SUEP_cluster(processor.ProcessorABC):
    def __init__(self, isMC, era=2017, sample="DY", do_syst=False, syst_var='', weight_syst=False, haddFileName=None, flag=False):
        self._flag = flag
        self.do_syst = do_syst
        self.era = era
        self.isMC = isMC
        self.sample = sample
        self.syst_var, self.syst_suffix = (syst_var, f'_sys_{syst_var}') if do_syst and syst_var else ('', '')
        self.weight_syst = weight_syst
        self.outfile = haddFileName

        #Set up for the histograms
        self._accumulator = processor.dict_accumulator({
            "uncleaned_tracks": hist.Hist(
                "Events",
                hist.Bin("Uncleaned_Ntracks", "Uncleaned NTracks", 10000, 0, 10000)
            ),
            "nCleaned_Cands": hist.Hist(
                "Events",
                hist.Bin("nCleaned_Cands", "NTracks", 200, 0, 200)
            ),
            "pT" : hist.Hist(
                "Events", 
                hist.Bin("pT", "$pT$ [GeV]", 200, 0, 200)
            ),
            "ngood_fastjets" : hist.Hist(
                "Events",
                hist.Bin("ngood_fastjets", "# Fastjets", 15, 0, 15)
            ),
            "SUEP_mult_nconst" : hist.Hist(
                "Events",
                hist.Bin("SUEP_mult_nconst", "# Tracks", 250, 0, 250)
            ),
            "SUEP_mult_pt" : hist.Hist(
                "Events",
                hist.Bin("SUEP_mult_pt", "pT", 100, 0, 2000)
            ),
            "SUEP_mult_eta" : hist.Hist(
                "Events",
                hist.Bin("SUEP_mult_eta", "eta", 100, -5, 5)
            ),
            "SUEP_mult_phi" : hist.Hist(
                "Events",
                hist.Bin("SUEP_mult_phi", "phi", 100, 0, 6.5)
            ),
            "SUEP_mult_mass" : hist.Hist(
                "Events",
                hist.Bin("SUEP_mult_mass", "mass", 150, 0, 4000)
            ),
            "SUEP_mult_spher" : hist.Hist(
                "Events",
                hist.Bin("SUEP_mult_spher", "Sphericity", 100, 0, 1)
            ),
            "SUEP_mult_aplan" : hist.Hist(
                "Events",
                hist.Bin("SUEP_mult_aplan", "Aplanarity", 100, 0, 1)
            ),
            "SUEP_mult_FW2M" : hist.Hist(
                "Events",
                hist.Bin("SUEP_mult_FW2M", "2nd Fox Wolfram Moment", 100, 0, 1)
            ),
            "SUEP_mult_D" : hist.Hist(
                "Events",
                hist.Bin("SUEP_mult_D", "D", 100, 0, 1)
            ),
            "SUEP_pt_nconst" : hist.Hist(
                "Events",
                hist.Bin("SUEP_pt_nconst", "# Tracks", 250, 0, 250)
            ),
            "SUEP_pt_pt" : hist.Hist(
                "Events",
                hist.Bin("SUEP_pt_pt", "pT", 100, 0, 2000)
            ),
            "SUEP_pt_eta" : hist.Hist(
                "Events",
                hist.Bin("SUEP_pt_eta", "eta", 100, -5, 5)
            ),
            "SUEP_pt_phi" : hist.Hist(
                "Events",
                hist.Bin("SUEP_pt_phi", "phi", 100, 0, 6.5)
            ),
            "SUEP_pt_mass" : hist.Hist(
                "Events",
                hist.Bin("SUEP_pt_mass", "mass", 150, 0, 4000)
            ),
            "SUEP_pt_spher" : hist.Hist(
                "Events",
                hist.Bin("SUEP_pt_spher", "Sphericity", 100, 0, 1)
            ),  
            "SUEP_pt_aplan" : hist.Hist(
                "Events",
                hist.Bin("SUEP_pt_aplan", "Aplanarity", 100, 0, 1)
            ),  
            "SUEP_pt_FW2M" : hist.Hist(
                "Events", 
                hist.Bin("SUEP_pt_FW2M", "2nd Fox Wolfram Moment", 100, 0, 1)
            ),  
            "SUEP_pt_D" : hist.Hist(
                "Events", 
                hist.Bin("SUEP_pt_D", "D", 100, 0, 1)
            ),
            "SUEP_pt_mean_pt" : hist.Hist(
                "SUEP_pt_mean_pt",
                hist.Bin("SUEP_pt_mean_pt","Mean constituent pT",100,0,2000)
            ),
            "SUEP_multi_mean_pt" : hist.Hist(
                "SUEP_multi_mean_pt",
                hist.Bin("SUEP_multi_mean_pt","Mean constituent pT",100,0,2000)
            ),
            

            # boosted variables

            "b_SUEP_mult_pt" : hist.Hist(
                "Events",
                hist.Bin("b_SUEP_mult_pt", "pT", 100, 0, 2000)
            ),
            "b_SUEP_mult_eta" : hist.Hist(
                "Events",
                hist.Bin("b_SUEP_mult_eta", "eta", 100, -5, 5)
            ),
            "b_SUEP_mult_phi" : hist.Hist(
                "Events",
                hist.Bin("b_SUEP_mult_phi", "phi", 100, 0, 6.5)
            ),
            "b_SUEP_mult_mass" : hist.Hist(
                "Events",
                hist.Bin("b_SUEP_mult_mass", "mass", 150, 0, 4000)
            ),
            "b_SUEP_mult_spher" : hist.Hist(
                "Events",
                hist.Bin("b_SUEP_mult_spher", "Sphericity", 100, 0, 1)
            ),
            "b_SUEP_mult_aplan" : hist.Hist(
                "Events",
                hist.Bin("b_SUEP_mult_aplan", "Aplanarity", 100, 0, 1)
            ),
            "b_SUEP_mult_FW2M" : hist.Hist(
                "Events",
                hist.Bin("b_SUEP_mult_FW2M", "2nd Fox Wolfram Moment", 100, 0, 1)
            ),
            "b_SUEP_mult_D" : hist.Hist(
                "Events",
                hist.Bin("b_SUEP_mult_D", "D", 100, 0, 1)
            ),
            "b_SUEP_pt_pt" : hist.Hist(
                "Events",
                hist.Bin("b_SUEP_pt_pt", "pT", 100, 0, 2000)
            ),
            "b_SUEP_pt_eta" : hist.Hist(
                "Events",
                hist.Bin("b_SUEP_pt_eta", "eta", 100, -5, 5)
            ),
            "b_SUEP_pt_phi" : hist.Hist(
                "Events",
                hist.Bin("b_SUEP_pt_phi", "phi", 100, 0, 6.5)
            ),
            "b_SUEP_pt_mass" : hist.Hist(
                "Events",
                hist.Bin("b_SUEP_pt_mass", "mass", 150, 0, 4000)
            ),
            "b_SUEP_pt_spher" : hist.Hist(
                "Events",
                hist.Bin("b_SUEP_pt_spher", "Sphericity", 100, 0, 1)
            ),  
            "b_SUEP_pt_aplan" : hist.Hist(
                "Events",
                hist.Bin("b_SUEP_pt_aplan", "Aplanarity", 100, 0, 1)
            ),  
            "b_SUEP_pt_FW2M" : hist.Hist(
                "Events", 
                hist.Bin("b_SUEP_pt_FW2M", "2nd Fox Wolfram Moment", 100, 0, 1)
            ),  
            "b_SUEP_pt_D" : hist.Hist(
                "Events", 
                hist.Bin("b_SUEP_pt_D", "D", 100, 0, 1)
            ),
            "b_SUEP_pt_mean_pt" : hist.Hist(
                "b_SUEP_pt_mean_pt",
                hist.Bin("b_SUEP_pt_mean_pt","Mean constituent pT",100,0,2000)
            ),
            "b_SUEP_multi_mean_pt" : hist.Hist(
                "SUEP_multi_mean_pt",
                hist.Bin("b_SUEP_multi_mean_pt","Mean constituent pT",100,0,2000)
            ),
        
        })

    @property
    def accumulator(self):
        return self._accumulator


    def sphericity(self, particles, r):
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
        evals = np.sort(np.linalg.eigvalsh(np.squeeze(np.moveaxis(s, 2, 0),axis=3)))
        return evals


    def process(self, events):
        output = self.accumulator.identity()
        dataset = events.metadata['dataset']

        #Prepare the clean track collection
        Cands = ak.zip({
            "pt": events.PFCands_trkPt,
            "eta": events.PFCands_trkEta,
            "phi": events.PFCands_trkPhi,
            "mass": events.PFCands_mass
        }, with_name="Momentum4D")

        cut = (events.PFCands_fromPV > 1) & (events.PFCands_trkPt >= 1) & (events.PFCands_trkEta <= 2.5)
        Cleaned_cands = Cands[cut]
        Cleaned_cands = ak.packed(Cleaned_cands)

        #The jet clustering part
        jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 1.5)
        cluster = fastjet.ClusterSequence(Cleaned_cands, jetdef) 
        ak_inclusive_jets = ak.with_name(cluster.inclusive_jets(min_pt=3),"Momentum4D")     # dim:  events * jets * 4 momenta
        ak_inclusive_cluster = ak.with_name(cluster.constituents(min_pt=3),"Momentum4D")    # dim: events * jets * consitutents * 4 momenta

        #remove events without a cluster
        ak_inclusive_cluster = ak_inclusive_cluster[ak.num(ak_inclusive_jets, axis=1)>0]
        ak_inclusive_jets = ak_inclusive_jets[ak.num(ak_inclusive_jets, axis=1)>0]

        
        #SUEP_mult

        # how many constituents in the jet
        chonkocity = ak.num(ak_inclusive_cluster, axis=2)           # dim: events * jets (values are # of constituents) 
        chonkiest_jet = ak.argsort(chonkocity, axis=1, ascending=True, stable=True)[:, ::-1]
        # ordered jets, by multiplicity
        thicc_jets = ak_inclusive_jets[chonkiest_jet]   


        #SUEP_pt
        #highpt_jet = ak.argsort(ak_inclusive_jets.pt, axis=1, ascending=False, stable=True)
        highpt_jet = ak.argsort(ak_inclusive_jets.pt, axis=1, ascending=True, stable=True)[:, ::-1]
        SUEP_pt = ak_inclusive_jets[highpt_jet]
        SUEP_pt_ncost = chonkocity[highpt_jet]

        #Sphericity tensor
        chonkiest_cands = ak_inclusive_cluster[chonkiest_jet][:,0]
        mult_eigs = self.sphericity(chonkiest_cands,2.0)

        highpt_cands = ak_inclusive_cluster[highpt_jet][:,0]
        pt_eigs = self.sphericity(highpt_cands,2.0)

        # mean of the constituents' pT for each jet (dim: events * jets, ordered by multiplicity)
        thicc_cands_mean_pt = ak.mean(chonkiest_cands.pt, axis = -1)
        highpt_cands_mean_pt = ak.mean(highpt_cands.pt, axis = -1)

        # boost - pT
        
        # v.boost(v) does NOT boost to CM, need to invert 3D components
        boost_SUEP_pt = SUEP_pt[:,0]                    # highest pT jets
        boost_SUEP_pt = ak.zip({
            "px": boost_SUEP_pt.px*-1,
            "py": boost_SUEP_pt.py*-1,
            "pz": boost_SUEP_pt.pz*-1,
            "mass": boost_SUEP_pt.mass
        }, with_name="Momentum4D")

        # boost constituents and jets
        boosted_highpt_cands = highpt_cands.boost(boost_SUEP_pt)  
        boosted_highpt_jets = SUEP_pt[:,0].boost(boost_SUEP_pt)

        # debugging
        # boosted_highpt_jets = [SUEP_pt[i,0].boost(boost_SUEP_pt[i]) for i in range(len(boost_SUEP_pt))]
        # print(boosted_highpt_jets[0])
        # boosted_highpt_jets = ak.Array(boosted_highpt_jets)


        # Sphericity tensor, in boosted frames
        b_pt_eigs = self.sphericity(boosted_highpt_cands,2.0)


        # boost - multiplicity

        # v.boost(v) does NOT boost to CM, need to invert 3D components
        boost_thicc_jets = thicc_jets[:,0]                    # highest multiplicity jets
        boost_thicc_jets = ak.zip({
            "px": boost_thicc_jets.px*- 1,
            "py": boost_thicc_jets.py*-1,
            "pz": boost_thicc_jets.pz*-1,
            "mass": boost_thicc_jets.mass
        }, with_name="Momentum4D")

        # boost constituents and jets
        boosted_chonkiest_cands = chonkiest_cands.boost(boost_thicc_jets)    
        boosted_chonkiest_jets = thicc_jets[:,0].boost(boost_thicc_jets)

        # Sphericity tensor, in boosted frames
        b_mult_eigs = self.sphericity(boosted_chonkiest_cands,2.0)

        # mean constituent pT
        b_thicc_cands_mean_pt = ak.mean(boosted_chonkiest_cands.pt, axis = -1)
        b_highpt_cands_mean_pt = ak.mean(boosted_highpt_cands.pt, axis = -1)


        #Now we will fill
        output["uncleaned_tracks"].fill(
            Uncleaned_Ntracks = ak.num(Cands),
        )   
        output["nCleaned_Cands"].fill(
            nCleaned_Cands = ak.num(Cleaned_cands),
        )
        output["pT"].fill(
            pT = (Cleaned_cands.px[0]),
        )
        output["ngood_fastjets"].fill(
            ngood_fastjets = ak.num(ak_inclusive_jets),
        )
        output["SUEP_mult_nconst"].fill(
            SUEP_mult_nconst = ak.max(ak.num(ak_inclusive_cluster, axis=2),axis=1)
        )
        output["SUEP_mult_pt"].fill(
            SUEP_mult_pt = thicc_jets[:,0].pt
        )
        output["SUEP_mult_eta"].fill(
            SUEP_mult_eta = thicc_jets[:,0].eta
        )
        output["SUEP_mult_phi"].fill(
            SUEP_mult_phi = thicc_jets[:,0].phi
        )
        output["SUEP_mult_mass"].fill(
            SUEP_mult_mass = thicc_jets[:,0].mass
        )
        output["SUEP_mult_spher"].fill(
            SUEP_mult_spher = 1.5 * (mult_eigs[:,1]+mult_eigs[:,0])
        )
        output["SUEP_mult_aplan"].fill(
            SUEP_mult_aplan = 1.5 * mult_eigs[:,0]
        )
        output["SUEP_mult_FW2M"].fill(
            SUEP_mult_FW2M = 1.0 - 3.0 * (mult_eigs[:,2]*mult_eigs[:,1] + mult_eigs[:,0]*mult_eigs[:,2] + mult_eigs[:,1]*mult_eigs[:,0])
        )
        output["SUEP_mult_D"].fill(
            SUEP_mult_D = 27.0 * mult_eigs[:,2]*mult_eigs[:,1]*mult_eigs[:,0]
        )
        output["SUEP_pt_nconst"].fill(
            SUEP_pt_nconst = SUEP_pt_ncost[:,0]
        )
        output["SUEP_pt_pt"].fill(
            SUEP_pt_pt = SUEP_pt[:,0].pt
        )
        output["SUEP_pt_eta"].fill(
            SUEP_pt_eta = SUEP_pt[:,0].eta
        )
        output["SUEP_pt_phi"].fill(
            SUEP_pt_phi = SUEP_pt[:,0].phi
        )
        output["SUEP_pt_mass"].fill(
            SUEP_pt_mass = SUEP_pt[:,0].mass
        ) 
        output["SUEP_pt_spher"].fill(
            SUEP_pt_spher = 1.5 * (pt_eigs[:,1]+pt_eigs[:,0])
        )
        output["SUEP_pt_aplan"].fill(
            SUEP_pt_aplan = 1.5 * pt_eigs[:,0]
        )
        output["SUEP_pt_FW2M"].fill(
            SUEP_pt_FW2M = 1.0 - 3.0 * (pt_eigs[:,2]*pt_eigs[:,1] + pt_eigs[:,2]*pt_eigs[:,0] + pt_eigs[:,1]*pt_eigs[:,0])
        )
        output["SUEP_pt_D"].fill(
            SUEP_pt_D = 27.0 * pt_eigs[:,2]*pt_eigs[:,1]*pt_eigs[:,0]
        )
        output["SUEP_multi_mean_pt"].fill(
            SUEP_multi_mean_pt = thicc_cands_mean_pt
        )
        output["SUEP_pt_mean_pt"].fill(
            SUEP_pt_mean_pt = highpt_cands_mean_pt
        )


        # boosted variables

        output["b_SUEP_mult_pt"].fill(
            b_SUEP_mult_pt = boosted_chonkiest_jets.pt
        )
        output["b_SUEP_mult_eta"].fill(
            b_SUEP_mult_eta = boosted_chonkiest_jets.eta
        )
        output["b_SUEP_mult_phi"].fill(
            b_SUEP_mult_phi = boosted_chonkiest_jets.phi
        )
        output["b_SUEP_mult_mass"].fill(
            b_SUEP_mult_mass = boosted_chonkiest_jets.mass
        )
        output["b_SUEP_mult_spher"].fill(
            b_SUEP_mult_spher = 1.5 * (b_mult_eigs[:,1]+b_mult_eigs[:,0])
        )
        output["b_SUEP_mult_aplan"].fill(
            b_SUEP_mult_aplan = 1.5 * b_mult_eigs[:,0]
        )
        output["b_SUEP_mult_FW2M"].fill(
            b_SUEP_mult_FW2M = 1.0 - 3.0 * (b_mult_eigs[:,2]*b_mult_eigs[:,1] + b_mult_eigs[:,0]*b_mult_eigs[:,2] + b_mult_eigs[:,1]*b_mult_eigs[:,0])
        )
        output["b_SUEP_mult_D"].fill(
            b_SUEP_mult_D = 27.0 * b_mult_eigs[:,2]*b_mult_eigs[:,1]*b_mult_eigs[:,0]
        )

        output["b_SUEP_pt_pt"].fill(
            b_SUEP_pt_pt = boosted_highpt_jets.pt
        )
        output["b_SUEP_pt_eta"].fill(
            b_SUEP_pt_eta = boosted_highpt_jets.eta
        )
        output["b_SUEP_pt_phi"].fill(
            b_SUEP_pt_phi = boosted_highpt_jets.phi
        )
        output["b_SUEP_pt_mass"].fill(
            b_SUEP_pt_mass = boosted_highpt_jets.mass
        ) 
        output["b_SUEP_pt_spher"].fill(
            b_SUEP_pt_spher = 1.5 * (b_pt_eigs[:,1]+b_pt_eigs[:,0])
        )
        output["b_SUEP_pt_aplan"].fill(
            b_SUEP_pt_aplan = 1.5 * b_pt_eigs[:,0]
        )
        output["b_SUEP_pt_FW2M"].fill(
            b_SUEP_pt_FW2M = 1.0 - 3.0 * (b_pt_eigs[:,2]*b_pt_eigs[:,1] + b_pt_eigs[:,2]*b_pt_eigs[:,0] + b_pt_eigs[:,1]*b_pt_eigs[:,0])
        )
        output["b_SUEP_pt_D"].fill(
            b_SUEP_pt_D = 27.0 * b_pt_eigs[:,2]*b_pt_eigs[:,1]*b_pt_eigs[:,0]
        )
        output["b_SUEP_multi_mean_pt"].fill(
            b_SUEP_multi_mean_pt = b_thicc_cands_mean_pt
        )
        output["b_SUEP_pt_mean_pt"].fill(
            b_SUEP_pt_mean_pt = b_highpt_cands_mean_pt
        )
    
        return output

    def postprocess(self, accumulator):
        return accumulator
