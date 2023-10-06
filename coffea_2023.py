import dask
import dask_awkward as dak
import hist
import hist.dask as hda

from distributed import Client

from coffea.nanoevents import NanoEventsFactory
from coffea.nanoevents.schemas import NanoAODSchema


client = Client()

redirector = "root://cmseos.fnal.gov/"
path = "/store/user/lpcsuep/SUEPNano_skimmed_merged/"
primary_dataset = "DY1JetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8+"
secondary_dataset = "RunIISummer19UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1+MINIAODSIM/"

events = NanoEventsFactory.from_root(
    redirector + path + primary_dataset + secondary_dataset + "skim_1.root",
    treepath="Events",
    schemaclass=NanoAODSchema,
    permit_dask=True,
    metadata={"dataset": "DY1JetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8"},
).events()

clean_muons = (
    (events.Muon.mediumId)
    & (events.Muon.pt > 3)
    & (abs(events.Muon.dxy) <= 0.02)
    & (abs(events.Muon.dz) <= 0.1)
    & (abs(events.Muon.eta) < 2.4)
)
muons = events.Muon[clean_muons]
muons = muons[dak.num(muons, axis=-1) >= 4]

skim = dak.to_parquet(muons, "/uscms_data/d3/chpapage/SUEPs/MuonTriggers/muon_branches/SUEPCoffea_dask/test.parquet")

#_ = dask.compute(skim)
