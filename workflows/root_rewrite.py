import uproot
import awkward as ak


def rewrite(infile):

    f = uproot.open(infile)
    to_write = "f_new['tree'] = {"
    for entry in f["mmtree/tree;4"]:
        if type(entry) == uproot.models.TBranch.Model_TBranch_v13: continue
        if entry.name == "hltResultName": continue
        if "PFcand_pt" in entry.name: out_name = entry.name.replace("PFcand_pt","PFCands_trkPt")
        elif "PFcand_eta" in entry.name: out_name = entry.name.replace("PFcand_eta","PFCands_trkEta")
        elif "PFcand_phi" in entry.name: out_name = entry.name.replace("PFcand_phi","PFCands_trkPhi")
        elif "PFcand_m" in entry.name: out_name = entry.name.replace("PFcand_m","PFCands_mass") 
        elif "Jet_m" in entry.name: out_name = entry.name.replace("Jet_m","Jet_mass")
    
        else: out_name = entry.name #rename PFCands class for NanoAODSchema
        to_write = to_write + "'{}':f['mmtree/tree;4']['{}'].array(),".format(out_name,entry.name)    
    to_write = to_write[:-1]+"}"
    
    f_new = uproot.recreate("rewrite.root")
    
    exec(to_write)
    
    return
