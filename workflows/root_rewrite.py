import awkward as ak
import uproot

# mass is expected as an input in NanoAODSchema methods. Must rewrite "m" to "mass" in order to catch the 4-vector until vector is implemented in coffea
# See here:
# https://github.com/CoffeaTeam/coffea/blob/master/coffea/nanoevents/methods/vector.py#L753


def rewrite(infile):

    f = uproot.open(infile)
    to_write = "f_new['tree'] = {"
    for entry in f["mmtree/tree"]:
        # if type(entry) == uproot.models.TBranch.Model_TBranch_v13: continue
        if entry.name == "hltResultName":
            continue
        if "PFcand_m" in entry.name:
            out_name = entry.name.replace("PFcand_m", "PFcand_mass")
        elif "Jet_m" in entry.name:
            out_name = entry.name.replace("Jet_m", "Jet_mass")

        else:
            out_name = entry.name  # rename PFCands class for NanoAODSchema
        to_write = to_write + "'{}':f['mmtree/tree']['{}'].array(),".format(
            out_name, entry.name
        )
    to_write = to_write[:-1] + "}"

    f_new = uproot.recreate("rewrite.root")

    exec(to_write)

    return
