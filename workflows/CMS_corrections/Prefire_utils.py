def GetPrefireWeights(self, events, output):
    if self.era == 2016 or self.era == 2017:
        output["vars"]["prefire_nom"] = events.L1PreFiringWeight.Nom
        output["vars"]["prefire_up"] = events.L1PreFiringWeight.Up
        output["vars"]["prefire_down"] = events.L1PreFiringWeight.Dn
    else:
        output["vars"]["prefire_nom"] = 1.0
        output["vars"]["prefire_up"] = 1.0
        output["vars"]["prefire_down"] = 1.0
    return
