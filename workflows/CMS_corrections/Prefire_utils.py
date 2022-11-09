def GetPrefireWeights(self, events):

    if self.era == 2016 or self.era == 2017:
        self.out_vars["prefire_nom"] = events.L1PreFiringWeight_Nom
        self.out_vars["prefire_up"] = events.L1PreFiringWeight_Up
        self.out_vars["prefire_down"] = events.L1PreFiringWeight_Dn
    else:
        self.out_vars["prefire_nom"] = 1.0
        self.out_vars["prefire_up"] = 1.0
        self.out_vars["prefire_down"] = 1.0
    return
