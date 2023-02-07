def GetPrefireWeights(self, events):
    if self.era == 2016 or self.era == 2017:
        self._accumulator["vars"]["prefire_nom"] = events.L1PreFiringWeight.Nom
        self._accumulator["vars"]["prefire_up"] = events.L1PreFiringWeight.Up
        self._accumulator["vars"]["prefire_down"] = events.L1PreFiringWeight.Dn
    else:
        self._accumulator["vars"]["prefire_nom"] = 1.0
        self._accumulator["vars"]["prefire_up"] = 1.0
        self._accumulator["vars"]["prefire_down"] = 1.0
    return
