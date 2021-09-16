"""
GenSumWeight.py
Workspace producers using coffea. Need the GenSumWeight to do normalization properly
"""

from coffea import hist, processor
from coffea.processor import ProcessorABC, LazyDataFrame, dict_accumulator
from uproot3 import recreate
import numpy as np

class XsecSumWeight(ProcessorABC):
    """
    A coffea Processor which produces a workspace.
    This is just for grabbing the NanoAOD genEventSum weight for normalization.
    """

    def __init__(self, isMC, xsec=1.0,  era=2017, sample="DY", do_syst=False, syst_var='', weight_syst=False, haddFileName=None, flag=False):
        self._flag = flag
        self.do_syst = do_syst
        self.era = era
        self.isMC = isMC
        self.xsec = xsec
        self.sample = sample
        self._accumulator = dict_accumulator({
            "genEventSumw": hist.Hist(
                "Runs",
		hist.Bin("genEventSumw", "genEventSumw",1,0,1),
            ),
        })
        self.outfile = haddFileName

    #add to print out the results of self.XXX
    #def __repr__(self):
    #    return f'{self.__class__.__name__}(era: {self.era}, isMC: {self.isMC}, sample: {self.sample}, do_syst: {self.do_syst}, syst_var: {self.syst_var}, weight_syst: {self.weight_syst}, output: {self.outfile})'

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df, *args):
        output = self.accumulator.identity()

        xsec = self.xsec
        weights = df.genEventSumw
        weight_array = np.array(weights)
        genweight = xsec / weight_array.sum()
        output["genEventSumw"].fill(
            genEventSumw=0.5,
            weight=genweight
        )

        return output

    def postprocess(self, accumulator):
        return accumulator
