import dask_awkward as ak
from coffea import processor


class GenSumWeightExtractor(processor.ProcessorABC):
    def __init__(self) -> None:
        self._accumulator = 0

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        dataset = events.metadata["dataset"]
        output = ak.sum(events.genEventSumw)
        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator
