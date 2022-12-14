import awkward as ak
from coffea import processor


class GenSumWeightExtractor(processor.ProcessorABC):
    def __init__(self) -> None:
        self._accumulator = processor.value_accumulator(float, 0)

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        dataset = events.metadata["dataset"]
        output = processor.value_accumulator(float, ak.sum(events.genEventSumw))
        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator
