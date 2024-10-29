from reagency.dataset import Dataset, Row, subset
from scientist.config import ai
from scientist.models import Experiment, ExperimentName


class ExperimentRow(Row):
    experiment: Experiment


@ai.dataset("experiments")
class ExperimentDataset(Dataset[ExperimentRow]):
    @subset()
    def grokking(self, row: ExperimentRow):
        return row.experiment.name == ExperimentName.GROKKING
