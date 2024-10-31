from agentlens.dataset import Dataset, Row, subset

from scientist.config import ai
from scientist.models import Experiment, ExperimentName


class ExperimentRow(Row):
    experiment: Experiment


@ai.dataset("experiments")
class ExperimentDataset(Dataset[ExperimentRow]):
    @subset()
    def grokking(self, row: ExperimentRow):
        return row.experiment.name == ExperimentName.GROKKING

    def get_experiment(self, name: ExperimentName):
        for row in self:
            if row.experiment.name == name.value:
                return row.experiment
        raise ValueError(f"Experiment {name} not found")
