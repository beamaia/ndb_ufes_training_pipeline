from abc import ABC
from enum import StrEnum

class JobType(StrEnum):
    train = "Train"
    test = "Test"
    origin_test = "Origin Test"
    none = "None"

class BaseJob(ABC):
    job: JobType = JobType.none

    def __init__(self, params):
        self.params = params
        self.device = self.params.device 

    def _validate_job_type(self):
        if self.job not in [JobType.train, JobType.test, JobType.origin_test]:
            raise ValueError("Invalid job type for pipeline.")

    


