from dataclasses import dataclass 
from abc import ABC, abstractmethod
from typing import override

class TemperatureScheduler(ABC):
    """
    Abstract temperature scheduler
    """

    @abstractmethod
    def temperature(self, step) -> float:
        """
        Return temperature based on the current step
        """


@dataclass
class AlphazeroScheduler(TemperatureScheduler):
    """
    Alphazero scheduler
    Use 1 for moves before a limit and then use absolute best
    """
    limit: int = 10
    starting_point: float = 2
    end_point: float = 1e-15

    @override
    def temperature(self, step) -> float:
        st = self.starting_point
        lb = self.end_point
        step = min(step, self.limit) / self.limit
        return st * (1 - step) +  step * lb
