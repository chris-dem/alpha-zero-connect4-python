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
    limit: int = 30

    @override
    def temperature(self, step) -> float:
        return 1 if step < self.limit else 1e-15
