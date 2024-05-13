"""Base classes for CyberMice tasks."""

from typing import Callable, Union, Sequence
from abc import ABC, abstractmethod
import numpy as np

from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable

class CyberMiceTask(composer.Task, ABC):
    """Base class for all tasks with CyberMice."""

    def __init__(self):
        pass

    @abstractmethod
    def action_spec(self):
        pass

    @property
    def walker(self):
        return self.walker

class Walking(CyberMiceTask):
    """Base class for all walking tasks."""

    def __init__():
        pass