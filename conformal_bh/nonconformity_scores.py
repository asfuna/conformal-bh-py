from abc import ABC, abstractmethod


class NonconformityScore(ABC):
    """
    The optimal function is P(Y<=y|X=x)
    """
    @abstractmethod
    def get_score(self, x, y) -> float:
        pass


class ModelZeroNonconformityScore(NonconformityScore):
    def __init__(self, model):
        self.model = model

    def get_score(self, x, y):
        return -self.model(x)


class ModelNonconformityScore(NonconformityScore):
    def __init__(self, model):
        self.model = model

    def get_score(self, x, y):
        return y - self.model(x)


class CliffModelNonconformityScore(NonconformityScore):
    def __init__(self, model, cliff, max_model=None):
        self.model = model
        self.cliff = cliff
        self.max_model = max_model * 2 if max_model is not None else None

    def get_score(self, x, y):
        return (self.max_model or y) * (1 if y > self.cliff else 0) - self.model(x)
