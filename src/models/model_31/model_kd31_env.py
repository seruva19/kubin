from dataclasses import dataclass


@dataclass
class Model_KD31_Environment:
    def from_config(self, params):
        optimization_flags = [
            value.strip() for value in params("native", "optimization_flags").split(";")
        ]

        return self
