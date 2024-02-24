from dataclasses import dataclass


@dataclass
class Model_KD3_Environment:
    kd30_low_vram: bool = False

    def from_config(self, params):
        optimization_flags = [
            value.strip() for value in params("native", "optimization_flags").split(";")
        ]

        self.kd30_low_vram = "kd30_low_vram" in optimization_flags
        return self
