from dataclasses import dataclass


@dataclass
class Model_KD31_Environment:
    kd31_low_vram: bool = False

    def from_config(self, params):
        optimization_flags = [
            value.strip() for value in params("native", "optimization_flags").split(";")
        ]

        self.kd31_low_vram = "kd31_low_vram" in optimization_flags
        return self
