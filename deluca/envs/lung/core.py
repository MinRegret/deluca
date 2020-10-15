import matplotlib.pyplot as plt
import numpy as np

from deluca.envs.core import Env


class Lung(Env):
    """
    Todos:
        - Plot incrementally
        - Save fig as fig.self
        - Modify close to delete fig and accumulated data
    """

    def render(self):
        fig, ax = plt.subplots()
        buf, (nrow, ncol) = fig.canvas.print_to_buffer()
        rgb_data = np.frombuffer(buf, dtype=np.uint8).reshape(nrow, ncol, 4)[:, :, :3]

        return rgb_data
