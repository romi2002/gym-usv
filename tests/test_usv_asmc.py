import numpy as np
import pytest
from gym_usv.control.usv_asmc import UsvAsmc


class TestUsvAsmc():
    n = 1000
    def test_no_movement(self):
        asmc = UsvAsmc()
        position = np.zeros(3)
        velocity = np.zeros(3)
        for _ in range(self.n):
            [position, velocity] = asmc.compute(np.zeros(2), position, velocity)

        assert np.allclose(position, np.zeros(3))
        assert np.allclose(velocity, np.zeros(3))

    def test_forward_movement(self):
        asmc = UsvAsmc()
        position = np.zeros(3)
        velocity = np.zeros(3)
        for _ in range(self.n):
            [position, velocity] = asmc.compute(np.array([10,0]), position, velocity)

        assert position[0] > 10
        assert np.all(np.abs(position[1:]) < 1)
        assert velocity[0] > 1
        assert np.all(np.abs(velocity[1:]) < 1)

    def test_rotation(self):
        asmc = UsvAsmc()
        position = np.zeros(3)
        velocity = np.zeros(3)
        for _ in range(self.n):
            [position, velocity] = asmc.compute(np.array([0,10]), position, velocity)

        assert position[2] > 5
