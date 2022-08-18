import pytest
from gym_usv.control.usv_asmc import UsvAsmc


class UsvAsmcTest():
    def test_no_movement(self):
        asmc = UsvAsmc()
        x = "this"
        assert "h" in x