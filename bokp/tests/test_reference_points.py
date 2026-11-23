import tempfile
import unittest
from pathlib import Path

from bokp_ks.gurobi_model import GurobiModel
from bokp_ks.instance_generator import InstanceGenerator
from bokp_ks.reference_points import ReferencePointCalculator


class ReferencePointTests(unittest.TestCase):
    def test_weighted_sum_endpoints_recompute(self):
        instance = InstanceGenerator().generate(10)
        with tempfile.TemporaryDirectory() as directory:
            references, endpoint_p, endpoint_q = ReferencePointCalculator().calculate(GurobiModel(instance, Path(directory) / "solve.log"))
        self.assertEqual((endpoint_p.p, endpoint_p.q), (references.p_max, 0))
        self.assertEqual((endpoint_q.p, endpoint_q.q), (0, references.q_max))
        self.assertEqual(instance.objectives(endpoint_p.x)[0] - references.f1_nadir, endpoint_p.p)
        self.assertEqual(instance.objectives(endpoint_p.x)[1] - references.f2_nadir, endpoint_p.q)
        self.assertEqual(instance.objectives(endpoint_q.x)[0] - references.f1_nadir, endpoint_q.p)
        self.assertEqual(instance.objectives(endpoint_q.x)[1] - references.f2_nadir, endpoint_q.q)
