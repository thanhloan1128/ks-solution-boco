import tempfile
import unittest
from pathlib import Path

from bokp_ks.gurobi_model import GurobiModel
from bokp_ks.instance_generator import InstanceGenerator
from bokp_ks.phase_one import PhaseOneSearch
from bokp_ks.reference_points import ReferencePointCalculator


class PhaseOneTests(unittest.TestCase):
    def test_rho_uses_exact_cross_product_and_search_records(self):
        instance = InstanceGenerator().generate(10)
        with tempfile.TemporaryDirectory() as directory:
            solver = GurobiModel(instance, Path(directory) / "solve.log")
            references, endpoint_p, endpoint_q = ReferencePointCalculator().calculate(solver)
            result = PhaseOneSearch().run(solver, references, endpoint_p, endpoint_q)
        self.assertIn(result["outcome"], {"KS_POINT", "C_SN"})
        self.assertGreater(result["iterations"], 0)
        self.assertTrue(result["trace"])
        point = endpoint_p
        self.assertEqual(references.rho_sign(point), 1)
