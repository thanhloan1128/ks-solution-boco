import tempfile
import unittest
from pathlib import Path

from bokp_ks.gurobi_model import GurobiModel
from bokp_ks.instance_generator import InstanceGenerator
from bokp_ks.phase_one import PhaseOneSearch
from bokp_ks.phase_two import PhaseTwoSearch
from bokp_ks.reference_points import ReferencePointCalculator


class PhaseTwoTests(unittest.TestCase):
    def test_both_searches_use_epsilon_constraints(self):
        instance = InstanceGenerator().generate(50)
        with tempfile.TemporaryDirectory() as directory:
            solver = GurobiModel(instance, Path(directory) / "solve.log")
            references, endpoint_p, endpoint_q = ReferencePointCalculator().calculate(solver)
            phase_one = PhaseOneSearch().run(solver, references, endpoint_p, endpoint_q)
            self.assertEqual(phase_one["outcome"], "C_SN")
            positive = phase_one["positive"]
            negative = phase_one["negative"]
            from bokp_ks.domain import ObjectivePoint
            positive = ObjectivePoint(positive["P"], positive["Q"], tuple(positive["x"]))
            negative = ObjectivePoint(negative["P"], negative["Q"], tuple(negative["x"]))
            p_result = PhaseTwoSearch().run_p(solver, references, positive, negative)
            q_result = PhaseTwoSearch().run_q(solver, references, positive, negative)
        self.assertIsNotNone(p_result["point"])
        self.assertIsNotNone(q_result["point"])
        self.assertTrue(all(step["solution"]["Q"] >= step["epsilon_m"] for step in p_result["trace"]))
        self.assertTrue(all(step["solution"]["P"] >= step["epsilon_m"] for step in q_result["trace"]))
