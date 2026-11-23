import tempfile
import unittest
from pathlib import Path

from bokp_ks.domain import BOKPInstance
from bokp_ks.proportional_fairness import PFOptimizer, defining_pf_holds, enumerate_objective_points


class ProportionalFairnessTests(unittest.TestCase):
    def test_paper_example_has_no_pf_point(self):
        points = {(2, 2), (1, 4)}
        self.assertFalse(any(defining_pf_holds(point, points) for point in points))

    def test_enumerated_bokp_pf_point_satisfies_definition(self):
        instance = BOKPInstance("small", 4, 1, {}, {}, (1, 1, 1, 1), (3, 2, 1, 1), (1, 2, 3, 4), 2)
        feasible_points = enumerate_objective_points(instance)
        with tempfile.TemporaryDirectory() as directory:
            outcome = PFOptimizer(instance, Path(directory) / "pf.log").run()
        self.assertEqual(outcome.status, "PF_FOUND")
        self.assertIsNotNone(outcome.point)
        self.assertIn((outcome.point.F1, outcome.point.F2), feasible_points)
        self.assertTrue(defining_pf_holds((outcome.point.F1, outcome.point.F2), feasible_points))
        self.assertFalse(outcome.positivity_assumption_holds)


if __name__ == "__main__":
    unittest.main()
