import unittest

from bokp_ks.instance_generator import InstanceGenerator


class InstanceGenerationTests(unittest.TestCase):
    def test_seed_rule_and_ranges(self):
        instance = InstanceGenerator().generate(50)
        self.assertEqual(instance.seed, 20261017 + 50 * 1009)
        self.assertTrue(all(1 <= value <= 20 for value in instance.weights))
        self.assertTrue(all(1 <= value <= 30 for value in instance.p))
        self.assertTrue(all(10 <= value <= 40 for value in instance.q))
        self.assertEqual(instance.capacity, sum(instance.weights) // 2)

    def test_indexed_instances_extend_seed_rule(self):
        instance = InstanceGenerator().generate(50, 2)
        self.assertEqual(instance.instance_id, "bokp_n0050_i002")
        self.assertEqual(instance.seed, 20261017 + 50 * 1009 + 1)
