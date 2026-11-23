import json
import tempfile
import unittest
from pathlib import Path

from bokp_ks.instance_generator import InstanceGenerator
from bokp_ks.result_writer import ResultWriter


class ResultWriterTests(unittest.TestCase):
    def test_instance_is_written_before_result(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = ResultWriter(Path(directory))
            instance = InstanceGenerator().generate(10)
            input_path = writer.write_instance(instance)
            self.assertTrue(input_path.exists())
            self.assertEqual(json.loads(input_path.read_text())["seed"], instance.seed)
            writer.write_result({"status": "COMPLETED", "instance": instance.as_dict(), "validation": {"passed": [], "failed": []}, "final_ks_points": []})
            self.assertTrue((Path(directory) / "instance_results" / f"{instance.instance_id}.json").exists())
            self.assertTrue((Path(directory) / "traces" / f"{instance.instance_id}.json").exists())
