"""Tests for component configuration behavior."""

import unittest
from pathlib import Path

import wetb

wetb.__path__ = [str(Path(__file__).resolve().parents[3])]

from wetb.dlb.report_generator.dlb_components import Component


class ComponentTests(unittest.TestCase):
    def test_component_configuration_is_a_behavior_object(self):
        component = Component(
            "Tower Base",
            sections=["extreme", "extreme", "statistics"],
            fatigue_m=4,
            match="Tower Base",
        )

        self.assertEqual(component.label, "Tower Base")
        self.assertEqual(component.enabled_sections, ["extreme", "statistics"])
        self.assertEqual(component.fatigue_m, 4)
        self.assertEqual(
            component.match,
            {"name": "Tower Base", "description": "Tower Base"},
        )


if __name__ == "__main__":
    unittest.main()
