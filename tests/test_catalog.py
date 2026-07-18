"""Tests for the programmatic exploration catalog."""

import unittest

import optimizer as opt


class CatalogTests(unittest.TestCase):
    def test_catalog_groups_and_namespace_lists_are_structured(self):
        groups = opt.catalog.groups()
        self.assertIn("optimizers", groups)
        self.assertIn("common_returns", groups["optimizers"])

        listing = opt.optimizers.list()
        self.assertEqual(listing["group"], "optimizers")
        self.assertIn("common_inputs", listing)
        self.assertIn("common_returns", listing)
        self.assertTrue(any(item["name"] == "adam" for item in listing["items"]))

    def test_human_list_contains_group_context_and_item_details(self):
        text = opt.optimizers.list(h=True)

        self.assertIn("optimizers", text)
        self.assertIn("Common inputs:", text)
        self.assertIn("Common returns:", text)
        self.assertIn("adam", text)
        self.assertIn("requires:", text)

    def test_method_info_is_attached_to_namespace_and_root_shortcuts(self):
        namespace_info = opt.optimizers.adam.info()
        root_info = opt.adam.info()

        self.assertEqual(namespace_info["id"], "optimizers.adam")
        self.assertEqual(root_info["id"], "optimizers.adam")
        self.assertEqual(namespace_info["returns"]["type"], "OptimizerResult")
        self.assertIn("variant", namespace_info["inputs"])
        self.assertIn("adam", namespace_info["variants"])
        self.assertIn("signature", namespace_info)
        self.assertTrue(callable(opt.optimizers.adam))
        self.assertTrue(callable(opt.adam))

    def test_method_human_info_includes_inputs_returns_and_example(self):
        text = opt.optimizers.adam.info(h=True)

        self.assertIn("optimizers.adam", text)
        self.assertIn("Inputs:", text)
        self.assertIn("Returns:", text)
        self.assertIn("Example:", text)
        self.assertIn("step_size", text)

    def test_group_info_and_catalog_attribute_views_work(self):
        via_group = opt.guesses.info("random_fourier_guess")
        via_catalog = opt.catalog.guesses.random_fourier_guess.info()

        self.assertEqual(via_group["id"], "guesses.random_fourier_guess")
        self.assertEqual(via_catalog["id"], "guesses.random_fourier_guess")
        self.assertIn("seed", via_group["inputs"])
        self.assertEqual(opt.schedules.AdaptiveStepSchedule.info()["id"], "schedules.AdaptiveStepSchedule")

    def test_filters_search_and_paths_support_recovery_when_lost(self):
        derivative_tools = opt.utils.list(kind="derivative")
        names = {item["name"] for item in derivative_tools["items"]}
        self.assertIn("verify_gradient", names)
        self.assertNotIn("repair_newton", names)

        search = opt.catalog.search("residual")
        self.assertGreater(search["count"], 0)
        self.assertTrue(any("residual" in " ".join(item.get("tags", [])) for item in search["items"]))

        path = opt.catalog.path("optimize_then_polish")
        self.assertIn("steps", path)
        self.assertTrue(any("adam" in step for step in path["steps"]))


if __name__ == "__main__":
    unittest.main()
