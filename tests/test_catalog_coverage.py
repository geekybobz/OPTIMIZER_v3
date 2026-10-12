"""Public exports and the catalog declarations stay one and the same.

Namespaces build ``__all__`` from their catalog declarations
(``opt.catalog.exports``), so a declaration is the only way to export a tool.
These tests check what that generation cannot: every exported name resolves,
no library function or class is reachable in a namespace without a
declaration, every declaration resolves to its object and names its families,
and the standalone ``optimizer_log`` reader, which keeps a literal list so it
never imports the optimizer package, matches its declarations.
"""

from __future__ import annotations

import collections.abc
import importlib
import inspect
import types
import unittest

import optimizer as opt
from optimizer import catalog
from optimizer.campaigns import MethodSpec
from optimizer.continuation import _registered_optimizer
from optimizer.families import FAMILY_NAMES


NAMESPACES = {
    "optimizers": "optimizer.optimizers",
    "guesses": "optimizer.guesses",
    "utils": "optimizer.utils",
    "schedules": "optimizer.schedules",
    "core": "optimizer.core",
    "campaigns": "optimizer.campaigns",
    "tracker": "optimizer.tracker",
    "visualization": "optimizer.visualization",
    "system_olgs": "optimizer.system_olgs",
    "system_olss": "optimizer.system_olss",
    "shooting": "optimizer.shooting",
    "optimizer_log": "optimizer_log",
}
HELPERS = {"info", "list", "search", "path"}


def _modules():
    yield "base", opt
    for group, module_name in NAMESPACES.items():
        yield group, importlib.import_module(module_name)


def _declared_objects() -> set[int]:
    return {
        id(item.public_object())
        for group_name in catalog.groups()
        for item in catalog.group(group_name).items
    }


def _loose_library_objects(module, declared: set[int]) -> list[str]:
    """Public functions and classes of a namespace's own package that no declaration covers.

    Objects imported from other packages are dependencies: an implementation
    module such as ``optimizer.campaigns`` imports tools for its own use. For
    the root package every library object counts.
    """

    exported = set(getattr(module, "__all__", ()))
    package = module.__name__
    loose = []
    for name, obj in vars(module).items():
        if name.startswith("_") or name in exported or isinstance(obj, types.ModuleType):
            continue
        if not (inspect.isfunction(obj) or inspect.isclass(obj)) or id(obj) in declared:
            continue
        origin = getattr(obj, "__module__", "") or ""
        if (origin == package or origin.startswith(package + ".")) and origin != "optimizer.catalog":
            loose.append(name)
    return sorted(loose)


class CatalogCoverageTests(unittest.TestCase):
    def test_every_exported_name_resolves(self):
        for group, module in _modules():
            with self.subTest(namespace=group):
                missing = [name for name in module.__all__ if not hasattr(module, name)]
                self.assertEqual(missing, [])
                exec(f"from {module.__name__} import *", {})

    def test_namespaces_export_exactly_their_declarations(self):
        for group, module in _modules():
            with self.subTest(namespace=group):
                exported = set(module.__all__) - HELPERS
                if group == "base":
                    exported = {name for name in exported if not isinstance(getattr(opt, name), (types.ModuleType, catalog.CatalogGroupView))}
                self.assertEqual(exported, set(catalog.exports(group)))

    def test_no_library_object_is_reachable_without_a_declaration(self):
        declared = _declared_objects()
        for group, module in _modules():
            with self.subTest(namespace=group):
                self.assertEqual(_loose_library_objects(module, declared), [])

    def test_every_declaration_resolves_and_names_its_families(self):
        for group_name in catalog.groups():
            for item in catalog.group(group_name).items:
                with self.subTest(item=item.id):
                    self.assertIsNotNone(item.public_object())
                    self.assertTrue(item.family)
                    self.assertTrue(set(item.family) <= set(FAMILY_NAMES))
                    if item.root:
                        self.assertIn(item.object_name or item.name, opt.__all__)

    def test_root_names_resolve_to_their_root_declaration(self):
        self.assertEqual(opt.info("finite_difference_jacobian")["id"], "utils.finite_difference_jacobian")
        self.assertEqual(opt.info("ChunkRecord")["id"], "base.ChunkRecord")
        self.assertEqual(opt.info("bind")["id"], "base.context")
        self.assertEqual(opt.info("follow_timeline")["id"], "tracker.follow_timeline")
        self.assertEqual(opt.finite_difference_jacobian.info()["id"], "utils.finite_difference_jacobian")
        self.assertEqual(opt.ChunkRecord.info()["id"], "base.ChunkRecord")
        self.assertEqual(opt.to_plain.info()["family"], ["olgs", "olss"])
        self.assertTrue(opt.info("adam")["root"])

    def test_protocols_and_typing_aliases_get_no_info_attribute(self):
        for protocol in (opt.OLGSystem, opt.SupportsHVP, opt.OpenLoopShootingSystem, opt.core.RunObserver):
            with self.subTest(protocol=protocol.__name__):
                self.assertNotIn("info", vars(protocol))
        self.assertFalse(hasattr(collections.abc.Mapping, "info"))

    def test_optimizer_names_come_from_the_declarations(self):
        names = catalog.exports("optimizers")
        self.assertIn("ncg", names)
        self.assertIs(_registered_optimizer("NCG"), opt.optimizers.nonlinear_cg)
        self.assertIs(MethodSpec("adam", {"maxiter": 1}).optimizer, opt.optimizers.adam)
        with self.assertRaisesRegex(KeyError, "valid names are"):
            _registered_optimizer("not_an_optimizer")
        with self.assertRaisesRegex(KeyError, "valid names are"):
            MethodSpec("not_an_optimizer", {"maxiter": 1})

    def test_methods_cover_every_root_function(self):
        methods = opt.methods()
        functions = {name for name in catalog.exports("base") if inspect.isfunction(getattr(opt, name))}
        self.assertEqual(set(methods), functions)
        self.assertEqual(methods["ncg"].module, "optimizer/optimizers/nonlinear_cg.py")
        self.assertEqual(methods["solve_shooting"].phase, "OLSS")
        self.assertEqual(methods["trace"].module, "optimizer/logs/trace.py")


if __name__ == "__main__":
    unittest.main()
