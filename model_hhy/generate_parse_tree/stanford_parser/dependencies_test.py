import unittest


class TestCase(unittest.TestCase):
    def testChildren(self):
        import dependencies
        hierarchy = dependencies.StanfordDependencyHierarchy()
        self.assertEqual(hierarchy.isa("agent", "arg"), True)

        self.assertEqual(hierarchy.isa("ref", "dep"), True)
        self.assertEqual(hierarchy.isa("dep", "dep"), False)

        self.assertEqual(hierarchy.isa("predet", "mod"), True)

        
        
    
