import unittest
from standoff import TextStandoff

class StandoffTestCase(unittest.TestCase):
    def testOverlaps(self):
        self.assertEqual(TextStandoff("Testing 123", (0, 1)).overlaps(TextStandoff("Testing 123", (0, 1))), True)
        self.assertEqual(TextStandoff("Testing 123", (0, 1)).overlaps(TextStandoff("Testing 123", (1, 2))), False)
        self.assertEqual(TextStandoff("Testing 123", (0, 10)).overlaps(TextStandoff("Testing 123", (1, 2))), True)
        self.assertEqual(TextStandoff("Testing 123", (0, 10)).overlaps(TextStandoff("Testing 123", (9, 10))), True)
        self.assertEqual(TextStandoff("Testing 123", (0, 10)).overlaps(TextStandoff("Testing 123", (10, 11))), False)
        self.assertEqual(TextStandoff("Testing 123", (10, 11)).overlaps(TextStandoff("Testing 123", (0, 10))), False)
        
        

    def testBefore(self):
        self.assertEqual(TextStandoff("Testing 123", (0, 1)).before(TextStandoff("Testing 123", (0, 1))), False)
        self.assertEqual(TextStandoff("Testing 123", (0, 1)).before(TextStandoff("Testing 123", (1, 2))), True)
        self.assertEqual(TextStandoff("Testing 123", (1, 2)).before(TextStandoff("Testing 123", (0, 1))), False)
        self.assertEqual(TextStandoff("Testing 123", (0, 10)).before(TextStandoff("Testing 123", (1, 2))), False)
        self.assertEqual(TextStandoff("Testing 123", (0, 10)).before(TextStandoff("Testing 123", (9, 10))), False)
        self.assertEqual(TextStandoff("Testing 123", (0, 10)).before(TextStandoff("Testing 123", (10, 11))), True)

        

    def testContains(self):
        self.assertEqual(TextStandoff("Testing 123", (0, 1)).contains(TextStandoff("Testing 123", (0, 1))), True)
        self.assertEqual(TextStandoff("Testing 123", (0, 1)).contains(TextStandoff("Testing 123", (1, 2))), False)
        self.assertEqual(TextStandoff("Testing 123", (1, 2)).contains(TextStandoff("Testing 123", (0, 1))), False)
        self.assertEqual(TextStandoff("Testing 123", (0, 10)).contains(TextStandoff("Testing 123", (1, 2))), True)
        self.assertEqual(TextStandoff("Testing 123", (0, 10)).contains(TextStandoff("Testing 123", (9, 10))), True)
        self.assertEqual(TextStandoff("Testing 123", (0, 10)).contains(TextStandoff("Testing 123", (10, 11))), False)
    def testDegreeOfOverlap(self):
        self.assertEqual(TextStandoff("Testing 123", (0, 1)).degreeOfOverlap(TextStandoff("Testing 123", (0, 1))), 1)
        self.assertEqual(TextStandoff("Testing 123", (0, 1)).degreeOfOverlap(TextStandoff("Testing 123", (1, 2))), 0)
        self.assertEqual(TextStandoff("Testing 123", (1, 2)).degreeOfOverlap(TextStandoff("Testing 123", (0, 1))), 0)
        self.assertEqual(TextStandoff("Testing 123", (0, 10)).degreeOfOverlap(TextStandoff("Testing 123", (1, 2))), 1)
        self.assertEqual(TextStandoff("Testing 123", (0, 10)).degreeOfOverlap(TextStandoff("Testing 123", (9, 10))), 1)
        self.assertEqual(TextStandoff("Testing 123", (0, 10)).degreeOfOverlap(TextStandoff("Testing 123", (10, 11))), 0)
        self.assertEqual(TextStandoff("Testing 123", (0, 10)).degreeOfOverlap(TextStandoff("Testing 123", (8, 11))), 2)
        self.assertEqual(TextStandoff("Testing 123", (8, 11)).degreeOfOverlap(TextStandoff("Testing 123", (0, 10))), 2)

        self.assertEqual(TextStandoff("Testing 123", (0, 5)).degreeOfOverlap(TextStandoff("Testing 123", (-1, 10))), 5)

        self.assertEqual(TextStandoff("Testing 123", (0, 5)).degreeOfOverlap(TextStandoff("Testing 123", (6, 125))), 0)
