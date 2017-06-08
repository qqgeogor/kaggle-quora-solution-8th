import unittest


class TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from stanford_parser.parser import Parser        
        cls.parser = Parser()
    
    def testParse(self):
        
        dependencies = self.parser.parseToStanfordDependencies("Pick up the tire pallet.")

        tupleResult = [(rel, gov.text, dep.text) for rel, gov, dep in dependencies.dependencies]
        self.assertEqual(tupleResult, [('prt', 'Pick', 'up'),
                                       ('det', 'pallet', 'the'),
                                       ('nn', 'pallet', 'tire'),
                                       ('dobj', 'Pick', 'pallet')])

        self.assertEqual(dependencies.tagForTokenStandoff(gov), "VB")
        self.assertEqual(dependencies.tagForTokenStandoff(dep), "NN") 

    def testParseRefexpNextTo(self):        
        dependencies = self.parser.parseToStanfordDependencies("Pick up the tire pallet next to the truck.")
        
        tupleResult = [(rel, gov.text, dep.text) for rel, gov, dep in dependencies.dependencies]

        
        self.assertEqual(tupleResult,
                         [('prt', 'Pick', 'up'),
                          ('det', 'pallet', 'the'),
                          ('nn', 'pallet', 'tire'),
                          ('dobj', 'Pick', 'pallet'),
                          ('det', 'truck', 'the'),
                          ('prep_next_to', 'pallet', 'truck')])


    def testParseRefexpNear(self):                
        dependencies =self.parser.parseToStanfordDependencies("Pick up the tire pallet near the truck.")
        
        tupleResult = [(rel, gov.text, dep.text) for rel, gov, dep in dependencies.dependencies]
        self.assertEqual(tupleResult,
                         [('prt', 'Pick', 'up'),
                          ('det', 'pallet', 'the'),
                          ('nn', 'pallet', 'tire'),
                          ('dobj', 'Pick', 'pallet'),
                          ('det', 'truck', 'the'),
                          ('prep_near', 'pallet', 'truck')])

        

    def testParseLong(self):                

        # this sentence has a self dependency that the python code filters out.
        # between drop and drop.
        dependencies = self.parser.parseToStanfordDependencies("Grab the skid of tires right in front of you " +
                                                               "and drop it off just in front and to the " +
                                                               "right of the far skid of tires.")
    
        tupleResult = [(rel, gov.text, dep.text) for rel, gov, dep in dependencies.dependencies]
        self.assertEqual(tupleResult,
                         [('det', 'skid', 'the'), ('dobj', 'Grab', 'skid'),
                          ('prep_of', 'skid', 'tires'), ('dep', 'Grab', 'right'),
                          ('prep_in', 'Grab', 'front'), ('prep_of', 'front', 'you'),
                          ('conj_and', 'Grab', 'drop'), ('dobj', 'drop', 'it'), ('prt', 'drop', 'off'),
                          ('advmod', 'drop', 'just'), ('prep_in', 'drop', 'front'), ('det', 'right', 'the'),
                          ('prep_to', 'drop', 'right'), ('det', 'skid', 'the'), ('amod', 'skid', 'far'),
                          ('prep_of', 'right', 'skid'), ('prep_of', 'skid', 'tires')])
                         
       
    
    def testAllCaps(self):
        dependencies = self.parser.parseToStanfordDependencies("GO TO THE TIRE PALLET NEXT TO THE TRUCK.")
        tupleResult = [(rel, gov.text, dep.text) for rel, gov, dep in dependencies.dependencies]
        self.assertEqual(tupleResult,
                         [('nn', 'PALLET', 'GO'),
                          ('nn', 'PALLET', 'TO'),
                          ('nn', 'PALLET', 'THE'),
                          ('nn', 'PALLET', 'TIRE'),
                          ('nsubj', 'NEXT', 'PALLET'),
                          ('dep', 'NEXT', 'TO'),
                          ('det', 'TRUCK', 'THE'),
                          ('dobj', 'TO', 'TRUCK')])
        

