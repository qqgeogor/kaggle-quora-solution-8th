import jpype
from standoff import TextStandoff

class ParserError(Exception):
    def __init__(self, *args, **margs):
        Exception.__init__(self, *args,**margs)


def standoffFromToken(txt, token):
    return TextStandoff(txt, (token.beginPosition(), token.endPosition()))


                   
           


class Dependencies:
    def __init__(self, sentence, tokens, posTags, dependencies):
        self.sentence = sentence

        self.posTags = posTags        
        
        self.tokens = tokens

        self.tokensToPosTags = dict(zip(self.tokens, self.posTags))

        self.dependencies = dependencies
        
        self.govToDeps = {}
        self.depToGov = {}
        self.constituentsToRelation = {}

        # there is a bug where sometimes there is a self dependency.
        self.dependencies = [(relation, gov, dep) for relation, gov, dep in self.dependencies
                             if gov != dep]


        for relation, gov, dep in self.dependencies:

            self.govToDeps.setdefault(gov, [])
            self.govToDeps[gov].append(dep)
            assert not dep in self.depToGov, (dep.text, [(key.text, value.text)
                                                         for key, value in self.depToGov.iteritems()])
            self.depToGov[dep] = gov
            self.constituentsToRelation[(gov,dep)] = relation
            
        self.checkRep()

    def tagForTokenStandoff(self, tokenStandoff):
        return self.tokensToPosTags[tokenStandoff]
        
        
    def checkRep(self):
        assert len(self.posTags) == len(self.posTags)        
        for t in self.tokens:
            assert t.entireText == self.sentence



    def govForDep(self, dep):
        return self.depToGov[dep]
    def depsForGov(self, gov):
        return self.govToDeps[gov]

    def relForConstituents(self, gov, dep):
        return self.constituentsToRelation((gov, dep))
    
    def __str__(self):
        result = ""
        result += "sentence=" + repr(self.sentence) + "\n"
        for relation, gov, dep in self.dependencies:
            result += relation + "(" + gov.text + ", " + dep.text + ")\n"
        return result

stanford_parser_home = None

def startJvm():
    import os
    os.environ.setdefault("STANFORD_PARSER_HOME", "3rdParty/stanford-parser/stanford-parser-2010-08-20")
    global stanford_parser_home
    #stanford_parser_home = os.environ["STANFORD_PARSER_HOME"]
    stanford_parser_home = '/home/hhyhhyhy/stanford-parser-2010-08-20'
    jpype.startJVM(jpype.getDefaultJVMPath(),
                   "-ea",
                   "-Djava.class.path=%s/stanford-parser.jar" % (stanford_parser_home),)
startJvm() # one jvm per python instance.

class Parser:

    def __init__(self, pcfg_model_fname=None):
        if pcfg_model_fname == None:
            #self.pcfg_model_fname = "%s/englishPCFG.ser" % stanford_parser_home
            #self.pcfg_model_fname = "%s/englishFactored.ser" % stanford_parser_home
            #self.pcfg_model_fname = "../stanford-parser/englishPCFG.July-2010.ser"
            self.pcfg_model_fname = "%s/englishPCFG.July-2010.ser"%stanford_parser_home
        else:
            self.pcfg_model_fname = pcfg_model_fname



        self.package_lexparser = jpype.JPackage("edu.stanford.nlp.parser.lexparser")
        
        self.parser = self.package_lexparser.LexicalizedParser(self.pcfg_model_fname)
        self.package = jpype.JPackage("edu.stanford.nlp")

        tokenizerFactoryClass = self.package.process.__getattribute__("PTBTokenizer$PTBTokenizerFactory")
        self.tokenizerFactory = tokenizerFactoryClass.newPTBTokenizerFactory(True, True)

        self.documentPreprocessor = self.package.process.DocumentPreprocessor(self.tokenizerFactory)
        
        
        self.parser.setOptionFlags(["-retainTmpSubcategories"])




    def printInfo(self):

        Numberer = self.package.util.Numberer
        print ("Grammar\t" +
               `Numberer.getGlobalNumberer("states").total()` + '\t' +
               `Numberer.getGlobalNumberer("tags").total()` + '\t' +
               `Numberer.getGlobalNumberer("words").total()` + '\t' +
               `self.parser.pparser.ug.numRules()` + '\t' +
               `self.parser.pparser.bg.numRules()` + '\t' +
               `self.parser.pparser.lex.numRules()`)

        print "ParserPack is ", self.parser.op.tlpParams.getClass()
        print "Lexicon is ", self.parser.pd.lex.getClass()        
        print "Tags are: ", Numberer.getGlobalNumberer("tags")
        self.parser.op.display()
        print "Test parameters"
        self.parser.op.tlpParams.display();
        self.package_lexparser.Test.display()
    def parse(self, sentence):
        """
        Parses the sentence string, returning the tokens, and the parse tree as a tuple.
        tokens, tree = parser.parse(sentence)
        """

        tokens = self.documentPreprocessor.getWordsFromString(sentence)
        for token in tokens:
            if token.word() in ["down"]:
                #print "setting tag"
                token.setTag("IN")
                pass
            if token.word().lower() in ["bot"]:
                token.setTag("NN")
                pass
        
        wasParsed = self.parser.parse(tokens)
        # if not wasParsed:
        #     raise ParserError("Could not parse " + sentence)
        return tokens, self.parser.getBestParse()

    #add tokens
    def getPosTag(self,tree):
        posTags = [token.tag() for token in tree.taggedYield()]
        return posTags

    def parseToStanfordDependencies(self, sentence):

        tokens, tree = self.parse(sentence)
        standoffTokens = [standoffFromToken(sentence, token)
                          for token in tokens]
        posTags = [token.tag() for token in tree.taggedYield()]
        #print " ".join(["%s/%s" % (word.text, tag) for word, tag in zip(standoffTokens, posTags)])
        #print tree.taggedYield().toString(False)
        result = self.package.trees.EnglishGrammaticalStructure(tree)
        
        returnList = []
        for dependency in result.typedDependenciesCollapsedTree():

            govStandoff = standoffTokens[dependency.gov().index() - 1]
            depStandoff = standoffTokens[dependency.dep().index() - 1]

            returnList.append((str(dependency.reln()),
                               govStandoff,
                               depStandoff))



        return Dependencies(sentence, standoffTokens, posTags, returnList)
                              
