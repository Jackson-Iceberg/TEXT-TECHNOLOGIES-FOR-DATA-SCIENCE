import re
import xml.etree.ElementTree as ElementTree
from stemming.porter2 import stem
# import nltk
import numpy as np
class IR_tool(object):

    def __init__(self):
        # initilize the stopwords as set
        self.STOPWORDS = set()
        # initilize the docnums as list
        self.docnums = []
        # initilize the headlines
        self.headlines = []
        # initilize the texts
        self.texts = []
        # initilize pos_index
        self.pos_index = {}

    # read the file and store the English stopwords as set
    def generate_stopwords(self,filename):
        with open(filename) as file:
            self.STOPWORDS = set(file.read().splitlines())

    # read the xml file, store the relevant data into docnums,headlines,texts.
    def generate_collections(self,xmlFile):
        tree = ElementTree.parse(xmlFile)
        root = tree.getroot()
        for childs in root:
            for i in range(len(childs)):
                if childs[i].tag.lower() == 'docno':
                    self.docnums.append(childs[i].text.strip())
                elif childs[i].tag.lower() == 'headline':
                    self.headlines.append(childs[i].text.strip())
                elif childs[i].tag.lower() == 'text':
                    self.texts.append(childs[i].text.strip())

    # preprocessing the input data
    def preprocess(self,texts): 
        # tokenisation
        tokens = re.sub(r"[^\w\s]|_"," ",texts).split()
        # case folding and remove non-linear and non-digit tokens
        tokens = [token.lower() for token in tokens if token.isalpha() or token.isdigit()]
        # remove stop words and stemming
        tokens = [stem(token) for token in tokens if token not in self.STOPWORDS]
        return tokens
    
    # build the IR system
    def generate_pos_index(self):
        # index to get the relevant document id
        index = 0
        
        for token in self.texts:
            # preprocess the token
            token = ir_tool.preprocess(token)
            # get the document id
            fileno = self.docnums[index]
            index+=1

            for pos,term in enumerate(token):
                if term in self.pos_index:
                    # Increment total freq by 1.
                    self.pos_index[term][0] = self.pos_index[term][0] + 1
                        
                    # Check if the term has existed in that DocID before.
                    if fileno in self.pos_index[term][1]:
                        # pos+1 because pos starts at 0, but the requirement: pos starts at 1
                        self.pos_index[term][1][fileno].append(pos+1)
                            
                    else:
                        self.pos_index[term][1][fileno] = [pos+1]
                else:
                    # Initialize the list.
                    self.pos_index[term] = []
                    # The total frequency is 1.
                    self.pos_index[term].append(1)
                    # The postings list is initially empty.
                    self.pos_index[term].append({})     
                    # Add doc ID to postings list.
                    self.pos_index[term][1][fileno] = [pos+1]  
        
        # sort the dictionary in keys
        self.pos_index = dict(sorted(self.pos_index.items()))
           
    def generateIndexFile(self):
            # save the index.txt file
        with open("index.txt", "w") as f:
            for word in ir_tool.pos_index:
                df = len(ir_tool.pos_index[word][1])
                f.write('{}:{}\n'.format(word, df))
                for docno,position_list in ir_tool.pos_index[word][1].items():
                    idx = ','.join(map(str, position_list))
                    f.write('\t{}: {}\n'.format(str(docno), idx))
            
            f.close()

class searchEnginer(object):
    
    def __init__(self,docnums,pos_index,STOPWORDS):
        self.docnums = set(docnums) # efficient than list
        self.pos_index = pos_index
        self.queriesID = []
        self.queries = []
        self.result = []
        self.queriesID_rank = []
        self.queries_rank = []
        self.result_rank = {}
        self.STOPWORDS = STOPWORDS

    def process_queriesBooleanTxt(self,booleanTxt):
        with open(booleanTxt) as file:
            content= file.read().splitlines()
            for element in content:
                element = element.split(" ",1)
                self.queriesID.append(element[0])
                self.queries.append(element[1])
            file.close()

    def process_queriesRankedTxt(self,RankedTxt):
        with open(RankedTxt) as file:
            content= file.read().splitlines()
            for element in content:
                element = element.split(" ",1)
                self.queriesID_rank.append(element[0])
                self.queries_rank.append(element[1])
            file.close()

    def Querydiversion(self): # diversion based on the AND,OR. Otherwise, directly using the booleanSearch
        for element in self.queries:
            if (' AND ' in element): 
                LeftTerm, RightTerm = element.split(' AND ')
                LeftSet = self.booleanSearch(LeftTerm)  
                RightSet = self.booleanSearch(RightTerm)
                if (len(LeftSet) == 0 or len(RightSet) == 0): None # avoid empty set
                result = list(LeftSet & RightSet)  # AND
                self.result.append(result)

            elif(' OR ' in element):
                LeftTerm, RightTerm = element.split(' OR ')
                LeftSet = self.booleanSearch(LeftTerm)  
                RightSet = self.booleanSearch(RightTerm)
                result = list(LeftSet | RightSet)  # OR
                self.result.append(result)               
            else:
                result = self.booleanSearch(element) # the output of self.booleanSearch(element) is a set. Converted into list.
                if(len(result)==0): None # avoid empty set.
                else: self.result.append(list(result)) 

    def booleanSearch(self,term): # return set.
        NOTexist = False
        result = set()
        if(term.startswith('NOT')): # check NOT exist.
            term = term[4:]
            NOTexist = True
        
        if(term.startswith('"')):# terms belong to "A B" using phrase search.
            result = self.phraseSearch(term)
            if NOTexist: # if the NOT in front of this term. Get the different set of this term.
                result = (self.docnums-result) # difference set
                return result
            else: return result
                
        elif(term.startswith('#')): # terms belong to #distance (A,B) using proximity search
            result = self.proximitySearch(term)
            if NOTexist: # if the NOT in front of this term. Get the different set of this term.
                result = (self.docnums-result) # difference set
                return result
            else: return result

        else: # single term belong to single Term Search
            result = self.singleTermSearch(term) # result is a set
            if NOTexist: # if the NOT in front of this term. Get the different set of this term.
                result = (self.docnums-result) # difference set
                return result
            else: return result

    def preprocess(self,term):
        term = re.sub(r"[^\w\s]|_"," ",term).split()
        term = stem(term[0].lower())
        return term     
            
    def phraseSearch(self,term):
        result = set()
        LeftTerm, RightTerm = term.split(" ")
        LeftTerm = self.preprocess(LeftTerm)
        RightTerm = self.preprocess(RightTerm)
        setL = self.singleTermSearch(LeftTerm,True)
        setR = self.singleTermSearch(RightTerm,True)
        
        if(len(setL) == 0 or len(setR) == 0): # ensure both term has this word. 
            return result 
        intersection = setL & setR

        if (len(intersection) == 0): # represent the set is empty, there are no intersection document ID.
            return result
        else: 
            # most efficient way to get docno between two terms.
            for docno in intersection: 
                for indexL in self.pos_index[LeftTerm][1][docno]:   # LTermPositionList = self.pos_index[LeftTerm][1][docno]
                    if (indexL+1) in self.pos_index[RightTerm][1][docno]: # check indexL + 1 == indexR
                        result.add(docno)
                        break
            return result
        return result

    def proximitySearch(self,term):
        result = set()
        term = re.sub(r"[^\w\s]|_"," ",term).split()
        distance,LeftTerm, RightTerm = int(stem(term[0].lower())),stem(term[1].lower()),stem(term[2].lower())
        # distance,LeftTerm, RightTerm = term[0],term[1],term[2]
        setL = self.singleTermSearch(LeftTerm,True)
        setR = self.singleTermSearch(RightTerm,True)
        if(len(setL) == 0 or len(setR) == 0): # ensure both term has this word. 
            return result 
        intersection = setL & setR
        if (len(intersection) == 0): # represent the set is empty, there are no intersection document ID.
            return result
        else:
            for docno in intersection:   
                for indexL in self.pos_index[LeftTerm][1][docno]:
                    for indexR in self.pos_index[RightTerm][1][docno]: 
                        if(indexL+distance>=indexR and indexL < indexR): # ensure indexL in front of indexR
                            result.add(docno)
                            break
        return result

    def singleTermSearch(self,term,preprocess=False): # search the single term in the IR dictionary, return the set of DocumentID has this term. 
        if preprocess==False:
            term = self.preprocess(term)
        if term not in self.pos_index: 
            return set()
        return set(self.pos_index[term][1].keys())

    def generateBooleanTxt(self):
        with open('results.boolean.txt', 'w') as f:
            for id in self.queriesID:
                id = int(id)
                for res in search_tool.result:
                    for docID in res:
                        f.write('{},{}\n'.format(id, int(docID)))
            f.close()

    def preprocessQueries(self,queries):
        tokens = re.sub(r"[^\w\s]|_"," ",queries).split()
        # case folding and remove non-linear and non-digit tokens
        tokens = [token.lower() for token in tokens if token.isalpha() or token.isdigit()]
        # remove stop words and stemming
        tokens = [stem(token) for token in tokens if token not in self.STOPWORDS]
        return tokens

    def queriesRanked(self): # return [index,dictionary_ID_score]
        
        docnum_length = len(self.docnums)
        result = [None] * len(self.queries_rank)

        # result = [index:dicitonary]
        # dictionary_docID_score = {docID:score}
        for i in range(len(self.queries_rank)):
            query = self.queries_rank[i] # 
            query = self.preprocessQueries(query)
            dictionary_docID_score = {}
            for docno in self.docnums:
                score = 0
                for word in query:
                    if word not in self.pos_index.keys(): continue # this word doestn't in this document
                    else:
                        if docno not in self.pos_index[word][1]:continue
                        tf = len(self.pos_index[word][1][docno])
                        df = len(self.pos_index[word][1])
                        score += (1 + np.log10(tf)) * np.log10(docnum_length / df)
                if score > 0:
                    dictionary_docID_score[docno] = score
            # sort the dictionary
            dictionary_docID_score = sorted(dictionary_docID_score.items(), key=lambda x: x[1], reverse=True)
            result[i] = dictionary_docID_score
        
        self.result_rank = result

    def generateRankedTxt(self):
        with open('results.ranked.txt', 'w') as f:
            for i in range(len(self.queries_rank)):
                index = self.queriesID_rank[i]
                dictionary_docID_score = self.result_rank[i]
                for j in range(len(dictionary_docID_score)):
                    if j>150: # only need top 150
                        break
                    docID_score = dictionary_docID_score[j]
                    f.write('{},{},{:.4f}\n'.format(index, int(docID_score[0]), docID_score[1]))    
            f.close()

if __name__ == "__main__":
    ir_tool = IR_tool()
    ir_tool.generate_stopwords('./collections/englishST.txt')
    ir_tool.generate_collections('./collections/trec.5000.xml')
    ir_tool.generate_pos_index()
    ir_tool.generateIndexFile()
    search_tool = searchEnginer(ir_tool.docnums,ir_tool.pos_index,ir_tool.STOPWORDS)
    search_tool.process_queriesBooleanTxt("./collections/queries.boolean.txt")
    search_tool.Querydiversion()
    search_tool.generateBooleanTxt()
    search_tool.process_queriesRankedTxt("./collections/queries.ranked.txt")
    search_tool.queriesRanked()
    search_tool.generateRankedTxt()