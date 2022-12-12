import math
import pandas as pd
import numpy as np
import time
from collections import defaultdict
import re
import scipy
from scipy.sparse import dok_matrix
from stemming.porter2 import stem
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from sklearn.svm import SVC
from sklearn.metrics import classification_report


class IR_evaluation(object):

    def __init__(self):
        # Dictionary where key = systemID and value = Dictionary, where key = queryID and value = [(docID, rank, score)]
        self.systemResult = defaultdict(lambda: defaultdict(list))
        # Dictionary where key = queryID and value = [(documentID, relevance value)]
        self.qrels = defaultdict(list)

    def eval(self):
        # Open the output file in write mode
        output_file = open("ir_eval.csv", "w")
        # Write the header to the output file
        header = "system_number,query_number,P@10,R@50,r-precision,AP,nDCG@10,nDCG@20\n"
        output_file.write(header)
        # Set the Precision and Recall values
        Precision = 10
        Recall = 50

        # Iterate over the system IDs
        for systemID in range(1, 7):
            # Initialize the precision and recall sums to 0
            precisionList = []
            recallList = []
            r_precisionList = []
            averagePrecisionList = []
            # Initialize the dictionary for storing the DCG values
            DCGDict = {10: [], 20: []}
            # Iterate over the query IDs
            for queryID in range(1, 11):
                # Get the documents for the current system and query
                docs = self.systemResult[systemID][queryID]  # value = [(docID, rank, score)]
                # Calculate the precision@10 and add it to the precision sum
                precision_10 = self.calculatePrecision(docs, queryID, Precision)
                precisionList.append(precision_10)

                # Get the length of the all documents for the current query
                length = len(self.qrels[queryID])

                # Calculate the recall@50 and add it to the recall List
                recall_50 = self.calculateRecall(docs, queryID, Recall, length)
                recallList.append(recall_50)

                # Calculate the r-precision and add it to the r-precision List
                r_precision = self.calculatePrecision(docs, queryID, length)
                r_precisionList.append(r_precision)

                # Calculate the average precision and add it to the average precision List
                average_precision = self.calculateAveragePrecision(docs, queryID)
                averagePrecisionList.append(average_precision)

                # Iterate over the cutoffs
                for cutoff in [10, 20]:
                    # Calculate the score and DCG for the current cutoff
                    score = self.calculateScore(cutoff, docs, queryID)
                    DCG = self.calculateDCG(score, cutoff)

                    # Calculate the ideal score and iDCG for the current cutoff
                    iscore = self.calculateiScore(cutoff, queryID)
                    iDCG = self.calculateDCG(iscore, cutoff)
                    # Calculate the nDCG for the current cutoff and append it to the DCG dictionary
                    DCGDict[cutoff].append(DCG / iDCG)

                # Format the results for the current system and query as a string and write them to the output file
                line = "{},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n".format(systemID, queryID, precisionList[-1],
                                                                                  recallList[-1], r_precisionList[-1],
                                                                                  averagePrecisionList[-1],
                                                                                  DCGDict[10][-1], DCGDict[20][-1])

                output_file.write(line)

            # Format the results for the current system and query as a string and write them to the output file
            line = "{},mean,{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n".format(systemID,
                                                                                np.mean(precisionList),
                                                                                np.mean(recallList),
                                                                                np.mean(r_precisionList),
                                                                                np.mean(averagePrecisionList),
                                                                                np.mean(DCGDict[10]),
                                                                                np.mean(DCGDict[20]))
            output_file.write(line)

        output_file.close()

    def calculateScore(self, cutoff, docs, queryID):
        # Initialize a list to store the scores for each document
        grade = [0] * cutoff

        # Iterate over the documents up to the cutoff number
        for i in range(cutoff):
            # Look up the score for the current document in the qrels dictionary
            for doc_id in self.qrels[queryID]:
                if docs[i][0] == doc_id[0]:
                    # If a score is found, add it to the grade list
                    grade[i] = doc_id[1]
                    break

        # Return the list of scores
        return grade

    def calculateDCG(self, grade, k):
        # Calculate the DCG for the first two elements in the grade list
        DCG = grade[0] + grade[1]

        # Iterate over the remaining elements in the grade list
        for i in range(2, k):
            # Add the current element divided by the logarithm (base 2) of the index
            # to the DCG variable
            DCG += grade[i] / math.log2(i + 1)

        # Return the calculated DCG value
        return DCG

    def calculateiScore(self, cutoff, queryID):
        # initialize the iscore list with the values from self.qrels
        iscore = [rel[1] for rel in self.qrels[queryID]]

        # add zeros to the iscore list if necessary
        iscore.extend([0] * (cutoff - len(iscore)))

        # return the iscore list
        return iscore[:cutoff]

    def calculateAveragePrecision(self, docs, queryID):
        # initialize the result variable
        result = 0

        # iterate over the docs and calculate the average precision
        for i, doc in enumerate(docs):
            precision = self.calculatePrecision(docs, queryID, i + 1)
            if precision > 0:
                for docID in self.qrels[queryID]:
                    if doc[0] == docID[0]:
                        result += precision * 1
                        break

        # return the calculated average precision
        return result / len(self.qrels[queryID])

    def calculatePrecision(self, docs, queryID, rangeOfdocs):
        # Initialize a variable to count the number of relevant documents
        count = 0

        # Iterate over the first rangeOfdocs documents in the docs list
        for doc in docs[:rangeOfdocs]:
            # Look up the document's relevance in the qrels dictionary
            for docID in self.qrels[queryID]:
                if doc[0] == docID[0]:
                    # If the document is relevant, increment the count
                    count += 1
                    break

        # Return the count divided by the rangeOfdocs parameter as the precision
        return count / rangeOfdocs

    def calculateRecall(self, docs, queryID, rangeOfdocs, length):
        # Initialize a variable to count the number of relevant documents
        count = 0

        # Iterate over the first rangeOfdocs documents in the docs list
        for doc in docs[:rangeOfdocs]:
            # Look up the document's relevance in the qrels dictionary
            for docID in self.qrels[queryID]:
                if doc[0] == docID[0]:
                    # If the document is relevant, increment the count
                    count += 1
                    break

        # Return the count divided by the length parameter as the recall
        return count / length

    def system_result_csv_reader(self, path):
        # Read the CSV file into a DataFrame
        csv_file = pd.read_csv(path)

        # Iterate over the rows in the DataFrame
        for i in range(len(csv_file)):
            # Create an empty dictionary, where key = queryID and value = [(docID, rank, score)]
            values = {}
            # Get the values from the row
            systemID = csv_file['system_number'][i]
            queryID = csv_file['query_number'][i]
            docID = csv_file['doc_number'][i]
            rank = csv_file['rank_of_doc'][i]
            score = csv_file['score'][i]

            # Create a tuple with the values
            items = (docID, rank, score)
            # Add the items to the dictionary
            values[queryID] = items

            # If the queryID is already in the systemResult dictionary, append the items to the list
            if queryID in self.systemResult[systemID]:
                self.systemResult[systemID][queryID].append(items)
            # Otherwise, create a new entry in the dictionary for the queryID and append the items to the list
            else:
                self.systemResult[systemID].setdefault(queryID, []).append(items)

    def qrels_csv_reader(self, path):
        # Read the CSV file into a DataFrame
        csv_file = pd.read_csv(path)
        # Iterate over the rows in the DataFrame
        for i in range(len(csv_file)):
            # Get the values from the row
            queryID = csv_file['query_id'][i]
            docID = csv_file['doc_id'][i]
            relevance = csv_file['relevance'][i]
            # Create a tuple with the values
            items = (docID, relevance)
            # If the queryID is already in the qrels dictionary, append the items to the list
            if queryID in self.qrels:
                self.qrels[queryID].append(items)
            # Otherwise, create a new entry in the dictionary for the queryID and append the items to the list
            else:
                self.qrels.setdefault(queryID, []).append(items)


class Text_analysis(object):

    def __init__(self):
        # Initialize the trainDict attribute to an empty dictionary
        self.trainDict = defaultdict()
        # Initialize the termsList attribute to an empty list
        self.termsList = []
        # Initialize the STOPWORDS attribute to an empty set
        self.STOPWORDS = set()

    def generate_stopwords(self, path):
        # Open the file at the given path
        with open(path) as file:
            # Read in the stopwords from the file and add them to the STOPWORDS attribute
            self.STOPWORDS = set(file.read().splitlines())
        # Close the file
        file.close()

    def train_dev_csv_reader(self, path):
        # Read in the data from the CSV file using pandas
        file = pd.read_csv(path, sep='\t', header=None)

        # Initialize an empty string to hold the text of the file
        text = ''

        # Loop through each row in the file
        for i in range(len(file)):
            # Get the corpora and verses values from the row
            corpora = file[0][i]
            verses = file[1][i]

            # If the corpora value is already in the trainDict, append the verses value to the corresponding list
            if corpora in self.trainDict:
                self.trainDict[corpora].append(verses)
            # Otherwise, add the corpora value as a new key in the trainDict and add the verses value as the first element in the corresponding list
            else:
                self.trainDict.setdefault(corpora, []).append(verses)

            # Add the verses value to the text string
            text += verses + '\n'

        # Loop through each corpora in the trainDict
        for corpora in self.trainDict:
            # Loop through each verses value in the corresponding list
            for i, verses in enumerate(self.trainDict[corpora]):
                # Preprocess the verses value and replace the original value in the list with the preprocessed value
                self.trainDict[corpora][i] = self.preprocessText(verses)

        # Preprocess the entire text of the file and set the termsList attribute to the resulting list of unique terms
        self.termsList = list(set(self.preprocessText(text)))

    def preprocessText(self, text):
        # Tokenize the text by splitting it into individual words, removing punctuation, and stripping leading and trailing whitespace
        tokens = re.sub(r"[^\w\s]|_", " ", text).split()

        # Perform case folding on the tokens (convert them all to lowercase) and remove any non-alphanumeric tokens and stopwords
        tokens = [stem(token.lower()) for token in tokens if token.lower() not in self.STOPWORDS]

        # Return the list of preprocessed tokens
        return tokens

    def WordLevel(self):
        # Compute MI and CHI scores for different orders of corpora
        orders = [['Quran', 'OT', 'NT'], ['OT', 'Quran', 'NT'], ['NT', 'Quran', 'OT']]
        N = len(self.trainDict['Quran']) + len(self.trainDict['OT']) + len(self.trainDict['NT'])
        for order in orders:
            target = order[0]  # The corpus for which MI and CHI scores will be calculated
            othersTargets = [order[1], order[2]]  # The other corpora
            targetsize = len(self.trainDict[target])
            othersize = N - targetsize

            chi = []  # List to store CHI results for the target corpus
            mi = []  # List to store MI results for the target corpus
            mires = []  # List to store intermediate MI results
            chires = []  # List to store intermediate CHI results

            # Loop through all terms in the list of terms
            for term in self.termsList:
                N11 = 0
                N10 = 0
                # Count the number of documents in the target corpus containing the term
                for item in self.trainDict[target]:
                    if term in item:
                        N11 += 1
                # Total number of documents in target corpus not containing term
                N01 = targetsize - N11

                # Count the number of documents in the other corpora containing the term
                for corpora in othersTargets:
                    for item in self.trainDict[corpora]:
                        if term in item:
                            N10 += 1

                # Total number of documents in other corpora not containing term
                N00 = othersize - N10

                # Compute the remaining counts needed to calculate MI and CHI values
                N1x = N11 + N10
                Nx1 = N11 + N01
                N0x = N00 + N01
                Nx0 = N00 + N10

                # Compute MI and CHI values for the term
                if N * N11 != 0 and N1x * Nx1 != 0:
                    sub1 = np.log2(N * N11 / (N1x * Nx1))
                else:
                    sub1 = 0
                if N * N01 != 0 and N0x * Nx1 != 0:
                    sub2 = np.log2(N * N01 / (N0x * Nx1))
                else:
                    sub2 = 0
                if N * N10 != 0 and N1x * Nx0 != 0:
                    sub3 = np.log2(N * N10 / (N1x * Nx0))
                else:
                    sub3 = 0
                if N * N00 != 0 and N0x * Nx0 != 0:
                    sub4 = np.log2(N * N00 / (N0x * Nx0))
                else:
                    sub4 = 0

                below = Nx1 * N1x * Nx0 * N0x

                # Store MI and CHI values for the term in intermediate lists
                mivalue = (N11 / N) * sub1 + (N01 / N) * sub2 + (N10 / N) * sub3 + (N00 / N) * sub4
                mires.append([term, mivalue])
                chivalue = N * (N11 * N00 - N10 * N01) ** 2 / below if below != 0 else 0
                chires.append([term, chivalue])

            # Sort the intermediate MI and CHI results in descending order
            mi.append(sorted(mires, key=lambda x: x[-1], reverse=True))
            chi.append(sorted(chires, key=lambda x: x[-1], reverse=True))

            # Print the top 10 MI and CHI results for each target corpus
            print(target + "-MI result")
            for mivalue in mi:
                print(mivalue[:10])

            print(target + "-CHI result")
            for chivalue in chi:
                print(chivalue[:10])

    def TopicLevel(self):
        # Create a list of texts from each of the three sub-datasets
        Quran = self.trainDict['Quran']
        OT = self.trainDict['OT']
        NT = self.trainDict['NT']
        texts = Quran + OT + NT

        # Create a dictionary of words from the texts
        dictionary = Dictionary(texts)

        # Create a corpus of bag-of-words representations of the texts
        corpus = [dictionary.doc2bow(text) for text in texts]

        # Train a Latent Dirichlet Allocation (LDA) model on the corpus to identify 20 topics
        LDA = LdaModel(corpus, id2word=dictionary, num_topics=20, random_state=1000)

        # Calculate the top 3 topics and top 10 scores for the Quran sub-dataset
        quran_topic_scores, quran_tokens = self.calculateTop3_topic_Top10_score(Quran, dictionary, LDA)

        # Calculate the top 3 topics and top 10 scores for the OT sub-dataset
        ot_topic_scores, ot_tokens = self.calculateTop3_topic_Top10_score(OT, dictionary, LDA)

        # Calculate the top 3 topics and top 10 scores for the NT sub-dataset
        nt_topic_scores, nt_tokens = self.calculateTop3_topic_Top10_score(NT, dictionary, LDA)

        # Print the results for each of the three sub-datasets
        print('Quran Topic', quran_tokens)
        print('Quran Score', quran_topic_scores)
        print('OT Topic', ot_tokens)
        print('OT Score', ot_topic_scores)
        print('NT Topic', nt_tokens)
        print('NT Score', nt_topic_scores)

    def calculateTop3_topic_Top10_score(self, targetText, dictionary, LDA):
        # Create a list of scores for each text in the targetText dataset
        score_list = []
        for text in targetText:
            score_list.append(LDA.get_document_topics(bow=dictionary.doc2bow(text)))

        # Calculate the average score for each topic
        # and select the top 3 topics based on the average scores
        top3_topic_scores = []
        topic_ids, avg_scores = self.calculate_avg_score_topic(score_list)
        topic_ids, avg_scores = topic_ids[:3], avg_scores[:3]
        for topic_id, avg_score in zip(topic_ids, avg_scores):
            top3_topic_scores.append([topic_id, avg_score])

        # Use the LDA.show_topic() method to get the top 10 tokens (words)
        # for each of the top 3 topics
        top_10_tokens = []
        for topic_score in top3_topic_scores:
            top_10_tokens.append(LDA.show_topic(topic_score[0]))

        # Return the resulting scores and tokens
        return top3_topic_scores, top_10_tokens

    def calculate_avg_score_topic(self, all_topic_score_list):
        # Initialize a list of 20 zeros to keep track of the total score for each topic
        topic_scores = [0] * 20

        # Iterate over the list of scores for each text and add the score for each topic
        # to the corresponding entry in the topic_scores list
        for doc_topic_scores in all_topic_score_list:
            for (topic_id, topic_score) in doc_topic_scores:
                topic_scores[topic_id] += topic_score

        # Calculate the average score for each topic by dividing the topic_scores list
        # by the number of texts in the dataset
        avg_scores = np.array(topic_scores) / len(all_topic_score_list)

        # Sort the topics by their average scores in descending order
        topic_ids = np.argsort(avg_scores)[::-1].tolist()
        avg_scores = avg_scores[topic_ids].tolist()

        # Return the resulting list of topic IDs and average scores
        return topic_ids, avg_scores


class Text_classification(object):

    def __init__(self):
        self.STOPWORDS = set()

    def generate_stopwords(self, path):
        # Open the file at the given path
        with open(path) as file:
            # Read in the stopwords from the file and add them to the STOPWORDS attribute
            self.STOPWORDS = set(file.read().splitlines())
        # Close the file
        file.close()

    def preprocessText(self, text, stopKeep=False):
        # Tokenize the text by splitting it into individual words, removing punctuation, and stripping leading and trailing whitespace
        tokens = re.sub(r"[^\w\s]|_", " ", text).split()
        if stopKeep:
            tokens = [stem(token.lower()) for token in tokens]
        else:
            # Perform case folding on the tokens (convert them all to lowercase) and remove any non-alphanumeric tokens and stopwords
            tokens = [stem(token.lower()) for token in tokens if token.lower() not in self.STOPWORDS]

        # Return the list of preprocessed tokens
        return tokens

    def generate_train_dev_dataSet(self, path, stopKeep=False):
        # Read the file using the pandas library and shuffle the rows
        file = pd.read_csv(path, sep='\t', header=0)
        file = file.sample(frac=1)

        # Split the file into training and development datasets, with 90% of the rows in the training set
        # and the remaining 10% in the development set
        length = int(0.9 * len(file))
        train = file[:length]
        dev = file[length:]

        # Extract the text and labels for each dataset
        train_tweet = train['tweet'][:]
        train_label = train['sentiment'][:]
        dev_tweet = dev['tweet'][:]
        dev_label = dev['sentiment'][:]

        # Create a set of all the unique words (terms) in the training dataset
        termSet = set()

        # Preprocess the text in the tweet column of each dataset, using the preprocessText() method
        # and passing in the stopKeep flag
        train_data = []
        for line in train_tweet:
            texts = self.preprocessText(line, stopKeep)
            train_data.append(texts)
            termSet.update(set(texts))

        dev_data = []
        for line in dev_tweet:
            texts = self.preprocessText(line, stopKeep)
            dev_data.append(texts)

        # Return the preprocessed text and labels for each dataset, as well as the set of terms
        return train_data, train_label, dev_data, dev_label, termSet

    def generate_test_dataSet(self, path, stopKeep=False):
        # Read the file using the pandas library and extract the text and labels
        file = pd.read_csv(path, sep='\t', header=0)
        test_tweet = file['tweet'][:]
        test_label = file['sentiment'][:]


        # Preprocess the text using the preprocessText() method and passing in the stopKeep flag
        test_data = []
        for line in test_tweet:
            texts = self.preprocessText(line, stopKeep)
            test_data.append(texts)

        return test_data, test_label

    def generate_word2id(self, termSet):
        # Initialize an empty dictionary to store the word-to-ID mapping
        word2id = {}

        # Iterate over the words in the set and assign each word a unique ID, starting from 0
        for word_id, word in enumerate(termSet):
            # Add the ID for each word to the word2id dictionary as a key-value pair, where
            # the word is the key and the ID is the value
            word2id[word] = word_id

        # Return the word2id dictionary
        return word2id

    # build a BOW representation of the files: use the scipy
    # data is the preprocessed_data
    # word2id maps words to their ids
    def convert_to_bow_matrix(self, preprocessed_data, word2id):
        # Initialize a sparse matrix with size len(preprocessed_data) rows and len(word2id) + 1 columns
        # to represent the BOW matrix. The extra column is for out-of-vocabulary (OOV) words.
        matrix_size = (len(preprocessed_data), len(word2id) + 1)
        oov_index = len(word2id)
        X = scipy.sparse.dok_matrix(matrix_size)

        # Iterate through all documents in the dataset
        for doc_id, doc in enumerate(preprocessed_data):
            # Iterate through all words in the current document
            for word in doc:
                # If the word is not in the word2id dictionary, it is considered an OOV word
                # and the count is added to the last column of the BOW matrix
                X[doc_id, word2id.get(word, oov_index)] += 1

        # Return the resulting BOW matrix
        return X

    def baseLine(self, Trainpath, TestPath):
        # Preprocess the training and test data and split the training data into training and development sets
        train_data, train_label, dev_data, dev_label, termSet = self.generate_train_dev_dataSet(Trainpath)
        test_data, test_label = self.generate_test_dataSet(TestPath)

        # Generate a word-to-ID mapping using the set of words (terms) extracted from the training data
        word2id = self.generate_word2id(termSet)

        # Convert the preprocessed training, development, and test data into BOW matrices using the generated
        # word-to-ID mapping
        x_train = self.convert_to_bow_matrix(train_data, word2id)
        x_dev = self.convert_to_bow_matrix(dev_data, word2id)
        x_test = self.convert_to_bow_matrix(test_data, word2id)

        # Train a support vector machine (SVM) model using the BOW representation of the training data
        model = SVC(kernel='linear', C=1000)
        model.fit(x_train, train_label)

        # Use the trained model to make predictions on the BOW representation of the development, training,
        # and test data
        x_train_pred = model.predict(x_train)
        x_dev_pred = model.predict(x_dev)
        x_test_pred = model.predict(x_test)
        print('baseline dev label real')
        print(dev_label)
        print('baseline predict dev label')
        print(x_dev_pred)
        print('baseline dev label real')
        print(test_label)
        print('baseline predict dev label')
        print(x_test_pred)
        # Calculate the precision, recall, and f1-score for each of the three classes, as well as the macro-averaged
        # precision, recall, and f1-score across all three classes for the development, training, and test datasets
        train_report = classification_report(train_label, x_train_pred,
                                             target_names=['negative', 'neutral', 'positive'], output_dict=True)
        dev_report = classification_report(dev_label, x_dev_pred, target_names=['negative', 'neutral', 'positive'],
                                           output_dict=True)
        test_report = classification_report(test_label, x_test_pred, target_names=['negative', 'neutral', 'positive'],
                                            output_dict=True)
        return train_report, dev_report, test_report

    def improved(self, Trainpath, TestPath):
        # Preprocess the training and test data and split the training data into training and development sets.
        # The preprocessing includes keeping stopwords this time.
        train_data, train_label, dev_data, dev_label, termSet = self.generate_train_dev_dataSet(Trainpath, True)
        test_data, test_label = self.generate_test_dataSet(TestPath, True)

        # Generate a word-to-ID mapping using the set of words (terms) extracted from the training data
        word2id = self.generate_word2id(termSet)

        # Convert the preprocessed training, development, and test data into BOW matrices using the generated word-to-ID mapping
        x_train = self.convert_to_bow_matrix(train_data, word2id)
        x_dev = self.convert_to_bow_matrix(dev_data, word2id)
        x_test = self.convert_to_bow_matrix(test_data, word2id)

        # Train a support vector machine (SVM) model using the BOW representation of the training data and the RBF kernel
        model = SVC(kernel='rbf', C=10)
        model.fit(x_train, train_label)

        # Use the trained model to make predictions on the BOW representation of the development, training, and test data
        x_train_pred = model.predict(x_train)
        x_dev_pred = model.predict(x_dev)
        x_test_pred = model.predict(x_test)
        print('improved dev label real')
        print(dev_label)
        print('improved predict dev label')
        print(x_dev_pred)
        print('improved dev label real')
        print(test_label)
        print('improvedimproved predict dev label')
        print(x_test_pred)
        # Calculate the precision, recall, and f1-score for each of the three classes, as well as the macro-averaged
        # precision, recall, and f1-score across all three classes for the development, training, and test datasets
        train_report = classification_report(train_label, x_train_pred,
                                             target_names=['negative', 'neutral', 'positive'], output_dict=True)
        dev_report = classification_report(dev_label, x_dev_pred, target_names=['negative', 'neutral', 'positive'],
                                           output_dict=True)
        test_report = classification_report(test_label, x_test_pred, target_names=['negative', 'neutral', 'positive'],
                                            output_dict=True)

        return train_report, dev_report, test_report

    def generate_classification_file(self, report, system, split):
        # Open the classification.csv file in append mode
        f = open("classification.csv", "a")
        # If the system is 'baseline' and the split is 'train', write a header row to the file
        if system == 'baseline' and split == 'train':
            header = "system,split,p-pos,r-pos,f-pos,p-neg,r-neg,f-neg,p-neu,r-neu,f-neu,p-macro,r-macro,f-macro\n"
            f.write(header)
        # Construct a line of data for the current classification report and write it to the file
        line = system + ',' + split + ',' + "{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n".format(
            report['positive']['precision'], report['positive']['recall'], report['positive']['f1-score'],
            report['negative']['precision'], report['negative']['recall'], report['negative']['f1-score'],
            report['neutral']['precision'], report['neutral']['recall'], report['neutral']['f1-score'],
            report['macro avg']['precision'], report['macro avg']['recall'], report['macro avg']['f1-score'])
        f.write(line)


if __name__ == "__main__":
    timeStart = time.time()
    ir_eval = IR_evaluation()
    ir_eval.system_result_csv_reader('./docs/system_results.csv')
    ir_eval.qrels_csv_reader('./docs/qrels.csv')
    ir_eval.eval()
    print("time spend for evaluation = ", time.time() - timeStart)
    text_analysis = Text_analysis()
    text_analysis.generate_stopwords('./docs/englishST.txt')
    text_analysis.train_dev_csv_reader('./docs/train_and_dev.tsv')
    text_analysis.WordLevel()
    text_analysis.TopicLevel()
    print("time spend for evaluation = ", time.time() - timeStart)
    text_classification = Text_classification()
    text_classification.generate_stopwords('./docs/englishST.txt')
    train_report_basic, dev_report_basic, test_report_basic = text_classification.baseLine('./docs/train.txt',
                                                                                           './docs/test.txt')
    train_report_improv, dev_report_improv, test_report_improv = text_classification.improved('./docs/train.txt',
                                                                                              './docs/test.txt')
    text_classification.generate_classification_file(train_report_basic, 'baseline', 'train')
    text_classification.generate_classification_file(dev_report_basic, 'baseline', 'dev')
    text_classification.generate_classification_file(test_report_basic, 'baseline', 'test')

    text_classification.generate_classification_file(train_report_improv, 'improved', 'train')
    text_classification.generate_classification_file(dev_report_improv, 'improved', 'dev')
    text_classification.generate_classification_file(test_report_improv, 'improved', 'test')

    print(train_report_basic)
    print(dev_report_basic)
    print(test_report_basic)

    print(train_report_improv)
    print(dev_report_improv)
    print(test_report_improv)
    print("time spend for evaluation = ", time.time() - timeStart)
