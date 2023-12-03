

#Basic Imports For NLP, File Processing, and Math
import nltk
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import glob





#These are downloads required for nltk to run. Once downloaded, they do not need to be downloaded again.

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')






#This performs NLP preprocessing on the text. First it creates
#a set of stopwords which we downloaded from nltk.  
#Then it tokenizes the text into individual phrases.
#Next we use list comprehension to remove all non-alphanumeric characters and stopwords and convert all words to lowercase.
#Finally, we return the preprocessed text as a string.
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    return ' '.join(words)



#This is a simple function that reads all the files in a directory into a list.
#For our case it takes the documents about different animals and stores them as individual strings in a list.
def read_files_into_list(directory):
    documents = []
    for filename in glob.glob(os.path.join(directory, '*')):
        with open(filename, 'r') as f:
            documents.append(f.read())
    return documents








def find_answer2(question, document):
    
    #First we tokenize the question and the document into words and sentences.
    question_words = word_tokenize(question)
    sentences = sent_tokenize(document)


    #Next we use Named Entity Recognition. This identifies and categorizs words
    #Into predefined categories such as names of persons, organizations, locations, expressions of times, etc.
    named_entities = Tree('S', ne_chunk(pos_tag(question_words)))


    #Then we initialize a TfidVectorizer object.
    vectorizer = TfidfVectorizer()

    #Then we transform the question and each sentence in the document into a matrix of
    #TF-IDF features. This stands for term frequency-inverse document frequency.
    #The term frequency is the number of times a word appears in a document.
    #The inverse document frequency is the log of the number of documents divided by the number of documents that contain the word.
    #This statistic reflects how important a wordis to a document.
    #The tfidf matrix is a matrix where each row represents a document
    #and each column represents a unique word in our vocabulary.
    #The value in each cell is the TF-IDF score of a word in a document.
    #The TF-IDF score is the product of the term frequency and the inverse document frequency.
    
    tfidf_matrix = vectorizer.fit_transform([question] + sentences)
    
    #Then we iterate through each sentence in the document and compute a score for each sentence.
    #The score is the cosine similarity between the question and the sentence plus the number of named entities in the sentence.
    #The cosine similarity is a measure of similarity between two vectors.
    #The cosine similarity between two vectors is the dot product of the two vectors divided by the product of the two vectors' magnitudes.
    best_match = None
    max_score = -1

    for i in range(1, tfidf_matrix.shape[0]):
        # Compute the cosine similarity between the question and the sentence
        similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[i])

        # Find matches with named entities
        sentence_words = set(word_tokenize(sentences[i-1]))
        named_entity_matches = len(sentence_words & set(named_entities.leaves()))

        # Compute a score for the sentence
        score = similarity + named_entity_matches
        if score > max_score:
            best_match = sentences[i-1]
            max_score = score

    return best_match




def find_answer(question):

    #This just gets the filepath of the directory that our documents are in and calls our function.
    filepath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    root = os.path.dirname(filepath)
    documents = read_files_into_list(root + '/info/Christopher_Boese_HW2_doc/documents')


    #Then we pre process our documents.
    processed_documents = [preprocess(doc) for doc in documents]

    #Here we prompt the user for a question and preprocess it.
    user_query = question
    processed_query = preprocess(user_query)

    #Here we use the tfidf vectorizer to convert our documents into vectors. First we initialize a TfidVectorizer object.
    #This will be used to convert our text into a matrix of TF-IDF features.
    vectorizer = TfidfVectorizer()
    
    #This line fits the vectorizer to the data and then transforms the data into a tfidf matrix.
    #Fit_transform is a method of the TfidfVectorizer class.
    #It fits the vectorizer to the data and then transforms the data into a tfidf matrix.
    tfidf_matrix = vectorizer.fit_transform([processed_query] + processed_documents)
    #Our resulting tfidf_matrix is a matrix where each row represents a document
    #and each column represents a unique word in our vocabulary. 
    #The value in each cell is the TF-IDF score of a word in a document. 
    #The TF-IDF score is the product of the term frequency and the inverse document frequency.
    #The term frequency is the number of times a word appears in a document.
    #The inverse document frequency is 
    #the log of the number of documents divided by the number of documents that contain the word.


    #Next we use cosine similarity to find the document that is most similar to our query.
    similarity_scores = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])
    #Then we find the index of the document with the highest similarity score.
    most_relevant_doc_index = similarity_scores.argmax()

    #Then we call find_answer with our query and the most relevant document
    most_relevant_doc = documents[most_relevant_doc_index]
    answer = find_answer2(user_query, most_relevant_doc)
    if answer is None:
        answer = "I don't know"
    return answer

if __name__ == '__main__':
    question = "What is a dog?"
    answer = find_answer(question)
    print(answer)




