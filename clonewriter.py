import javalang
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import json
import jinja2.ext
import pandas as pd
from nltk.corpus import wordnet
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import nltk
import re
from more_itertools import unique_everseen
# import os
# from rake_nltk import Rake
from sklearn.metrics.pairwise import cosine_similarity

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config
)

app = Flask(__name__)

tokenizer = GPT2Tokenizer.from_pretrained('DeepClone')
config = GPT2Config.from_pretrained('DeepClone')
model = GPT2LMHeadModel.from_pretrained('DeepClone', config=config)
model.to('cpu')

@app.route('/')
def index():
    # return render_template('form.html')
    return render_template('editor.html')


@app.route('/process', methods=['POST'])
def process():
    email = request.form['email']
    name = request.form['name']

    if name and email:
        newName = name[::-1]

        return jsonify({'name': newName, 'codearea': 'yes'})

    return jsonify({'error': 'Missing data!'})


@app.route('/getTop10PredictionedTokens', methods=['POST'])
def getTop10PredictionedTokens():
    inputString = request.form['inputString']
    #topp=request.form['topp']
    #tokenSize=request.form['tokenSize']

    tokenize = list(javalang.tokenizer.tokenize(inputString))
    tokens = []
    for m in range(0, len(tokenize)):
        tokentype = tokenize[m].__class__.__name__
        if (tokentype == 'DecimalFloatingPoint' or tokentype == 'DecimalInteger' or tokentype == 'Integer' or
                tokentype == 'OctalInteger' or tokentype == 'BinaryInteger' or tokentype == 'HexInteger' or
                tokentype == 'FloatingPoint' or tokentype == 'HexFloatingPoint' or tokentype == 'Boolean' or tokentype == 'Literal'):
            tokens.append("<num_val>")
        elif (tokentype == 'Character' or tokentype == 'String'):
            tokens.append("<str_val>")
        else:
            tokens.append(tokenize[m].value)
    #selectedTokens=[]

    #if tokenSize<len(tokens):
    #    selectedTokens=tokens[:-tokenSize]
    #else:
    #    selectedTokens=tokens

    inputSample = " ".join(tokens)

    tokens, scores = generateTop10Suggestions(inputSample, model, tokenizer)
    return jsonify({'tokens': tokens, 'scores': scores})


@app.route('/getRecommendations', methods=['POST'])
def getRecommendations():
    inputString = request.form['inputString']
    topp=float(request.form['topp'])
    tokenSize=int(request.form['tokenSize'])
    kvalue=int(request.form['kvalue'])

    tokenize = list(javalang.tokenizer.tokenize(inputString))
    tokens = []
    for m in range(0, len(tokenize)):
        tokentype = tokenize[m].__class__.__name__
        if (tokentype == 'DecimalFloatingPoint' or tokentype == 'DecimalInteger' or tokentype == 'Integer' or
                tokentype == 'OctalInteger' or tokentype == 'BinaryInteger' or tokentype == 'HexInteger' or
                tokentype == 'FloatingPoint' or tokentype == 'HexFloatingPoint' or tokentype == 'Boolean' or tokentype == 'Literal'):
            tokens.append("<num_val>")
        elif (tokentype == 'Character' or tokentype == 'String'):
            tokens.append("<str_val>")
        else:
            tokens.append(tokenize[m].value)
    
    selectedTokens=[]

    if tokenSize<len(tokens):
        selectedTokens=tokens[:-tokenSize]
    else:
        selectedTokens=tokens
    
    inputSample = " ".join(selectedTokens) + " <soc>"

    clonesnippet = getModelClone(inputSample, model, tokenizer,topp)
    if clonesnippet != "":
        cloneLibrary = readFile("Data/tokenizeCloneList.txt")
        identifierList = readFile("Data/identifiersList.txt")
        topKPredicted = getTopKCloneResultsDocSim(list(set(cloneLibrary)), clonesnippet, kvalue)
        output = list(get_col(topKPredicted, 1))
        scores = list(get_col(topKPredicted, 0))

        text_file = open("Data/orignalCodeList.txt", 'r', encoding="utf8")
        data = text_file.read()
        orignalClones = data.split(" <CODESPLIT> ")

        orignalRecommendations = []
        recommendedIdentifiers = []
        for m in output:
            matchedIndex = -1
            for i in range(0, len(cloneLibrary)):
                if m == cloneLibrary[i]:
                    matchedIndex = i
                    break

            orignalRecommendations.append(orignalClones[matchedIndex])
            recommendedIdentifiers.append(identifierList[matchedIndex])
        return jsonify({'recommendations': orignalRecommendations, 'scores': scores,
                        'recommendedIdentifiers': recommendedIdentifiers})
    return jsonify({'missing': "No recommendations"})


def generateTop10Suggestions(sample, model, tokenizer):
    sequence = tokenizer.encode(sample, return_tensors="pt")
    next_word_logits = model(sequence)[0][0, -1].detach()
    probabilities, word_ids = next_word_logits.topk(5)
    wordList = []
    scoreList = []
    for ind in range(len(word_ids)):
        wordList.append(tokenizer.decode([word_ids[ind]]).strip())
        scoreList.append(str(probabilities[ind]))
    return wordList, scoreList


def get_col(arr, col):
    return map(lambda x: x[col], arr)


def readFile(filename):
    text_file = open(filename, 'r', encoding="utf8")
    data = text_file.read()
    listOfSentences = data.split("\n")
    return listOfSentences


def getTopKCloneResultsDocSim(cloneLibrary, clone, k):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([clone] + cloneLibrary)
    # Calculate the word frequency, and calculate the cosine similarity of the search terms to the documents
    cosine_similarities = linear_kernel(vectors[0:1], vectors).flatten()
    document_scores = [item.item() for item in cosine_similarities[1:]]  # convert back to native Python dtypes
    total_score_snippets = [(score, title) for score, title in zip(document_scores, cloneLibrary)]
    topk = sorted(total_score_snippets, reverse=True, key=lambda x: x[0])[:k]
    return topk


def getModelClone(sample, model, tokenizer,topp):
    encoded_prompt = tokenizer.encode(sample, add_special_tokens=True, return_tensors="pt")
    encoded_prompt = encoded_prompt.to('cpu')

    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_length=50 + len(encoded_prompt[0]),
        temperature=1.0,
        top_p=topp,
        repetition_penalty=1,  # args.repetition_penalty,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Remove the batch dimension when returning multiple sequences
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    generated_sequence = ""

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        generated_sequence = generated_sequence.tolist()
        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        if (text.find('<eoc>') ==-1):
            return text
        else:
            # Remove all text after the stop token
            text3 = text[: text.find('<eoc>') if '<eoc>' else None]
            text4 = text3 + "<eoc>"
            clonesnippet = text4[text4.find('<soc>'):]
            return clonesnippet

##################################Clone-Seeker###############

def removeStopWords(queryTokens):
    filtered_tokens = [word for word in queryTokens if word not in stopwords.words('english')]
    return filtered_tokens


def removePrepositions(tokens):
    tagged = nltk.pos_tag(tokens)
    filtered_tokens = tagged[0:6]
    return filtered_tokens


def stemming(tokens):
    stemmer = SnowballStemmer(language='english')
    # stemmer = PorterStemmer()
    filtered_tokens = []
    for token in tokens:
        filtered_tokens.append(stemmer.stem(token))
    return filtered_tokens


def camel_case_split(tokens):
    filtered_tokens = []
    for i in tokens:
        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', i)
        filtered_token = [m.group(0) for m in matches]
        filtered_tokens = filtered_tokens + filtered_token
    return filtered_tokens


def splitUnderscore(tokens):
    filtered_tokens = []

    for tok in tokens:
        matches = tok.split('_')
        filtered_tokens = filtered_tokens + matches
    return filtered_tokens




def removeSingleCharacters(query):
    document = [w for w in query if len(w) > 1]
    # (re.sub(r'\s+[a-zA-Z]\s+', ' ', ' '.join(query))).split(' ')
    return document


def convertLowerCase(tokens):
    lower = []
    for x in tokens:
        lower.append(x.lower())
    return lower


def getIdentifiers(code):
    tokenizeCode = list(javalang.tokenizer.tokenize(code))
    queryTokens = []
    for m in range(len(tokenizeCode)):
        tokentype = tokenizeCode[m].__class__.__name__
        if (tokentype == "Null" or tokentype == "Keyword" or tokentype == "Separator" or tokentype == "Modifier" or
                tokentype == "Operator" or tokentype == 'DecimalFloatingPoint' or tokentype == 'DecimalInteger' or
                tokentype == 'Integer' or tokentype == 'OctalInteger' or tokentype == 'BinaryInteger' or
                tokentype == 'HexInteger' or tokentype == 'FloatingPoint' or tokentype == 'HexFloatingPoint' or
                tokentype == 'Boolean' or tokentype == 'Literal' or tokentype == 'Character' or tokentype == 'String' or tokentype == 'BasicType'):
            continue
        else:
            queryTokens.append(tokenizeCode[m].value)
    return queryTokens


def normalizeNaturalLanguageQuery(docString):
    queryTokens = []
    # # Remove stop words
    docstringFilter2 = nltk.word_tokenize(docString)
    filter2 = removeStopWords(docstringFilter2)
    filter3 = filter2  # queryTokens#filterS+
    filter4 = filter3
    # Perform camel case
    filter5 = camel_case_split(filter4)
    # Perform underscore splittings
    filter6 = splitUnderscore(filter5)
    filter7 = removeSingleCharacters(filter6)
    # Convert into lower case
    filter8 = convertLowerCase(filter7)
    filter9 = stemming(filter8)

    # filter7 = getSynonyms(set(filter6))
    filter10 = list(unique_everseen(filter9))  # set(filter9)
    return ' '.join(filter10)


def normalizeCodeQuery(tokenizeCode):
    queryTokens = []
    for m in range(len(tokenizeCode)):
        tokentype = tokenizeCode[m].__class__.__name__
        # Remove reserve words, operators,seperators,
        # Modifier such as public, static
        # Keyword such as void
        # Seperator such as (, )
        ###Need to think for values for String value
        # Null values

        if (tokentype == "Null" or tokentype == "Keyword" or tokentype == "Separator" or tokentype == "Modifier" or
                tokentype == "Operator" or tokentype == 'DecimalFloatingPoint' or tokentype == 'DecimalInteger' or
                tokentype == 'Integer' or tokentype == 'OctalInteger' or tokentype == 'BinaryInteger' or
                tokentype == 'HexInteger' or tokentype == 'FloatingPoint' or tokentype == 'HexFloatingPoint' or
                tokentype == 'Boolean' or tokentype == 'Literal' or tokentype == 'Character' or tokentype == 'String'):
            continue
        else:
            queryTokens.append(tokenizeCode[m].value)


    filter3 = queryTokens

    filter4 = filter3
    # Perform camel case
    filter5 = camel_case_split(filter4)
    # Perform underscore splittings
    filter6 = splitUnderscore(filter5)
    filter7 = removeSingleCharacters(filter6)
    # Convert into lower case
    filter8 = convertLowerCase(filter7)
    filter9 = stemming(filter8)

    filter10 = list(unique_everseen(filter9))  # set(filter9)
    return ' '.join(filter10)


def semanticSearch(cloneLibrary, clone, k, isSemantic):
    if isSemantic == 1:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import linear_kernel
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([clone] + cloneLibrary)
        # Calculate the word frequency, and calculate the cosine similarity of the search terms to the documents
        cosine_similarities = linear_kernel(vectors[0:1], vectors).flatten()
        document_scores = [item.item() for item in cosine_similarities[1:]]  # convert back to native Python dtypes
        total_score_snippets = [(score, title) for score, title in zip(document_scores, cloneLibrary)]
        topk = sorted(total_score_snippets, reverse=True, key=lambda x: x[0])[:k]
        return topk


@app.route('/getSimilarityValues', methods=['POST'])
def getSimilarityValues():
    cloneCodeList = json.loads(request.form['cloneCodeList'])#request.POST.get('cloneCodeList'))

    tfidf_vectorizer = TfidfVectorizer()


    X_train_counts = tfidf_vectorizer.fit_transform(cloneCodeList)
    similarities = cosine_similarity(X_train_counts)

    return jsonify(
        {'scores': json.dumps(similarities.tolist())})


@app.route('/getCloneSeekerRecommendations', methods=['POST'])
def getCloneSeekerRecommendations():
    query = request.form['query']
    methodType = request.form['methodType']  # Automatic, Manual, Baseline
    queryType = request.form['queryType']  # Natural Language or Code Query ---NLQ or CQ
    kvalue=int(request.form['kvalue'])
    normalizedQuery = ''
    searchcorpus = []
    if queryType == 'Text':
        normalizedQuery = normalizeNaturalLanguageQuery(query)
    else:
        tokenizeCode = list(javalang.tokenizer.tokenize(query))
        normalizedQuery = normalizeCodeQuery(tokenizeCode)

    if methodType == 'Automatic':
        searchcorpus = readFile("Data/automatic_bcb_5.txt")

    if methodType == 'Manual':
        searchcorpus = readFile("Data/manual_bcb.txt")

    if methodType == 'Baseline':
        searchcorpus = readFile("Data/baseline_bcb.txt")

    functionalityTypeId = readFile("Data/functionTypeID_bcb.txt")

    topKSearchResults = semanticSearch(list(set(searchcorpus)), normalizedQuery, kvalue, 1)

    topKDocString = list(get_col(topKSearchResults, 1))
    scores = list(get_col(topKSearchResults, 0))

    text_file = open("Data/orignalCodeList.txt", 'r', encoding="utf8")
    data = text_file.read()
    orignalClones = data.split(" <CODESPLIT> ")

    orignalRecommendations = []
    recommendedIdentifiers = []
    for m in topKDocString:
        matchedIndex = -1
        for i in range(0, len(searchcorpus)):
            if m == searchcorpus[i]:
                matchedIndex = i
                break

        orignalRecommendations.append(orignalClones[matchedIndex])
        identifiersList = getIdentifiers(orignalClones[matchedIndex])
        recommendedIdentifiers.append(" ".join(identifiersList))

    return jsonify(
        {'recommendations': orignalRecommendations, 'scores': scores, 'recommendedIdentifiers': recommendedIdentifiers})


if __name__ == '__main__':
    app.run(debug=True)
