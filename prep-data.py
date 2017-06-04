import json
from nltk.tokenize import word_tokenize
import pickle

tokenized = []

#abstract_tokens = []
#content_tokens = []

i = 0
n = 0

def tokenize_sentence(sentence):
    return ' '.join(list(filter(
        lambda x: x.lower() != "advertisement",
        word_tokenize(sentence))))

with open('signalmedia-1m.jsonl') as json_corpus:
    n = len(json_corpus.readlines())
    print(str(n) + ' lines')

with open('signalmedia-1m.jsonl') as json_corpus:
    for line in json_corpus:
        json_object = json.loads(line)
        #print d['title']
        #abstract_tokens.append(nltk.word_tokenize(json_object['title']))
        #content_tokens.append(nltk.word_tokenize(json_object['content']))
        tokenized.append((tokenize_sentence(json_object['title']), tokenize_sentence(json_object['content'])))

        i += 1
        if i % 1000 == 0:
            print(str(i) + ' / ' + str(n))
        if i == 50000:
            break

#tpl = (abstract_tokens, content_tokens)

with open('data/data.pkl', 'wb') as myfile:
    pickle.dump(tuple(map(list, zip(*tokenized))), myfile, 2)
