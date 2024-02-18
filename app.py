import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from tensorflow.keras.models import load_model

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize WordNet lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load intents and model
intents = json.loads(open('intents.json').read())
model = load_model('chatbot_model.h5')

# Load preprocessed data
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

def clean_up_sentence(sentence):
    # Tokenize the sentence
    sentence_words = nltk.word_tokenize(sentence)
    # Remove stopwords, lemmatize, and convert to lowercase
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words if word.lower() not in stop_words]
    return sentence_words

def bow(sentence, words, show_details=False):
    # Tokenize the sentence
    sentence_words = clean_up_sentence(sentence)
    # Create bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence):
    # Filter below threshold predictions
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.7  # Adjust this threshold as needed
    results = [[i, r] for i, r in enumerate(res)]

    # Sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
   # print("Predictions:", results)  # Debug output
    
    return_list = []
    for r in results:
        if r[1] > ERROR_THRESHOLD:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    if not ints:
        return "I'm sorry, I didn't understand that question. Could you please try asking in a different way?"
    else:
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if(i['tag']== tag):
                result = random.choice(i['responses'])
                break
        return result

print("|============= Welcome to iServe Africa Inquiry Chatbot System! =============|")
print("|============================== Feel Free ============================|")
print("|================================== To ===============================|")
print("|=============== Ask your any query about iServe Africa ================|")

# Main interaction loop
while True:
    message = input("| You: ").lower()
    if message in ["bye", "goodbye"]:
        print("| Bot: Goodbye!")
        print("|===================== The Program Ends here! =====================|")
        break
    else:
        ints = predict_class(message)
        res = get_response(ints, intents)
        print("| Bot:", res)
