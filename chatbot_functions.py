from main import *
from speak import speak

def predict_tag(sentence):
    p = clean_up_sentence(sentence)
    bow = np.array(bag_of_words(p,words))
    res = model.predict(np.array([bow]))[0]  #Generates output predictions for the input samples
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': tags[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(ints, intents_json):
    try:
        tag = ints[0]['intent']
        list_of_intents = intents_json['data']
        for i in list_of_intents:
            if i['tag']  == tag:
                result = random.choice(i['responses'])
                break
    except IndexError:
        result = "I don't understand!"
    return result

print("gooooo!!!")

while True:
    message = input("")
    tag = predict_tag(message)
    res = get_response(tag , data)
    print(res)
    speak(res)

