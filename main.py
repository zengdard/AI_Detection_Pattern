import streamlit as st
st.set_page_config(layout="wide")

st.title('StendhalGPT')
from nltk import bigrams
from collections import Counter
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from scipy.spatial import distance
import openai

openai_api_key = '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'

openai.api_key = "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"

from langchain.chat_models import ChatOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import kl_div
import numpy as np
import string

def grammatical_richness(text):
    words = word_tokenize(text)
    words = [word for word in words if word.isalnum()]
    words_pos = pos_tag(words)
    words_pos = [word for word in words_pos if word[0] not in stop_words]
    pos = [pos for word, pos in words_pos]
    fdist = FreqDist(pos)
    types = len(fdist.keys())
    tokens = len(words)
    return types / tokens

def verbal_richness(text):
    words = word_tokenize(text)
    words = [word for word in words if word.isalnum()]
    words_pos = pos_tag(words)
    words_pos = [word for word in words_pos if word[0] not in stop_words]
    verbs = [word for word, pos in words_pos if pos[:2] == 'VB']
    fdist = FreqDist(verbs)
    types = len(fdist.keys())
    tokens = len(words)
    return types / tokens

def nettoyer_texte(texte):
    # Supprimer les chiffres et les caractères spéciaux
    texte = ''.join(c for c in texte if c not in string.punctuation and not c.isdigit())
    # Convertir en minuscules
    texte = texte.lower()
    # Supprimer les mots vides
    stopwords = set(nltk.corpus.stopwords.words('french'))
    texte = ' '.join(w for w in nltk.word_tokenize(texte) if w.lower() not in stopwords)
    return texte

def lexical_richness(text):
    # Tokenization du texte
    words = nltk.word_tokenize(text)
    
    # Calcul de l'étendue du champ lexical
    type_token_ratio = len(set(words)) / len(words)
    return type_token_ratio





def lexical_richness_normalized(text1):
    # Tokenization
    tokens1 = nltk.word_tokenize(text1)
    
    # Removing punctuation
    table = str.maketrans('', '', string.punctuation)
    tokens1 = [w.translate(table) for w in tokens1]
    
    # Number of unique words
    unique1 = len(set(tokens1))
    
    # Total number of words
    total1 = len(tokens1)
    
    # Type-token ratio
    #ttr1 = unique1 / total1
    
    # Measure of Textual Lexical Diversity
    mtd1 = len(set(tokens1)) / len(tokens1)
    
    # Return normalized values in a dictionary
    return [unique1,total1,mtd1]


stop_words = set(stopwords.words("french"))

col5, col6 = st.columns(2)
col2, col1 = st.columns(2)

bar = st.progress(0)
bar.progress(0) 



def compare_markov_model(text1, text2):
    global vector1
    global vector2
      # tokenize the two texts
    tokens1 = nltk.word_tokenize(text1)
    tokens2 = nltk.word_tokenize(text2)

    # create bigrams for the two texts
    bigrams1 = list(bigrams(tokens1))
    bigrams2 = list(bigrams(tokens2))

    # count the number of occurrences of each bigram in the two texts
    count1 = Counter(bigrams1)
    count2 = Counter(bigrams2)

    # calculate the transition probability for each bigram in the two texts
    prob1 = {bigram: count/len(bigrams1) for bigram, count in count1.items()}
    prob2 = {bigram: count/len(bigrams2) for bigram, count in count2.items()}

    common_bigrams = set(count1.keys()) & set(count2.keys())

    # Only keep the common bigrams in the probability distribution
    prob1 = {bigram: prob1[bigram] for bigram in common_bigrams}
    prob2 = {bigram: prob2[bigram] for bigram in common_bigrams}

    # Transformez vos probabilités en vecteurs
    vector1 = [prob1.get(bigram, 0) for bigram in common_bigrams]
    vector2 = [prob2.get(bigram, 0) for bigram in common_bigrams]
    if not vector1 or not vector2:
        return 0, 1, 1

    # Calculez la similarité cosinus et la distance euclidienne
    cos_sim = cosine_similarity([vector1], [vector2])[0][0]
    euclid_dist = distance.euclidean(vector1, vector2)

    return cos_sim, euclid_dist, len(vector1)



def plot_text_relations_LANG(texts):

    '''text1 = [unique1, total1, mtd1]
                        text2 = [unique2, total2, mtd2]

                        texts = [text1, text2]'''
    # Créer une figure 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    texts = list(filter(None, texts))


    # Tracer un nuage de points pour chaque texte avec les valeurs des paramètres en x, y et z
    for i, text in enumerate(texts):
        unique, total, mtd = text
        ax.scatter(unique, total, mtd, c=f'C{i}', marker='o', label=f'Texte {i+1}')

    # Ajouter des labels aux axes x, y et z
    ax.set_xlabel('Single word ratio')
    ax.set_ylabel('Total word ratio')
    ax.set_zlabel('MTD Ratio')

    # Ajouter une légende
    ax.legend()

    fig.text(0.5, 0.5, '© StendhalGPT', ha='center')

    # Afficher le graphique dans Streamlit
    st.pyplot(fig)


def plot_texts_3d_LANG(*args):
    
    '''

plot_texts_3d((x1, y1, z1), (x2, y2, z2), (x3, y3, z3))
'''
    # Créer une figure 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    args = list(filter(None, args))
    # Ajouter les textes sur le graphique
    for i, data in enumerate(args):
        color = 'C' + str(i % 10)  # Définir une couleur différente pour chaque texte
        marker = 'o' if i == 0 else '^'  # Utiliser un marqueur différent pour le premier texte
        label = f'Texte {i+1}'
        ax.scatter(data[0], data[1], data[2], c=color, marker=marker, label=label)
    
    # Définir les étiquettes des axes
    ax.set_xlabel('Lexical Richness')
    ax.set_ylabel('Grammatical Richness')
    ax.set_zlabel('Verbal Richness')
    ax.legend()

    fig.text(0.5, 0.5, '© StendhalGPT', ha='center')
    
    # Afficher le graphique
    st.pyplot(fig)

st.info('Below 130 words, it is best to use the Expert function application. ')

def is_within_10_percent(x, y):
    threshold = 0.29  # 29%
    difference = abs(x - y)
    avg = (x + y) / 2
    return difference <= (avg * threshold)


def generation2(thm):
    bar.progress(32)
    response = llm.generate_text(prompt="tu es une intelligence artificielle qui reformule un texte dans la même langue qu'on lui donne et de même taille. Tu retourneras uniquement le texte reformulé sans phrase supplémentaire" + thm)
    return response

def generation(thm):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
                {"role": "system", "content": "tu es une intelligence artificielle qui reformule un texte et de même taille. Tu retourneras uniquement le texte reformulé sans phrase supplémentaire"},
                {"role": "user", "content": f"{thm}"},
            ]
    )
    #print(response['choices'][0]['message']['content'])
    return response['choices'][0]['message']['content']



def compare_markov_model_2(text1, text2):
    # tokenize the two texts
    tokens1 = nltk.word_tokenize(text1)
    tokens2 = nltk.word_tokenize(text2)

    # create bigrams for the two texts
    bigrams1 = list(bigrams(tokens1))
    bigrams2 = list(bigrams(tokens2))

    # count the number of occurrences of each bigram in the two texts
    count1 = Counter(bigrams1)
    count2 = Counter(bigrams2)

    # calculate the transition probability for each bigram in the two texts
    prob1 = {bigram: count/len(bigrams1) for bigram, count in count1.items()}
    prob2 = {bigram: count/len(bigrams2) for bigram, count in count2.items()}

    common_bigrams = set(count1.keys()) & set(count2.keys())

    # Only keep the common bigrams in the probability distribution
    prob1 = {bigram: prob1[bigram] for bigram in common_bigrams}
    prob2 = {bigram: prob2[bigram] for bigram in common_bigrams}

    # Sort the common bigrams
    sorted_common_bigrams = sorted(common_bigrams)

    # Convert to lists to ensure the same order
    prob1_list = np.array([prob1[bigram] for bigram in sorted_common_bigrams])
    prob2_list = np.array([prob2[bigram] for bigram in sorted_common_bigrams])

    # Calculate KL divergence
    kl_divergence = kl_div(prob1_list, prob2_list).sum()

    # Scale the KL divergence based on the number of common bigrams
    scaled_kl_divergence = kl_divergence * (1 - len(common_bigrams) / (len(set(count1.keys())) + len(set(count2.keys()))))
    
    return scaled_kl_divergence


def measure_text_distribution(text: str, num_parts: int = 2) -> list:
    """Divide a text into several parts and measure the distribution of Markov model similarity between the parts."""
    words = text.split()  # Split the text into words
    part_length = len(words) // num_parts
    parts = [words[i*part_length:(i+1)*part_length] for i in range(num_parts)]

    divergences = []
    for i in range(num_parts):
        for j in range(i+1, num_parts):
            divergence = compare_markov_model_2(' '.join(parts[i]), ' '.join(parts[j]))
            divergences.append(divergence)

    pairs = [(i,j) for i in range(num_parts) for j in range(i+1, num_parts)]
    pair_labels = [f"{pair[0]}-{pair[1]}" for pair in pairs]

    # Plotting the divergences
    with col16:
        fig, ax = plt.subplots()
        ax.bar(pair_labels, divergences)
        ax.set_xlabel('Pair of Parts')
        ax.set_ylabel('KL Divergence')
        ax.set_title('KL Divergence Between Parts of Text')
        fig.text(0.5, 0.5, '© StendhalGPT', ha='center') # Adding copyright text
        st.pyplot(fig)  # Display the plot in Streamlit

    return divergences, parts, pairs
def somme_coefficients(coefficients):
    somme = 0
    for coefficient in coefficients:
        somme += coefficient
    return somme

coefficients = [0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.017045454545454544, 0.005681818181818182, 0.005681818181818182, 0.011363636363636364, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.017045454545454544, 0.005681818181818182, 0.017045454545454544, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.005681818181818182, 0.022727272727272728, 0.005681818181818182]

print(somme_coefficients(coefficients))
def display_text(text: str, num_parts: int = 2):
    global col16
    
    col10, col16 = st.columns(2)
    divergences, parts, pairs = measure_text_distribution(text, num_parts)
    max_divergence = max(divergences)

    
    for i, part in enumerate(parts):
        # Calculate the maximum divergence of this part
        part_divergences = [divergences[j] for j, pair in enumerate(pairs) if i in pair]
        max_part_divergence = max(part_divergences, default=0)
        
        # If the divergence is more than half of the maximum divergence, color the text red
        with col10 :
            if max_part_divergence > max_divergence / 2:
                st.markdown(f"<p style='color:red;'>{' '.join(part)}</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p>{' '.join(part)}</p>", unsafe_allow_html=True)


llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name='gpt-3.5-turbo-16k',
        temperature=1
    )



text = st.text_area("Insert your text here.", '')
text_ref = text

st.write(len(text.split()))
if text != '' and text_ref != '' :
    if   len(text.split()) <= 130 :
        st.warning('Your text is to short or too long. Pleas use the free expert mod, or StendhalGPT+.')
    else:
        if st.button('Check'):
            try:
                    text_ref = generation2(text_ref)
            except:
                try:    
                    text_ref = generation(text_ref)
                except:
                    st.warning('The service is overloaded, please use another method.')
                
            try : 
                print(text_ref)
                cos_sim, euclid_dist, vec1 = compare_markov_model(nettoyer_texte(text), nettoyer_texte(text_ref))
                max_coef = abs(somme_coefficients(vector1)-somme_coefficients(vector2))
                print(cos_sim, euclid_dist, vec1)
                max_coef = log((1/max_coef))#CAS = 0
                resul = (euclid_dist*max_coef)/log(cos_sim) ##GERER 3 CAS 
                print(resul)
                if resul >= 9 :
                    resul = 1 
                else:
                    pass


                st.markdown(f'The relative Euclidean distance is :red[{round((resul),4)}.] Indice of similarity : {vec1}. Cosinus Similarity : {cos_sim}')
            
                if resul > 1 or is_within_10_percent(0.96,resul) == True :
                    st.markdown('It seems your text was written by a human.')
                elif is_within_10_percent(resul,2) == True :
                    st.markdown('It is safe that your text has been generated.')
                else:
                    st.markdown('It is certain that your text has been generated.')

                with col2 : 
                    try: 
                        plot_texts_3d_LANG((lexical_richness(text),grammatical_richness(text),verbal_richness(text)),(lexical_richness(text_ref),grammatical_richness(text_ref),verbal_richness(text_ref)))

                    except:
                                st.warning('An error has occurred in the processing of your texts.')

                    
                with col1:
                    try:    
                        text_dt_tt = lexical_richness_normalized(text)
                        text_ref_dt_tt = lexical_richness_normalized(text_ref)

                        texts = [text_dt_tt, text_ref_dt_tt]

                        plot_text_relations_LANG(texts)



                    except:
                            st.warning("An error has occurred in the processing of your texts.")
                                    
                display_text(text, len(text.split())%10) 
                
                bar.progress(100)

            except:
               st.warning('Problem occurred, try again. ')
            
