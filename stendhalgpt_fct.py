
import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import string

import re

import statistics


from nltk import bigrams
from collections import Counter

import math


import PyPDF2

import openai
openai.api_key = "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"

import numpy as np


import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

from nltk.probability import *

from math import sqrt

import streamlit as st
bar = st.progress(0)



nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
stop_words = set(stopwords.words("french") + list(string.punctuation))

def calculate_stats(texts):
    # Séparer les textes en une liste
    

    # Supprimer les éléments vides
    texts = list(filter(None, texts))
    num_texts = []
    avg_length = []
    # Calculer le nombre de textes
    for x in texts : 
        num_texts.append(verbal_richness(x))
        avg_length.append(compute_text_metrics(lexical_richness(x),grammatical_richness(x),verbal_richness(x)))
    # Créer un DataFrame pandas pour stocker les résultats
    stats_df = pd.DataFrame({
        "Nombre": num_texts,##Verbal
        "Longueur": avg_length##Combiné et pondération
    })

    colors = plt.cm.Set1(np.linspace(0, 1, len(num_texts)))


    fig, ax = plt.subplots()
    for i in range(0, len(num_texts)):
        ax.scatter(x=num_texts[i], y=avg_length[i], c=[colors[i]], label=f"Texte {i+1}")
    ax.scatter(x=np.mean(num_texts), y=np.mean(avg_length), c="black", marker=".", s=300, label="Moyenne")
    max_num_texts = max(num_texts)
    min_num_texts = min(num_texts)
    max_avg_length = max(avg_length)
    min_avg_length = min(avg_length)

    # Définir les limites du graph en utilisant les valeurs maximales et minimales
    ax.set_xlim(min_num_texts - 0.1, max_num_texts + 0.1)
    ax.set_ylim(min_avg_length - 0.1, max_avg_length + 0.1)


    ax.set_xlabel("Taille du champ verbal")
    ax.set_ylabel("Taille de la richesse générale")
    ax.set_title(f"Visualisation de la richesse lexicale par rapport à la richesse générale.")
    ax.legend()
    st.pyplot(fig)



def kl_div_with_exponential_transform(mat1, mat2, alpha):
    # Transform matrices
    mat1_transformed = np.exp(alpha * mat1)
    mat2_transformed = np.exp(alpha * mat2)
    
    # Normalize matrices
    mat1_normalized = mat1_transformed / np.sum(mat1_transformed, keepdims=True)
    mat2_normalized = mat2_transformed / np.sum(mat2_transformed, keepdims=True)
    
    # Calculate KL divergence
    kl_div = np.sum(mat1_normalized * np.log(mat1_normalized / mat2_normalized))
    
    return kl_div


def shifted_exp(x, shift=0):
    return 1+(1 / ((x - shift)**10))




def scaled_manhattan_distance(a, b):
    scale_factor = sum(a.shape) + sum(b.shape)
    return np.sum(np.abs(a - b)) / scale_factor

def is_within_10_percent(x, y):
    threshold = 0.29  # 29%
    difference = abs(x - y)
    avg = (x + y) / 2
    return difference <= (avg * threshold)

def nettoyer_texte(texte):
    # Supprimer les chiffres et les caractères spéciaux
    texte = ''.join(c for c in texte if c not in string.punctuation and not c.isdigit())
    # Convertir en minuscules
    texte = texte.lower()
    # Supprimer les mots vides
    stopwords = set(nltk.corpus.stopwords.words('french'))
    texte = ' '.join(w for w in nltk.word_tokenize(texte) if w.lower() not in stopwords)
    return texte

def create_markdown_table(similarity_measures):
    table_header = '| Mesure | Taux de similarité |\n'
    table_divider = '| ------- | ---------- |\n'
    table_rows = ''
    for measure, similarity in similarity_measures.items():
        table_rows += f'| {measure} | {similarity} |\n'
    markdown_table = table_header + table_divider + table_rows
    return markdown_table


def generation2(thm):
    result = ''
    bar.progress(32)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a chatbot"},
                {"role": "user", "content": f"écris moi uniquement un texte suivant ces paramètres {thm}"},
            ]
    )
    bar.progress(89)
    for choice in response.choices:
        result += choice.message.content + '\n'
    return result

def generation(thm):
    
    bar.progress(32)
    openai.api_key = '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'
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=thm,
    max_tokens=2048,
    temperature=0
        )
    bar.progress(80)
    answer = response.choices[0].text
    return answer


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

def lexical_field(text):
    # Tokenization du texte
    words = nltk.word_tokenize(text)
    words = [word for word in words if word.lower() not in stop_words]
    # Calcul des fréquences des mots
    fdist = FreqDist(words)
    return fdist


def lexical_richness(text):
    # Tokenization du texte
    words = nltk.word_tokenize(text)
    
    # Calcul de l'étendue du champ lexical
    type_token_ratio = len(set(words)) / len(words)
    return type_token_ratio

def highlight_text2(text, words_to_highlight):
    # Compile la liste de mots à surligner en une expression régulière
    words_regex = "|".join(words_to_highlight)
    pattern = re.compile(r"\b(" + words_regex + r")\b")
    
    # Remplace les mots correspondant à la condition par des mots entourés de balises HTML <mark>
    highlighted_text = pattern.sub(r"**<span style='background-color: blue;'>\1</span>**", text)
    return highlighted_text

def highlight_text(text, words_to_highlight):
    # Compile la liste de mots à surligner en une expression régulière
    words_regex = "|".join(words_to_highlight)
    pattern = re.compile(r"\b(" + words_regex + r")\b")
    
    # Remplace les mots correspondant à la condition par des mots entourés de balises HTML <mark>
    highlighted_text = pattern.sub(r"**<span style='background-color: red;'>\1</span>**", text)
    return highlighted_text



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

def lexical_richness_normalized_ttr(text1):
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
    ttr1 = unique1 / total1
    
    # Measure of Textual Lexical Diversity
    mtd1 = len(set(tokens1)) / len(tokens1)
    
    # Return normalized values in a dictionary
    return [unique1,ttr1,mtd1]

def lexical_richness_normalized_only(text):

    # Tokenization
    tokens = nltk.word_tokenize(text)
    
    # Removing punctuation
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    
    # Number of unique words
    unique_words = len(set(tokens))
    
    # Total number of words
    total_words = len(tokens)
    
    # Type-token ratio
    ttr_ratio = unique_words / total_words
    
    # Measure of Textual Lexical Diversity
    mtd_ratio = len(set(tokens)) / len(tokens)
    
    # Normalize values
    unique_words_ratio = unique_words / total_words
    ttr_ratio_normalized = (ttr_ratio - 0.2) / (0.8 - 0.2) # Normalize between 0 and 1, assuming ttr_ratio >= 0.2
    mtd_ratio_normalized = mtd_ratio / unique_words
    
    # Return normalized values in a list
    return [unique_words_ratio, ttr_ratio_normalized, mtd_ratio_normalized]




def count_words(text):
    words = text.split()
    return len(words)

def compute_text_metrics(lexical_density,grammatical_density,verbal_density):
 
    # Ajuster les pondérations pour donner plus de poids au taux verbal
    lexical_weight = 0.25
    grammatical_weight = 0.55
    verbal_weight = 1.75
    
    # Calculer la mesure agrégée en combinant les trois taux avec des pondérations ajustées
    text_metric = (lexical_weight * lexical_density) + (grammatical_weight * grammatical_density) + (verbal_weight * verbal_density)
    ### Plus proche de 1 alors texte très riche et donc suspect 

    return text_metric 


import statistics

def measure_lexical_richness(text, punctuation):
    # Diviser le texte en phrases
    sentences = nltk.sent_tokenize(text.lower())
    
    # Diviser chaque phrase en blocs en utilisant la ponctuation spécifiée
    blocks = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        sentence_blocks = []
        current_block = ''
        for word in words:
            if word in punctuation:
                if current_block:
                    sentence_blocks.append(current_block)
                sentence_blocks.append(word)
                current_block = ''
            else:
                current_block += ' ' + word
        if current_block:
            sentence_blocks.append(current_block.strip())
        blocks.append(sentence_blocks)
    
    # Calculer le taux lexical de chaque bloc
    lexical_densities = [len(set(block))/len(block) for sentence in blocks for block in sentence if len(block) > 0]
    
    # Calculer la moyenne et l'écart-type des taux lexicaux
    avg_ld = sum(lexical_densities)/len(lexical_densities)
    std_ld = statistics.stdev(lexical_densities)
    
    # Trouver les blocs anormaux (ceux dont le taux lexical est en dehors de deux écarts-types de la moyenne)
    abnormal_blocks = [block for sentence in blocks for block in sentence if len(block) > 0 and (len(set(block))/len(block) > (avg_ld + 2*std_ld) or len(set(block))/len(block) < (avg_ld - 2*std_ld))]
    
    return abnormal_blocks 

def plot_text_relations_2d(*texts):
    # Créer une figure
    fig, ax = plt.subplots()
    
    # Tracer un nuage de points pour chaque texte avec les valeurs des paramètres en x et y
    for i, text in enumerate(texts):
        x_data = text[0] # unique_words_ratio
        y_data = text[1] # ttr_ratio
        ax.scatter(x_data, y_data, label=f'Texte {i+1}')
    
    # Ajouter des labels aux axes x et y
    ax.set_xlabel('Ratio de mots uniques')
    ax.set_ylabel('Ratio Type-Token (TTR)')
    
    # Ajouter une légende
    ax.legend()
    
    # Afficher le graphique dans Streamlit
    st.pyplot(fig)




def detect_generated_text(text):
    # Tokenize le texte en mots
    words = text.lower().split()
    
    # Calculer la fréquence de chaque mot
    word_freq = {}
    for word in words:
        if word not in word_freq:
            word_freq[word] = 0
        word_freq[word] += 1
    
    # Calculer le nombre de mots uniques
    num_unique_words = len(word_freq)
    
    # Calculer la proportion de mots uniques dans le texte
    unique_word_ratio = num_unique_words / len(words)
    
    # Calculer le poids de chaque mot en fonction de sa fréquence et de la diversité de mots dans le texte
    word_weights = {}
    for word in word_freq:
        # Le poids de chaque mot est calculé en multipliant la fréquence du mot par un facteur de normalisation basé sur la diversité de mots dans le texte
        weight = word_freq[word] * (1 - (math.log(num_unique_words) / math.log(len(words))))
        word_weights[word] = weight
        
    # Calculer le poids total du texte en faisant la somme des poids de chaque mot
    total_weight = sum(word_weights.values())
    
    # Appliquer une formule pour donner un poids plus important aux textes ayant une proportion élevée de mots uniques
    if unique_word_ratio >= 0.4:
        score = (unique_word_ratio - 0.4) / 0.6
    else:
        score = 0
        
    # Ajouter le score basé sur la proportion de mots uniques au poids total du texte
    weighted_score = total_weight + score
    
    return weighted_score
import string

def plot_texts_3d(*args):
    
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
    ax.set_xlabel('Richesse Lexicale')
    ax.set_ylabel('Richesse Grammaticale')
    ax.set_zlabel('Richesse Verbale')
    ax.legend()
    
    # Afficher le graphique
    st.pyplot(fig)



def plot_text_relations(texts):

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
    ax.set_xlabel('Ratio de mots uniques')
    ax.set_ylabel('Ratio de mots totaux')
    ax.set_zlabel('Ratio de MTD')

    # Ajouter une légende
    ax.legend()

    # Afficher le graphique dans Streamlit
    st.pyplot(fig)
  

def count_characters(text):
    return len(text)
