import streamlit as st
st.set_page_config(layout="wide")

st.title('StendhalGPT')
from st_on_hover_tabs import on_hover_tabs

from stendhalgpt_fct import *
import pandas as pd


import nltk

import streamlit.components.v1 as components

col5, col6 = st.columns(2)
col2, col1 = st.columns(2)

bar.progress(0) 




def compare_markov_model(text1, text2):
    # tokenize les deux textes
    tokens1 = nltk.word_tokenize(text1)
    tokens2 = nltk.word_tokenize(text2)

    # créer des bigrames pour les deux textes
    bigrams1 = list(bigrams(tokens1))
    bigrams2 = list(bigrams(tokens2))

    # compter le nombre d'occurences de chaque bigramme  A MODIFIER PAR LA TF IDF
    count1 = Counter(bigrams1)
    count2 = Counter(bigrams2)
 
    # mesurer la probabilité de transition pour chaque bigramme dans les deux textes
    prob1 = {bigram: count/len(bigrams1) for bigram, count in count1.items()}
    prob2 = {bigram: count/len(bigrams2) for bigram, count in count2.items()}


    common_bigrams = set(count1.keys()) & set(count2.keys())
    # Obtenir les probabilités pour chaque bigramme commun
    prob1 = {bigram: count1[bigram] / sum(count1.values()) for bigram in common_bigrams}
    prob2 = {bigram: count2[bigram] / sum(count2.values()) for bigram in common_bigrams}
    
    return [prob1, prob2]

def plot_text_relations_2d(*texts):
    # Créer une figure
    fig, ax = plt.subplots()
    
    # Tracer un nuage de points pour chaque texte avec les valeurs des paramètres en x et y
    for i, text in enumerate(texts):
        x_data = text[0] # unique_words_ratio
        y_data = text[1] # ttr_ratio
        ax.scatter(x_data, y_data, label=f'Texte {i+1}')
    
    # Ajouter des labels aux axes x et y
    ax.set_xlabel('Single word ratio')
    ax.set_ylabel('Ratio Type-Token (TTR)')
    
    # Ajouter une légende
    ax.legend()
    
    # Afficher le graphique dans Streamlit
    st.pyplot(fig)

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
    ax.set_xlabel('Lexical Richness')
    ax.set_ylabel('Grammatical Richness')
    ax.set_zlabel('Verbal Richness')
    ax.legend()
    
    # Afficher le graphique
    st.pyplot(fig)

st.info('Below 130 words, it is best to use the Expert function application. ')


with col5:

    text = st.text_input("Insert a referent text(s) in this column.", '')


with col6:
    nbr_mots_text = len(text.split(" "))
    text_ref = st.text_input("Insert a description of your text (size, type, subject, level of study.)")
    try:
        
        text_ref = generation2('Génére uniquement un text dans la même langue respectant ces critères : '+text_ref+' en '+str(nbr_mots_text)+'nombre de mots')
    except:
        try:
            text_ref = generation('Génére uniquement un texte dans la même langue en respectant ces critères : '+text_ref+' en '+str(nbr_mots_text)+'nombre de mots')
        except:
            st.warning('The service is overloaded, please use another method.')


if st.button('Check'):
        
    try : 
        diff = compare_markov_model(nettoyer_texte(text), nettoyer_texte(text_ref))
        vec1 = np.array([diff[0][bigram] for bigram in diff[0]] +[verbal_richness(text)]+[grammatical_richness(text)]+[lexical_richness(text)] )
        vec2 = np.array([diff[1][bigram] for bigram in diff[1]] +[verbal_richness(text_ref)]+[grammatical_richness(text_ref)]+[lexical_richness(text_ref)])
                    
        x = len(vec1)
        A = vec1
        B= vec2
        distance = np.sqrt(np.sum((A - B) ** 2))
        resul = (1/distance)/x


        st.markdown(f'The relative Euclidean distance is :red[{round((resul),4)}.]')
    
        if resul > 1 or is_within_10_percent(0.96,resul) == True :
            st.markdown('It seems your text was written by a human.')
        elif is_within_10_percent(resul,2) == True :
            st.markdown('It is safe that your text has been generated.')
        else:
            st.markdown('It is certain that your text has been generated.')

        with col2 : 
            try: 
                            plot_texts_3d((lexical_richness(text),grammatical_richness(text),verbal_richness(text)),(lexical_richness(text_ref),grammatical_richness(text_ref),verbal_richness(text_ref)))

            except:
                           st.warning('An error has occurred in the processing of your texts.')

            
        with col1:
            try:    
                text_dt_tt = lexical_richness_normalized(text)
                text_ref_dt_tt = lexical_richness_normalized(text_ref)



                P1 =  np.array(text_dt_tt)
                P2 = np.array(text_ref_dt_tt)

                dist=  np.sqrt(np.sum((P1 - P2) ** 2))


                texts = [text_dt_tt, text_ref_dt_tt]

                plot_text_relations(texts)

                st.markdown(f"Euclidean distance between points {dist}.")
            except:
                    st.warning("An error has occurred in the processing of your texts.")
        bar.progress(100)

    except:
        st.warning('Problem occurred, try again. ')
    
