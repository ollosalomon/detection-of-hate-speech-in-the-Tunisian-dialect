
import tensorflow as tf
from tensorflow import keras
import transformers
from transformers import BertTokenizer,AutoTokenizer
import arabert 
import numpy as np
import streamlit as st  
import arabert 
import altair as alt
from PIL import Image
import emoji


from keras import backend as K
from tensorflow import keras 
from keras.utils.vis_utils import plot_model

#les differentes metriques
#les metriques sont définies afin de charger le modele
#mon bb dani, le modele a été évalué à partir de ces metriques, alors pour le charger il faut forcement ces metriques
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
    
#chargement du modele
new_model = tf.keras.models.load_model('/content/drive/MyDrive/NLP/Deux_classes', custom_objects={'f1':f1_m,
                                    'recall':recall_m,
                                    'precision_m':precision_m},compile=False)

#chargement du tokeniseur
tokenizer = AutoTokenizer.from_pretrained('aubmindlab/bert-base-arabertv02')

#fonction pour pretraiter les entrées (les mots)
def prepare_data(input_text, tokenizer):
    token = tokenizer.encode_plus(
        input_text,
        max_length=256, 
        truncation=True, 
        padding='max_length', 
        add_special_tokens=True,
        return_tensors='tf'
    )
    return {
        'input_ids': tf.cast(token.input_ids, tf.float64),
        'attention_mask': tf.cast(token.attention_mask, tf.float64)
    }
#fonction pour faire la prediction
def make_prediction(new_model, processed_data, classes=['normal','hate']):
    probs = new_model.predict(processed_data)[0]
    return classes[np.argmax(probs)]

#les deux fonctions seront utilisées pour le traitement qui va suivre


def main():

	#st.markdown("<h1 style='text-align: center; color: black;'>Système de détection et classification de discours haineux du dialecte tunisien</h1>", unsafe_allow_html=True)
	st.markdown("<h1 style='text-align: center; color: black;'>Detection et classification automatique des discours haineux du dialecte Tunisien</h1>", unsafe_allow_html=True)

#automatic detection of hate speech

#	st.markdown("<h2 style='text-align: center; color: black;'>du dialecte tunisien </h2>", unsafe_allow_html=True)
#menu de l'application (page d'accueil)
#l'application a deux page (Accueil et A propos)


	menu = ["Accueil","À propos"]
	choice = st.sidebar.selectbox("Menu",menu)

#sur le coté gauche de la page d'accueil, il ya la possibilié de choisir la page qu'on veut afficher
	if choice == "Accueil": #si on choisit la page d'accueil
		image = Image.open('/content/drive/MyDrive/NLP/emotions.jpg') # l'image qui s'affiche à la page d'accueil

		st.image(image, caption='emotions') #affichage de l'image

		with st.form(key='nlpForm'): #un genre de formulaire avec zone de texte et un boutton de soumission
			st.write(emoji.emojize('Aimons nous les uns les autres :red_heart: love ',use_aliases=True)) #
			raw_text = st.text_area("Entrez votre phrase") #zone de texte
			submit_button = st.form_submit_button(label='Analyser') #boutton de soumission

		# layout
#les resultats seront afichés sur deux colonnes lorsqu'on va cliquer sur le boutton de soumission
		col1,col2 = st.columns(2) #creation des deux colonnes
		if submit_button: #si le boutton de soumission est declanché

			with col1: #ce qui doit etre affiché sur la premiere colonne
				st.info("Resultat") #message affiché comme titre
				sentiment = prepare_data(raw_text,tokenizer) #fonction de pretraitement
				result =  make_prediction(new_model, sentiment)#fonction pour la prediction

				if result == 'normal':
					st.markdown("Votre phrase est: Neutre :smiley: ")
				elif result == 'hate':
					st.markdown("Votre phrase est: haineuse :angry: ")
				else:
					st.markdown("Sentiment:: Neutre 😐 ")


			with col2:#ce qui doit etre affiché sur la deuxieme colonne
				st.info("Astuce")#message affiché comme titre
				st.write("Partagez l'amour autour de vous, un sourire vaut mille fois mieux qu'une haine :smiley: ")#message affiché 

	else: #si on choisit la section A propos

 #message et image à afficher 
		st.info("Created by Ollo Salomon Pale with Streamlit")
		fig = Image.open('/content/drive/MyDrive/NLP/photo1.jpg')
		st.image(fig, caption='ollosalomon@gmail.com Jesus Saves')
		st.markdown("<h2 style='text-align: center; color: black;'>Ce système permet de détecter et de classifier un discours comme haineux, injurieux ou neutre. Notre modèle est basé sur AraBERT</h2>", unsafe_allow_html=True)



if __name__ == '__main__':
	main()