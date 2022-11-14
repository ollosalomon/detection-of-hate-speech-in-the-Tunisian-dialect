
import tensorflow as tf
from tensorflow import keras
import transformers
from transformers import BertTokenizer,AutoTokenizer
import arabert 
import numpy as np
import streamlit as st  
import arabert 
import altair as alt


from keras import backend as K
from tensorflow import keras 
from keras.utils.vis_utils import plot_model


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
    

new_model = tf.keras.models.load_model('/content/drive/MyDrive/NLP/Deux_classes', custom_objects={'f1':f1_m,
                                    'recall':recall_m,
                                    'precision_m':precision_m},compile=False)


tokenizer = AutoTokenizer.from_pretrained('aubmindlab/bert-base-arabertv02')

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

def make_prediction(new_model, processed_data, classes=['normal','hate']):
    probs = new_model.predict(processed_data)[0]
    return classes[np.argmax(probs)]


def main():
	st.title("Sentiment Analysis NLP App")
	st.subheader("Streamlit Projects")

	menu = ["Home","About"]
	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Home":
		st.subheader("Home")
		with st.form(key='nlpForm'):
			raw_text = st.text_area("Enter Text Here")
			submit_button = st.form_submit_button(label='Analyze')

		# layout
		col1,col2 = st.columns(2)
		if submit_button:

			with col1:
				st.info("Results")
				sentiment = prepare_data(raw_text,tokenizer)#.sentiment
				result =  make_prediction(new_model, sentiment)
				st.write(result)

				# Emoji
				if result == 'normal':
					st.markdown("Sentiment:: Positive :smiley: ")
					st.write("Partagez l'amour autour de vous, un sourire est milles fois mieux qu'une haine :smiley: ")
				else :
					st.markdown("Sentiment:: Negative :angry: ")
					st.write("Partagez l'amour autour de vous, un sourire est milles fois mieux qu'une haine :smiley: ")
				#else:
				#	st.markdown("Sentiment:: Neutral 😐 ")


			with col2:
				st.info("Astuce")
				st.write("Partagez l'amour autour de vous, un sourire est milles fois mieux qu'une haine :smiley: ")

	else:
		st.subheader("Nous détectons le discours haineux avec le deep learing. Notre modèle est basé sur AraBERT.")


if __name__ == '__main__':
	main()