import streamlit as st
from transformers import MarianMTModel, MarianTokenizer, BartForConditionalGeneration, BartTokenizer

# Load translation models
def load_translation_model(model_name):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

# Load summarization model
def load_summarization_model():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    return tokenizer, model

# Translation function
def translate_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

# Summarization function
def summarize_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=50, length_penalty=2.0, num_beams=4)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Load models
st.session_state.en_fr_tokenizer, st.session_state.en_fr_model = load_translation_model("Helsinki-NLP/opus-mt-en-fr")
st.session_state.en_es_tokenizer, st.session_state.en_es_model = load_translation_model("Helsinki-NLP/opus-mt-en-es")
st.session_state.summarizer_tokenizer, st.session_state.summarizer_model = load_summarization_model()

# Streamlit UI
st.title("Insurance Query Chatbot")

# User Input
user_input = st.text_area("Enter your text:")

# Option Selection
option = st.selectbox("Select an option:", ["Translate to French", "Translate to Spanish", "Summarize", "Exit"])

if st.button("Submit"):
    if option == 'Translate to French':
        translation = translate_text(user_input, st.session_state.en_fr_tokenizer, st.session_state.en_fr_model)
        st.success(f"**Translated to French:** {translation}")
    
    elif option == 'Translate to Spanish':
        translation = translate_text(user_input, st.session_state.en_es_tokenizer, st.session_state.en_es_model)
        st.success(f"**Translated to Spanish:** {translation}")
    
    elif option == 'Summarize':
        summary = summarize_text(user_input, st.session_state.summarizer_tokenizer, st.session_state.summarizer_model)
        st.success(f"**Summarized Text:** {summary}")
    
    else:
        st.warning("Exiting the chatbot.")
