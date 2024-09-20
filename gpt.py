import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import pipeline

# Streamlit app title and header
st.title("GPT-2 Text Generator App")
st.write("This app uses GPT-2 to generate text based on your input prompt.")

# Load pre-trained GPT-2 model and tokenizer
@st.cache_resource
def load_model():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generator

generator = load_model()

# Text input from the user
user_input = st.text_area("Enter a prompt:")

# Slider for max length of generated text
max_length = st.slider("Max length of generated text:", min_value=50, max_value=300, value=100)

# Generate text button
if st.button("Generate Text"):
    if user_input:
        # Generate text from the input prompt
        output = generator(user_input, max_length=max_length, num_return_sequences=1)
        # Display the generated text
        st.write("Generated Text:")
        st.write(output[0]['generated_text'])
    else:
        st.write("Please enter a text prompt to generate text.")

# Run Streamlit app using: streamlit run gpt_streamlit.py
