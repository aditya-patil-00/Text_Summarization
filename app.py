from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import streamlit as st

tokenizer = AutoTokenizer.from_pretrained("saved_model")
model = AutoModelForSeq2SeqLM.from_pretrained("saved_model")

def generate_summary(input_text):
    
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(**inputs, max_length=150, min_length=10, length_penalty=2.0, num_beams=4)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

st.title("Text Summarization App")
input_text = st.text_area("Enter your text here")
if st.button("Generate Summary"):
    if input_text:
        summary = generate_summary(input_text)
        st.write("Summary:")
        st.write(summary)
    else:
        st.warning("Please enter text to summarize.")