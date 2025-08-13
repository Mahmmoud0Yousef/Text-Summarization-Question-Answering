import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline

# Load the summarization model
save_dir = "t5-Text-Summarizer"
model = T5ForConditionalGeneration.from_pretrained(save_dir)
tokenizer = T5Tokenizer.from_pretrained(save_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Q&A pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Summarization function
def summarize_text(article):
    inputs = tokenizer(
        article,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=150,
        min_length=30,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ---------------- Streamlit UI ----------------
st.title("📄 Text Summarization & Q&A Tool")

# Step 1: Enter text and summarize
text_input = st.text_area("✏️ Enter text to summarize:")

if st.button("🔍 Summarize"):
    if text_input.strip():
        with st.spinner("Summarizing..."):
            summary = summarize_text(text_input)
        st.session_state["summary"] = summary
        st.subheader("📝 Summary:")
        st.write(summary)
    else:
        st.warning("Please enter some text first.")

# Step 2: Ask questions about the summary
if "summary" in st.session_state:
    st.subheader("❓ Ask questions about the summary")
    questions_input = st.text_area(
        "Enter questions (one per line):",
        placeholder="Example:\nWho is the comedian?\nWhere did the comedian travel?"
    )

    if st.button("💬 Get Answers"):
        if questions_input.strip():
            questions = [q.strip() for q in questions_input.split("\n") if q.strip()]
            for q in questions:
                answer = qa_pipeline(question=q, context=st.session_state["summary"])
                st.write(f"**Q:** {q}")
                st.write(f"**A:** {answer['answer']}")
        else:
            st.warning("Please enter at least one question.")
