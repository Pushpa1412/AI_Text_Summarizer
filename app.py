import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="🧠",
    layout="centered"
)


@st.cache_resource
def load_summarizer():
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )


st.title("🧠 AI Text Summarizer")
st.write("Paste your text below and get a concise summary instantly!")

text_input = st.text_area(
    "Enter text to summarize:",
    height=250
)

if st.button("Summarize"):
    if text_input.strip():
        try:
            summarizer = load_summarizer()

            summary = summarizer(
                text_input,
                max_length=130,
                min_length=30,
                do_sample=False
            )

            st.subheader("✨ Summary:")
            st.success(summary[0]["summary_text"])

        except Exception as e:
            st.error(f"Error: {e}")

    else:
        st.warning("Please enter some text before summarizing.")


st.markdown("---")
st.caption(
    "Built with ❤️ by Pushpa Bachar using Streamlit & Hugging Face Transformers"
)