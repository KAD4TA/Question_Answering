from transformers import pipeline
import streamlit as st

# models "question answering"
models={"distilbert":"distilbert-base-cased-distilled-squad","roberta-base":"deepset/roberta-base-squad2","mdeberta":"timpal0l/mdeberta-v3-base-squad2","bertQA":"alphakavi22772023/bertQA","deberta-v3":"LLukas22/deberta-v3-base-qa-en"}
qa_model = st.selectbox('MODEL OPTİONS',models.keys())
qa_pipeline = pipeline("question-answering", model=models.get(qa_model), tokenizer=models.get(qa_model))


def main():
    st.title("QUESİON ANSWERING APP")
    
    # İNPUT THE TEXT
    context = st.text_area("İNPUT YOUR TEXT İN THİS AREA")
    
    # İNPUT A QUESTİON
    question = st.text_input("ASK THE QUESTİON")
    
    # FİND ANSWER PART
    if st.button("FİND ASNWER"):
        if context and question:
            answer = qa_pipeline(question=question, context=context)
            st.write(f"ANSWER: {answer['answer']}")
        else:
            st.warning("PLEASE FİLL THE BLANK PARTS")

if __name__ == "__main__":
    main()