import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import PyPDF2
import tempfile

from giskard.rag import KnowledgeBase, generate_testset
import pandas as pd
from giskard.rag import evaluate, RAGReport
from giskard.rag.metrics.ragas_metrics import ragas_context_recall, ragas_context_precision
from giskard.llm.client.openai import OpenAIClient
import giskard

load_dotenv()

giskard.llm.set_llm_api("openai")
oc = OpenAIClient(model="gpt-3.5-turbo")
giskard.llm.set_default_client(oc)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def create_qa_chain(uploaded_file_path):
    # Load document if file is uploaded
    if uploaded_file is not None:
        documents = [extract_text_from_pdf(uploaded_file_path)]
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)

        # Select embeddings
        embeddings = OpenAIEmbeddings()
        # Create a vectorstore from documents
        db = Chroma.from_documents(texts, embeddings)
        # Create retriever interface
        retriever = db.as_retriever()
        # Create QA chain
        qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0), chain_type='stuff', retriever=retriever)
        return qa, texts

qa, texts = None, None

def generate_response(query_text):
    return qa.run(query_text)

# # Streamlit app
# def main():
# Page title
st.set_page_config(page_title='Talk to my PDF')
st.title('Talk to my PDF')

# File upload
uploaded_file = st.file_uploader('Upload an article')
if uploaded_file is not None:
    # Load and process the PDF
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name
    # Define input and output paths
    filename = temp_file_path
    qa, texts = create_qa_chain(filename)
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted:
        with st.spinner('Looking for response...'):
            response = generate_response(query_text)
            result.append(response)

if len(result):
    st.info(response)
knowledge_base = None
st.title("Giskard RAG Diagnostics")
st.write("There are 4 steps: \n 1. Create a KB \n 2. Creating a Testset \n 3. Evaluation \n 4. Reports")
if st.button("Start with KB"):
    knowledge_base_df = pd.DataFrame([t for t in texts[0].page_content.split("\n")], columns=["text"])
    knowledge_base = KnowledgeBase(knowledge_base_df)
    st.dataframe(knowledge_base_df, use_container_width=True)
    # if st.button("Generate Samples"):
    num_question = st.text_input("Please enter the number of samples you wish to create for testset(it would take long time to generate):", 10)

    with st.spinner('Creating Testset and then generating report...'):
        testset = generate_testset(knowledge_base,
                                num_questions=int(num_question),
                                agent_description="A chatbot answering questions about the document")
        st.write("Testset Samples:")
        st.dataframe(testset.to_pandas(), use_container_width=True)

    
        report = evaluate(generate_response,
                testset=testset,
                knowledge_base=knowledge_base,
                metrics=[ragas_context_recall, ragas_context_precision])
        
        txt_path = report.to_html("file.html")
        report_html = open("file.html").read()
        st.components.v1.html(report_html, scrolling=True, height=500)
        # st.markdown(report_html, unsafe_allow_html=True)

        # Correctness on each topic of the Knowledge Base
        st.write("Report Analysis metrics:")

        st.write("Report correctness by topic metrics:")
        st.write(report.correctness_by_topic())

        # Correctness on each type of question
        st.write("Report correctness by question type metrics:")
        st.write(report.correctness_by_question_type())

        # get all the failed questions
        st.write("Report failures:")
        st.write(report.failures)

        # get the failed questions filtered by topic and question type
        st.write("Report failures by topic:")
        st.write(report.get_failures(topic="Topic from your knowledge base", question_type="simple"))

        st.write("Evaluation Report of RAG:")
        print(report.to_pandas())
        st.dataframe(report.to_pandas(), use_container_width=True)
#st.markdown(text_file, unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()