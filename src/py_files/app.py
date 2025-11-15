import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.tools.retriever import create_retriever_tool

from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from typing import Annotated
from langchain_openai import ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate
# from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
import streamlit as st

from dotenv import load_dotenv
load_dotenv(override=True)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


llm = ChatOpenAI(model="gpt-4.1", temperature=0)


## processing Policy RAG + FAISS

#pdf_folder = r"C:\AgenticAI\Projects_AgentAI\AgenticAI_04_companypolicy\pdf_files"
current_dir = os.path.dirname(os.path.abspath(__file__))
pdf_folder = os.path.abspath(os.path.join(current_dir, "..", "..", "pdf_files"))

pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")] ## collect all PDF files
docs = []
for file in pdf_files:
    loader = PyPDFLoader(file)
    docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs_splitter = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(
    documents = docs_splitter,
    embedding = embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k":3})


## prepare for SQL_Server connection
import pyodbc

def run_sql_query(query: str):
    """ execute the sql query"""

    # SQL server connection details
    server = r"DESKTOP-6SIQQDV" ## DESKTOP-6SIQQDV\INSTANCE2022
    # server = "192.168.10.36"    
    database = "ABC_Company"
    username = "sa"
    password = "Sagar@12"


    # Connection string for SQL Authentication
    conn_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        f"Encrypt=no;"
    )

    # connect to database
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()            # ferch rows
    cursor.close()
    conn.close()

    return rows


## define State
class State(TypedDict):
    question: str
    query_type: str  ## policy or employee
    context: str
    sql_query: str
    answer: str

## Classify User Query
def classify(state: State):
    """Classify if User question is about policy or employee"""
    question = state["question"]

    system_prompt = """
    You are a routing assistant.
    You need to read the question carefully and understand it. 
    If question is about company policy, leave policy, rules, working hours then output should "policy".
    if questions is about employees details, employees information like salary, department, name then output should "employee".

    Please note: respond with only one word either "policy" or "employee"
    """

    messages = [
        {"role":"system", "content":system_prompt},
        {"role":"user", "content":question}
    ]

    result = llm.invoke(messages)
    query_type = result.content.strip().lower()
    return {"query_type": query_type}

## policy agent
def policy_agent(state: State):
    """llm's will generate precise answer from pdf documents relevant to User's question"""
    question = state["question"]
    # relevant_docs = retriever.get_relevant_documents(question)
    relevant_docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    system_text = """
    You are an expert HR policy assistant.
    Answer the question strictly based on the provided policy document context.
    Be concise, factual, and do not add extra assumptions.
    If the answer is not in the document, say: "The policy document does not mention this specifically.
    """

    messages = [
        {"role":"system","content":system_text},
        {"role":"user", "content":f"Question: {question}\n\n Context:\n{context}"}
    ]

    response = llm.invoke(messages)

    return {"context": context, "answer": response.content}




# First define a structured output
class EvaluatorOutput(BaseModel):
    sql_query: str = Field(description="structured SQL query response from the User's question")


## sql agent
def sql_agent(state: State):
    """llm will generate natural language to structured sql query"""
    question = state["question"]


    SQL_SYSTEM_MESSAGE = """
    You are a SQL Query Generator. Please note You are SQL Query Generator for Microsoft SQL Server.
    Always use Miscrosoft SQL Server syntax. 

    *** Do NOT use backticks (`). Use square brackets [ ] for column or table names if needed.

    Table Name: Employees
    Columns:
    - EmployeeID (int, primary key)
    - FirstName (varchar)
    - LastName (varchar)
    - Gender (varchar)
    - Age (int)
    - Department (varchar)
    - JobTitle (varchar)
    - Salary (decimal)
    - JoinDate (datetime)
    - ManagerID (int)
    - Email (varchar)
    - working_hours (int)



    Given an question, create a syntactically correct query to run to help find the answer.
    Unless the User specifies in his questions a specific number of examples they want to obtain, always limit your query to at most top 6 results. 
    You can order the results by a relevant columns to return the most interesting examples in the database.

    Pay attention to use only the column names that you can see in the schema description.
    Be careful to not query for columns that do not exist.

    """

    query_prompt_template = ChatPromptTemplate.from_messages([
            ("system", SQL_SYSTEM_MESSAGE), 
            ("user", question)
            ])

    structured_llm = llm.with_structured_output(EvaluatorOutput)
    # print(structured_llm)

        
    ## format the prompt
    messages = query_prompt_template.format_messages(input_text=state["question"])
        
    ## run llm to generate sql query
    result: EvaluatorOutput = structured_llm.invoke(messages)
        
    ## update state with query
    state["query"] = result.sql_query

    sql_query = state["query"]

    ## execute sql query
    try:
        rows = run_sql_query(sql_query)
    except Exception as e:
        return {
            "sql_query": sql_query,
            "answer": f"❌ SQL execution failed with error: {e}"
            }
    





    # sql_messages = [
    #     {"role":"system", "content":SQL_SYSTEM_MESSAGE},
    #     {"role":"user", "content":question}
    # ]

    # sql_text = llm.invoke(sql_messages)
    # sql_query = sql_text.content.strip()
    # print(f"\n Generated sql query: \n {sql_query}")

    
    #     ## execute sql query
    # try:
    #     rows = run_sql_query(sql_query)
    # except Exception as e:
    #     return {
    #         "sql_query": sql_query,
    #         "answer": f"❌ SQL execution failed with error: {e}"
    #         }
    

    
    ## Explain back structured SQL query result in simple language to User
    explain_query = f"""
    You are a helpful assistant who explains SQL query results in simple, natural language. 
    Here are the results : {rows}


    Output Style:
    - Briefly restate the user's request
    - Summarize what the results show
    - Bullet list of key rows (Name and Salary or similar fields)
    - One concluding sentence
    - End with a short, helpful conclusion



    Do NOT use tables.
    Just speak like you're summarizing to a colleague.
    """

    explain_sql = [
        {"role":"system", "content":explain_query}
    ]
    explain_text = llm.invoke(explain_sql)
    return {"sql_query":sql_query, "answer": explain_text.content}


## build the langgraph
graph = StateGraph(State)

graph.add_node("classify_query", classify)
graph.add_node("policy_agent", policy_agent)
graph.add_node("sql_agent", sql_agent)


graph.set_entry_point("classify_query")

graph.add_conditional_edges("classify_query", lambda state: state["query_type"], {"policy":"policy_agent", "employee":"sql_agent"})

graph.add_edge("policy_agent", END)
graph.add_edge("sql_agent", END)

state_graph = graph.compile()


# -------------------------------------------------------------------
# 🎨 Streamlit UI
# -------------------------------------------------------------------
st.title("🤖 Company Assistant (Policy + Employee Info)")
st.write("Ask about company policies or employee details. The system will automatically decide where to look.")

user_question = st.text_input("💬 Enter your question:")

if st.button("Run Query"):
    if user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing your query..."):
            result = state_graph.invoke({"question": user_question})
        st.subheader("🧠 Final Answer")
        st.write(result["answer"])


