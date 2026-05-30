from langchain_cerebras import ChatCerebras
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableParallel
load_dotenv()
import os



model1=ChatCerebras(
  model="gpt-oss-120b", 
api_key=os.environ.get("CEREBRAS_API_KEY")
)

model2=ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
    
)

prompt=PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}",
    input_variables=["text"]
)

prompt1=PromptTemplate(
    template="Generate  5 short question answer from following text \n {text}",
    input_variables=["text"]
)

prompt3=PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n notes ->{notes} and {quiz}",
    input_variables=["notes","quiz"]
)

parser=StrOutputParser()

parallel_chain=RunnableParallel(
    {
        'notes': prompt | model1 | parser,
        'quiz':prompt1 | model2 | parser
    }
)

merge_chain=prompt3 | model1 | parser

chain= parallel_chain | merge_chain
text="""Layoffs are a major challenge in today’s corporate world, especially in the IT industry. A layoff happens when a company reduces its workforce to cut costs, improve efficiency, or handle financial problems. In recent years, many technology companies have announced layoffs due to economic slowdown, reduced demand, automation, and changing business strategies.
"""
result=chain.invoke({'text':text})
print(result)

chain.get_graph().print_ascii()

