from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os
load_dotenv()

prompt1=PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template="Generate 5 pointer summary from following text \n {text}",
    input_variables=['text']
)

model=ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)
parser=StrOutputParser()
chain=prompt1 | model | parser | prompt2 | model | parser

result=chain.invoke({'topic':'Layoff in india IT'})
print(result)
