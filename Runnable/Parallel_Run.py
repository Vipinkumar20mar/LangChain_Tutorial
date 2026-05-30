from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableSequence
import os

load_dotenv()

prompt1=PromptTemplate(
    template="Generates a tweet about \n {topic}",
    input_variables=["topic"]   
)
prompt2=PromptTemplate(
    template="Generates a Linkedin post about \n {topic}",
    input_variables=["topic"]   
)

model=ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
    
)

parser=StrOutputParser()

chain=RunnableParallel({
"tweet":RunnableSequence(prompt1,model,parser),
"linkedin":RunnableSequence(prompt2,model,parser)

})

result=chain.invoke({
    "topic":"Layoff"
})
print(result['linkedin'])