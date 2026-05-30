from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal
from langchain_core.runnables import RunnableBranch,RunnableLambda
import os
load_dotenv()


 # structued output


class Feedback(BaseModel):
    sentiment:Literal['Positive','Negative']=Field(description="Give the sentiment of the feedback")

parser2=PydanticOutputParser(pydantic_object=Feedback)


prompt1=PromptTemplate(
    template=" classify sentiment of the following  feedback text into positive or negative \n{feedback} \n {format_instruction}",
    input_variables=["feedback"],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

model=ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
    
)
parser=StrOutputParser()

classfier_chain= prompt1 | model | parser2

#print(classfier_chain.invoke({'feedback':'This is a wonderful smartphone'}))

prompt = PromptTemplate(
    template="""
You are a sentiment analysis expert.

Classify the sentiment of the following feedback
as either Positive or Negative.

Feedback:
{feedback}

Return only one word:
Positive or Negative
""",
    input_variables=["feedback"]
)
branch_chain=RunnableBranch(
(lambda x:x.sentiment =='Positive',prompt | model| parser),
(lambda x:x.sentiment =='Negative',prompt | model| parser),
RunnableLambda(lambda x:"could not find the sentiment")

)

chain=classfier_chain | branch_chain

print(chain.invoke({'feedback':'this is a poor quality phone'}))