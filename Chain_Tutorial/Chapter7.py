from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os
load_dotenv()


os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

prompt=PromptTemplate(
    template="Generate 5 interesting facts about  {topic}" ,
    input_variables={'topic'}
   
)




llm=init_chat_model(
    model="llama-3.3-70b-versatile",
    model_provider="groq"

)

parser=StrOutputParser()


chain=prompt | llm | parser
result=chain.invoke({'topic':'Cricket'})
print(result)

chain.get_graph().print_ascii()