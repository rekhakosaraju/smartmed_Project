
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
# # from sqlalchemy import create_engine
from dotenv import load_dotenv
# # import os
# # from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()
# # llm = ChatGoogleGenerativeAI(
# #     model="gemini-pro",
# #     temperature=0,
# # )


from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase

# Initialize the database connection
db = SQLDatabase.from_uri("sqlite:///hospital.db")
# db = SQLDatabase.from_uri("")

# Initialize the language model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Create the SQL agent executor
# agent = create_sql_agent(
#     llm, 
#     db=db,
#     # toolkit=SQLDatabaseToolkit(db=db, llm=llm),
#     agent_type="openai-tools", 
#     verbose=True
# )
# db =SQLDatabase(create_engine(f"sqlite:///hospital.db"))
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(llm=llm,toolkit=toolkit,verbose=True,agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
