from langchain.agents import Tool
# from langchain_core.tools import tool
from doctor_agent import agent
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from general_bot import prompt_template,appointment_booking_prompt, parser, book_appointment_helper
from langchain.prompts import PromptTemplate
from langchain.tools.render import render_text_description
from langchain import LLMChain
from typing import List, Union
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate, ChatPromptTemplate
import re
import sqlite3

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
prompt_react = hub.pull("hwchase17/react")
retrieved_text = ""


def get_answer(query: str) -> str:
    tprompt = prompt_template.format(chat_history=st.session_state.messages,user_question=query) 
    result = llm.invoke(tprompt)
    return result.content

def get_doctors_data(query: str) -> str:
    print("===========================================================")
    print("QUERY : ",query, st.session_state.messages[-1])
    print("===========================================================")

    response =agent.invoke(st.session_state.messages[-1])
    return f"\n{response['output']}\n"

def book_appointment(query:str) -> str:
    # tprompt = appointment_booking_prompt.format(chat_history=st.session_state.messages,user_question=query) 
    # result = llm.invoke(tprompt)
    # print("===========================================================")
    # print(result.content)
    # print("===========================================================")
    chain = appointment_booking_prompt | llm | parser

    print("MESSAGES : ",st.session_state.messages)
    result = chain.invoke({"chat_history":st.session_state.messages,"query": query})
    
    print("===========================================================")
    print(result)
    print("===========================================================")
    if result['doctor_name'] and result['patient_name'] and result['time_slot'] and result['availability_date'] and result['can_book']:
       return book_appointment_helper(result)
            
    return result["followup_question"] #"Appointment Booked"


# Updated Tool Descriptions:

# Tool for general health information, disease, symptoms, etc.
get_answer_tool = Tool(
    name="General Health Chatbot",
    func=get_answer,
    description="Useful for answering health-related questions, symptoms, diseases,general conversations related to health and  greetings."
)

# Tool for fetching doctors data from the database
get_doctors_data_tool = Tool(
    name="Get Doctors Data",
    func=get_doctors_data,
    description="Useful for retrieving detailed data about doctors, including their specializations, availability slots, and appointment timings."
)


# Tool for booking, blocking, or modifying appointments
book_appointment_tool = Tool(
    name="Book Appointment",
    func=book_appointment,
    description="Useful for booking, blocking, or modifying appointment slots with doctors. Can be used to book a slot or block a slot for a specific time."
)



class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
            if observation and 'None' not in observation:
                prompt= f"Action : Just print below Statement \n Statement \n '{action.log}\nFinal Answer: {observation}'"
                data= [HumanMessage(content=prompt)]
                return data
            
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]

from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage
# LLM wrapper

class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        
        # Check if agent should finish
        if "Final Answer" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        
        # Parse out the action and action input
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        
        # If it can't parse the output it raises an error
        # You can add your own logic here to handle errors in a different way i.e. pass to a human, give a canned response
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input, log=llm_output)
    
def get_response(query):
    # template = """Answer the following question. You have access to the following tools:
    # If the `input` is a Greeting, reply with a friendly Greeting Message. 

    # {tools}

    # Use the following format:

    # Question: the input question you must answer
    # Thought: you should always think about what to do
    # Action: the action to take, should be one of [{tool_names}]
    # Action Input: the input to the action
    # Observation: the result of the action,
    # ...(this Thought/Action/Action Input/Observation)

    # Final Answer: the final answer to the original input question .

    # Begin!

    # Question: {input}
    # {agent_scratchpad}"""
    template = """Answer the following question, considering the context from the chat history. You have access to the following tools:
    If the input is a Greeting, reply with a friendly Greeting Message.
    If the input relates to booking an appointment or blocking a slot, use the appropriate appointment tool.
    If the input is a health-related question, use the general health tool to provide an answer.
    If the input asks for doctor data, use the doctor data tool.

    {tools}

    Use the following format:

    Question: the input question you must answer
    Chat History: Chat history of the user , that you need to take as reference for making decisions.
    Thought: Based on the chat history and input, I should decide which action to take. Consider the context of the conversation so far and the user's needs.
    Action: The action to take, should be one of [{tool_names}]
    Action Input: The input to the action (you may extract relevant details from the chat history or input)
    Observation: The result of the action
    ...(this Thought/Action/Action Input/Observation)

    Final Answer: the final answer to the original input question, considering both the chat history and the action taken.

    Begin!

    Question: {input}
    
    Chat History: {chat_history}
    {agent_scratchpad}"""

    tools = [get_answer_tool, get_doctors_data_tool,book_appointment_tool]
    prompt = CustomPromptTemplate(
        template=template,
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "chat_history","intermediate_steps"]
    )
    output_parser = CustomOutputParser()
    llm_chain = LLMChain(llm=llm, prompt=prompt)

# Using tools, the LLM chain and output_parser to make an agent
    tool_names = [tool.name for tool in tools]

    agent = LLMSingleActionAgent(
        llm_chain=llm_chain, 
        output_parser=output_parser,
        # We use "Observation" as our stop sequence so it will stop when it receives Tool output
        # If you change your prompt template you'll need to adjust this as well
        stop=["\nObservation:"], 
        allowed_tools=tool_names
    )
    # Initiate the agent that will respond to our queries
# Set verbose=True to share the CoT reasoning the LLM goes through
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, max_iterations=5)
    res = agent_executor.run({"input":query, "chat_history":st.session_state.messages})
    return res
# Streamlit app layout

def main():
    st.set_page_config(page_title=" Conversational Bot!")
    st.title("Conversational Chatbot 💬")

    # initialize the messages key in streamlit session to store message history
    if "messages" not in st.session_state:
        # add greeting message to user
        st.session_state.messages = [
            AIMessage(content="Hello, I am a bot. How can I help you?")
        ]

    # if there are messages already in session, write them on app
    for message in st.session_state.messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)

    prompt = st.chat_input("Say Something")

    if prompt is not None and prompt != "":
        # add the message to chat message container
        # if not isinstance(st.session_state.messages[-1], HumanMessage):
            # display to the streamlit application
        message = st.chat_message("user")
        message.write(f"{prompt}")
        st.session_state.messages.append(HumanMessage(content=prompt))


        # if not isinstance(st.session_state.messages[-1], AIMessage):
        with st.chat_message("assistant"):
            # use .write() method for non-streaming, which means .invoke() method in chain
            response = get_response(prompt)
            st.markdown(response)
        
        st.session_state.messages.append(AIMessage(content=response))

if __name__=="__main__":
    main()