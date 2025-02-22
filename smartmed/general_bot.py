import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
prompt_template = PromptTemplate.from_template(
    template="""You are a health management assistant. Your role is to provide accurate information and guidance on health-related topics, including diseases, symptoms, dietary recommendations, and food suggestions. 
    You have a list of medical specialties: Cardiology, Gastroenterology, Psychiatry, General Surgery, Neurology, Gynecology, Orthopedics, Pediatrics, Dermatology, and Family Medicine. 
    Based on the symptoms provided by the user, suggest which specialty doctor they should consult. 
    If the symptoms are unclear, ask the user for more details about their condition.Take all the symptoms from patient then suggest the doctor. Confirm with the user if he mentioned all the symptoms or not. Always respond with 'I can't help you with that' for questions outside health management. Utilize the chat history to provide relevant answers
    %CHAT HISTORY
    chat history : {chat_history}
    
    %USER QUESTION
    Question : {user_question}
    """) 

# Create a prompt using the prompt template 
# question="No, I haven't eaten anything unusual and i am not taking any medications also."
# prompt = prompt_template.format(chat_history=history,user_question=question) 
# print("The prompt is:",prompt) 
# history.append(HumanMessage(content=question))

# # Generate results using the LLM application 
# result = llm.invoke(prompt) 
# history.append(HumanMessage(content=result.content))

# print("The output is:",result.content) 
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

class UserDetails(BaseModel):
    doctor_name: str = Field(description="Name of the Doctor")
    patient_name: str = Field(description="Name of the Patient")
    time_slot: str = Field(description="Time Slot")
    availability_date: str = Field(description="Availability Date")
    can_book: bool = Field(description="Can we book appointment with given data")
    followup_question: str = Field(description="Generate Followup question you don't fin any one of the datails (doctor_name, patient_name, time_slot, availability_date)")

parser = JsonOutputParser(pydantic_object=UserDetails)
appointment_booking_prompt = PromptTemplate(
    template = """
You are an Appointment Booking Agent. Your task is to extract the details (doctor_name, patient_name, time_slot, available_date) from the given chat history or the user's query.

First, analyze the chat history and try to extract the details. If any of the details (doctor_name, patient_name, time_slot, available_date) are missing or unclear, ask the user for the missing information. Ensure that all details are clear before proceeding with the appointment booking.

Ensure the time_slot is provided in the format: 'hh:mm AM/PM - hh:mm AM/PM' (e.g., '01:00 PM - 02:00 PM').

You should extract details in the following format:
- doctor_name: [Extracted doctor's name]
- patient_name: [Extracted patient's name]
- time_slot: [Extracted time slot, e.g., '02:00 PM - 03:00 PM']
- available_date: [Extracted date, e.g., 'tomorrow', 'Monday', '2024-11-12']
- Can_book: [boolean indicating whether the appointment can be booked based on the extracted details]
{format_instructions}
%CHAT HISTORY
chat history: {chat_history}

%USER Query
query: {query}

If the details cannot be extracted from the chat history, ask the user for more information. Once you have gathered all necessary details, confirm them with the user. If any required detail is missing (doctor_name, patient_name, time_slot, or available_date), set `Can_book` to `False` and ask for the missing information. If all details are valid and complete, set `Can_book` to `True`.
""",

        input_variables=["chat_history", "query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
)

import sqlite3

def book_appointment_helper(result):
    connection = sqlite3.connect('hospital.db')
    cursor = connection.cursor()

    # Check if the record exists
    check_query = '''
    SELECT COUNT(*) FROM availability
    WHERE doctor_id = (SELECT doctor_id FROM doctors WHERE name = ?)
    AND available_date = ?
    AND time_slot LIKE ?;
    '''

    cursor.execute(check_query, (result["doctor_name"], result["availability_date"], f'%{result["time_slot"]}%'))
    record_exists = cursor.fetchone()[0] > 0

    if record_exists:
        # Proceed to update the record
        sql_query = '''
        UPDATE availability
        SET slot_booked = TRUE,
            patient_name = ?
        WHERE doctor_id = (SELECT doctor_id FROM doctors WHERE name = ?)
        AND available_date = ?
        AND time_slot LIKE ?;
        '''
        
        cursor.execute(sql_query, (result["patient_name"], result["doctor_name"], result["availability_date"], f'%{result["time_slot"]}%'))
        connection.commit()
        message = "Appointment Booked"
    else:
        sql_query_insert = '''
        INSERT INTO availability (doctor_id, available_date, time_slot, slot_booked, patient_name)
        SELECT doctor_id, ?, ?, TRUE, ?
        FROM doctors
        WHERE name = ?;
        '''
        
        cursor.execute(sql_query_insert, (result["availability_date"], result["time_slot"], result["patient_name"], result["doctor_name"]))
        connection.commit()
        message = "New appointment created"

    # Closing the connection
    cursor.close()
    connection.close()
    
    return message