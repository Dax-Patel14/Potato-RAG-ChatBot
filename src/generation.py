# File: src/4_generation.py
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI 

load_dotenv()

def generate_answer(prompt: str) -> str:
    print("Calling OpenAI API for generation...")

    # Initialize the ChatOpenAI model with gpt-4o-mini
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0.1
    )
    
    try:
        # The response is an AIMessage object, so we access its content
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"An error occurred during the API call: {e}")
        return "Sorry, an error occurred while trying to generate a response."