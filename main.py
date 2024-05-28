from langchain_community.llms import Ollama
from crewai import Agent, Task, Crew, Process
import os

os.environ["OPENAI_API_BASE"] = 'https://api.groq.com/openai/v1'
os.environ["OPENAI_MODEL_NAME"] = 'llama3-8b-8192'
os.environ["OPENAI_API_KEY"] = 'my-api-key'

model = Ollama(model = "llama3")

email = "Congratulations! You have been accepted into Harvard University. We are excited to have you as a student. Please confirm your acceptance by replying to this email."

classifier = Agent(
    role = "email_classifier",
    goal = "accurately classify emails based on their importance. Classify them as important, casual, or spam.",
    backstory = "You are an AI assistant that helps people manage their emails. Your only job is to clqassify emails based on their importance. Do not be afraid to give a bad classification to an email if it seems to be less important. Your job is to help users manage their inbox.",
    verbose = True,
    allow_delegation = False,
)  

responder = Agent(
    role = "email_responder",
    goal = "respond to emails based on their importance. Respond to important emails with a detailed response, casual emails with a short response, and spam emails with a deletion.",
    backstory = "You are an AI assistant that helps people manage their emails. Your only job is to respond to emails based on their importance. Do not be afraid to give a bad response to an email if it seems to be less important. Your job is to help users manage their inbox.",
    verbose = True,
    allow_delegation = False,
)

classify_email = Task(
    description = f"Classify the email: {email}",
    agent = classifier,
    expected_output = "One of these three options: 'important','casual' or 'spam'"
)

respond_email = Task(
    description = f"Respond to the email: {email} based on the importance provided by the 'classifier' agent.",
    agent = responder,
    expected_output = "A response to the email. If the email is classified as 'important', the response should be detailed. If the email is classified as 'casual', the response should be short. If the email is classified as 'spam', the response should be a deletion."
)

crew = Crew(
    agents=[classifier, responder],
    tasks = [classify_email, respond_email],
    verbose=2,
    process= Process.sequential
)

output = crew.kickoff()
print(output) 