from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
import os

load_dotenv()  


os.environ["OPENAI_API_BASE"] = 'https://api.groq.com/openai/v1'
os.environ["OPENAI_MODEL_NAME"] = 'Llama3-70b-8192'  # Adjust based on available model
os.environ["OPENAI_API_KEY"] = os.getenv("GROQ_API_KEY")


text = """
Bridging Technology and Business: Unlocking New Value for Clients

One of the great advantages of being independent is the freedom to experiment. 

I recently realized that focusing solely on technical solutions was like building a glass ceiling for myself. To truly enhance value, I needed to be intricately involved in the day-to-day business operations.

With this insight, for a recent project, instead of merely setting up a CRM and Customer Journey pipeline using tools like CG, BigQuery, Airflow, Segment, Hubspot, Python, NodeJS, or Java, I expanded my services to include a consultancy package. 

This wasn't just about building infrastructure—it was about maximizing lead generation and conversion rates through tailored CRM consultancy!

This experience was enlightening. I discovered significant synergies between the technical and business aspects of the project. 

I am now convinced that mastering both domains provides a unique selling proposition for my clients, enabling me to deliver unparalleled service."
"""

classifier = Agent(
    role = "AI classifier",
    goal = "accurately classify messages based on if they were created with AI or not. give every messages one of these ratings: AI or No-AI",
    backstory = "You are an AI assistant whose only job is to classify messages accurately and honestly. Your job is to help the user know if the messages was made by an AI.",
    verbose = True,
    allow_delegation = False,
)

responder = Agent(
    role = "AI responder",
    goal = "respond to the message based on their classification. If the messages is AI generated respond as an AI if it is not AI try to respond as a human",
    backstory = "You are an AI assistant whose only job is to respond to messages accurately and honestly. Your job is to help the user answer messages.",
    verbose = True,
    allow_delegation = False,
)

classify_email = Task(
    description = f"Classify the following text: '{text}'",
    agent = classifier,
    expected_output = "One of these two options: 'AI' or 'Not-AI'",
)

respond_to_email = Task(
    description = f"Respond to the message: '{text}' based on the asnwer provided by the 'classifier' agent.",
    agent = responder,
    expected_output = "a very concise response to the message based on the answer provided by the 'classifier' agent.",
)

crew = Crew(
    agents = [classifier, responder],
    tasks = [classify_email, respond_to_email],
    verbose = 2,
    process = Process.sequential

)
output = crew.kickoff()
print(output)
