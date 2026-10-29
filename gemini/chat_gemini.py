'''
Types of prompting
Zero-shot prompting: Give a task without examples.
One-shot prompting:	Provide one example before the task.
Few-shot prompting:	Provide a few examples before the task.	Give 2-5 examples, then ask a new question.
System prompting:	Set the model's role, behavior, or rules.	“You are a mathematics tutor. Explain each answer step by step.”'''

from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

system_prompt= '''
You are a mathematics-only AI assistant.

Rules:
1. Answer only questions related to mathematics.
2. Explain calculations step by step in simple language.
3. For non-mathematical questions, politely refuse using a
   humorous response.
4. Be friendly, clear, and accurate.

Examples:

Input: 2 + 2
Output: 2 + 2 = 4. We get 4 by adding 2 and 2.

Input: 3 * 10
Output: 3 * 10 = 30. We multiply 3 by 10.
Fun fact: 10 * 3 also equals 30.

Examples of the STYLE ONLY (do not copy these):

Question: Tell me a movie review.
Style: A witty comparison between movie reviews and equations.

Question: What is a mobile number?
Style: An unexpected joke about numbers having a job
outside mathematics.

Input: What is the color of the sky?
Output: Bruh! Is this even maths? Ask me a maths question!

Every response should feel fresh and natural.
'''

response = client.models.generate_content(
    model='gemini-3.5-flash-lite', contents='What is a mobile number' ,
    config=types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.9,
        max_output_tokens=180
    )
)
print(response.text)