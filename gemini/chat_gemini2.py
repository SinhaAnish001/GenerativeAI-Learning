from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

system_prompt= '''
You are an AI assistant who is expert in breaking down complex problems and then resolve the useer query.

For the given answer input , analyse the input and breakdown thw problem step by step.
Atleast think 5-6 steps on how to solve the problem before solving it down.

the The steps are you get a user input, you analyse , you think , you again think for several times and then return an output with explanation and then finally you validate the output as well before giving final result.

Follow this steps in sequence tthat is "analyse", "think", "output", "validate", "result".

Rules: 
1. Follow the strict JSON output as per the Output schema,
2. Always perform one step at a time and wait for next input.
3. Carefully analyse the user query

Output Format:
{{ step: "string" , content: "string"}}

Example:
Input: What is 2 + 2.
Output: {{step: "analyse", content: "Alright! The user is intrested in maths query and he is asking a basic arithmetic operation."}}
Output: {{step: "think", content: "To perform the addition i must go from left and right and add all the operands."}}
Output: {{step: "output", content: "4"}}
Output: {{step: "validate", content: "seems like 4 is the correct ans for 2 + 2"}}
Output: {{step: "result", content:"2 + 2 = 4 and that is calculated by adding all number}}
'''

# First API call
response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents="Analyse this problem: What is 3 + 4 * 5?",
    config={"system_instruction": system_prompt}
)

print("First response:")
print(response.text)

# Repeat until the final step
for step_number in range(5):
    print(f"\nStep {step_number + 1}:")
    print(response.text)

    if step_number == 4:
        break

    # Feed the previous response back automatically
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=f"""
        Previous response:
        {response.text}

        Continue with the NEXT step only.
        Follow the five-step sequence in your system prompt.
        Do not repeat the previous step.
        """,
        config={"system_instruction": system_prompt}
    )

# answer
'''Step 1:
{"step": "analyse", "content": "The user is asking a mathematical question involving both addition and multiplication: 'What is 3 + 4 * 5?'."}

Step 2:
{"step": "think", "content": "According to the order of operations (PEMDAS/BODMAS), multiplication must be performed before addition. Therefore, I need to first calculate 4 * 5, which is 20, and then add 3 to that result."}

Step 3:
{"step": "output", "content": "3 + 4 * 5 = 3 + 20 = 23"}

Step 4:
{"step": "validate", "content": "Checking the previous calculation: 4 multiplied by 5 is 20, and 3 plus 20 is indeed 23, following the standard order of operations (PEMDAS/BODMAS)."}}

Step 5:
{"step": "result", "content": "The final calculation for the expression 3 + 4 * 5 is 23, correctly determined by applying the standard order of operations where multiplication is performed before addition."}
'''