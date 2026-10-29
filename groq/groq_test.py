
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

system_prompt = '''
You are an AI assistant who is an expert at breaking down
complex problems and solving user queries.

For each user input, follow these steps in sequence:
"analyse", "think", "output", "validate", "result".

Rules:
1. Return one JSON object per response.
2. Each object must contain "step" and "content".
3. Perform only the next step requested by the program.
4. Do not repeat a previous step.
5. Give a brief, clear, user-visible explanation.
6. In the "think" step, provide a concise explanation,
   not private internal reasoning.

Output format:
{"step": "string", "content": "string"}

Example:
{"step": "analyse", "content": "The user asks for an arithmetic calculation."}
{"step": "think", "content": "Multiplication is performed before addition."}
{"step": "output", "content": "4 * 5 = 20, then 3 + 20 = 23."}
{"step": "validate", "content": "The calculation follows the order of operations."}
{"step": "result", "content": "3 + 4 * 5 = 23."}
'''


# Start the conversation with the system prompt and user question

steps = ["analyse", "think", "output", "validate", "result"]

messages = [
    {"role": "system", "content": system_prompt},
    {
        "role": "user",
        "content": (
            "Solve 3 + 4 * 5. Give only the analyse step "
            "as one JSON object."
        )
    }
]

for i, step in enumerate(steps):
    try:
        response = client.chat.completions.create(
    model="qwen/qwen3.8-27b",
    messages=messages,
    temperature=0.3,
    max_tokens=300,
    response_format={"type": "json_object"},
    reasoning_effort="none"
)

        content = response.choices[0].message.content

        if not content or not content.strip():
            print(f"No content returned for step: {step}")
            break

        print(f"\nStep {i + 1} ({step}):")
        print(content)

        messages.append({
            "role": "assistant",
            "content": content
        })

        if i < len(steps) - 1:
            next_step = steps[i + 1]
            messages.append({
                "role": "user",
                "content": (
                    f"Give only the {next_step} step next. "
                    "Return one JSON object with step and content."
                )
            })

    except Exception as e:
        print(f"Error at step {step}: {e}")
        break
