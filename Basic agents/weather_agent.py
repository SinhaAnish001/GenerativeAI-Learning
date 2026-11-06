
from dotenv import load_dotenv
from groq import Groq
import os
import json
import requests

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def get_weather(city: str):
    print("🔧 Tool called: get_weather", city)

    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}."
    return "Something went wrong"

available_tools = {
    "get_weather": {
        "fn": get_weather,
        "description": "Takes a city name as an input and returns the current weather for the city"
    }
}


system_prompt = '''
    You are a helpful AI assistant who is specialized in solving user queries.
    You work on start, plan, action, observe mode.

    For the given user query and available tools, plan the step-by-step execution.
    Based on the planning, select the relevant tool and perform an action to call it.
    Wait for the observation and, based on the observation from the tool call,
    resolve the user query.

    Rules:
    1. Return one JSON object per response.
    2. Always perform one step at a time.
    3. Carefully analyse the user query.
    4. Never invent tool observations.
    5. For multiple cities, call the weather tool for each city whose data is needed.
    6. Use previous observations when answering follow-up questions.
    7. Give a clear final answer.

    Output: JSON format:
    {
        "step": "string",
        "content": "string",
        "function": "The name of the function if the step is action",
        "input": "The input parameter for the function"
    }

    Available tools:
    - get_weather: Takes a city name as an input and returns the current weather for the city.

    Examples:
    User query: What is the weather of New York?

    Output: {"step": "plan", "content": "The user is interested in weather data of New York."}
    Output: {"step": "plan", "content": "I should call get_weather for New York."}
    Output: {"step": "action", "function": "get_weather", "input": "New York"}
    Output: {"step": "observe", "content": "Actual result returned by the tool"}
    Output: {"step": "output", "content": "The weather in New York is ..."}
'''


messages = [
    {"role": "system", "content": system_prompt}
]


while True:
    user_query = input("> ")

    if user_query.lower() in ["exit", "quit"]:
        break

    messages.append({"role": "user", "content": user_query})

    while True:
        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            response_format={"type": "json_object"},
            messages=messages
        )

        parsed_output = json.loads(response.choices[0].message.content)

        messages.append({
            "role": "assistant",
            "content": json.dumps(parsed_output)
        })

        if parsed_output.get("step") == "plan":
            print(f"🧠: {parsed_output.get('content')}")
            continue

        if parsed_output.get("step") == "action":
            tool_name = parsed_output.get("function")
            tool_input = parsed_output.get("input")

            if tool_name in available_tools:
                output = available_tools[tool_name]["fn"](tool_input)

                messages.append({
                    "role": "user",
                    "content": json.dumps({
                        "step": "observe",
                        "content": output
                    })
                })

                continue

        if parsed_output.get("step") == "observe":
            print(f"👀: {parsed_output.get('content')}")
            continue

        if parsed_output.get("step") == "output":
            print(f"🤖: {parsed_output.get('content')}")
            break


# output
#> what is the temperature of purnea
#🧠: The user is asking for the current temperature of Purnea.
#🧠: I need to call the get_weather tool with 'Purnea' as the input to retrieve the current weather and temperature.
#🔧 Tool called: get_weather Purnea
#🤖: The current temperature in Purnea is +26°C with overcast weather.
#> what is the temperature of gaya
#🧠: The user is asking for the current temperature of Gaya.
#🧠: I need to call the get_weather tool with 'Gaya' as the input to retrieve the current weather and temperature.
#🔧 Tool called: get_weather Gaya
#🤖: The current temperature in Gaya is +28°C with patchy rain nearby.
#> what is the average wheather of purnea , gaya and patna
#🧠: The user is asking for the average weather of Purnea, Gaya, and Patna. I already have data for Purnea (+26°C) and Gaya (+28°C). I need to get the weather for Patna first, then calculate the average temperature.
#🔧 Tool called: get_weather Patna
#🧠: Now I have the temperatures for all three cities: Purnea (+26°C), Gaya (+28°C), and Patna (+27°C). I can calculate the average temperature.
#🤖: Here is the current weather for the three cities:
#- **Purnea**: +26°C (Overcast)
#- **Gaya**: +28°C (Patchy rain nearby)
#- **Patna**: +27°C (Light rain)
#
#The **average temperature** across these three cities is **27°C**.
#> exit