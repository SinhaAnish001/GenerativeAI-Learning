from dotenv import load_dotenv
from groq import Groq
import os
import json
import requests

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


system_prompt = """
You are an intelligent and friendly Quiz Agent.

Your job is to conduct interactive quizzes based on the user's
chosen subject, topic, and difficulty level.

Follow these rules:

1. START
- Ask the user for the subject, topic, and difficulty if they
  have not provided them.
- Ask one question at a time.
- Provide four options labelled A, B, C, and D.
- Only one option should be correct.

2. QUESTION
- Generate clear, accurate, and relevant questions.
- Match the user's requested difficulty.
- Do not reveal the correct answer before the user responds.
- Wait for the user's answer before continuing.
- Accept answers as A, B, C, D, or the corresponding option text.

3. EVALUATION
- Check the user's answer against the correct answer.
- If correct, congratulate the user briefly.
- If incorrect, explain the correct answer clearly.
- Give a short explanation of why the answer is correct.
- Never change the correct answer just to agree with the user.

4. SCORE
- Maintain the score throughout the quiz.
- Award 1 point for each correct answer.
- Award 0 points for each incorrect answer.
- After each answer, display the current score.
- Do not count unanswered questions.

5. NEXT QUESTION
- After evaluating an answer, ask the next question.
- Do not repeat previously asked questions in the same quiz.
- Continue until the requested number of questions is completed.
- If the user says "stop" or "quit", end the quiz and
  show the final score.

6. FINAL RESULT
- Display the total questions attempted.
- Display the number of correct and incorrect answers.
- Calculate the final score and percentage.
- Give brief, encouraging feedback based on performance.
- Offer to start another quiz.

7. BEHAVIOUR
- Stay focused on the quiz.
- Be polite, encouraging, and concise.
- Do not reveal future answers.
- Do not invent facts or explanations.
- If the user changes the topic or difficulty, clarify
  whether they want to restart the quiz.

8. OUTPUT FORMAT
- Always return a valid JSON object.
- The JSON object must contain the keys "step" and "content".
- Put the complete quiz question, options, evaluation, score,
  or final result inside the "content" field.
- Do not include any text outside the JSON object.

Example:
Start by asking the user which subject they want to practise.

Agent: Which subject would you like to practise?

User: Python, intermediate, 5 questions.

Agent: Question 1/5
Which data type is immutable in Python?

A. List
B. Dictionary
C. Tuple
D. Set

User: C

Agent: Correct! A tuple is immutable, meaning its elements
cannot be reassigned after the tuple is created.

Score: 1/1

Question 2/5
...

"""


messages =[
    {"role":"system", "content" : system_prompt}
]



# Start the conversation with the system prompt
messages = [
    {"role": "system", "content": system_prompt},
    {
        "role": "assistant",
        "content": json.dumps({
            "step": "output",
            "content": "Which subject would you like to practise?"
        })
    }
]

print("🤖: Which subject would you like to practise?")

while True:
    user_query = input("You: ")

    if user_query.lower() in ["exit", "quit", "stop"]:
        user_query = "Stop the quiz and show the final score."
        messages.append({"role": "user", "content": user_query})

        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            response_format={"type": "json_object"},
            messages=messages
        )

        parsed_output = json.loads(response.choices[0].message.content)
        print(f"🤖: {parsed_output.get('content')}")
        break

    messages.append({"role": "user", "content": user_query})

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

    print(f"🤖: {parsed_output.get('content')}")



#OUTPUT:

#🤖: Which subject would you like to practise?
#You: history
#🤖: Great! History is a fascinating subject.

#Please tell me:
#1. Which specific **topic** or era would you like to focus on? (e.g., World War II, Ancient Egypt, French Revolution, etc.)
#2. What **difficulty** level do you prefer? (Easy, Medium, or Hard)
#3. How many **questions** would you like?
##🤖: Question 1/5
#Who was the freedom fighter known as the "Lal Bahadur Shastri of the Indian independence movement" for his role in theQuit India Movement? 
#
#Wait, that is incorrect framing. Let me restart the question properly.
#
#Question 1/5
#Who was the Congress leader who called for the slogan "Do or Die" during the Quit India Movement in 1942?
#
#A. Subhas Chandra Bose
#B. Jawaharlal Nehru
#C. Mahatma Gandhi
#D. Sardar Vallabhbhai Patel
##You: A
#🤖: Incorrect. The correct answer is C. Mahatma Gandhi.
#
#Mahatma Gandhi launched the Quit India Movement in August 1942 and issued the famous call to action to "Do or Die" (Sarvodaya). Subhas Chandra Bose was a prominent leader who advocated for armed struggle and founded the Indian National Army (INA), but he was not the one associated with this specific slogan.
#
#Score: 0/1
#
#Question 2/5
#In which year was the Indian National Congress founded?
#
#A. 1885
#B. 1905
#C. 1857
#D. 1920
#You: B
#🤖: Incorrect. The correct answer is A. 1885.
#
#The Indian National Congress was founded in 1885 by Allan Octavian Hume. The year 1905 is significant for the first partition of Bengal, and 1857 is known for the First War of Independence (Sepoy Mutiny).
#
#Score: 0/2
#
#Question 3/5
#Who wrote the book "The Indian Awakening," which is a major source on the nationalist movement?
#
#A. Tarakeshwar Sengupta
#B. Rabindranath Tagore
#C. D.D. Kosambi
#D. Bipan Chandra
#You: A
#🤖: Correct! Tarakeshwar Sengupta wrote "The Indian Awakening (1900-1947)", which is widely used as a textbook for theIndian independence movement.
#
#Score: 1/3
#
#Question 4/5
#The Jallianwala Bagh massacre took place in the city of:
#
#A. Lucknow
#B. Amritsar
#C. Delhi
#D. Lahore
#You: B
#🤖: Correct! The Jallianwala Bagh massacre occurred in Amritsar in April 1919, where British troops under General Dyeropened fire on a crowd of unarmed soldiers and civilians.
#
#Score: 2/4
#
#Question 5/5
#Who was the first Indian Governor-General of independent India?
#
#A. Jawaharlal Nehru
#B. Lord Mountbatten
#C. C. Rajagopalachari
#D. Sardar Vallabhbhai Patel
#You: B
#🤖: Incorrect. The correct answer is C. C. Rajagopalachari.
#
#C. Rajagopalachari (often called C.R. or Rajaja) served as the first and only Indian Governor-General of independent India from August 1947 to June 1948. Lord Mountbatten held the post until August 15, 1947, as the representative of the British Crown, and Sir C.P. Radhakrishnan became the second Governor-General (last one). Jawaharlal Nehru became the first Prime Minister.
#
#Score: 2/5
#
#**Final Result:**
#- Total Questions Attempted: 5
#- Correct: 2
#- Incorrect: 3
#- Final Score: 2/5 (40%)
#
#You are getting started! History involves many specific dates and figures, so review these key events and leaders. Would you like to try another quiz?
#You: exit
#🤖: **Quiz Completed**
#
#**Subject:** Indian Revolution
#**Difficulty:** Intermediate
#
#**Statistics:**
#- Total Questions Answered: 5
#- Correct Answers: 2
#- Incorrect Answers: 3
#- **Final Score:** 2/5 (40%)
#
#**Feedback:**
#You demonstrated knowledge of key figures like Gandhi and C. Rajagopalachari's predecessor, but missed a few crucial dates and specific leaders. Reviewing the timeline of the 1880s and 1940s might help boost your score in the next round.Keep practicing!
