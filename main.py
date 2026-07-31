from openai import OpenAI
import requests
import os
import json

# ----------------------------------------
# STEP 1: Create OpenAI Client
# ----------------------------------------
# Reads your OpenAI API key from environment variables
# Example:
# export OPENAI_API_KEY="your-key"
# ----------------------------------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------------------
# STEP 2: Create a Tool
# ----------------------------------------
# This tool calls the GitHub API and fetches
# information about a GitHub user.
#
# IMPORTANT:
# The LLM DOES NOT execute this function.
# Python executes this function.
# ----------------------------------------

def get_github_profile(username):

    print(f"\nFetching GitHub profile for {username}...\n")

    response = requests.get(
        f"https://api.github.com/users/{username}"
    )

#https://api.github.com/users/torvalds

    data = response.json()

    return {
        "name": data.get("name"),
        "bio": data.get("bio"),
        "followers": data.get("followers"),
        "following": data.get("following"),
        "public_repos": data.get("public_repos"),
        "location": data.get("location"),
        "html_url": data.get("html_url")
    }


# ----------------------------------------
# STEP 3: Define Tool Metadata
# ----------------------------------------
# This information is sent to the LLM.
#
# We are NOT sending the Python code.
#
# We only send:
# - Tool name
# - Description
# - Parameters
#
# The LLM uses this information to decide
# when the tool should be called.
# ----------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_github_profile",
            "description": "Get information about a GitHub user profile",
            "parameters": {
                "type": "object",
                "properties": {
                    "username": {
                        "type": "string",
                        "description": "GitHub username"
                    }
                },
                "required": ["username"]
            }
        }
    }
]

# ----------------------------------------
# STEP 4: User Question
# ----------------------------------------

user_question = input("Ask me anything: ")

print(user_question)

# ----------------------------------------
# STEP 5: Send Question + Tool Definitions
# to OpenAI
# ----------------------------------------

# Sent to the LLM:
#
# tools = [
#   {
#     "type": "function",
#     "function": {
#       "name": "get_github_profile",
#       "description": "Fetches a GitHub user's profile",
#       "parameters": {
#         "username": "string"
#       }
#     }
#   }
# ]
#
# NOTE:
# We are NOT sending the Python function itself.
# We are only sending its metadata (name, description, and parameters).

response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {
            "role": "user",
            "content": user_question
        }
    ],
    tools=tools
)



# ----------------------------------------
# STEP 6: Check if LLM wants to use a tool
# ----------------------------------------


# The LLM has two choices:
#
# 1. It already knows the answer
#    -> Returns the answer in "content"
#
# 2. It needs external information
#    -> Returns a "tool_calls" request
#       asking our application to execute a tool.
#
# In this example, the LLM chose option 2.


# API Response
# {
#   "choices": [
#     {
#       "message": {
#         "content": null,
#         "tool_calls": [
#           {
#             "function": {
#               "name": "get_github_profile",
#               "arguments": "{\"username\":\"torvalds\"}"
#             }
#           }
#         ]
#       }
#     }
#   ]
# }

tool_call = response.choices[0].message.tool_calls[0]

tool_name = tool_call.function.name

print("\nLLM Selected Tool:")
print(tool_name)

# ----------------------------------------
# STEP 7: Extract Tool Arguments
# ----------------------------------------
# Example:
#
# {
#   "username": "torvalds"
# }
# ----------------------------------------

arguments = json.loads(
    tool_call.function.arguments
)

username = arguments["username"]

print("\nUsername Received:")
print(username)

# ----------------------------------------
# STEP 8: Execute Tool
# ----------------------------------------
# THIS IS THE MOST IMPORTANT STEP
#
# The LLM selected the tool.
#
# Python executes the tool.
#
# The LLM never directly calls GitHub.
# ----------------------------------------

profile = get_github_profile(username)

print("\nGitHub Profile Retrieved:\n")
print(json.dumps(profile, indent=4))

# ----------------------------------------
# STEP 9: Send Tool Result Back To LLM
# ----------------------------------------
# Now the LLM can use the tool result
# to generate a final answer.
# ----------------------------------------

final_response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {
            "role": "user",
            "content": user_question
        },
        response.choices[0].message,
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(profile)
        }
    ]
)

# ----------------------------------------
# STEP 10: Final Answer
# ----------------------------------------

print("\nFinal Answer:\n")

print(
    final_response.choices[0].message.content
)