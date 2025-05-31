import uuid

from langgraph_memory.server import serve

thread_id = str(uuid.uuid4())

# test fixture to simulate a chat session
# calling the serve function with a unique thread_id

while True:
    input_message = input("Enter message: ")
    if input_message.lower() == "exit":
        print("Exiting...")
        break
    response = serve(thread_id, input_message)
    print(f"Response: {response}")
