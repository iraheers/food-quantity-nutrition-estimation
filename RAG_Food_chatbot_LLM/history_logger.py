history = []

def log_interaction(user_input: str, response: str):
    history.append({"input": user_input, "response": response})

def get_history() -> list:
    return history