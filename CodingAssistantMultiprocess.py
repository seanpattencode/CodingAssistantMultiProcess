import openai
import re
import multiprocessing
from multiprocessing import Queue
import time
import os

# Constants for configuration
API_KEY = ""
NUM_GPT35_TURBOS = 5
NUM_GPT4_1106_PREVIEWS = 1
NUM_GPT4_2024_08_06 = 1
NUM_CHATGPT4_LATEST = 1
MAX_TOKENS = 1000
OUTPUT_DIRECTORY = "ai_generated_code"
MAX_EXECUTION_TIME = 60  # Maximum execution time in seconds

# Initialize OpenAI client
client = openai.Client(api_key=API_KEY)

# Ensure output directory exists
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

def generate_text_gpt35(prompt: str, start_time: float) -> str:
    """Generates text using GPT-3.5 Turbo."""
    if time.time() - start_time > MAX_EXECUTION_TIME:
        return ""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating text with GPT-3.5 Turbo: {e}")
        return ""

def generate_text_gpt4(prompt: str, model_name: str, start_time: float) -> str:
    """Generates text using GPT-4 models."""
    if time.time() - start_time > MAX_EXECUTION_TIME:
        return ""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating text with {model_name}: {e}")
        return ""

def extract_python_code(text: str) -> str:
    """Extracts Python code blocks from the given text."""
    python_code_search = re.search(r'(?<=```python)[\s\S]+?(?=```)', text)
    if python_code_search:
        return python_code_search.group().strip()
    else:
        return ""

def execute_python_code(code: str, start_time: float) -> bool:
    """Executes the given Python code and returns True if successful, False otherwise."""
    if time.time() - start_time > MAX_EXECUTION_TIME:
        return False
    try:
        exec(code, globals())
        return True
    except Exception as e:
        print(f"Error executing code: {e}")
        return False

def save_code_to_file(code: str, filename: str):
    """Saves the given code to a file with the specified filename."""
    try:
        with open(filename, "w") as file:
            file.write(code)
        print(f"Code saved to: {filename}")
    except Exception as e:
        print(f"Error saving code to file: {e}")

def model_worker(model_name: str, prompt: str, result_queue: Queue, start_time: float):
    """Worker function for each model process."""
    if time.time() - start_time > MAX_EXECUTION_TIME:
        return

    if model_name == "gpt-3.5-turbo":
        output = generate_text_gpt35(prompt, start_time)
    else:
        output = generate_text_gpt4(prompt, model_name, start_time)
    
    if time.time() - start_time > MAX_EXECUTION_TIME:
        return

    code = extract_python_code(output)
    if code:
        success = execute_python_code(code, start_time)
        if success:
            # Save the code immediately upon successful execution
            filename = f"{OUTPUT_DIRECTORY}/code_{model_name}_{int(time.time())}.py"
            save_code_to_file(code, filename)
            result_queue.put((model_name, output, filename))

def evaluate_responses(responses: list, prompt: str) -> tuple:
    """Evaluates the responses and chooses the best one."""
    if not responses:
        return None, None, None
    
    # Simple evaluation: Choose the longest response for now
    # TODO: Implement more sophisticated evaluation logic
    best_response = max(responses, key=lambda x: len(x[1]))
    return best_response

def generate_with_all_models(prompt: str):
    """Generates text using all specified models, evaluates the results, and saves code to files."""
    model_names = (
        ["gpt-3.5-turbo"] * NUM_GPT35_TURBOS +
        ["gpt-4-1106-preview"] * NUM_GPT4_1106_PREVIEWS +
        ["gpt-4o-2024-08-06"] * NUM_GPT4_2024_08_06 +
        ["chatgpt-4o-latest"] * NUM_CHATGPT4_LATEST
    )

    result_queue = Queue()
    start_time = time.time()

    processes = []
    for model_name in model_names:
        process = multiprocessing.Process(target=model_worker, args=(model_name, prompt, result_queue, start_time))
        processes.append(process)
        process.start()

    for process in processes:
        process.join(timeout=max(0, MAX_EXECUTION_TIME - (time.time() - start_time)))
        if process.is_alive():
            process.terminate()
            process.join()

    successful_responses = []
    while not result_queue.empty():
        successful_responses.append(result_queue.get())

    best_model, best_response, best_filename = evaluate_responses(successful_responses, prompt)

    # Mark the best response
    if best_model:
        new_best_filename = f"{OUTPUT_DIRECTORY}/code_BEST_{best_model}.py"
        os.rename(best_filename, new_best_filename)
        print(f"Best response marked: {new_best_filename}")

    print("Successful Responses:")
    for model_name, response, filename in successful_responses:
        print(f"Model: {model_name}\nResponse saved to: {filename}\n")

    print("\nBest Response:")
    if best_response:
        print(f"Model: {best_model}\nResponse saved to: {new_best_filename}")
    else:
        print("No successful responses found.")

if __name__ == "__main__":
    original_prompt = "Write a python code for a neural network which uses an existing dataset and makes a token text generation prediction"
    modified_prompt = f"Write code to accomplish this. If there are files required, mock them up in code: '{original_prompt}'"

    generate_with_all_models(modified_prompt)