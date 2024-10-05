import cohere

co = cohere.Client('Ozozzmd98oFBuOXfk4LMNKszGrgjjyyenDN7cNBG')  # Use your Cohere API key

def generate_question(level):
    prompt = f"Create a one sentence {level} english question."
    response = co.generate(
        model='command-xlarge-nightly',   # This is their language model
        prompt=prompt,
        max_tokens=50)
    return response.generations[0].text.strip()

def questions_answer(question):
    prompt = f"translate '{question}' to spanish."
    response = co.generate(
        model='command-xlarge-nightly',
        prompt=prompt,
        max_tokens=50)
    return response.generations[0].text.strip()



user_input = input("Enter the difficulty level (easy, medium, hard): ")
if(user_input.lower() == "easy"):
    easy_questions = generate_question("easy")
    print("Translate the following question to Spanish:")
    print(easy_questions)
    user_answers = input("Enter translation")
    print(f"Your translation: {user_answers}. Correct translation: {questions_answer(easy_questions)}")
elif(user_input.lower() == "medium"):
    medium_questions = generate_question("medium")
    print("Translate the following question to Spanish:")
    print(medium_questions)
    user_answers = input("Enter translation: ")
    print(f"Your translation: {user_answers}. Correct translation: {questions_answer(medium_questions)}")
elif(user_input.lower() == "hard"):
    hard_questions = generate_question("hard")
    print("Translate the following question to Spanish:")
    print(hard_questions)
    user_answers = input("Enter translation")
    print(f"Your translation: {user_answers}. Correct translation: {questions_answer(hard_questions)}")
else:
    print("Invalid difficulty level. Please enter 'easy','medium', or 'hard'.")

