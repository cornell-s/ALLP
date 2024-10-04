import cohere

co = cohere.Client('Ozozzmd98oFBuOXfk4LMNKszGrgjjyyenDN7cNBG')  # Use your Cohere API key

def generate_question(level):
    prompt = f"Create a {level} in english and ask what the translation would be in spanish."
    response = co.generate(
        model='command-xlarge-nightly',   # This is their language model
        prompt=prompt,
        max_tokens=50)
    return response.generations[0].text.strip()

beginner_question = generate_question('beginner')
print(beginner_question)
