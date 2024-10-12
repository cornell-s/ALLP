# Required Libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer, MarianMTModel, MarianTokenizer
import torch

# Load the GPT-2 model and tokenizer for text generation
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load MarianMT model and tokenizer for translation
marian_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-es')
marian_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-es')

def generate_question(level):
    # Create a prompt based on the difficulty level
    prompt = f"Create a one sentence {level} english question."
    
    # Tokenize the prompt
    inputs = gpt2_tokenizer(prompt, return_tensors="pt")
    
    # Generate a response with GPT-2
    outputs = gpt2_model.generate(inputs.input_ids, max_length=50)
    
    # Decode and return the generated question
    return gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)

def questions_answer(question):
    # Translate the English question into Spanish using MarianMT
    translated = marian_model.generate(**marian_tokenizer(question, return_tensors="pt"))
    return marian_tokenizer.decode(translated[0], skip_special_tokens=True)

# Example usage:
user_input = input("Enter the difficulty level (easy, medium, hard): ").lower()

if user_input in ["easy", "medium", "hard"]:
    question = generate_question(user_input)
    print("Translate the following question to Spanish:")
    print(question)
    
    user_answers = input("Enter translation: ")
    correct_translation = questions_answer(question)
    print(f"Your translation: {user_answers}. Correct translation: {correct_translation}")
else:
    print("Invalid difficulty level. Please enter 'easy', 'medium', or 'hard'.")

gpt2_model.save_pretrained('./gpt2_model')
gpt2_tokenizer.save_pretrained('./gpt2_model')

marian_model.save_pretrained('./marian_model')
marian_tokenizer.save_pretrained('./marian_model')

import coremltools as ct
import torch

# Load the GPT-2 model
model = GPT2LMHeadModel.from_pretrained('./gpt2_model')
model.eval()

# Convert the GPT-2 model to Core ML
traced_model = torch.jit.trace(model, torch.ones(1, 1, dtype=torch.long))
mlmodel = ct.convert(traced_model, inputs=[ct.TensorType(shape=(1, 1))])
mlmodel.save("GPT2_text_generation.mlmodel")

# Repeat the process for MarianMT (translation) model
model = MarianMTModel.from_pretrained('./marian_model')
model.eval()

traced_model = torch.jit.trace(model, torch.ones(1, 1, dtype=torch.long))
mlmodel = ct.convert(traced_model, inputs=[ct.TensorType(shape=(1, 1))])
mlmodel.save("MarianMT_translation.mlmodel")
