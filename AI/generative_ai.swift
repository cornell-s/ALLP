import CoreML

func generateQuestion(level: String) -> String {
    let prompt = "Create a one sentence \(level) english question."
    
    // Load the GPT-2 Core ML model
    let model = try! GPT2_text_generation(configuration: MLModelConfiguration())
    
    // Generate the text using the model
    let input = GPT2_text_generationInput(input_text: prompt)
    let output = try! model.prediction(input: input)
    
    return output.generated_text
}

func translateToSpanish(text: String) -> String {
    // Load the MarianMT Core ML model
    let model = try! MarianMT_translation(configuration: MLModelConfiguration())
    
    // Translate the text using the model
    let input = MarianMT_translationInput(input_text: text)
    let output = try! model.prediction(input: input)
    
    return output.translated_text
}

// Example usage
let difficulty = "easy"
let question = generateQuestion(level: difficulty)
let translation = translateToSpanish(text: question)

print("English Question: \(question)")
print("Spanish Translation: \(translation)")
