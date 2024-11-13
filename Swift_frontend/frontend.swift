import SwiftUI
import CoreML

struct ContentView: View {
    @State private var englishText: String = ""
    @State private var translatedText: String = "Translation will appear here"

    var body: some View {
        VStack(spacing: 20) {
            Text("English to Spanish Translator")
                .font(.title)
                .fontWeight(.bold)

            TextField("Enter English text", text: $englishText)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .padding()

            Button(action: translateText) {
                Text("Translate")
                    .foregroundColor(.white)
                    .padding()
                    .background(Color.blue)
                    .cornerRadius(10)
            }

            Text("Spanish Translation:")
                .font(.headline)

            Text(translatedText)
                .font(.body)
                .padding()
                .foregroundColor(.green)
        }
        .padding()
    }

    func translateText() {
        guard let model = try? EnglishToSpanishTranslationModel(configuration: MLModelConfiguration()) else {
            translatedText = "Model failed to load."
            return
        }

        // Prepare the input for the model
        if let input = try? MLMultiArray(shape: [1], dataType: .int32) {
            // Assuming a simple input text, tokenize it according to your model's needs
            // Here we would need an English text tokenizer similar to the one in Python
            // For now, this is placeholder code to demonstrate
            
            // Perform the translation
            do {
                let prediction = try model.prediction(input: input)
                translatedText = prediction.output
            } catch {
                translatedText = "Failed to translate."
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
