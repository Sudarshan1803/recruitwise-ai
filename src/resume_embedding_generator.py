import os
import openai

# Set your OpenAI API key
openai.api_key = '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'

def generate_embeddings(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate over preprocessed resume files
    for filename in os.listdir(input_directory):
        input_file_path = os.path.join(input_directory, filename)
        output_file_path = os.path.join(output_directory, filename)

        # Read preprocessed text from the input file
        with open(input_file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # Generate semantic embeddings for the preprocessed text
        # embeddings = openai.embeddings.create(model="text-embedding-ada-002", input=text)
        embeddings = openai.embeddings.create(model="text-embedding-3-small", input=text)

        # Save the embeddings to the output file
        with open(output_file_path, 'w', encoding='utf-8') as file:
            for embedding in embeddings:
                file.write(str(embedding) + '\n')

        print(f'Generated embeddings for: {filename}')

    print('Embedding generation complete.')

def main():
    # Define input and output directories
    input_directory = 'preprocessed_resumes'
    output_directory = 'resume_embeddings'

    # Generate embeddings for preprocessed resumes
    generate_embeddings(input_directory, output_directory)

if __name__ == "__main__":
    main()