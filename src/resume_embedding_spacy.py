import os
import spacy
import numpy as np

def generate_embeddings(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Load English model with pre-trained word vectors
    nlp = spacy.load("en_core_web_sm")

    # Read preprocessed resume files
    for filename in os.listdir(input_directory):
        input_file_path = os.path.join(input_directory, filename)
        with open(input_file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            doc = nlp(text)

            # Extract word vectors and save to output file
            output_file_path = os.path.join(output_directory, filename.replace('.txt', '.vec'))
            with open(output_file_path, 'w', encoding='utf-8') as out_file:
                for token in doc:
                    if token.has_vector:
                        embedding = token.vector
                        embedding_str = ' '.join(str(value) for value in embedding)
                        out_file.write(f'{embedding_str}\n')

    print('Embedding generation complete.')

def main():
    # Define input and output directories
    input_directory = 'preprocessed_resumes'
    output_directory = 'resume_embeddings_spacy'

    # Generate embeddings for preprocessed resumes
    generate_embeddings(input_directory, output_directory)

if __name__ == "__main__":
    main()