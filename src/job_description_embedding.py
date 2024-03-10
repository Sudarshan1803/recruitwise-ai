import os
import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    # Tokenize the text and remove stopwords and punctuation
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return tokens

def generate_embeddings(job_descriptions):
    embeddings = []
    for job_description in job_descriptions:
        # Preprocess the job description text
        tokens = preprocess_text(job_description)
        # Convert tokens to string and append to embeddings list
        embeddings.append(" ".join(tokens))
    return embeddings

def save_embeddings_to_file(embeddings, output_file):
    with open(output_file, "w") as file:
        for embedding in embeddings:
            file.write(embedding + "\n")

def main():
    # Sample job descriptions
    job_descriptions = [
        "We are seeking a DevOps engineer to automate and streamline our organization's software development and deployment processes. The candidate should have experience with continuous integration/continuous deployment (CI/CD) tools and cloud platforms like AWS or Azure. Responsibilities include building and maintaining CI/CD pipelines, provisioning infrastructure, and monitoring system performance."
    ]

    # Generate embeddings for job descriptions
    job_embeddings = generate_embeddings(job_descriptions)

    # Save the embeddings to a text file
    output_file = "job_description_embeddings.txt"
    save_embeddings_to_file(job_embeddings, output_file)
    print(f"Embeddings saved to {output_file}")

if __name__ == "__main__":
    main()