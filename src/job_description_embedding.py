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
        "We are seeking a talented software engineer to join our team. The ideal candidate should have a strong background in software development, including experience with languages such as Python, Java, or C++. Responsibilities include designing, developing, and testing software applications, as well as collaborating with cross-functional teams to deliver high-quality solutions.",
        "We are looking for a data scientist with expertise in machine learning and statistical analysis. The candidate should have experience working with large datasets and using tools like Python, R, or MATLAB. Responsibilities include developing predictive models, performing data analysis, and communicating insights to stakeholders.",
        "We are hiring a network administrator to manage our organization's computer networks. The ideal candidate should have knowledge of network protocols, security practices, and troubleshooting techniques. Responsibilities include configuring and maintaining network infrastructure, monitoring network performance, and ensuring data security.",
        "We are seeking a cybersecurity analyst to protect our organization's information systems from cyber threats. The candidate should have experience with security tools and techniques, such as intrusion detection systems, firewalls, and penetration testing. Responsibilities include monitoring for security breaches, conducting risk assessments, and implementing security measures.",
        "We are looking for a skilled web developer to build and maintain websites and web applications. The candidate should have proficiency in HTML, CSS, JavaScript, and other web development technologies. Responsibilities include designing user interfaces, coding responsive web pages, and troubleshooting website issues.",
        "We are hiring an IT project manager to oversee our organization's IT projects from initiation to completion. The ideal candidate should have experience in project management methodologies and tools, as well as strong leadership and communication skills. Responsibilities include planning project timelines, allocating resources, and managing project budgets.",
        "We are seeking a database administrator to manage our organization's databases and ensure their security and integrity. The candidate should have expertise in database management systems like MySQL, Oracle, or SQL Server. Responsibilities include installing and configuring database software, optimizing database performance, and performing data backups.",
        "We are looking for a systems analyst to analyze our organization's information systems and processes and recommend improvements. The ideal candidate should have a strong understanding of business requirements and technical solutions. Responsibilities include gathering and documenting user requirements, designing system workflows, and coordinating system implementations.",
        "We are hiring an IT support specialist to provide technical assistance to our organization's users. The candidate should have experience troubleshooting hardware and software issues, as well as excellent customer service skills. Responsibilities include diagnosing and resolving IT problems, installing and configuring software, and training users on IT systems.",
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