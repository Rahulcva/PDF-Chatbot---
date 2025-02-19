# PDF Chatbot 📚

This is a simple chatbot application that allows you to ask questions based on the contents of a PDF file. The app reads the PDF and returns answers based on the information contained within it.

## Requirements

Before running the application, make sure you have the necessary dependencies installed. You can install them by running:

Step-by-Step Instructions:

    Install Ollama:
    Before running the app, you need to install Ollama. Run the following command to install it:

curl https://ollama.ai/install.sh | sh

Run Ollama Orca Mini Model:
Once Ollama is installed, you can run the Orca Mini model with the following command:

ollama run orca-mini

Install Dependencies:
Install the required Python libraries by running:

pip install -r requirements.txt

How to Use

    app.py: This version answers questions once based on the PDF content.
    app1.py: This version continuously loops and answers multiple questions. It's great if you want the chatbot to interact with you over time.
