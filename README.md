
# DocuGenie: Conversational Document Assistant

[Click here to see a video about this work](https://youtu.be/4XwLu-2z2dY)

Welcome to DocuGenie, your intelligent assistant for interactive conversations with PDF documents. This project leverages advanced AI models to make document navigation and information retrieval more intuitive and efficient.



## Introduction
- **DocuGenie** is designed to transform the way you interact with your PDF documents. Whether you're conducting research, extracting information, or simply navigating through large documents, DocuGenie provides an interactive chat-based interface that makes it easy to find the information you need quickly. To make the model accessible to users, **Streamlit** has been employed to develop a simple and interactive user interface.


-  **Streamlit**
Streamlit allows for the rapid development of web applications for machine learning and data science projects. With this interface, users can enter any tweet or comment, and the application will instantly provide a sentiment prediction. This integration of Streamlit ensures that the sentiment analysis process is not only efficient but also user-friendly, enabling individuals without a technical background to easily interact with and benefit from the machine learning model.
## Documentation


This README file serves as the main documentation for the DocuGenie project. For detailed API documentation, refer to the API Reference section below.
## API Reference




| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `groq_api_key` | `string` | **Required**. Your API key |
`huggingface_api_key` | `string`| **Required**. Your API key|
`mistral_api_key`| `string`| **Optional**. Your API key|
 






## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`GROQ_API_KEY`

`HUGGINGFACE_API_KEY`

`MISTRALAI_API_KEY` --> **Optional**


## Screenshots
- Home Page
![Home page](https://github.com/user-attachments/assets/d8fa1b4f-e537-44e0-aeb0-b1a9bb7c4c3d)


- Upload PDF
![upload](https://github.com/user-attachments/assets/dfcc4d06-3638-4b2e-9e7e-b2fd025398af)


- Chat Interface
![convo](https://github.com/user-attachments/assets/ffba443c-d067-4b64-bddd-aca26d6c59d3)

## Features

- **Interactive Chat Interface:** Engage in conversations with your PDF documents.
- **Session Management:** Create and switch between different sessions with unique conversation histories.
- **PDF Upload and Processing:** Upload multiple PDFs and have them processed and ready for querying.
- **History-Aware Querying:** Retrieve context-aware answers based on the document content and conversation history.
- **Customizable AI Model:** Integrate various AI models from Groq or HuggingFace to power the conversational assistant.


## Experimentation 
- **Customizing Models**
You can replace the default models with other models from Groq or HuggingFace by modifying the initialization code. Experiment with different model configurations to achieve optimal results based on your document types and use cases.

- **Session Customization**
Modify the session handling logic to suit different use cases, such as persistent session storage, user authentication, or multi-user support.

- **Extending the Chat Interface**
Customize the chat interface with additional features like rich text formatting, inline file previews, or even voice input and output.
## Conclusion
**DocuGenie** is a versatile and powerful tool designed to make working with PDFs more interactive and efficient. By using cutting-edge AI models, it transforms the way you access and interact with information within documents.

**I hope you find this project useful and look forward to your contributions and feedback. Happy document chatting!**
