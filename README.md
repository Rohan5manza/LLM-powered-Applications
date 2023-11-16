# LLM-powered-Applications
My projects that implement deployment of various Large language models-powered web applications 
What this project is about:
1. A Streamlit-powered web application where we can upload text, multiple pdf files, and csv data files of excel tabular format, and ask any questions about them and get insightful answers. Advantages are many, such as summarization of a lengthy pdf user doesnt want to read, or getting data insights from csv files.
2. A LLM cannot understand data in the format of csv docs, or big chunks of pdf files, so we embed our data and store them in vector database like FAISS or Chroma or Pinecone, and then send those embeddings to LLM for output. Langchain does this for us. We make use of APIs to the LLMs.
3. Use of OpenAI API is limited as it has billing limitations, so we can just chooose from a plethora of HuggingFace LLM model card APIs like Flan,etc.
4. My app includes the choice for user to choose from a variety of different LLMs given as options, we can choose from different AI models to compare performance and the worth of outputs.

Things this project covered:
1. Working with Langchain for building LLM-powered applications
2. Working with vector databases for storing text embeddings
3. Use of Langchain agent, tools, etc. (PS: langchain agent is deprecated for csv tasks, although the concept of agent is really good and clever, use csv loader function for future projects)
4. Streamlit for frontend ( and deploying on Streamlit cloud if needed)

Future project directions:
1. Youtube video transcripter
2. web scraping
3. 
