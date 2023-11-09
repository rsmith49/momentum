# momentum
Enable your LLM app to continually learn from its success and crowdsource quality improvements.

## Architecture

As shown in the following diagram, the ideal flow of this package assumes that your app can make use of an LLM (via Langchain) and a vector store that supports real-time writes (via Astra DB). The typical flow is that a "user" (can be any consumer of the application) sends a query to the model, and that query is augmented with similar examples from the model that have been shown to work well. This is accomplished by storing each model response in the vector store. These responses can be answers to a question, ReAct/Reflexion/etc. thought traces, or any other generated response. For each request that comes in, the model finds similar examples that have proven to be effective from the vector store to give the model effective context. As the number of stored responses increases, and the amount of feedback increases to determine good examples from bad, the coverage of being able to find similar few shot examples for the model will increase as well. This momentum allows your Gen AI application to self-improve online, in production.

![Screen Shot 2023-11-03 at 4 35 18 PM](https://github.com/rsmith49/momentum/assets/17658617/0eb16aeb-7db1-4e8d-b2aa-dd0e2afec02c)
