# AI use policy and guidelines

The goal of the MalariaGEN data API is to make access, use, and interpretation of the genomic data collected by our partners as easy and intuitive as possible. Maintainers have limited time and attention to focus on reviews, which means that each review request has to be for code that you can be proud of.

Any tool that can help produce better code and understand better the existing codebase, including AI tools, can be used. The only key questions are: “Is this an improvement?” and “Why is the code better now?”.

NEVER submit an AI-generated PR if you are not able to understand and explain the changes and why they matter. Maintainers WILL close PRs without reviewing them if they feel like they are a waste of time.

## Using AI as a coding assistant

1.	Understanding and familiarising yourself with the codebase is key. No matter how good the AI code assistant, it will return useless code if you do not provide a smart and accurate enough prompt.
2.	Always check that your changes make sense. LLMs are terrible at saying no to a prompt and will lie and make false claims if they can’t do otherwise. It is particularly true if they lack key information.
3.	Each commit should be its own piece of coherent change. LLMs like to do everything at once but digestible change is easier to understand and process.
4.	Commenting your code is important, but LLMs really like to listen to themselves talking and will be very verbose. A small comment explaining why you made a choice is better than a paragraph explaining how a loop iterates through a list.

## Using AI for communication

As noted above, maintainers have a limited amount of time to spend on malariaGEN data API maintenance and do not want to waste it going through long, sloppy PR descriptions of simple issue. We strongly prefer clear and concise communication, even if it means we have to ask questions when more details are needed.

You are responsible for your own PRs and comments. Even if you use an LLM to write a PR description or comment, you are expected to read through everything and make sure that it accurately and concisely reflects your opinions, ideas and contributions. If reading your own PRs and comments is too much work for you, it is going to be the same for everyone else.
Here are some concrete guidelines for using AI as part of your communication toolbox.

1.	In general, the question that needs answering is why not what. Maintainers can see the files and lines of codes that were modified, what they will want to know is the reasoning behind the choices. Sadly, LLMs are not great at explaining their reasoning so you probably will have to chip in.
2.	In the same way, if you are responding to a comment or a review, you will need to justify your choice and explain how you made the decision.
3.	Make sure that the description of your work is accurate. Errors can happen but it is fairly obvious when an LLM claims more than it delivers.
4.	We are aware that English is not everyone’s first language. The grammar of your communications isn’t as important as the quality of your contribution. Feel free to use AI to improve your writing style but make sure that you still understand the message, that its content is conserved and that it doesn’t turn into an epic poem.
5.	Maintainers are more interested in your ideas and thoughts than in the standard answer provided by an LLM. We work with genomic data, and contributors are not expected to be experts in computer science, software engineering, genomics, entomology, …  You are allowed not to know or not to be sure and it is miles better to say so than it is to regurgitate an answer that you do not understand.
