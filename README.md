# STEER: Systematic and Tuneable Evaluation of Economic Rationality in LLMs

A multiple-choice question-answering (MCQA) benchmark containing test questions on elements of economic rationality. 

These elements are organized into a four order taxonomy: ![](taxonomy.jpg)

## Schema
Each element contains four dataframes: 
- questions (columns: question_id, question_text, explanation) 
- options (columns: question_id, option_id, option_text)
- answers (columns: question_id, option_id, correct)
- questions_metadata (columns: question_id, type, domain, difficulty_level, tags)

Each test question is associated with a unique question_id where the first number indicates the position in the dataframe and the second denotes its order as a subquestion. E.g., "1_0" in the questions dataframe means the question_text is the first subquestion for the second question.

There is an element that does not have an answers in the answers dataframe. In this case answers scored by being consistent:
- independence: the option selected in the first part should be the same in the second