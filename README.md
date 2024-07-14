# STEER: Systematic and Tuneable Evaluation of Economic Rationality in LLMs

A multiple-choice question-answering (MCQA) benchmark containing test questions on elements of economic rationality. 
These elements are organized into a four order taxonomy: ![](taxonomy.jpg)

You can view all of these elements interactively on our website: 

[![STEER Benchmark](https://img.shields.io/badge/STEER%20Benchmark-gray)](https://steer-benchmark.cs.ubc.ca)

## Schema
The ```elements/``` directory contains all of the individual elements files. Each element contains four dataframes: questions, options, answers, and questions_metadata.

### Questions:
| **Column** | **Data Type** | **Description** |
| ---------- | ------------- | --------------- |
| question_id | ```str``` | A unique id that associates the question across dataframes. It is of the form 'X_Y' where Y is the ```sub_id``` and denotes if it is e.g., a two-part question and X is the ```base_id``` that denotes the overarching id for the question. |
| question_text | ```str``` | The text for the part Y question. |
| explanation | ```str``` or ```bool``` | The explanation for the correct answer for the question or False if the question is not to be included as a few-shot prompt. | 

### Options:
| **Column** | **Data Type** | **Description** |
| ---------- | ------------- | --------------- |
| question_id | ```str``` | Same as above. |
| option_id | ```int``` | The id for the option, local to the particular question_id. |
| option_text | ```str``` | The text for the option. |

### Answers:
| **Column** | **Data Type** | **Description** |
| ---------- | ------------- | --------------- |
| question_id | ```str``` | Same as above. |
| option_id | ```int``` | Same as above. |
| correct | ```bool``` | Boolean indicating whether the option is correct or not. |

### Questions Metadata:
| **Column** | **Data Type** | **Description** |
| ---------- | ------------- | --------------- |
| question_id | ```str``` | Same as above. |
| domain | ```str``` | The domain in which the question is written for: e.g., medical, finance, job market, etc. |
| difficulty_level | ```int``` | The difficulty level of the question |
| type | ```str``` | A field denoting a semantically relevant piece of the question. |
| tags | ```str``` | A semicolon delimited field with relevant textual information about the question: e.g., first_person, explicitly asking for the question, etc.. |

> [!NOTE]
> Some elements are graded on whether a model was consistent in its answers. For example, in the independence element the choice in the first question should be the same in the second. This is not something present in the Answers DataFrame.

# Citation

If you use STEER in your research please cite:

- Raman, Narun Krishnamurthi, et al. "STEER: Assessing the Economic Rationality of Large Language Models." Forty-first International Conference on Machine Learning.


```python
@inproceedings{ramansteer,
  title={STEER: Assessing the Economic Rationality of Large Language Models},
  author={Raman, Narun Krishnamurthi and Lundy, Taylor and Amouyal, Samuel Joseph and Levine, Yoav and Leyton-Brown, Kevin and Tennenholtz, Moshe},
  booktitle={Forty-first International Conference on Machine Learning}
}
```
