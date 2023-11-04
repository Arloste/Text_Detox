# Text Detoxification with LSTM
---
## Personal Information
Name: Danil Meshcherekov

Email: d.meshcherekov@innopolis.university

Group number: B21-DS-02
## Description
In this assignment I attempt to use an LSTM neural network to solve the text de-toxification.

Text detoxification problem can be defined as transforming text with high level of toxicity - that is, containing rude, disrespectful, or inapproproate words or phrases, that is likely to make another people leave a conversation - into the text with the same meaning, but with non-toxic style by removing or replacing the toxic words.

Firstly, I scan the vocabulary of the provided dataset of toxic comments and their non-toxic equivalens (you can see more about the dataset in the 'how to use' section) and tokenize each sentence.

Then, I train the neural network on these tokenized sentences.

Finally, to make a prediction on a sentence I tokenize with the obtained in the first step dictionary, make a prediction with the trained neural network, and then de-tokenize it back into a readable sentence.

Also, I have considered several other techniques to solve this problem:
- Using a ready solution made by Skoltech. This, however, seemed like cheating; besides, it is quite complicated to use and I do not have enought computational resources to train the model.
- Transformer-based neural network. I discarded this idea because I, too, did not have enough resources to fine-tune even a trained neural network.
- Detecting toxic words in the text and replacing them with non-toxic alternatives. However, it is unclear how to pinpoint the exact word toxic word and what alternatives to use in each context; moreover, using different words may require changing other non-toxic words in the sentence, which is such a model is not capable of.

In this situation, using an LSTM seemed to be reasonable in terms of both complexity ('translating' one sentence to another does not require pinpointing and replacing the exact words) and computational resources (LSTM is not as demanding as a transformer architecture).

However, the technical limitations became a bottleneck anyway - it was not possible to load the whole dataset into the memory, so I had to use only a subset thereof; a simple NN with a single LSTM layer and one fully connected layer still was taking a lot of time to train; and I ran out of GPU units very fast. As a result, the existing model is very limited and mostly outputs garbage ):

## How to Use the Program
### Training the model
1. Put a `raw.tsv` dataset into the `/data/` folder. It must contain a `reference` column (with toxic sentences) and a `translation` column (with their non-toxic equivalents). Optionally, there can be `ref_tox` and `trn_tox` which show the numerical value of toxicity in the reference and translation sentences; I use them because in the original Scoltech dataset some of the sentences are placed incorrectly.
2. Open the terminal in the root folder of the project.
3. Type `sh script.sh train <model_name>` to train the model. You can specify the model name to create and use several different models. Ideally, the output should be as follows:
![image](https://github.com/Arloste/Text_Detox/assets/88305350/a8283270-d330-42ab-a725-6996741f6fcb)

### Making predictions in a command line
It is possible to make a prediction on a single sentence from the terminal.
1. Open the terminal in the root folder of the project
2. Type `sh script.sh predict <model_name> terminal <prompt>`. The program should write the result in the terminal. Here is an example:
![image](https://github.com/Arloste/Text_Detox/assets/88305350/e6bf6fec-e07c-4829-952a-c660895107a9)

### Making batch predictions from txt file
1. Put a `references.txt` file into the `/data/` folder. This file should have one sentence on each line.
2. Open the terminal in the root folder of the project
3. Type `sh script.sh predict <model_name> txt`. The program will read the txt file line by line and make separate predictions for each line. Here is an example, although the output is not good:
![image](https://github.com/Arloste/Text_Detox/assets/88305350/713c1bb8-6b7c-4eaf-a2df-79e1dc6e58a2)

### Making batch predictions from pandas dataframe
1. Put a `references.tsv` file into the `/data/` folder. The program will read the sentences in this file in the `references` column.
2. Open the terminal in the root folder of the project
3. Type `sh script.sh predict <model_name> dataframe`. The output is similar to the previous example.
