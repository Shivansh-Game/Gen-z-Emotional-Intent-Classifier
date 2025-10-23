import torch
import torch.utils.data as data
import json
from pps import tokenize, preprocess
from torch.nn.utils.rnn import pad_sequence

class EmotionDataset(data.Dataset):

    def __init__(self, intents_file_path):
        with open(intents_file_path, 'r') as f:
            intents = json.load(f)

        all_words = []
        tags = []
        xy = []

        for intent in intents['intents']:
            tag = intent['tag']
            if tag not in tags:
                tags.append(tag)
            for pattern in intent['patterns']:
                words = tokenize(pattern)
                all_words.extend(words)
                xy.append((words, tag))

        ignore_words = ['?', '.', '!']
        all_words = [preprocess(w) for w in all_words if w not in ignore_words]

        
        # Vocab and a dict that maps each word to a unique number
        self.vocabulary = sorted(list(set(all_words)))
        self.vocabulary.insert(0, "<PAD>") # Padding token at idx 0 because that's the padding idx we gave the model
        self.vocabulary.insert(1, "<UNK>") # Unknown (place holder for words not in vocab)
        self.word_to_idx = {word: i for i, word in enumerate(self.vocabulary)}
        
        self.tags = sorted(list(set(tags)))
        
        X_train = []
        y_train = []

        for (pattern_sentence, tag) in xy:
            # Turn every word into it's corresponding number
            sentence_indices = [self.word_to_idx[preprocess(w)] for w in pattern_sentence if preprocess(w) in self.word_to_idx]
            X_train.append(torch.tensor(sentence_indices, dtype=torch.long)) # torch.long = long integer 
            
            label = self.tags.index(tag)
            y_train.append(label)

        self.X_data = X_train
        self.y_data = torch.tensor(y_train, dtype=torch.long)
        self.n_samples = len(self.X_data)

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples
    


def collate_fn(batch, pad_idx=0):
    # Separates the sentences and labels from the batch
    sentences, labels = zip(*batch)

    # Use pad_sequence to pad all sentences to the same length
    padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=pad_idx)
    
    # Stack labels into a single tensor
    labels = torch.stack(labels, 0)
    
    return padded_sentences, labels
    
    
# This block will only run when you execute this file directly
if __name__ == '__main__':
    # Create an instance of the dataset
    dataset = EmotionDataset(intents_file_path='Unk_infested_intents.json')
    
    # Get the first sample
    first_sample_X, first_sample_y = dataset[0]
    
    print("Successfully loaded the dataset.")
    print(f"Number of samples: {len(dataset)}")
    print("\n--- First Sample ---")
    print("X (Bag of Words):", first_sample_X)
    print("y (Label):", first_sample_y)
    print("Shape of X:", first_sample_X.shape)