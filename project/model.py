import torch
import torch.nn as nn
import torch.nn.functional as F

class IntentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_tags):
        super(IntentClassifier, self).__init__()
        
        self.attention_weights = nn.Linear(embedding_dim, 1)

        # num_embeddings = number of words to be embedded which here is the size of vocab
        # embedding_dim = dimensions of the embedding, basically how big each vector can be for each word
        self.embedding = nn.Embedding(num_embeddings=vocab_size, 
                                      embedding_dim=embedding_dim,
                                      padding_idx=0)

        # input --> hidden layer 
        self.fc1 = nn.Linear(in_features=embedding_dim, # number of inputs must match the output size from last step
                             out_features=hidden_size) # out features is basically the amount of neurons in HL
        
        self.relu = nn.ReLU()

        # hidden ---> output
        self.fc2 = nn.Linear(in_features=hidden_size, 
                             out_features=num_tags) # number of intents

    def forward(self, x):
        embedded = self.embedding(x)
        
        #--- ATTENTION POOLING ---#
        # learns how important each word is 
        attn_weights = torch.tanh(self.attention_weights(embedded)) # restricts attn_weights between -1 and 1  
        
        # softmax used to convert into probabilities 
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # a vector of (input vector * weight of each word)
        context_vector = torch.sum(attn_weights * embedded, dim=1)
        
        # passing the pooled sentence vector through the fc1
        out = self.fc1(context_vector)
        
        # apply the ReLU activation function
        out = self.relu(out)
        
        # pass through the final output layer
        # these are the final logits
        # it's softmaxed in the entropyloss function dw
        out = self.fc2(out)
        
        return out
    
    
    
# This block will only run when you execute this file directly
if __name__ == '__main__':
    from dataset import EmotionDataset

    
    dataset = EmotionDataset(intents_file_path="Unk_infested_intents.json")
    vocab_size = len(dataset.vocabulary)
    num_tags = len(dataset.tags)

    EMBEDDING_DIM = 100
    HIDDEN_SIZE = 64

    model = IntentClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_SIZE, num_tags)

    sample_input = torch.randint(0, vocab_size, (5, 10)) 

    output = model(sample_input)

    print("Successfully created the model.")
    print("Sample input shape:", sample_input.shape)
    print("Model output shape:", output.shape)
    print("Expected output shape:", (5, num_tags))
