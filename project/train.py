import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import EmotionDataset, collate_fn
from model import IntentClassifier

intent_path = "Unk_infested_intents.json"
with open(intent_path, 'r') as f:
    intents = json.load(f)

batch_size = 32
embedding_dim = 128
hidden_size = 128
learning_rate = 0.001
num_epochs = 40

dataset = EmotionDataset(intents_file_path=intent_path)
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0,
                          collate_fn=collate_fn)

vocab_size = len(dataset.vocabulary)
num_tags = len(dataset.tags)
# we need this info from the class so get it here 


device = "cuda" if torch.cuda.is_available() else "cpu" # Me no gpu so useless line for me but your mileage may vary 
model = IntentClassifier(vocab_size, embedding_dim, hidden_size, num_tags).to(device)

# loss func
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


print("Starting training...")
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        # Move data to the configured device again useless for me but y'all's mileage may differ 
        words = words.to(device)
        labels = labels.to(device)
        
        # FP, get logits
        outputs = model(words)
        
        # Calculate the loss
        loss = criterion(outputs, labels)
        
        # Backpropagation
        optimizer.zero_grad() # Clear previously gradients
        loss.backward()      # Calculate gradients again for this batch
        
        # Update the weights
        optimizer.step()
        
    
    print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'Final loss: {loss.item():.4f}')
print("Training complete.")

# --- Save the Trained Model --- # 
data_to_save = {
    "model_state": model.state_dict(),
    "vocab_size": vocab_size,
    "embedding_dim": embedding_dim,
    "hidden_size": hidden_size,
    "num_tags": num_tags,
    "vocabulary": dataset.vocabulary,
    "tags": dataset.tags
}

FILE = "trained_data.pth"
torch.save(data_to_save, FILE)

print(f'Training complete. Model saved to {FILE}')