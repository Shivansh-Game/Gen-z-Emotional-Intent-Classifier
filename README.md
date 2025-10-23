An emotional intent classifier trained on specially made data which helps it gain around mid to high 90% accuracy in classifying the emotional intent of the user.

NOTE: The emotional intents patterns are specifically made to be similar for some emotions to encourage the AI to find what the difference between a loving and a caring intent is

About the model
- Trained using word embeddings
- Attention pooling was used as it is immensely more effective for this use case (since yk humans usually don't infer emotions by giving every word in a sentence the same importance)
- A placeholder token was made "<UNK>" and used in the training data as well to teach the model that "Yo want to go get <UNK>" is usually someone asking to hang out. This type of pattern is sprinkled into the training data to encourage the AI to learn properly instead of just attributing an intent to a single word no matter the context
- Padding was implemented so batch sizes could work properly 
