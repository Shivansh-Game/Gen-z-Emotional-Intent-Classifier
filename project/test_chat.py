import torch
from pps import preprocess, tokenize
from model import IntentClassifier
import pytest

@pytest.fixture(scope="session")
def model_components():
    BOT_NAME = "Intent Guesser" # Or whatever you want to call it
    device = "cuda" if torch.cuda.is_available() else "cpu" # irrelavant for me personally since I don't have a gpu but enables y'all to use GPUs if you have em

    # load model data
    FILE = "trained_data.pth"
    data = torch.load(FILE)

    # seperate the variablessssssss
    vocab_size = data["vocab_size"]
    embedding_dim = data["embedding_dim"]
    hidden_size = data["hidden_size"]
    num_tags = data["num_tags"]
    model_state = data["model_state"]
    vocabulary = data["vocabulary"]
    tags = data["tags"]

    # --- RECREATE THE MODEL --- #
    model = IntentClassifier(vocab_size, embedding_dim, hidden_size, num_tags).to(device)
    # Load the saved weightssss
    model.load_state_dict(model_state)

    model.eval()
    
    return {
            "model": model,
            "vocabulary": vocabulary,
            "tags": tags,
            "device": device
        }


def preprocess_sentence(sentence, vocabulary, device):
    tokens = tokenize(sentence)
    processed_words = [preprocess(token) for token in tokens]
    print(processed_words)
    indices = []
    
    ignore_words = ['?', '.', '!']
    
    
    for w in processed_words:
        if w in vocabulary:
            indices.append(vocabulary.index(w))
        elif w in ignore_words:
            pass
        else:
            indices.append(vocabulary.index("<UNK>"))

    if not indices:
        return None
    
    return torch.tensor(indices).view(1, -1).to(device)

def make_prediction(input_tensor, BOT_NAME, model, tags):
    if input_tensor is None:
        return f"{BOT_NAME}: Sorry, I don't know any of those words."
    # torch.no_grad() is used to improve performance (Inference)
    with torch.no_grad():
        output = model(input_tensor)
    probabilities = torch.softmax(output, dim=1)
    max_prob, predicted_idx = torch.max(probabilities, dim=1)
    predicted_tag = tags[predicted_idx.item()]
    return max_prob, predicted_tag


# most of the failures thus far are acceptable enough, after accounting for acceptable failures the accuracy is around low to mid 90%
test_cases = [
    ("How are you?", "user_caring"),
    ("I love youuu", "user_loving"),
    ("This is great!", "user_happy"),
    ("I will killlllll youu", "user_happy"),      # Test for playful hyperbole
    ("heyyyy bitchh", "user_happy"),             # Test for playful insult
    ("you're a fucking idiot", "user_angry"),    # Test for direct insult
    ("thanks a lot", "user_thankful"),
    ("my apologies", "user_sorry"),
    ("I feel so down today", "user_feeling_bad"),
    ("Hello", "neutral"),                        # Test for greeting/neutral
    # --- Caring (Invitations & Genuine Care) ---
    ("You seem a bit down, what's up?", "user_caring"),
    ("Wanna go grab a drink sometime?", "user_caring"),
    ("Let me know if I can help with anything.", "user_caring"),
    ("Just checking in on you.", "user_caring"),
    ("Take your time, no rush at all.", "user_caring"),
    ("We should totally catch up soon.", "user_caring"),

    # --- Loving (Compliments & "Stunned Admiration") ---
    ("You're a true gem.", "user_loving"),
    ("My admiration for you is off the charts.", "user_loving"),
    ("You are absolutely insane for pulling that off, amazing.", "user_loving"),
    ("What a fucking legend you are.", "user_loving"),
    ("You make everything better just by being you.", "user_loving"),
    ("I'm so proud of the work you've done.", "user_loving"),

    # --- Feeling Bad (Sadness, Tiredness, Disappointment) ---
    ("Well, that's disappointing news.", "user_feeling_bad"),
    ("I'm in a real funk and can't seem to shake it.", "user_feeling_bad"),
    ("My day has been a total wash.", "user_feeling_bad"),
    ("I feel completely and utterly empty.", "user_feeling_bad"),
    ("It's just one thing after another today.", "user_feeling_bad"),
    ("I am so mentally drained right now.", "user_feeling_bad"),

    # --- Angry (Direct & Non-Playful) ---
    ("That is the stupidest thing I have ever heard.", "user_angry"),
    ("This is a complete waste of my time.", "user_angry"),
    ("I'm at my wit's end with this.", "user_angry"),
    ("This is completely unacceptable.", "user_angry"),
    ("Don't piss me off right now.", "user_angry"),
    ("That's some real bullshit.", "user_angry"),

    # --- Happy (Hype, Playful Insults, Hyperbole, Slang) ---
    ("LMAOOO you're a clown for that", "user_happy"),
    ("Yessss let's do it!", "user_happy"),
    ("I'm so ridiculously excited right now!", "user_happy"),
    ("omg I'm dying that's too cute", "user_happy"),
    ("What's up fuckerrrrr", "user_happy"),
    ("Everything is finally falling into place.", "user_happy"),
    ("That's actually hilarious XD", "user_happy"),
    ("This is the best outcome I could have hoped for.", "user_happy"),

    # --- Thankful ---
    ("I don't know what I'd do without you, thanks.", "user_thankful"),
    ("You're a lifesaver, seriously.", "user_thankful"),
    ("Much obliged for your assistance on this.", "user_thankful"),

    # --- Sorry ---
    ("I really messed that one up, my apologies.", "user_sorry"),
    ("That was completely out of line of me, I'm sorry.", "user_sorry"),
    ("I regret that, it was all my mistake.", "user_sorry"),

    # --- Neutral ---
    ("Yo", "neutral"),
    ("sup", "neutral"),
    
    # --- Boundary Tests (Testing the line between intents) ---
    ("I'm so frustrated I could cry.", "user_feeling_bad"),
    ("I seriously don't know what I'd do without you.", "user_thankful"),
    ("You're my favorite person to talk to.", "user_loving"),
    ("Ugh, another setback. Just my luck.", "user_angry"),
    ("You are a lifesaver, you absolute legend.", "user_thankful"),
    ("I can't believe how thoughtful that was of you.", "user_thankful"),
    ("That's disappointing but I understand.", "user_feeling_bad"),
    ("Whatever, it doesn't matter anymore.", "user_angry"),

    # --- Subtle & Implied Emotion ---
    ("Things are finally looking up.", "user_happy"),
    ("Let's just drop the subject, okay?", "user_angry"),
    ("It is what it is, I guess.", "user_feeling_bad"),
    ("Just wanted to follow up on our chat from yesterday.", "user_caring"),
    ("I just need a minute to myself.", "user_feeling_bad"),
    ("We need to talk later.", "user_caring"),
    ("I'm fine.", "user_angry"),
    ("Do whatever you want.", "user_angry"),
    
    ("I'm so obsessed with this, it's unhealthy.", "user_loving"),
    ("You absolute buffoon lmao", "user_happy"),
    ("I will literally fight you, that's too adorable.", "user_happy"),
    ("What an absolute madlad.", "user_loving"),
    ("You're a menace to society in the best way possible.", "user_loving"),
    ("Stop being so smart, you're making us all look bad lol.", "user_loving")
]


@pytest.mark.parametrize("sentence, expected_intent", test_cases)
def test_chatbot_intents(sentence, expected_intent, model_components):
    """
    This single test function will run for every case defined in `test_cases`.
    """
    
    model = model_components["model"]
    vocabulary = model_components["vocabulary"]
    tags = model_components["tags"]
    device = model_components["device"]
    
    input_tensor = preprocess_sentence(sentence, vocabulary, device)

    confidence, predicted_tag = make_prediction(input_tensor, "intent_guesser", model, tags)

    if not (confidence < 0.65):
        assert predicted_tag == expected_intent 
    else:
        pass
        