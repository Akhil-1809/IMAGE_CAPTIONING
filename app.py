from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import pickle
from collections import Counter
from torchtext.vocab import vocab
import nltk
import os
from tqdm import tqdm

nltk.download('punkt')

# Initialize Flask app
app = Flask(__name__)

# Define paths to model weights and feature files
BASE_DIR = 'C:/Users/akhil/OneDrive/Desktop/DL/dataset'
model_weights_path = 'C:/Users/akhil/OneDrive/Desktop/DL/RNN/sboreddy_aghanta_indushre_VGG16_RNN_1.pth'
vgg_model_path = 'C:/Users/akhil/OneDrive/Desktop/DL/LSTM/sboreddy_aghanta_indushre_features_LSTM.pkl'
# Load the pre-trained VGG16 model for feature extraction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg_model = models.vgg16(pretrained=True)  # Updated for the latest weight format
vgg_model.classifier = torch.nn.Sequential(*list(vgg_model.classifier.children())[:-1])
#vgg_model.load_state_dict(torch.load(vgg_model_path, map_location=device))

#vgg_model = vgg_model.to(device)
vgg_model.eval()


# Define preprocessing function
preprocess = transforms.Compose(
    transforms.Resize(256),[
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and process the captions
with open(os.path.join(BASE_DIR, 'captions.txt'), 'r', encoding='utf-8') as f:
    next(f) 
    captions_doc = f.read()

# Parse the captions into a mapping
mapping = {}
for line in tqdm(captions_doc.split('\n')):
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    image_id = image_id.split('.')[0]
    caption = " ".join(caption)
    if image_id not in mapping:
        mapping[image_id] = []
    mapping[image_id].append(caption)

# Clean the captions
def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i]
            caption = caption.lower()
            caption = caption.replace('[^A-Za-z]', '')
            caption = caption.replace('\s+', ' ')
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word) > 1]) + ' endseq'
            captions[i] = caption

clean(mapping)

# Function to tokenize captions
def tokenize(text):
    return nltk.word_tokenize(text.lower())

# Function to yield tokens for vocabulary building
def yield_tokens(all_captions):
    for caption in all_captions:
        yield tokenize(caption)

# Build the vocabulary using cleaned captions
all_captions = [caption for key in mapping for caption in mapping[key]]
counter = Counter()
for caption in all_captions:
    counter.update(tokenize(caption))

my_vocab = vocab(counter)

# Insert special tokens and set default index
default_tokens = {'<unk>': 0, '<pad>': 1, '<bos>': 2, '<eos>': 3}
for token, index in default_tokens.items():
    my_vocab.insert_token(token, index)

my_vocab.set_default_index(my_vocab['<unk>'])
vocab_size = len(my_vocab)

# Define the caption generation model
class CaptioningModel(nn.Module):
    def __init__(self, vocab_size, max_length, feature_dim=4096, embedding_dim=256, hidden_dim=256):
        super(CaptioningModel, self).__init__()
        self.dropout1 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(feature_dim, embedding_dim)  
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout2 = nn.Dropout(0.4)
        self.rnn = nn.RNN(2 * embedding_dim, hidden_dim, batch_first=True)  
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)
        self.outputs = nn.Linear(embedding_dim, vocab_size)  

    def forward(self, img_features, captions):
        img_features = self.dropout1(img_features)
        img_features = F.relu(self.fc1(img_features))  
        img_features_repeated = img_features.unsqueeze(1).repeat(1, captions.size(1), 1)  
        captions = self.embedding(captions)  
        captions = self.dropout2(captions)
        combined_inputs = torch.cat((img_features_repeated, captions), dim=2)  
        rnn_out, _ = self.rnn(combined_inputs)  
        decoded = F.relu(self.fc2(rnn_out))  
        outputs = self.outputs(decoded)  
        return outputs

# Initialize the captioning model with the correct vocab size
max_length = 20  # Example max length, replace with actual
caption_model = CaptioningModel(vocab_size, max_length)
caption_model = caption_model.to(device)

# Load the model weights
caption_model.load_state_dict(torch.load(model_weights_path, map_location=device))
caption_model.eval()

def decode_caption(output_tensor, vocab):
    """Convert model output tensor to a human-readable caption."""
    words = []
    for idx in output_tensor.squeeze().cpu().numpy():
        word = vocab.lookup_token(idx)  # Convert index to word using the vocab
        if word == '<eos>':
            break
        words.append(word)
    return ' '.join(words)

@app.route('/generate_caption', methods=['POST'])
def generate_caption():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Load and preprocess the image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        image = preprocess(image).unsqueeze(0).to(device)
        
        # Extract features using the pre-trained VGG16 model
        with torch.no_grad():
            features = vgg_model(image)
        
        # Prepare input for caption generation
        features = features.to(device)  # Ensure tensor is on correct device
        input_captions = torch.zeros((1, max_length), dtype=torch.long).to(device)  # Initial input caption sequence

        # Generate caption
        with torch.no_grad():
            outputs = caption_model(features, input_captions)
            # Get the predicted word indices
            _, predicted_indices = torch.max(outputs, dim=2)

            # Decode the output tensor to a human-readable caption
            caption = decode_caption(predicted_indices, my_vocab)

        return jsonify({'caption': caption}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
