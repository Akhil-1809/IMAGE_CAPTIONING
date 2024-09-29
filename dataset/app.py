


from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import pickle

# Initialize Flask app
app = Flask(__name__)

# Define paths to model weights and feature files
feature_path = r'C:\\Users\\akhil\\OneDrive\\Desktop\\DL\\RNN\\features_vgg_new.pkl'
model_weights_path = r'C:\\Users\\akhil\\OneDrive\\Desktop\\DL\\RNN\\sboreddy_aghanta_indushre_VGG16_RNN_1.pth'

# Load the pre-trained VGG16 model for feature extraction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg_model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')  # Updated for the latest weight format
vgg_model.classifier = torch.nn.Sequential(*list(vgg_model.classifier.children())[:-1])
vgg_model = vgg_model.to(device)
vgg_model.eval()

# Define preprocessing function
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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

# Initialize the captioning model with your parameters
vocab_size = 10000  # Example vocab size, replace with actual
max_length = 20     # Example max length, replace with actual
caption_model = CaptioningModel(vocab_size, max_length)
caption_model = caption_model.to(device)

# Load the model weights
caption_model.load_state_dict(torch.load(model_weights_path, map_location=device))
caption_model.eval()

# Load vocabulary (this should map indices to words, replace with your actual vocab)
vocab = {0: '<pad>', 1: '<start>', 2: '<end>', 3: 'a', 4: 'cat', 5: 'on', 6: 'the', 7: 'mat'}  # Example vocab
inv_vocab = {v: k for k, v in vocab.items()}  # Reverse mapping

def decode_caption(output_tensor, vocab):
    """Convert model output tensor to a human-readable caption."""
    words = []
    for idx in output_tensor.squeeze().cpu().numpy():
        word = vocab.get(idx, '<unk>')  # Replace unknown words with '<unk>'
        if word == '<end>':
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
            caption = decode_caption(predicted_indices, vocab)

        return jsonify({'caption': caption}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
