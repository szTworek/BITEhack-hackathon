import torch
from torchvision import transforms
import cv2
from model import Generator

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator(3, 3, 64, 9).to(device)
generator.load_state_dict(torch.load('./checkpoints/generator_final.pth', map_location=device))
generator.eval()

PATH = "/home/albert/bitehack/BITEhack-hackathon/backend/tests/image.png"
# Load and preprocess image
img = cv2.imread(PATH)  # CHANGE THIS
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256, 256))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
img_tensor = transform(img).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = generator(img_tensor)

# Convert back to image
output = output.squeeze(0).cpu() * 0.5 + 0.5
output = output.permute(1, 2, 0).numpy()
output = (output * 255).astype('uint8')

# Save
output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
cv2.imwrite('deblurred.jpg', output)
print("Done! Saved to deblurred.jpg")