import torch
from torchvision import transforms
import cv2
from model import NAFNet

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NAFNet(img_channel=3, width=32, middle_blk_num=12, 
               enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]).to(device)
model.load_state_dict(torch.load('./checkpoints/nafnet_final.pth', map_location=device))
model.eval()

# Load image
img = cv2.imread('image copy.png')  # CHANGE THIS
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256, 256))

# Transform
transform = transforms.ToTensor()
img_tensor = transform(img).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = model(img_tensor)

# Convert back
output = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
output = (output * 255).clip(0, 255).astype('uint8')

# Save
output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
cv2.imwrite('deblurred.jpg', output)
print("Done! Saved to deblurred.jpg")