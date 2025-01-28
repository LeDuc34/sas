import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50, ResNet50_Weights

# Load a pretrained ResNet model
weights = ResNet50_Weights.IMAGENET1K_V1
model = resnet50(weights=weights)
model.eval()  # Set model to evaluation mode

# Specify the target layers for GradCAM
target_layers = [model.layer4[-1]]

# Load and preprocess the input image
image_path = "input.jpg"  # Provide the path to your image
rgb_img = Image.open(image_path).convert("RGB")  # Open image and ensure RGB format

# Save the original dimensions
original_size = rgb_img.size  # (width, height)

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to model's expected input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
])
input_tensor = transform(rgb_img).unsqueeze(0)  # Add batch dimension

# Perform a forward pass to get predictions
output = model(input_tensor)
_, predicted_idx = torch.max(output, 1)  # Get the index of the highest score

# Get the class name using ResNet weights metadata
class_name = weights.meta["categories"][predicted_idx.item()]
print(f"Predicted Class: {class_name}")

# Define the target for GradCAM visualization
targets = [ClassifierOutputTarget(predicted_idx.item())]

# Convert the RGB image to numpy array (normalized between 0 and 1)
rgb_img = np.array(rgb_img) / 255.0

# Construct the CAM object and generate the visualization
with GradCAM(model=model, target_layers=target_layers) as cam:
    # Generate the CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]  # Extract the CAM for the first image in the batch

    np.save("activation_map.npy", grayscale_cam)

    # Resize the CAM to match the original image size
    from cv2 import resize
    grayscale_cam_resized = resize(grayscale_cam, (original_size[0], original_size[1]))



    # Overlay the CAM on the original image
    visualization = show_cam_on_image(rgb_img, grayscale_cam_resized, use_rgb=True)

    # Save or display the visualization
    from matplotlib import pyplot as plt
    plt.imshow(visualization)
    plt.axis("off")
    plt.title(f"Class: {class_name}")  # Display the class name as the title
    plt.show()
