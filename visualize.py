import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Data path
data_path = './datasets/mixed'

# Get class folder list
classes = sorted(os.listdir(data_path))
classes = [c for c in classes if os.path.isdir(os.path.join(data_path, c))]

# Select first 5 classes
n_classes = 5
n_samples = 5
selected_classes = classes[:n_classes]

# Visualization setup
fig, axes = plt.subplots(n_classes, n_samples, figsize=(15, 3*n_classes))
fig.suptitle('Sample Images from First 5 Classes', fontsize=16, y=0.995)

# Load and display sample images for each class
for i, class_name in enumerate(selected_classes):
    class_path = os.path.join(data_path, class_name)
    images = sorted(os.listdir(class_path))
    
    # Filter image files only (common image extensions)
    images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    # Select up to 5 samples
    sample_images = images[:n_samples]
    
    for j, img_name in enumerate(sample_images):
        img_path = os.path.join(class_path, img_name)
        img = Image.open(img_path)
        
        # Adjust axes indexing for single row case
        if n_classes == 1:
            ax = axes[j]
        else:
            ax = axes[i, j]
        
        ax.imshow(img)
        ax.axis('off')
        
        # Display class name on the first column only
        if j == 0:
            ax.set_ylabel(f'Class {i}: {class_name}', fontsize=10, rotation=0, 
                         labelpad=80, va='center')

plt.tight_layout()
plt.savefig('class_samples_visualization.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'class_samples_visualization.png'")
plt.show()

# Print class information
print(f"Total number of classes: {len(classes)}")
print(f"Visualized classes: {selected_classes}")
for class_name in selected_classes:
    class_path = os.path.join(data_path, class_name)
    n_images = len([f for f in os.listdir(class_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])
    print(f"  {class_name}: {n_images} images")