import os
import torch
import pandas as pd
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import redis

from utils_ds.saveToDB import saveToRedis

# Initialize MTCNN and ResNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20,
              thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
              device=device)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
img_list = set()
def collate_fn(x):
    return x[0]

def check_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            try:
                img = Image.open(os.path.join(root, file))
                img.verify()
            except (IOError, SyntaxError) as e:
                print(f'Bad file: {os.path.join(root, file)}')

def main():
    # Check data integrity
    check_images('facedata')
    r = redis.Redis(
    host='redis-16109.c328.europe-west3-1.gce.redns.redis-cloud.com',
    port=16109,
    password='bawEnIPG6hRrMNTs6lyQ0NF6tQq05nso')
    
    # Load the dataset
    dataset = datasets.ImageFolder('facedata')
    dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=0)  # Set num_workers=0

    # Align and embed dataset images
    aligned = []
    names = []
    for x, y in loader:
        x_aligned, prob = mtcnn(x, return_prob=True)
        if x_aligned is not None:
            print('Face detected with probability: {:8f}'.format(prob))
            aligned.append(x_aligned)
            names.append(dataset.idx_to_class[y])

    aligned = torch.stack(aligned).to(device)
    embeddings = resnet(aligned).detach().cpu()

    # Function to recognize a face from a single image
    def recognize_face(image_path,i):
        # Load and process the image
        img = Image.open(image_path)
        img_cropped, prob = mtcnn(img, return_prob=True)

        if img_cropped is not None:
            print('Face detected with probability: {:8f}'.format(prob))
            img_embedding = resnet(img_cropped.unsqueeze(0).to(device)).detach().cpu()

            # Calculate distances to the dataset embeddings
            dists = [(e - img_embedding).norm().item() for e in embeddings]
            min_dist_idx = torch.argmin(torch.tensor(dists))
            if(dists[min_dist_idx])  > .8:
                print(f"Closest match: {names[min_dist_idx]} with distance: {dists[min_dist_idx]}")
                print('out of database')
            else:
                print(f"Closest match: {names[min_dist_idx]} with distance: {dists[min_dist_idx]}")
                saveToRedis(i, r, names[min_dist_idx])
        else:
            print("No face detected.")

    while True:
        for i in os.listdir('output\img'):
            if i not in img_list:
                file_path = os.path.join('output\img', i)
                if os.path.exists(os.path.join('output\img', i)):
                    recognize_face(file_path ,i)
                    img_list.add(i)


if __name__ == '__main__':
    main()
