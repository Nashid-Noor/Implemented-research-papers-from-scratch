import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.datasets as datasets
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import argparse
import json
import random
import shutil
import gc
from pathlib import Path
from datasets import load_dataset
from datasets import load_from_disk
from torchvision.transforms import functional as TF
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR


class MultiDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.cumulative_lengths = np.cumsum(self.lengths)
        
        # For debugging/info
        print(f"Combined dataset with {len(self.datasets)} datasets")
        print(f"Individual dataset lengths: {self.lengths}")
        print(f"Total samples: {sum(self.lengths)}")

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        # Find which dataset the index belongs to
        dataset_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        
        # Calculate the index within that dataset
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_lengths[dataset_idx - 1]
        
        # Convert numpy integers to Python integers
        # This is crucial for HuggingFace datasets compatibility
        dataset_idx = int(dataset_idx)
        sample_idx = int(sample_idx)
        
        # Get the sample from the appropriate dataset
        try:
            return self.datasets[dataset_idx][sample_idx]
        except Exception as e:
            print(f"Error accessing dataset {dataset_idx}, index {sample_idx}: {e}")
            # Fallback to a default item in case of error
            dummy_img = torch.zeros((3, 224, 224))
            dummy_text = torch.zeros(77, dtype=torch.long)
            return dummy_img, dummy_text


def ensure_rgb(image):
    if hasattr(image, 'mode') and image.mode != 'RGB':
        return image.convert('RGB')
    return image


# Import our CLIP implementation
from clip_paper_implementation import create_clip_model
from clip_tokenizer_and_training import SimpleTokenizer

def load_class_mapping(dataset_name):
    mapping_path = os.path.join("./data", dataset_name, "class_mapping", f"{dataset_name}_class_mapping.json")

    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            class_mapping = json.load(f)
        print(f"Loaded class mapping for {dataset_name}")
        return {int(k): v for k, v in class_mapping.items()}
    else:
        print(f"Warning: No class mapping found for {dataset_name}. Using numeric labels.")
        return None



class CLIPDatasetWrapper(Dataset):
    def __init__(self, dataset, transform=None, context_length=77, dataset_name=None):
        self.dataset = dataset
        self.transform = transform
        self.tokenizer = SimpleTokenizer(context_length=context_length)
        self.dataset_name = dataset_name
        print("Dataset name:", self.dataset_name)

        # Try loading class mapping
        class_mapping = load_class_mapping(self.dataset_name)

        # Unwrap Subset to access base dataset
        base_dataset = dataset
        while isinstance(base_dataset, torch.utils.data.Subset):
            base_dataset = base_dataset.dataset

        # Get class to index mapping from different sources
        if class_mapping is not None:
            print("Using class mapping from file")
            self.idx_to_class = {int(k): v for k, v in class_mapping.items()}
        elif hasattr(base_dataset, "class_to_idx"):
            print("Using dataset's internal class_to_idx")
            self.idx_to_class = {v: k for k, v in base_dataset.class_to_idx.items()}
        else:
            print("Warning: No class mapping or class_to_idx found, using numeric labels.")
            unique_labels = set()
            # Try to sample a few items to find unique labels
            try:
                for i in range(min(len(dataset), 100)):
                    sample = dataset[i]
                    if isinstance(sample, tuple) and len(sample) > 1:
                        unique_labels.add(sample[1])
            except Exception as e:
                print(f"Error sampling dataset: {e}")
                # Default to a small number of classes
                unique_labels = set(range(10))
            
            num_classes = len(unique_labels)
            self.idx_to_class = {i: str(i) for i in range(num_classes)}

        # Format class names and get templates
        self.idx_to_readable = self._format_class_names()
        self.templates = self._get_templates()

    def _format_class_names(self):
        if self.dataset_name == "oxford_pets": # working
            return {idx: name.replace("_", " ").title() for idx, name in self.idx_to_class.items()}
        elif self.dataset_name == "stanford_cars": #working
            return {idx: name.replace("_", " ").title() for idx, name in self.idx_to_class.items()}
        elif self.dataset_name == "flowers102": # working with class mappings
            # Flowers dataset has numeric labels, we can use the class number
            return {idx: f"Flower Type {name}" for idx, name in self.idx_to_class.items()}
        elif self.dataset_name == "food101": # working
            return {idx: name.replace("_", " ").title() for idx, name in self.idx_to_class.items()}
        elif self.dataset_name == "fgvc_aircraft": #working
            return {idx: name.replace("_", " ").title() for idx, name in self.idx_to_class.items()}
        elif self.dataset_name == "sun397": #working
            return {idx: name.replace("_", " ").title() for idx, name in self.idx_to_class.items()}
        elif self.dataset_name == "dtd":   #working
            # DTD is about textures
            return {idx: name.replace("_", " ").lower() for idx, name in self.idx_to_class.items()}
        elif self.dataset_name == "eurosat": # working
            return {idx: name.replace("_", " ").title() for idx, name in self.idx_to_class.items()}
        elif self.dataset_name == "ucf101":
            # UCF101 is about actions
            return {idx: name.replace("_", " ").title() for idx, name in self.idx_to_class.items()}
        elif self.dataset_name == "caltech101": # downloaded manually, working
            return {idx: name.replace("_", " ").title() for idx, name in self.idx_to_class.items()}
        elif self.dataset_name == "combined":
            # For combined dataset, just use the class index and name directly
            return {idx: name.replace("_", " ").title() for idx, name in self.idx_to_class.items()}
        else:
            # Generic approach
            return {idx: name.replace("_", " ").title() for idx, name in self.idx_to_class.items()}
    
    def _get_templates(self):
        if self.dataset_name == "oxford_pets":
            return[
                "a photo of a {}.",
                "a close-up photo of a {}.",
                "a picture of a {} in a house.",
                "an image of a beautiful {}.",
                "a {} relaxing indoors.",
                "a photo of a cute {} resting on a bed.",
                "an adorable {} lying on the grass.",
                "a playful {} captured outdoors.",
                "a {} sitting quietly.",
                "an elegant {} looking at the camera.",
                "a close-up shot of a {}'s face.",
                "a {} with a fluffy coat.",
                "a {} pet posing for a photo.",
                "a happy {} enjoying the sun.",
                "a {} pet sitting on a couch.",
                "a {} playing with a toy.",
                "an excited {} in a garden.",
                "a sleepy {} curled up on a blanket.",
                "a smiling {} captured up close.",
                "a side profile of a {}.",
                "a {} standing near a window.",
                "a lovely {} looking around.",
                "a {} wagging its tail happily.",
                "a photo of a healthy {}.",
                "a majestic {} sitting upright.",
                "a young {} resting after play.",
                "an artistic shot of a {}.",
                "a portrait of a {} in natural light.",
                "an energetic {} playing outside.",
                "a {} captured mid-run."
            ]


        elif self.dataset_name == "stanford_cars":
            return [
                "a photo of a {} car.",
                "a side view of a {} car.",
                "a front view of a {}.",
                "a rear view of a {}.",
                "an angled shot of a {} car.",
                "a photo of a fast {} car.",
                "a luxury {} car.",
                "a photo of a vintage {} car.",
                "a photo of a new {} model.",
                "an advertisement for a {} car.",
                "a parked {} car.",
                "a racing {} car.",
                "a photo of a {} on a road.",
                "a photo of a {} at a showroom.",
                "a photo of a moving {} car.",
                "a photo of a modified {}.",
                "a close-up of a {} car grille.",
                "a clean {} car photo.",
                "a dirty {} car image.",
                "a photo of a shiny {}.",
                "a matte-finish {} car photo.",
                "a bright photo of a {}.",
                "a low-light photo of a {}.",
                "a detailed shot of {} headlights.",
                "a photo showing {} wheels.",
                "a {} drifting.",
                "a {} in a parking lot.",
                "a {} on a highway.",
                "an interior shot of a {} car.",
                "a {} at a car exhibition."
            ]


        elif self.dataset_name == "flowers102":
            return [
                "a photo of a {} flower.",
                "a close-up of a blooming {}.",
                "a macro shot of a {}.",
                "a photo of a {} garden.",
                "an artistic image of a {} flower.",
                "a watercolor painting of a {}.",
                "a digital painting of a {} flower.",
                "a side view of a {} blossom.",
                "a top view of a {} flower.",
                "a beautiful shot of a {}.",
                "a field of {} flowers.",
                "a bouquet of {} flowers.",
                "a single {} flower shot.",
                "a blurry photo of a {}.",
                "a vibrant {} in a garden.",
                "a flower bed full of {}.",
                "a fresh {} flower close-up.",
                "a dying {} flower image.",
                "a wet {} flower after rain.",
                "a photo of a {} under sunlight.",
                "a {} blooming in spring.",
                "a {} during autumn.",
                "a snow-covered {}.",
                "a pink {} blossom.",
                "a yellow {} in a meadow.",
                "a white {} flower picture.",
                "a colorful bouquet with {}.",
                "an artistic bouquet of {}.",
                "a bunch of {} flowers.",
                "a bouquet with mixed {}."
            ]


        elif self.dataset_name == "food101":
            return [
                "a photo of {} dish.",
                "a close-up of {} cuisine.",
                "a plate of {}.",
                "a freshly made {}.",
                "an overhead view of {}.",
                "a street food shot of {}.",
                "a fine dining {} plate.",
                "an image of {} being served.",
                "a cooking shot of {}.",
                "a restaurant menu photo of {}.",
                "a delicious {} meal.",
                "a photo of hot {}.",
                "a cold {} dessert image.",
                "a photo of spicy {}.",
                "a photo of sweet {}.",
                "an artistic plating of {}.",
                "a food photography of {}.",
                "a shot of {} with garnish.",
                "an image of {} on a table.",
                "a healthy {} dish.",
                "an unhealthy {} treat.",
                "a bowl of {}.",
                "a spoonful of {}.",
                "a slice of {} dessert.",
                "a serving of {}.",
                "a side dish of {}.",
                "a full meal with {}.",
                "a bite taken from {}.",
                "a zoomed-in shot of {}.",
                "a messy {} plate."
            ]

        elif self.dataset_name == "fgvc_aircraft":
            return [
                "a photo of a {} aircraft.",
                "an image of a flying {}.",
                "a parked {} plane photo.",
                "a side view of a {} jet.",
                "a close-up of a {} cockpit.",
                "an aerial shot of a {}.",
                "a vintage {} aircraft photo.",
                "a modern {} airplane.",
                "a jet-powered {} photo.",
                "a propeller-driven {} shot.",
                "a top view of a {}.",
                "a colorful {} aircraft image.",
                "a black and white {} photo.",
                "a photo of a {} during takeoff.",
                "a photo of a {} landing.",
                "a {} on a runway.",
                "a {} parked at an airport.",
                "a blurry photo of a flying {}.",
                "an action shot of a {} aircraft.",
                "an airline advertisement of a {}.",
                "an artistic image of a {}.",
                "a military {} aircraft.",
                "a passenger {} plane.",
                "a photo of {} at sunset.",
                "a plane race showing {}.",
                "a maintenance view of a {}.",
                "a historical photo of {}.",
                "an animated sketch of {}.",
                "a close-up of {} engines.",
                "a panoramic view of a {} jet."
            ]


        elif self.dataset_name == "sun397":
            return[
                "a photo of a {}.",
                "an outdoor photo showing a {}.",
                "a wide shot of a {} scene.",
                "an aerial view of a {}.",
                "a panoramic photo of a {}.",
                "a beautiful view of a {}.",
                "a landscape featuring a {}.",
                "a distant view of a {}.",
                "an image taken at a {}.",
                "a scenic photo of a {} place.",
                "a daytime photo of a {}.",
                "a nighttime view of a {}.",
                "a sunset photo showing a {}.",
                "a photo taken inside a {}.",
                "an indoor scene of a {}.",
                "a tourist photo at a {}.",
                "a photo of people at a {}.",
                "an empty {} location photo.",
                "an abandoned {} scene.",
                "a busy {} area.",
                "a peaceful {} setting.",
                "a bustling {} scene.",
                "a winter view of a {}.",
                "a summer photo at a {}.",
                "a rainy day at a {}.",
                "a photo of a {} during autumn.",
                "a sketch of a {} landscape.",
                "a minimalistic photo of a {}.",
                "an artistic interpretation of a {}.",
                "a photo looking out onto a {}."
            ]


        elif self.dataset_name == "dtd":
            return [
                "a photo showing a {} texture.",
                "a close-up of a {} surface.",
                "an image of a {} pattern.",
                "a macro shot of a {} material.",
                "a zoomed-in view of a {} texture.",
                "an artistic photo of {} texture.",
                "a surface with {} properties.",
                "a detailed view of a {} fabric.",
                "an object displaying {} texture.",
                "a rough {} surface photo.",
                "a smooth {} surface photo.",
                "a highly detailed {} pattern.",
                "a close-up of {} on a wall.",
                "a photo of a {} textile.",
                "an abstract {} pattern image.",
                "a blurry photo of a {} texture.",
                "a wet {} surface.",
                "a dry {} material image.",
                "an image showing {} textures in nature.",
                "a metallic {} surface.",
                "a photo of a {} wooden texture.",
                "a plastic surface with {}.",
                "an organic {} texture.",
                "an artificial {} pattern.",
                "an outdoor shot of a {} surface.",
                "an indoor shot of a {} wall.",
                "a painted surface with {} effect.",
                "a stone surface showing {}.",
                "an artistic abstraction of {}.",
                "a high contrast photo of {} texture."
            ]


        elif self.dataset_name == "eurosat":
            return [
                    "a satellite image of {}.",
                    "an aerial view of {}.",
                    "a top-down satellite photo showing {}.",
                    "a remote sensing image of {}.",
                    "an overhead photo of {} land.",
                    "a satellite picture of {} from space.",
                    "a landscape photo of {} seen from above.",
                    "a satellite map of {}.",
                    "a geospatial view showing {}.",
                    "an Earth observation photo of {}.",
                    "a cloudless satellite image of {}.",
                    "an urban {} satellite capture.",
                    "a rural {} land image.",
                    "a farmland area showing {}.",
                    "an aerial shot of {} fields.",
                    "a river captured from space showing {}.",
                    "a forest patch satellite image of {}.",
                    "a coastline with {} shown from above.",
                    "a snowy {} region from satellite.",
                    "a satellite view of {} infrastructure.",
                    "a green area of {} land.",
                    "a dry desert view showing {}.",
                    "an image of {} terrain from the sky.",
                    "a topographical map showing {}.",
                    "a satellite view showing agricultural {}.",
                    "a river system showing {} from orbit.",
                    "an environmental monitoring photo of {}.",
                    "an eco-region observed as {}.",
                    "a remote area captured showing {}.",
                    "an open landscape image of {}."
                ]


        elif self.dataset_name == "ucf101":
            return [
                    "a person {}.",
                    "someone is {}.",
                    "an individual {} outdoors.",
                    "an athlete performing {}.",
                    "a human {} in a competition.",
                    "a video of someone {}.",
                    "a person practicing {}.",
                    "an action scene of {}.",
                    "a sports event showing {}.",
                    "a performer {} on stage.",
                    "a training session involving {}.",
                    "a video recording of {}.",
                    "an artistic shot of someone {}.",
                    "a group of people {} together.",
                    "a solo person performing {}.",
                    "a photo capturing {} in motion.",
                    "a slow-motion capture of {}.",
                    "an intense action of {}.",
                    "someone {} at the beach.",
                    "someone {} on a mountain.",
                    "a professional athlete {}.",
                    "an amateur {} attempt.",
                    "someone {} indoors.",
                    "someone {} outdoors.",
                    "a video clip of {} activity.",
                    "a practice session showing {}.",
                    "a competition involving {}.",
                    "a match scene of {}.",
                    "a recreational act of {}.",
                    "a cinematic shot of someone {}."
                ]


        elif self.dataset_name == "caltech101":
            return[
                "a photo of a {} object.",
                "an image showing a {}.",
                "an artistic image of a {}.",
                "a sketch of a {}.",
                "a low-quality photo of a {}.",
                "a high-quality image of a {}.",
                "a drawing of a {}.",
                "a detailed photo of a {}.",
                "a centered photo of a {}.",
                "a close-up of a {}.",
                "an overhead shot of a {}.",
                "a zoomed-out photo of a {}.",
                "a blurry image of a {}.",
                "a clean image of a {}.",
                "a close-up photo showing a {}.",
                "a studio shot of a {}.",
                "a photo of a single {}.",
                "an artistic representation of a {}.",
                "a painting of a {}.",
                "a computer-generated image of a {}.",
                "a side view of a {}.",
                "a rear view of a {}.",
                "a macro shot of a {}.",
                "a photo of a {} toy.",
                "an abstract image of a {}.",
                "a photo of an old {}.",
                "a photo of a new {}.",
                "a vintage photo of a {}.",
                "a daytime photo of a {}.",
                "a nighttime shot of a {}."
            ]
            
        elif self.dataset_name == "combined":
            # Balanced selection of the most effective prompts from all datasets
            return [
                # General prompts
                "a photo of a {}.",
                "an image of a {}.",
                "a picture showing a {}.",
                "a clear image of a {}.",
                "a detailed photograph of a {}.",
                
                # Animal/Pet specific (from oxford_pets)
                "a photo of a cute {}.",
                "an adorable {} photographed outdoors.",
                "a close-up shot of a {}'s face.",
                
                # Vehicle specific (from stanford_cars)
                "a {} vehicle in a photograph.",
                "a side view of a {} on display.",
                "a professional photo of a {}.",
                
                # Nature specific (from flowers102)
                "a beautiful {} in bloom.",
                "a close-up of a {} flower.",
                "a photo of a colorful {}.",
                
                # Food specific (from food101)
                "a plate of {} on a table.",
                "a delicious looking {}.",
                "a serving of {} cuisine.",
                
                # Aircraft specific (from fgvc_aircraft)
                "a {} aircraft in flight.",
                "a photograph of a {} at an airport.",
                
                # Location specific (from sun397)
                "a scene showing a {}.",
                "a panoramic view of a {}.",
                "a location identified as {}.",
                
                # Texture specific (from dtd)
                "a surface with {} texture.",
                "a close-up of a {} pattern.",
                "a material with a {} appearance.",
                
                # Satellite specific (from eurosat)
                "a satellite view of {}.",
                "an aerial photograph showing {}.",
                
                # Action specific (from ucf101)
                "a person performing {}.",
                "someone in the middle of {}.",
                
                # Object specific (from caltech101)
                "a photograph of a {} object.",
                "a {} photographed in studio lighting.",
                "a well-defined image of a {}."
            ]
        else:
            # Generic templates
            return [
                "a photo of a {}.",
                "a picture of a {}.",
                "a close-up photo of a {}.",
                "an image of a {}.",
                "a photograph showing a {}.",
                "a detailed image of a {}.",
                "a clear photo of a {}.",
                "a well-lit image of a {}.",
                "a professional photo of a {}.",
                "a high-quality picture of a {}."
            ]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        try:
            # Get image and class index
            if isinstance(self.dataset[idx], tuple) and len(self.dataset[idx]) == 2:
                img, class_idx = self.dataset[idx]
            else:
                # Handle datasets with different return formats
                sample = self.dataset[idx]
                if hasattr(sample, 'items'):  # Check if it's a dictionary-like object (from datasets library)
                    img = sample['image'] if 'image' in sample else sample.get('img', None)
                    class_idx = sample['label'] if 'label' in sample else sample.get('class', 0)
                else:
                    img = sample[0] if isinstance(sample, tuple) else sample
                    class_idx = sample[1] if isinstance(sample, tuple) and len(sample) > 1 else 0
            
            # Make sure we have a valid image
            if img is None:
                raise ValueError(f"Could not extract image from dataset at index {idx}")
                
            # Apply transform if provided
            if self.transform:
                # Make sure image is a PIL image if transform expects it
                if torch.is_tensor(img):
                    # Convert tensor to PIL before transform
                    to_pil = transforms.ToPILImage()
                    img = to_pil(img)
                
                img = self.transform(img)
            
            # Ensure class_idx is a standard Python int
            class_idx = int(class_idx) if not isinstance(class_idx, int) else class_idx
            
            # Get class name
            if class_idx not in self.idx_to_readable:
                print(f"Warning: Class index {class_idx} not found in idx_to_readable. Using default.")
                class_name = f"unknown_class_{class_idx}"
            else:
                class_name = self.idx_to_readable[class_idx]
            
            # Create text description with random template
            template = random.choice(self.templates)
            text = template.format(class_name)
            
            # Tokenize text
            tokens = torch.tensor(self.tokenizer.encode(text))
            
            return img, tokens
            
        except Exception as e:
            print(f"Error getting item at index {idx}: {e}")
            # Return a default/fallback item
            # Create a blank image and empty text
            dummy_img = torch.zeros((3, 224, 224))
            dummy_text = torch.zeros(77, dtype=torch.long)
            return dummy_img, dummy_text


class UCF101Dummy(Dataset):
    def __init__(self, root="./data", train=True, transform=None):
        self.transform = transform
        self.train = train
        
        # Create dummy data
        self.num_samples = 1000 if train else 250
        self.num_classes = 101
        
        # Assign class names for UCF101
        self.class_names = [
            "ApplyEyeMakeup", "ApplyLipstick", "Archery", "BabyCrawling", "BalanceBeam",
            "BandMarching", "BaseballPitch", "Basketball", "BasketballDunk", "BenchPress",
            "Biking", "Billiards", "BlowDryHair", "BlowingCandles", "BodyWeightSquats",
            "Bowling", "BoxingPunchingBag", "BoxingSpeedBag", "BreastStroke", "BrushingTeeth",
            "CleanAndJerk", "CliffDiving", "CricketBowling", "CricketShot", "CuttingInKitchen",
            "Diving", "Drumming", "Fencing", "FieldHockeyPenalty", "FloorGymnastics",
            "FrisbeeCatch", "FrontCrawl", "GolfSwing", "Haircut", "Hammering",
            "HammerThrow", "HandstandPushups", "HandstandWalking", "HeadMassage", "HighJump",
            "HorseRace", "HorseRiding", "HulaHoop", "IceDancing", "JavelinThrow",
            "JugglingBalls", "JumpingJack", "JumpRope", "Kayaking", "Knitting",
            "LongJump", "Lunges", "MilitaryParade", "Mixing", "MoppingFloor",
            "Nunchucks", "OpeningBottle", "ParallelBars", "PizzaTossing", "PlayingCello",
            "PlayingDhol", "PlayingFlute", "PlayingGuitar", "PlayingPiano", "PlayingSitar",
            "PlayingTabla", "PlayingViolin", "PoleVault", "PommelHorse", "PullUps",
            "Punch", "PushUps", "Rafting", "RockClimbingIndoor", "RopeClimbing",
            "Rowing", "SalsaSpin", "ShavingBeard", "Shotput", "SkateBoarding",
            "Skiing", "Skijet", "SkyDiving", "SoccerJuggling", "SoccerPenalty",
            "StillRings", "SumoWrestling", "Surfing", "Swing", "TableTennisShot",
            "TaiChi", "TennisSwing", "ThrowDiscus", "TrampolineJumping", "Typing",
            "UnevenBars", "VolleyballSpiking", "WalkingWithDog", "WallPushups", "WritingOnBoard",
            "YoYo"
        ]
        
        # Create class to index mapping
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        
        print(f"Created UCF101Dummy dataset with {self.num_samples} samples, {self.num_classes} classes")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Create a dummy image (random color)
        img = torch.rand(3, 224, 224)
        
        # Assign a deterministic class based on index (cycle through classes)
        label = idx % self.num_classes
        
        if self.transform:
            # Convert to PIL for transform
            to_pil = transforms.ToPILImage()
            pil_img = to_pil(img)
            img = self.transform(pil_img)
        
        return img, label


class StanfordCarsLocal(torch.utils.data.Dataset):
    def __init__(self, split="train", transform=None):
        dataset_dir = "./local_datasets/stanford_cars"
        split_dir = os.path.join(dataset_dir, split)
        os.makedirs(dataset_dir, exist_ok=True)

        if not os.path.exists(os.path.join(split_dir, "dataset_info.json")):
            print(f"Downloading Stanford Cars {split} split...")
            dataset_dict = load_dataset("Donghyun99/Stanford-Cars", cache_dir=dataset_dir)
            split_dataset = dataset_dict[split]
            print(f"Saving {split} split to {split_dir}...")
            split_dataset.save_to_disk(split_dir)
            print(f"{split} split downloaded and saved successfully!")
        else:
            print(f"{split} split already exists locally. Loading from disk...")

        # Load split dataset
        self.dataset = load_from_disk(split_dir)
        self.transform = transform

        # Create class_to_idx mapping
        self.classes = self.dataset.features['label'].names
        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.classes)}
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample['image']
        label = sample['label']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class DatasetFactory:
    @staticmethod
    def get_dataset(name, root="./data", train=True, transform=None, download=True):
        name = name.lower()
        
        # Create dataset directory
        dataset_dir = os.path.join(root, name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        if name == "oxford_pets" or name == "oxfordpets" or name == "oxford_iiit_pet":
            split = "trainval" if train else "test"
            dataset = datasets.OxfordIIITPet(root=dataset_dir, split=split, transform=transform, download=download)
            return dataset, "oxford_pets"
            
        elif name == "caltech101":
            try:
                #dataset = datasets.Caltech101(root=dataset_dir, transform=transform, download=download)
                dataset = datasets.ImageFolder(root=os.path.join(dataset_dir, "101_ObjectCategories"), transform=transform)

                # Manually split into train/test since Caltech101 doesn't provide a split
                num_samples = len(dataset)
                if train:
                    indices = list(range(int(0.8 * num_samples)))
                else:
                    indices = list(range(int(0.8 * num_samples), num_samples))
                
                from torch.utils.data import Subset
                return Subset(dataset, indices), "caltech101"
            except Exception as e:
                print(f"Error loading Caltech101: {e}")
                raise
            
        elif name == "stanford_cars" or name == "stanfordcars":
            try:
                # Use custom Stanford Cars implementation
                split = "train" if train else "test"
                dataset = StanfordCarsLocal(split=split)
                return dataset, "stanford_cars"
            except Exception as e:
                print(f"Error loading Stanford Cars: {e}")
                raise
            
        elif name == "flowers102":
            split = "train" if train else "test"
            dataset = datasets.Flowers102(root=dataset_dir, split=split, transform=transform, download=download)
            return dataset, "flowers102"
            
        elif name == "food101":
            split = "train" if train else "test"
            dataset = datasets.Food101(root=dataset_dir, split=split, transform=transform, download=download)
            return dataset, "food101"
            
        elif name == "fgvc_aircraft" or name == "fgvcaircraft" or name == "aircraft":
            split = "train" if train else "test"
            dataset = datasets.FGVCAircraft(root=dataset_dir, split=split, transform=transform, download=download)
            return dataset, "fgvc_aircraft"
            
        elif name == "sun397":
            dataset = datasets.SUN397(root=dataset_dir, transform=transform, download=download)
            # Manually split into train/test
            num_samples = len(dataset)
            if train:
                indices = list(range(int(0.8 * num_samples)))
            else:
                indices = list(range(int(0.8 * num_samples), num_samples))
            
            from torch.utils.data import Subset
            return Subset(dataset, indices), "sun397"
            
        elif name == "dtd":
            split = "train" if train else "test"

            dataset = datasets.DTD(root=dataset_dir, split=split, transform=transform, download=download)
            return dataset, "dtd"
            
        elif name == "eurosat":
            dataset = datasets.EuroSAT(root=dataset_dir, transform=transform, download=download)
            # Manually split into train/test
            num_samples = len(dataset)
            if train:
                indices = list(range(int(0.8 * num_samples)))
            else:
                indices = list(range(int(0.8 * num_samples), num_samples))
            
            from torch.utils.data import Subset
            return Subset(dataset, indices), "eurosat"
            
        elif name == "ucf101":
            # ====== REVISED UCF101 HANDLING ======
            # Use our dummy implementation instead of the real UCF101
            print("Creating UCF101Dummy dataset instead of the original UCF101 dataset")
            dataset = UCF101Dummy(root=dataset_dir, train=train, transform=transform)
            return dataset, "ucf101"
            
        elif name == "all":
            # Special case for "all" - return a placeholder dataset
            # This should never actually be called since "all" is handled at a higher level
            print("WARNING: DatasetFactory.get_dataset() was called with name='all'")
            print("This is not supported - 'all' should be handled in train_clip_on_dataset()")
            dummy_data = torch.utils.data.TensorDataset(
                torch.zeros(10, 3, 224, 224),  # 10 dummy images
                torch.zeros(10, dtype=torch.long)  # 10 dummy labels
            )
            return dummy_data, "combined"
            
        else:
            raise ValueError(f"Dataset {name} not supported")

def train_clip_on_dataset(args):
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = create_clip_model(args.model_type).to(device)
    
    # Define transforms with enhanced augmentation
    train_transform = transforms.Compose([
        transforms.Lambda(ensure_rgb),
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
        ], p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Lambda(ensure_rgb),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create datasets
    print(f"Loading {args.dataset} dataset...")    
    
    if args.dataset.lower() == "all":
        # Define all datasets to combine
        all_datasets = ["oxford_pets", "caltech101", "flowers102", "food101", "stanford_cars",
                        "fgvc_aircraft", "sun397", "dtd", "eurosat", "ucf101"]

        print(f"Creating combined dataset from: {all_datasets}")
        train_wrappers, val_wrappers, test_wrappers = [], [], []
        
        # Loop through all datasets and prepare them for combining
        for ds_name in all_datasets:
            print(f"Loading {ds_name}...")
            try:
                # Attempt to load the dataset
                train_base, _ = DatasetFactory.get_dataset(ds_name, root=args.data_dir, train=True, transform=None, download=True)
                val_base, _ = DatasetFactory.get_dataset(ds_name, root=args.data_dir, train=False, transform=None, download=True)

                # Create CLIP dataset wrappers
                train_wrapper = CLIPDatasetWrapper(train_base, transform=train_transform, dataset_name=ds_name)
                val_wrapper = CLIPDatasetWrapper(val_base, transform=val_transform, dataset_name=ds_name)
                test_wrapper = CLIPDatasetWrapper(val_base, transform=val_transform, dataset_name=ds_name)

                # Make sure we can get at least one item without errors
                # This will catch potential issues early
                try:
                    _ = train_wrapper[0]
                    _ = val_wrapper[0]
                    
                    # Only add dataset if we can successfully retrieve items
                    train_wrappers.append(train_wrapper)
                    val_wrappers.append(val_wrapper)
                    test_wrappers.append(test_wrapper)
                    
                    print(f"✅ Added {ds_name} with {len(train_wrapper)} training and {len(val_wrapper)} validation samples")
                except Exception as e:
                    print(f"❌ Error accessing items in {ds_name}: {e}")
                    print(f"Skipping {ds_name} dataset")
                    continue
                    
            except Exception as e:
                print(f"Error loading {ds_name}: {e}")
                print(f"Skipping {ds_name} dataset")
                continue
        
        # Check if we have any datasets to combine
        if len(train_wrappers) == 0:
            raise ValueError("No datasets could be loaded successfully. Cannot continue with training.")
            
        # Combine all datasets
        train_dataset = MultiDataset(train_wrappers)
        val_dataset = MultiDataset(val_wrappers)
        test_dataset = MultiDataset(test_wrappers)
        
        # The dataset_name is "combined" for the combined dataset
        dataset_name = "combined"
        
        # Create a unified idx_to_class mapping for the combined dataset
        # This is necessary for saving class mappings
        combined_idx_to_class = {}
        combined_idx_to_readable = {}
        class_offset = 0
        
        for wrapper in train_wrappers:
            for idx, class_name in wrapper.idx_to_class.items():
                combined_idx_to_class[class_offset + idx] = f"{wrapper.dataset_name}_{class_name}"
                combined_idx_to_readable[class_offset + idx] = wrapper.idx_to_readable[idx]
            # Update offset for next dataset
            class_offset += len(wrapper.idx_to_class)
        
        # Assign the combined mappings to the MultiDataset
        train_dataset.idx_to_class = combined_idx_to_class
        train_dataset.idx_to_readable = combined_idx_to_readable
        
    # For single dataset training
    else:
        train_data, dataset_name = DatasetFactory.get_dataset(args.dataset, root=args.data_dir, train=True, transform=None, download=True)

        if args.val_split < 1.0 and args.val_from_train:
            train_len = int((1 - args.val_split) * len(train_data))
            val_len = len(train_data) - train_len
            train_data, val_data = random_split(train_data, [train_len, val_len])
        else:
            val_data, _ = DatasetFactory.get_dataset(args.dataset, root=args.data_dir, train=False, transform=None, download=True)

        test_data, _ = DatasetFactory.get_dataset(args.dataset, root=args.data_dir, train=False, transform=None, download=True)

        train_dataset = CLIPDatasetWrapper(train_data, transform=train_transform, dataset_name=dataset_name)
        val_dataset = CLIPDatasetWrapper(val_data, transform=val_transform, dataset_name=dataset_name)
        test_dataset = CLIPDatasetWrapper(test_data, transform=val_transform, dataset_name=dataset_name)
        
    print(f"Dataset splits: Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # For combined dataset, we need to handle class mapping differently
    if args.dataset.lower() == "all":
        if hasattr(train_dataset, 'idx_to_class') and hasattr(train_dataset, 'idx_to_readable'):
            # Class mapping is already set in MultiDataset during the combined dataset creation
            class_mapping = {
                'idx_to_class': train_dataset.idx_to_class,
                'idx_to_readable': train_dataset.idx_to_readable
            }
        else:
            # If not explicitly set, create a mapping of all datasets combined
            # This is a fallback in case the earlier assignment didn't work
            print("Creating combined class mapping from all datasets...")
            combined_idx_to_class = {}
            combined_idx_to_readable = {}
            
            class_index = 0
            for dataset in train_dataset.datasets:
                for idx, class_name in dataset.idx_to_class.items():
                    # Use dataset name as prefix to avoid collisions
                    combined_idx_to_class[class_index] = f"{dataset.dataset_name}_{class_name}"
                    combined_idx_to_readable[class_index] = dataset.idx_to_readable[idx]
                    class_index += 1
            
            class_mapping = {
                'idx_to_class': combined_idx_to_class,
                'idx_to_readable': combined_idx_to_readable
            }
    else:
        # Regular single dataset mapping
        class_mapping = {
            'idx_to_class': train_dataset.idx_to_class,
            'idx_to_readable': train_dataset.idx_to_readable
        }
    
    # Save the class mapping
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert all keys to strings for JSON serialization
    class_mapping_serializable = {
        'idx_to_class': {str(k): v for k, v in class_mapping['idx_to_class'].items()},
        'idx_to_readable': {str(k): v for k, v in class_mapping['idx_to_readable'].items()}
    }
    
    with open(os.path.join(args.output_dir, "class_mapping.json"), "w") as f:
        json.dump(class_mapping_serializable, f, indent=2)
    
    # Check that we have at least some data
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty. Cannot proceed with training.")
    
    # Create dataloaders with improved error handling
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,  # Reduced workers for stability
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if torch.cuda.is_available() else False,
        prefetch_factor=2,
        timeout=120,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if torch.cuda.is_available() else False,
    )
    
    # Create optimizer with improved parameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),  # Changed from (0.9, 0.98) to standard (0.9, 0.999)
        eps=1e-8,  # Changed from 1e-6 to 1e-8 for better stability
        weight_decay=args.weight_decay
    )
    
    # Improved learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    # First, create a warmup scheduler (linear warmup)
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.1,  # Start at 10% of base learning rate
        end_factor=1.0,    # End at 100% of base learning rate
        total_iters=warmup_steps
    )

    # Then create a cosine annealing scheduler with warm restarts
    # T_0 is the initial cycle length (in iterations)
    # T_mult is the factor by which T_0 is multiplied after each restart
    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=len(train_loader) * args.restart_epochs,  # First restart after restart_epochs epochs
        T_mult=2,                   # Double the cycle length after each restart
        eta_min=args.learning_rate * 0.01  # Minimum LR = 1% of base LR
    )

    # Combine both schedulers: first warmup, then cosine annealing with restarts
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    
    # Training loop with gradient accumulation
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    learning_rates = []  # Track learning rates
    
    # Set gradient accumulation steps
    gradient_accumulation_steps = args.gradient_accumulation_steps if hasattr(args, 'gradient_accumulation_steps') else 1
    
    # Use try-except to catch and handle errors during training
    try:
        for epoch in range(args.epochs):
            # Training
            model.train()
            train_loss = 0.0
            num_train_batches = 0
            optimizer.zero_grad()  # Zero gradients at the start of each epoch
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]") as pbar:
                for i, (images, texts) in enumerate(pbar):
                    try:
                        # Move to device
                        images = images.to(device)
                        texts = texts.to(device)
                        
                        # Forward pass
                        logits_per_image, logits_per_text = model(images, texts)
                        
                        # Compute loss with label smoothing
                        batch_size = images.size(0)
                        labels = torch.arange(batch_size, device=device)
                        
                        # Add label smoothing for better regularization
                        smoothing = getattr(args, 'label_smoothing', 0.1)
                        off_diag_scale = smoothing / (batch_size - 1)
                        
                        # Create soft targets
                        soft_targets = torch.zeros(batch_size, batch_size, device=device)
                        soft_targets.fill_(off_diag_scale)
                        soft_targets.fill_diagonal_(1.0 - smoothing)
                        
                        # Cross entropy with soft targets
                        log_probs_img = F.log_softmax(logits_per_image, dim=1)
                        log_probs_txt = F.log_softmax(logits_per_text, dim=1)
                        
                        loss_img = -torch.sum(soft_targets * log_probs_img) / batch_size
                        loss_txt = -torch.sum(soft_targets * log_probs_txt) / batch_size
                        
                        loss = (loss_img + loss_txt) / 2
                        
                        # Scale loss for gradient accumulation
                        loss = loss / gradient_accumulation_steps
                        
                        # Backward pass
                        loss.backward()
                        
                        # Update weights every gradient_accumulation_steps batches
                        if (i + 1) % gradient_accumulation_steps == 0:
                            # Clip gradients
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                            
                            # Update weights
                            optimizer.step()
                            scheduler.step()  # Step the scheduler after each optimizer step
                            optimizer.zero_grad()
                        
                        # Update progress bar
                        train_loss += loss.item() * gradient_accumulation_steps  # Scale back for reporting
                        num_train_batches += 1
                        
                        # Get current learning rate for logging
                        current_lr = optimizer.param_groups[0]['lr']
                        pbar.set_postfix({"loss": loss.item() * gradient_accumulation_steps, 
                                         "lr": f"{current_lr:.2e}"})
                    
                    except Exception as e:
                        print(f"Error in training batch: {e}")
                        # Skip this batch and continue with next one
                        continue
            
            # Final optimizer step for any remaining gradients
            if (i + 1) % gradient_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            
            # Record learning rate for this epoch
            learning_rates.append(optimizer.param_groups[0]['lr'])
            
            # Calculate average training loss (avoid division by zero)
            avg_train_loss = train_loss / max(1, num_train_batches)
            train_losses.append(avg_train_loss)
            
            # Validation
            model.eval()
            val_loss = 0.0
            num_val_batches = 0
            
            with torch.no_grad():
                with tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]") as pbar:
                    for images, texts in pbar:
                        try:
                            # Move to device
                            images = images.to(device)
                            texts = texts.to(device)
                            
                            # Forward pass
                            logits_per_image, logits_per_text = model(images, texts)
                            
                            # Compute loss
                            batch_size = images.size(0)
                            labels = torch.arange(batch_size, device=device)
                            
                            # Use the same label smoothing for consistency
                            smoothing = getattr(args, 'label_smoothing', 0.1)
                            off_diag_scale = smoothing / (batch_size - 1)
                            
                            soft_targets = torch.zeros(batch_size, batch_size, device=device)
                            soft_targets.fill_(off_diag_scale)
                            soft_targets.fill_diagonal_(1.0 - smoothing)
                            
                            log_probs_img = F.log_softmax(logits_per_image, dim=1)
                            log_probs_txt = F.log_softmax(logits_per_text, dim=1)
                            
                            loss_img = -torch.sum(soft_targets * log_probs_img) / batch_size
                            loss_txt = -torch.sum(soft_targets * log_probs_txt) / batch_size
                            
                            loss = (loss_img + loss_txt) / 2
                            
                            # Update validation loss
                            val_loss += loss.item()
                            num_val_batches += 1
                            pbar.set_postfix({"loss": loss.item()})
                        
                        except Exception as e:
                            print(f"Error in validation batch: {e}")
                            # Skip this batch and continue with next one
                            continue
            
            # Calculate average validation loss (avoid division by zero)
            avg_val_loss = val_loss / max(1, num_val_batches)
            val_losses.append(avg_val_loss)
            
            # Print epoch summary
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Save checkpoint
            checkpoint_path = os.path.join(args.output_dir, f"clip_checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, checkpoint_path)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = os.path.join(args.output_dir, "clip_best_model.pt")
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved to {best_model_path}")
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Save final model regardless of how training ended
        final_model_path = os.path.join(args.output_dir, "clip_final_model.pt")
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        # Plot training and validation loss if we have any data
        if train_losses:
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Training Loss')
            if val_losses:
                plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('CLIP Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            
            loss_plot_path = os.path.join(args.output_dir, "loss_plot.png")
            plt.savefig(loss_plot_path)
            plt.close()
            print(f"Loss plot saved to {loss_plot_path}")
            
            # Plot learning rate curve
            plt.figure(figsize=(10, 5))
            plt.plot(learning_rates)
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.grid(True)
            plt.savefig(os.path.join(args.output_dir, "learning_rate_plot.png"))
            plt.close()
            print(f"Learning rate plot saved to {os.path.join(args.output_dir, 'learning_rate_plot.png')}")
    
    # Return dataset for evaluation
    return {
        'final_model_path': final_model_path,
        'best_model_path': best_model_path if 'best_model_path' in locals() else final_model_path,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_dataset': test_dataset
    }


def main():
    parser = argparse.ArgumentParser(description="Train CLIP model on various datasets")
    
    # Data parameters
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["oxford_pets", "caltech101", "stanford_cars", "flowers102", 
                                 "food101", "fgvc_aircraft", "sun397", "dtd", "eurosat", "ucf101", "all"],
                        help="Dataset to train on")
    parser.add_argument("--data_dir", type=str, default="./data", 
                        help="Directory to store the dataset")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction of training data to use for validation")
    parser.add_argument("--val_from_train", action="store_true",
                        help="Use a portion of training data for validation")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, default="ViT-B/32", 
                        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-S/32", "toy"],
                        help="CLIP model variant to use")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.2, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.2, help="Fraction of training for warmup")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, 
                        help="Number of steps to accumulate gradients")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing factor for loss function")
    parser.add_argument("--restart_epochs", type=int, default=5,
                        help="Number of epochs between learning rate restarts")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory for models and logs")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train the model
    train_clip_on_dataset(args)


if __name__ == "__main__":
    main()