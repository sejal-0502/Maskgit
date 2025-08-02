import sys
sys.path.append('../')
import os
import io
import argparse
import importlib

import yaml
import random
import PIL
from PIL import Image
from scipy.linalg import sqrtm
from PIL import ImageDraw, ImageFont
import numpy as np
# also disable grad to save memory
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

import torch
torch.set_grad_enabled(False)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torchvision.utils import save_image
from torch.utils.data import Dataset, Subset
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models import inception_v3

import matplotlib.pyplot as plt
from omegaconf import OmegaConf
# from taming.models.vqgan import VQModel, GumbelVQ
# from taming.models.vqgan_with_entropy_loss import VQModel2WithEntropyLoss

from pytorch_fid.fid_score import calculate_fid_given_paths

try:
  sys.path.append('../Depth-Anything')
  from depth_anything.dpt import DepthAnything
  from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
except ImportError:
  print("Depth-Anything not found")

def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    print("Target is : ", config["target"])
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def load_maskgit(config, ckpt_path=None, is_gumbel=False):

  model = instantiate_from_config(config.model)
  # print(f"Model: {model.__class__.__name__} with config: {config.model}")
  # print("Hence Model : ", model)

  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    # remove weights that contain "dino"
    sd = {k: v for k, v in sd.items() if "dino" not in k}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # print("missing keys:", missing)
    # print("unexpected keys:", unexpected)
  return model

def preprocess_maskgit(x):
  x = 2.*x - 1.
  return x

def unnormalize_maskgit(x):
   if isinstance(x, torch.Tensor):
     x = x.cpu().detach().numpy()
   image = ((x+1)*127.5).astype(np.uint8)
   return image

def generate_with_maskgit(input_indices, model):
    # could also use model(x) for reconstruction but use explicit encoding and decoding here
    # print("Input Indices shape:", input_indices.shape)
    x, code_inds, mask_inds = model.sample()
    return x, code_inds, mask_inds

def preprocess(img, target_image_size=256, map_dalle=True):
    img = PIL.Image.open(img)
    s = min(img.size)

    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')

    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    if map_dalle:
      img = map_pixels(img)
    return img

def generation_pipeline(model, image, size):
  if len(image.shape) == 3:
    x_maskgit = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    # x_maskgit = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    # print("Generation pipeline shape :", x_maskgit.shape)
  else:
    x_maskgit = image.to(DEVICE)
  generated, code_indices, mask_codes = generate_with_maskgit(x_maskgit, model)
  # print("Generated 1 shape:", generated.shape)
  generated = generated[0].cpu().permute(1, 2, 0)
  # generated = generated[0].cpu().permute(0, 1, 2)
  # print("Generated 2 shape:", generated.shape)
  generated = ((generated+1)*127.5).clamp_(0, 255).numpy().astype(np.uint8)
  # print("Generated 3 shape:", generated.shape)
  return generated, code_indices, mask_codes

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

def get_inception_model():
    model = inception_v3(pretrained=True)
    model.fc = torch.nn.Identity()  # Remove the classification layer
    model = model.to(DEVICE)
    return model

class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        self.image_filenames.sort()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_filenames[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

class ImageRFIDDataset(ImageFolderDataset):
    def __init__(self, folder_path, generated_folder, transform=None):
        super().__init__(folder_path, transform)
        self.generated_folder = generated_folder

    def __getitem__(self, idx):
        orig_path = os.path.join(self.folder_path, self.image_filenames[idx])
        gen_path = os.path.join(self.generated_folder, self.image_filenames[idx])
        orig_image = Image.open(orig_path).convert('RGB')
        gen_image = Image.open(gen_path).convert('RGB')
        if self.transform is not None:
            orig_image = self.transform(orig_image)
            gen_image = self.transform(gen_image)
        return orig_image, gen_image
    
# Function to calculate the FID score
def calculate_fid(act1, act2):
    # Calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    # Calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)

    # Calculate sqrt of product between covariances
    covmean = sqrtm(sigma1.dot(sigma2))
    
    # Check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
       
# Function to get the features from images using Inception model
def get_inception_features(images, model, device, transform):
    model.eval()
    features = []

    with torch.no_grad():
        for img in tqdm(images):
            img = img.to(device).unsqueeze(0)
            feature = model(img)[0]
            features.append(feature.cpu().numpy())

    features = np.vstack(features)
    return features

def compute_rFID_score(model, path_original_images, path_recons_images):
    
    # Load the Inception model
    print(">> Loading Inception Model...")
    inception_model = get_inception_model()
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])

    original_images = ImageFolderDataset(folder_path=path_original_images, transform=transform)

    reconstructed_images = ImageFolderDataset(folder_path=path_recons_images, transform=transform)
    reconstructed_images.image_filenames = original_images.image_filenames  # let's make sure they match...
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if True:
      fid_score = calculate_fid_given_paths([path_original_images, path_recons_images], 16, DEVICE, 2048)
      # print("--------------------This fid is implemented")
    else:
      print(">> Getting Inception Features...")
      original_features = get_inception_features(original_images, inception_model, DEVICE, transform)
      recontructed_features = get_inception_features(reconstructed_images, inception_model, DEVICE, transform)

      print(">> Calculating FID Score...")
      fid_score = calculate_fid(original_features, recontructed_features)

    print("FID Score:", fid_score)


def process_images(model, dataset, num_images, codebook_size, exp_dir):
    """
    Process the images in the input folder.
    :param input_folder: The folder containing the images.
    :param num_images: The number of images to process.
    :param codebook_size: The size of the codebook.
    """

    if args.compute_rFID_score:
      # prepare folders
      path_original_images = os.path.join(args.exp_dir, "original_images")
      path_gen_images = os.path.join(args.exp_dir, "generated_images")
      
      os.makedirs(path_original_images, exist_ok=True)
      os.makedirs(path_gen_images, exist_ok=True)
      # remove folder contents
      for f in os.listdir(path_original_images):
        os.remove(os.path.join(path_original_images, f))
      for f in os.listdir(path_gen_images):
        os.remove(os.path.join(path_gen_images, f))

    # Randomly select num_images from the list
    selected_images = random.sample(range(len(dataset)), num_images)
    # print("Dataset size:", len(dataset))
    # print("Num Images to process:", num_images)
    # print("Selected images:", selected_images)

     # Generate random colors for each index
    colors = np.random.randint(0, 255, (codebook_size, 3))

    # Process each selected image
    for idx, image_idx in enumerate(selected_images):
        if idx%100==0:
            print(f'{idx} images processed')
        # input_path = os.path.join(input_folder, image_name)
        
        sample = dataset[image_idx]
        # print("Sample type:", type(sample))
        # print("Sample shape:", sample.shape if isinstance(sample, torch.Tensor) else None)

        image = sample.permute(1, 2, 0)

        # Modify the image
        generated, code_indices, mask_codes = generation_pipeline(model, image, size=256)
        # print("Generated process_img shape:", generated.shape)
        
        if args.compute_rFID_score:
          # save original images for rFID score TODO use orig image names
          # save_image(x_vqgan[0], os.path.join(path_original_images, f"original_image_{image_idx}.jpg"))
          # print("Original image shape:", image.shape)
          plt.imsave(os.path.join(path_original_images, f"image_{image_idx}.png"), unnormalize_maskgit(image))
          # print("Generated image shape:", generated.shape)
          plt.imsave(os.path.join(path_gen_images, f"image_{image_idx}.png"), generated)
          
    if args.compute_rFID_score:      
      compute_rFID_score(model, path_original_images, path_gen_images)

    #plot_histogram(histo)
    # print(100*sum([int(h!=0) for h in histo]) / len(histo), '% codebook usage')

def main(args):
  config = load_config(args.config_path, display=False) #99.85% zeros 
  # print(f"Checkpoint exists? {os.path.exists(args.ckpt_path)}")
  model = load_maskgit(config, ckpt_path=args.ckpt_path)
  # print("Loaded Model : ", model)
  model = model.to(DEVICE)
  try:
    codebook_size = config.model.params.quantizer_config.params.get("n_e", args.codebook_size)
  except:
    codebook_size = config.model.params.get("n_embed", args.codebook_size)
  
  # dataset
  if args.data_config is not None:
    data_config = load_config(args.data_config, display=False)
    data = instantiate_from_config(data_config.data)
  else:
    data = instantiate_from_config(config.data)
  data.prepare_data()
  data.setup()
  data = data.datasets['validation']
    
  process_images(model, data, num_images=args.num_images, codebook_size=codebook_size, exp_dir=args.exp_dir)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--exp_name", type=str, default=None, help="Experiment name")
  parser.add_argument("--config_path", type=str, default=None, help="Path to the config file")
  parser.add_argument("--ckpt_path", type=str, default=None, help="Path to the checkpoint file")

  # (optional) data config
  parser.add_argument("--data_config", type=str, default=None, help="Path to the data config file")

  parser.add_argument("--input_folder", type=str, default="./datasets/BDD100K/bdd100k/images/100k/test/", help="Path to input data folder")
  parser.add_argument("--create_index_visualization", action="store_true", help="Create index visualization")
  parser.add_argument("--cluster_indices", action="store_true", help="Cluster indices")
  parser.add_argument("--save_patches_by_index", action="store_true", help="Save patches by index")
  parser.add_argument("--compute_rFID_score", action="store_true", help="Compute rFID score")
  parser.add_argument("--num_images", type=int, default=1000, help="Number of images to process")
  parser.add_argument("--codebook_size", type=int, default=1024, help="Number of images to process")
  parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
  args = parser.parse_args()
  
  if (args.config_path is None or args.ckpt_path is None):
     assert args.exp_name is not None, "Please provide the experiment name"
     args.config_path = os.path.join(os.environ['VQ_WORK_DIR'], args.exp_name, "config.yaml")
     args.ckpt_path = os.path.join(os.environ['VQ_WORK_DIR'], args.exp_name, "checkpoints", "last.ckpt")
     if not (os.path.exists(args.config_path) and os.path.exists(args.ckpt_path)):
        args.config_path = os.path.join("./logs", args.exp_name, "config.yaml")
        args.ckpt_path = os.path.join("./logs", args.exp_name, "checkpoints", "last.ckpt")

  try:
    # directory name of config_path
    args.exp_dir = os.path.join(os.environ['VQ_WORK_DIR'], 'visualizations', args.exp_name) #, os.path.basename(os.path.dirname(args.config_path)))
    if not os.path.exists(args.exp_dir):
      os.makedirs(args.exp_dir)
  except:
    args.exp_dir = '/work/dlclarge2/mutakeks-titok/visualizations_gen/default'
  
  print(f"\n> Loading config from: {args.config_path}")
  print(f"> Loading checkpoint from: {args.ckpt_path}")
  print(f"> Saving visualizations to: {args.exp_dir}\n")

  # set seed
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  random.seed(args.seed)

  main(args)