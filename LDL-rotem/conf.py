from torchvision import transforms
from transforms.affine_transforms import RandomRotate


# Default train batch size
BATCH_SIZE_TR = 2

# Default test batch size
BATCH_SIZE_TS = 2

# Random seed for reproducibility across all modules of LDL the model
RANDOM_SEED = 42

# Path to 1,457 acne images
PATH_IMAGES = '/host_root/fastspace/datasets/external/Acne04/Classification/JPEGImages'

# Path to a trained LDL model
PATH_MODEL = '/host_root/home/rotem/Private/Academic/LDL-rotem/trained_model.npz'

# Path to .txt files with lists of train/test samples
PATH_IMAGE_SETS = '/host_root/fastspace/datasets/external/Acne04/Detection/VOC2007/ImageSets/Main'

# Init a normalization transform to be applied to tr/ts data
TRANSFROM_NORMALIZE = transforms.Normalize(mean=[0.45815152, 0.361242, 0.29348266], std=[0.2814769, 0.226306, 0.20132513])

# Chain together transformations to be applied to test data
TRANSFORM_TS = transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   TRANSFROM_NORMALIZE])

# Chain together transformations to be applied to train data
TRANSFORM_TR = transforms.Compose([transforms.Resize((256, 256)),
                                   transforms.RandomCrop(224),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   RandomRotate(rotation_range=20),
                                   TRANSFROM_NORMALIZE])
