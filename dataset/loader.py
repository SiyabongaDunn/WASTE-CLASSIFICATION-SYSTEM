import os
import pickle
import kagglehub
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from .preprocessing import preprocess_image
from .augmentation import augment_cnn


class GarbageDataset:
    
    def __init__(self, config):
        self.config = config
        self.data_dir = None
        self.class_names = config['dataset']['class_names']
        self.num_classes = config['dataset']['num_classes']
        self.image_size = config['dataset']['image_size']
        self.random_seed = config['dataset']['random_seed']
        self.test_size = config['dataset']['test_size']
        
        self.file_paths = None
        self.labels = None
    
    def load_data(self):
        local_path = self.config.get('dataset', {}).get('local_path', 'data/raw/garbage-classification')
        self.data_dir = os.path.abspath(os.path.expanduser(local_path))
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Dataset path not found! Check your folder at: {self.data_dir}")
            
        return self
    
    def load_paths(self):
        print(f"\nLoading dataset from: {self.data_dir}")
        
        self.file_paths = []
        self.labels = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.exists(class_dir):
                images = [
                    f for f in os.listdir(class_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ]
                self.file_paths.extend([
                    os.path.join(class_name, img_file) for img_file in images
                ])
                self.labels.extend([class_idx] * len(images))
                print(f"  {class_name}: {len(images)} images")
            else:
                print(f"  {class_name}: NOT FOUND")
        
        print(f"\nTotal images loaded: {len(self.file_paths)}")
        return self
    
    def split(self):
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            self.file_paths,
            self.labels,
            test_size=self.test_size,
            stratify=self.labels,
            random_state=self.random_seed
        )
        
        print(f"\n Training pool (80%): {len(train_paths)} images")
        print(f" Test pool (20%): {len(test_paths)} images - ISOLATED")
        
        self._print_distribution(train_labels, "Training pool")
        self._print_distribution(test_labels, "Test pool")
        
        return train_paths, test_paths, train_labels, test_labels
    
    def _print_distribution(self, labels, name):
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\n{name} distribution:")
        for idx, count in zip(unique, counts):
            percentage = count / len(labels) * 100
            print(f"  {self.class_names[idx]}: {count} ({percentage:.1f}%)")
    
    def save_test_set(self, test_paths, test_labels, output_path="data/splits/test_set.pkl"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data = {
            'paths': test_paths,
            'labels': test_labels,
            'class_names': self.class_names,
            'num_classes': self.num_classes
        }
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"\n Test set saved to '{output_path}' - DO NOT USE IN PHASE 1")


def create_dataset(paths, labels, batch_size, data_dir, image_size, num_classes,
                   augment=False, is_vit=False):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    dataset = dataset.map(
        lambda x, y: preprocess_image(x, y, data_dir, image_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    if augment:
        dataset = dataset.map(augment_cnn, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size)
    
    if is_vit:
        dataset = dataset.map(
            lambda x, y: (x, tf.cast(y, tf.int32)),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        dataset = dataset.map(
            lambda x, y: (x, tf.one_hot(y, depth=num_classes)),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset