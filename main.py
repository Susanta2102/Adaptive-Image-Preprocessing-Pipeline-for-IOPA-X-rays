import os
import sys
import numpy as np
import cv2
import pydicom

# Configure matplotlib for non-interactive backend by default
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
import pandas as pd
from scipy import ndimage
from skimage import filters, restoration, measure, exposure
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ImageQualityAnalyzer:
    """Analyzes various image quality metrics for IOPA X-rays."""
    
    def __init__(self):
        self.metrics_cache = {}
    
    def normalize_to_8bit(self, image):
        """Normalize image to 8-bit range."""
        if image.dtype != np.uint8:
            image_min, image_max = np.min(image), np.max(image)
            if image_max > image_min:
                image = ((image - image_min) / (image_max - image_min) * 255).astype(np.uint8)
            else:
                image = np.zeros_like(image, dtype=np.uint8)
        return image
    
    def calculate_brightness(self, image):
        """Calculate brightness using mean pixel intensity."""
        return np.mean(image)
    
    def calculate_contrast(self, image):
        """Calculate multiple contrast metrics."""
        # RMS Contrast
        rms_contrast = np.std(image)
        
        # Michelson Contrast
        max_val, min_val = np.max(image), np.min(image)
        if max_val + min_val > 0:
            michelson_contrast = (max_val - min_val) / (max_val + min_val)
        else:
            michelson_contrast = 0
        
        return {
            'rms_contrast': rms_contrast,
            'michelson_contrast': michelson_contrast
        }
    
    def calculate_sharpness(self, image):
        """Calculate sharpness using multiple methods."""
        # Ensure image is in proper format for OpenCV operations
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Convert to uint8 if not already
        if image.dtype != np.uint8:
            image = self.normalize_to_8bit(image)
        
        # Laplacian variance
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        laplacian_var = laplacian.var()
        
        # Tenengrad (Sobel gradient magnitude)
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = np.mean(sobel_x**2 + sobel_y**2)
        
        return {
            'laplacian_variance': laplacian_var,
            'tenengrad': tenengrad
        }
    
    def calculate_noise_level(self, image):
        """Estimate noise level using multiple methods."""
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = self.normalize_to_8bit(image)
        
        # Wavelet-based noise estimation (simplified)
        try:
            # Use median absolute deviation in high-frequency components
            gaussian = cv2.GaussianBlur(image, (3, 3), 0)
            noise_estimate = np.median(np.abs(image.astype(np.float32) - gaussian.astype(np.float32))) / 0.6745
        except:
            noise_estimate = np.std(image) * 0.1  # Fallback
        
        # Standard deviation in flat regions (simplified)
        try:
            # Use morphological operations to find flat regions
            kernel = np.ones((5, 5), np.uint8)
            eroded = cv2.erode(image, kernel, iterations=1)
            dilated = cv2.dilate(image, kernel, iterations=1)
            flat_mask = (dilated - eroded) < 10  # Flat regions
            
            if np.sum(flat_mask) > 100:  # Ensure we have enough flat pixels
                noise_std = np.std(image[flat_mask])
            else:
                noise_std = np.std(image) * 0.2  # Fallback
        except:
            noise_std = np.std(image) * 0.2  # Fallback
        
        return {
            'wavelet_noise_estimate': noise_estimate,
            'flat_region_noise': noise_std
        }
    
    def analyze_image(self, image, image_id=None):
        """Comprehensive image quality analysis."""
        if image_id and image_id in self.metrics_cache:
            return self.metrics_cache[image_id]
        
        metrics = {
            'brightness': self.calculate_brightness(image),
            'contrast': self.calculate_contrast(image),
            'sharpness': self.calculate_sharpness(image),
            'noise': self.calculate_noise_level(image)
        }
        
        if image_id:
            self.metrics_cache[image_id] = metrics
        
        return metrics

class DICOMHandler:
    """Handles DICOM and RVG file reading and processing."""
    
    def __init__(self):
        self.supported_formats = ['.dcm', '.rvg', '.jpg', '.png', '.tiff', '.bmp']
    
    def read_dicom_file(self, filepath):
        """Read DICOM file and extract pixel data and metadata."""
        try:
            dicom_data = pydicom.dcmread(filepath)
            
            # Extract pixel array
            pixel_array = dicom_data.pixel_array
            
            # Handle different photometric interpretations
            if hasattr(dicom_data, 'PhotometricInterpretation'):
                if dicom_data.PhotometricInterpretation == 'MONOCHROME1':
                    # Invert for proper display
                    pixel_array = np.max(pixel_array) - pixel_array
            
            # Normalize to 8-bit
            pixel_array = self.normalize_to_8bit(pixel_array)
            
            # Extract relevant metadata
            metadata = self.extract_metadata(dicom_data)
            
            return pixel_array, metadata
            
        except Exception as e:
            print(f"Error reading DICOM file {filepath}: {str(e)}")
            return None, None
    
    def read_image_file(self, filepath):
        """Read standard image files."""
        try:
            filepath_str = str(filepath)  # Convert Path object to string
            if filepath_str.lower().endswith('.rvg'):
                # RVG files are often proprietary, try reading as binary
                return self.read_rvg_file(filepath)
            else:
                image = cv2.imread(filepath_str, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    # Try with PIL
                    pil_image = Image.open(filepath).convert('L')
                    image = np.array(pil_image)
                return image, {}
        except Exception as e:
            print(f"Error reading image file {filepath}: {str(e)}")
            return None, None
    
    def read_rvg_file(self, filepath):
        """Attempt to read RVG files (proprietary format)."""
        try:
            # RVG files might be readable as DICOM
            return self.read_dicom_file(filepath)
        except:
            try:
                # Try reading as raw binary data (this is speculative)
                with open(filepath, 'rb') as f:
                    data = f.read()
                # This is a simplified approach - actual RVG format would need specific handling
                print(f"Warning: RVG file {filepath} format not fully supported. Skipping.")
                return None, None
            except Exception as e:
                print(f"Error reading RVG file {filepath}: {str(e)}")
                return None, None
    
    def normalize_to_8bit(self, image):
        """Normalize image to 8-bit range."""
        if image.dtype != np.uint8:
            image_min, image_max = np.min(image), np.max(image)
            if image_max > image_min:
                image = ((image - image_min) / (image_max - image_min) * 255).astype(np.uint8)
            else:
                image = np.zeros_like(image, dtype=np.uint8)
        return image
    
    def extract_metadata(self, dicom_data):
        """Extract relevant metadata from DICOM."""
        metadata = {}
        
        fields_of_interest = [
            'Modality', 'SOPClassUID', 'PhotometricInterpretation',
            'PixelSpacing', 'SliceThickness', 'KVP', 'Exposure',
            'StudyDate', 'Manufacturer', 'ManufacturerModelName',
            'Rows', 'Columns', 'BitsAllocated', 'BitsStored'
        ]
        
        for field in fields_of_interest:
            try:
                if hasattr(dicom_data, field):
                    value = getattr(dicom_data, field)
                    metadata[field] = str(value) if not isinstance(value, (int, float)) else value
            except:
                continue
        
        return metadata

class StaticPreprocessor:
    """Static preprocessing baseline pipeline."""
    
    def __init__(self):
        self.name = "Static Preprocessor"
    
    def preprocess(self, image):
        """Apply static preprocessing steps."""
        processed = image.copy()
        
        # Step 1: Histogram equalization
        processed = cv2.equalizeHist(processed)
        
        # Step 2: Gaussian denoising
        processed = cv2.GaussianBlur(processed, (3, 3), 0)
        
        # Step 3: Unsharp masking for sharpening
        gaussian = cv2.GaussianBlur(processed, (0, 0), 2.0)
        processed = cv2.addWeighted(processed, 1.5, gaussian, -0.5, 0)
        
        # Ensure values are in valid range
        processed = np.clip(processed, 0, 255).astype(np.uint8)
        
        return processed

class AdaptivePreprocessor:
    """Adaptive preprocessing pipeline that adjusts based on image characteristics."""
    
    def __init__(self, quality_analyzer):
        self.quality_analyzer = quality_analyzer
        self.name = "Adaptive Preprocessor"
        
        # Define thresholds for different quality categories
        self.brightness_thresholds = {'low': 50, 'high': 200}
        self.contrast_thresholds = {'low': 30, 'high': 80}
        self.sharpness_thresholds = {'low': 100, 'high': 500}
        self.noise_thresholds = {'low': 5, 'high': 20}
    
    def classify_image_quality(self, metrics):
        """Classify image into quality categories."""
        brightness = metrics['brightness']
        contrast = metrics['contrast']['rms_contrast']
        sharpness = metrics['sharpness']['laplacian_variance']
        noise = metrics['noise']['wavelet_noise_estimate']
        
        classification = {
            'brightness_level': self._classify_metric(brightness, self.brightness_thresholds),
            'contrast_level': self._classify_metric(contrast, self.contrast_thresholds),
            'sharpness_level': self._classify_metric(sharpness, self.sharpness_thresholds),
            'noise_level': self._classify_metric(noise, self.noise_thresholds)
        }
        
        return classification
    
    def _classify_metric(self, value, thresholds):
        """Classify a metric value into low/medium/high categories."""
        if value < thresholds['low']:
            return 'low'
        elif value > thresholds['high']:
            return 'high'
        else:
            return 'medium'
    
    def preprocess(self, image, image_id=None):
        """Apply adaptive preprocessing based on image characteristics."""
        # Analyze image quality
        metrics = self.quality_analyzer.analyze_image(image, image_id)
        classification = self.classify_image_quality(metrics)
        
        processed = image.copy().astype(np.float32)
        
        # Adaptive brightness/contrast adjustment
        processed = self._adaptive_contrast_enhancement(processed, classification)
        
        # Adaptive noise reduction
        processed = self._adaptive_noise_reduction(processed, classification)
        
        # Adaptive sharpening
        processed = self._adaptive_sharpening(processed, classification)
        
        # Normalize back to uint8
        processed = np.clip(processed, 0, 255).astype(np.uint8)
        
        return processed, metrics, classification
    
    def _adaptive_contrast_enhancement(self, image, classification):
        """Apply adaptive contrast enhancement."""
        if classification['contrast_level'] == 'low':
            # Low contrast - apply strong CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            return clahe.apply(image.astype(np.uint8)).astype(np.float32)
        elif classification['contrast_level'] == 'high':
            # High contrast - mild enhancement
            return exposure.adjust_gamma(image/255.0, gamma=0.8) * 255
        else:
            # Medium contrast - moderate CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            return clahe.apply(image.astype(np.uint8)).astype(np.float32)
    
    def _adaptive_noise_reduction(self, image, classification):
        """Apply adaptive noise reduction."""
        if classification['noise_level'] == 'high':
            # High noise - strong denoising
            return cv2.bilateralFilter(image.astype(np.uint8), 9, 75, 75).astype(np.float32)
        elif classification['noise_level'] == 'medium':
            # Medium noise - moderate denoising
            return cv2.bilateralFilter(image.astype(np.uint8), 5, 50, 50).astype(np.float32)
        else:
            # Low noise - minimal denoising
            return cv2.GaussianBlur(image, (3, 3), 0.5)
    
    def _adaptive_sharpening(self, image, classification):
        """Apply adaptive sharpening."""
        if classification['sharpness_level'] == 'low':
            # Low sharpness - strong sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            return cv2.filter2D(image, -1, kernel)
        elif classification['sharpness_level'] == 'high':
            # Already sharp - no sharpening
            return image
        else:
            # Medium sharpness - mild sharpening
            gaussian = cv2.GaussianBlur(image, (0, 0), 1.0)
            return cv2.addWeighted(image, 1.2, gaussian, -0.2, 0)

class ImageEvaluator:
    """Evaluates and compares preprocessing results."""
    
    def __init__(self):
        self.evaluation_results = []
    
    def calculate_psnr(self, original, processed):
        """Calculate Peak Signal-to-Noise Ratio."""
        mse = np.mean((original - processed) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    def calculate_ssim(self, original, processed):
        """Calculate Structural Similarity Index."""
        from skimage.metrics import structural_similarity as ssim
        return ssim(original, processed, data_range=255)
    
    def calculate_edge_preservation(self, original, processed):
        """Calculate how well edges are preserved."""
        edges_original = cv2.Canny(original, 50, 150)
        edges_processed = cv2.Canny(processed, 50, 150)
        
        # Calculate edge similarity
        edge_similarity = np.mean(edges_original == edges_processed)
        return edge_similarity
    
    def evaluate_preprocessing(self, original, static_result, adaptive_result, image_id):
        """Comprehensive evaluation of preprocessing results."""
        results = {
            'image_id': image_id,
            'static_psnr': self.calculate_psnr(original, static_result),
            'adaptive_psnr': self.calculate_psnr(original, adaptive_result),
            'static_ssim': self.calculate_ssim(original, static_result),
            'adaptive_ssim': self.calculate_ssim(original, adaptive_result),
            'static_edge_preservation': self.calculate_edge_preservation(original, static_result),
            'adaptive_edge_preservation': self.calculate_edge_preservation(original, adaptive_result)
        }
        
        self.evaluation_results.append(results)
        return results

class VisualizationManager:
    """Handles visualization and reporting."""
    
    def __init__(self, output_dir='results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def plot_quality_metrics_distribution(self, all_metrics, save=True, show=False):
        """Plot distribution of quality metrics across dataset."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Distribution of Image Quality Metrics', fontsize=16)
        
        # Extract metrics for plotting
        brightness_values = [m['brightness'] for m in all_metrics.values()]
        contrast_values = [m['contrast']['rms_contrast'] for m in all_metrics.values()]
        sharpness_values = [m['sharpness']['laplacian_variance'] for m in all_metrics.values()]
        noise_values = [m['noise']['wavelet_noise_estimate'] for m in all_metrics.values()]
        
        # Plot distributions
        axes[0,0].hist(brightness_values, bins=20, alpha=0.7, color='blue')
        axes[0,0].set_title('Brightness Distribution')
        axes[0,0].set_xlabel('Mean Pixel Intensity')
        
        axes[0,1].hist(contrast_values, bins=20, alpha=0.7, color='green')
        axes[0,1].set_title('Contrast Distribution')
        axes[0,1].set_xlabel('RMS Contrast')
        
        axes[1,0].hist(sharpness_values, bins=20, alpha=0.7, color='red')
        axes[1,0].set_title('Sharpness Distribution')
        axes[1,0].set_xlabel('Laplacian Variance')
        
        axes[1,1].hist(noise_values, bins=20, alpha=0.7, color='orange')
        axes[1,1].set_title('Noise Distribution')
        axes[1,1].set_xlabel('Estimated Noise Level')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'quality_metrics_distribution.png', dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()  # Close the figure to free memory
    
    def create_comparison_visualization(self, original, static_result, adaptive_result, 
                                     metrics, classification, image_id, save=True, show=False):
        """Create side-by-side comparison visualization."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Preprocessing Comparison - {image_id}', fontsize=14)
        
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(static_result, cmap='gray')
        axes[1].set_title('Static Preprocessing')
        axes[1].axis('off')
        
        axes[2].imshow(adaptive_result, cmap='gray')
        axes[2].set_title('Adaptive Preprocessing')
        axes[2].axis('off')
        
        # Add quality metrics as text
        metrics_text = f"""
        Brightness: {metrics['brightness']:.1f}
        Contrast: {metrics['contrast']['rms_contrast']:.1f}
        Sharpness: {metrics['sharpness']['laplacian_variance']:.1f}
        Noise: {metrics['noise']['wavelet_noise_estimate']:.1f}
        
        Classification:
        B: {classification['brightness_level']}
        C: {classification['contrast_level']}
        S: {classification['sharpness_level']}
        N: {classification['noise_level']}
        """
        
        fig.text(0.02, 0.02, metrics_text, fontsize=8, verticalalignment='bottom')
        
        if save:
            plt.savefig(self.output_dir / f'comparison_{image_id}.png', dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()  # Close the figure to free memory
    
    def plot_evaluation_results(self, evaluation_results, save=True, show=False):
        """Plot evaluation metrics comparison."""
        df = pd.DataFrame(evaluation_results)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # PSNR comparison
        x = range(len(df))
        axes[0].bar([i-0.2 for i in x], df['static_psnr'], width=0.4, label='Static', alpha=0.7)
        axes[0].bar([i+0.2 for i in x], df['adaptive_psnr'], width=0.4, label='Adaptive', alpha=0.7)
        axes[0].set_title('PSNR Comparison')
        axes[0].set_ylabel('PSNR (dB)')
        axes[0].set_xlabel('Image Index')
        axes[0].legend()
        
        # SSIM comparison
        axes[1].bar([i-0.2 for i in x], df['static_ssim'], width=0.4, label='Static', alpha=0.7)
        axes[1].bar([i+0.2 for i in x], df['adaptive_ssim'], width=0.4, label='Adaptive', alpha=0.7)
        axes[1].set_title('SSIM Comparison')
        axes[1].set_ylabel('SSIM')
        axes[1].set_xlabel('Image Index')
        axes[1].legend()
        
        # Edge Preservation comparison
        axes[2].bar([i-0.2 for i in x], df['static_edge_preservation'], width=0.4, label='Static', alpha=0.7)
        axes[2].bar([i+0.2 for i in x], df['adaptive_edge_preservation'], width=0.4, label='Adaptive', alpha=0.7)
        axes[2].set_title('Edge Preservation Comparison')
        axes[2].set_ylabel('Edge Similarity')
        axes[2].set_xlabel('Image Index')
        axes[2].legend()
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'evaluation_comparison.png', dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()  # Close the figure to free memory

class AdaptiveIOPAPreprocessor:
    """Main class that orchestrates the entire adaptive preprocessing pipeline."""
    
    def __init__(self, data_directory='data', output_directory='results'):
        self.data_dir = Path(data_directory)
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.dicom_handler = DICOMHandler()
        self.quality_analyzer = ImageQualityAnalyzer()
        self.static_preprocessor = StaticPreprocessor()
        self.adaptive_preprocessor = AdaptivePreprocessor(self.quality_analyzer)
        self.evaluator = ImageEvaluator()
        self.visualizer = VisualizationManager(self.output_dir)
        
        # Storage for results
        self.images = {}
        self.metadata = {}
        self.quality_metrics = {}
        self.preprocessing_results = {}
        self.evaluation_results = []
        
        # Display preferences
        self.show_plots = False  # Default to non-interactive mode
    
    def load_dataset(self):
        """Load all images from the data directory."""
        print("Loading dataset...")
        
        supported_extensions = ['.dcm', '.rvg', '.jpg', '.jpeg', '.png', '.tiff', '.bmp']
        image_files = []
        
        for ext in supported_extensions:
            image_files.extend(list(self.data_dir.glob(f'*{ext}')))
            image_files.extend(list(self.data_dir.glob(f'*{ext.upper()}')))
        
        # Remove duplicates (in case of case-insensitive file systems)
        unique_files = []
        seen_names = set()
        for file_path in image_files:
            if file_path.name.lower() not in seen_names:
                unique_files.append(file_path)
                seen_names.add(file_path.name.lower())
        
        print(f"Found {len(unique_files)} unique image files")
        
        for file_path in unique_files:
            image_id = file_path.stem
            print(f"Loading {file_path.name}...")
            
            if file_path.suffix.lower() in ['.dcm', '.rvg']:
                image, metadata = self.dicom_handler.read_dicom_file(file_path)
            else:
                image, metadata = self.dicom_handler.read_image_file(file_path)
            
            if image is not None:
                self.images[image_id] = image
                self.metadata[image_id] = metadata
                print(f"  ✓ Loaded {image_id}: {image.shape}")
            else:
                print(f"  ✗ Failed to load {image_id}")
        
        print(f"Successfully loaded {len(self.images)} images")
        return len(self.images) > 0
    
    def analyze_dataset_quality(self):
        """Analyze quality metrics for all loaded images."""
        print("\nAnalyzing image quality metrics...")
        
        for image_id, image in self.images.items():
            print(f"Analyzing {image_id}...")
            metrics = self.quality_analyzer.analyze_image(image, image_id)
            self.quality_metrics[image_id] = metrics
        
        # Create visualization of quality distribution
        self.visualizer.plot_quality_metrics_distribution(self.quality_metrics, 
                                                         save=True, show=self.show_plots)
        
        # Save metrics to JSON
        metrics_for_json = {}
        for img_id, metrics in self.quality_metrics.items():
            metrics_for_json[img_id] = {
                'brightness': float(metrics['brightness']),
                'rms_contrast': float(metrics['contrast']['rms_contrast']),
                'michelson_contrast': float(metrics['contrast']['michelson_contrast']),
                'laplacian_variance': float(metrics['sharpness']['laplacian_variance']),
                'tenengrad': float(metrics['sharpness']['tenengrad']),
                'wavelet_noise_estimate': float(metrics['noise']['wavelet_noise_estimate']),
                'flat_region_noise': float(metrics['noise']['flat_region_noise'])
            }
        
        with open(self.output_dir / 'quality_metrics.json', 'w') as f:
            json.dump(metrics_for_json, f, indent=2)
        
        print("Quality analysis complete!")
    
    def run_preprocessing_pipeline(self):
        """Run both static and adaptive preprocessing on all images."""
        print("\nRunning preprocessing pipeline...")
        
        for image_id, original_image in self.images.items():
            print(f"Processing {image_id}...")
            
            # Static preprocessing
            static_result = self.static_preprocessor.preprocess(original_image)
            
            # Adaptive preprocessing
            adaptive_result, metrics, classification = self.adaptive_preprocessor.preprocess(
                original_image, image_id
            )
            
            # Store results
            self.preprocessing_results[image_id] = {
                'original': original_image,
                'static': static_result,
                'adaptive': adaptive_result,
                'metrics': metrics,
                'classification': classification
            }
            
            # Evaluate results
            evaluation = self.evaluator.evaluate_preprocessing(
                original_image, static_result, adaptive_result, image_id
            )
            
            # Create comparison visualization
            self.visualizer.create_comparison_visualization(
                original_image, static_result, adaptive_result,
                metrics, classification, image_id, save=True, show=self.show_plots
            )
            
            print(f"  ✓ Processed {image_id}")
        
        print("Preprocessing pipeline complete!")
    
    def evaluate_and_report(self):
        """Generate comprehensive evaluation and report."""
        print("\nGenerating evaluation report...")
        
        # Plot evaluation results
        self.visualizer.plot_evaluation_results(self.evaluator.evaluation_results, 
                                               save=True, show=self.show_plots)
        
        # Calculate summary statistics
        df_results = pd.DataFrame(self.evaluator.evaluation_results)
        
        summary_stats = {
            'static_avg_psnr': df_results['static_psnr'].mean(),
            'adaptive_avg_psnr': df_results['adaptive_psnr'].mean(),
            'static_avg_ssim': df_results['static_ssim'].mean(),
            'adaptive_avg_ssim': df_results['adaptive_ssim'].mean(),
            'static_avg_edge_preservation': df_results['static_edge_preservation'].mean(),
            'adaptive_avg_edge_preservation': df_results['adaptive_edge_preservation'].mean(),
            'psnr_improvement': df_results['adaptive_psnr'].mean() - df_results['static_psnr'].mean(),
            'ssim_improvement': df_results['adaptive_ssim'].mean() - df_results['static_ssim'].mean(),
            'edge_improvement': df_results['adaptive_edge_preservation'].mean() - df_results['static_edge_preservation'].mean()
        }
        
        # Save detailed results
        df_results.to_csv(self.output_dir / 'evaluation_results.csv', index=False)
        
        with open(self.output_dir / 'summary_stats.json', 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Dataset size: {len(self.images)} images")
        print(f"Average PSNR - Static: {summary_stats['static_avg_psnr']:.2f} dB")
        print(f"Average PSNR - Adaptive: {summary_stats['adaptive_avg_psnr']:.2f} dB")
        print(f"PSNR Improvement: {summary_stats['psnr_improvement']:.2f} dB")
        print(f"Average SSIM - Static: {summary_stats['static_avg_ssim']:.3f}")
        print(f"Average SSIM - Adaptive: {summary_stats['adaptive_avg_ssim']:.3f}")
        print(f"SSIM Improvement: {summary_stats['ssim_improvement']:.3f}")
        print(f"Edge Preservation - Static: {summary_stats['static_avg_edge_preservation']:.3f}")
        print(f"Edge Preservation - Adaptive: {summary_stats['adaptive_avg_edge_preservation']:.3f}")
        print(f"Edge Preservation Improvement: {summary_stats['edge_improvement']:.3f}")
        print("="*50)
        
        return summary_stats
    
    def run_full_pipeline(self):
        """Execute the complete adaptive preprocessing pipeline."""
        print("Starting Adaptive IOPA Preprocessing Pipeline")
        print("="*50)
        
        # Step 1: Load dataset
        if not self.load_dataset():
            print("No images loaded. Please check your data directory.")
            return False
        
        # Step 2: Analyze quality metrics
        self.analyze_dataset_quality()
        
        # Step 3: Run preprocessing
        self.run_preprocessing_pipeline()
        
        # Step 4: Evaluate and report
        summary_stats = self.evaluate_and_report()
        
        print("\nPipeline execution complete!")
        print(f"Results saved to: {self.output_dir}")
        
        return True

def main():
    """Main function to run the adaptive preprocessing pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Adaptive IOPA X-ray Preprocessing Pipeline')
    parser.add_argument('--data-dir', default='data', help='Directory containing IOPA images')
    parser.add_argument('--output-dir', default='results', help='Directory to save results')
    parser.add_argument('--show-plots', action='store_true', 
                       help='Show plots interactively (default: False, just save them)')
    
    args = parser.parse_args()
    
    # Configure matplotlib backend based on user preference
    import matplotlib
    if args.show_plots:
        matplotlib.use('TkAgg')  # Interactive backend
        print("Interactive plot display enabled. You'll need to close each plot window.")
    else:
        matplotlib.use('Agg')    # Non-interactive backend
        print("Non-interactive mode. Plots will be saved automatically without display.")
    
    # Initialize and run pipeline
    pipeline = AdaptiveIOPAPreprocessor(args.data_dir, args.output_dir)
    pipeline.show_plots = args.show_plots  # Pass the show_plots preference
    success = pipeline.run_full_pipeline()
    
    if success:
        print("\n🎉 Pipeline completed successfully!")
        print(f"Check the '{args.output_dir}' directory for all results and visualizations.")
        if not args.show_plots:
            print("All plots have been saved as PNG files in the results directory.")
    else:
        print("\n❌ Pipeline failed. Please check your data directory and file formats.")

if __name__ == "__main__":
    main()