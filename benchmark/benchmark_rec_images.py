import os
import sys
import copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from utils.metrics import ImageMetric
from utils.image import read_image
from utils.args import test_options
from utils.config import get_device

def benchmark_rec_images(rec_images_dir, rec_images_ext='png', bin_rec_ext='bits'):
    '''
    Benchmark the reconstruction of images: 
    Input: rec_images_dir
            The directory should have the following structure:
            model_name/dataset/quality
            model_name: the name of the model
            dataset: the name of the dataset
            quality: the quality of the images
    Output: 
            two csv files: results.csv and results_avg.csv
    '''

    args = test_options()
    image_metrics = ImageMetric(get_device())

    original_images_dir = os.path.join(args.main_path, *args.test_dataset.split('/')[:-1])
    model_name = rec_images_dir.split('/')[-1]
    dataset_names = os.listdir(rec_images_dir)

    csv_dir = os.path.join(args.main_path, args.output_dir, 'metrics')
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, model_name + '.csv')
    csv_avg_path = os.path.join(csv_dir, 'average_' + model_name + '.csv')
    # model_name,dataset_name,psnr,ssim, ssim_multiscale, lpips, fid, inception_score_mean, inception_score_std, bpp
    with open(csv_path, 'w') as f:
        f.write('model_name,dataset_name,quality,psnr,ssim,ssim_multiscale,lpips,fid,inception_score_mean,inception_score_std,bpp\n')
    with open(csv_avg_path, 'w') as f:
        f.write('model_name,dataset_name,quality,psnr,ssim,ssim_multiscale,lpips,fid,inception_score_mean,inception_score_std,bpp\n')


    for dataset_name in dataset_names:
        dataset_dir = os.path.join(rec_images_dir, dataset_name)
        quality_names = os.listdir(dataset_dir)
        for quality in quality_names:
            quality_dir = os.path.join(dataset_dir, quality)
            files_quality = os.listdir(quality_dir)
            rec_image_paths = [os.path.join(quality_dir, img) for img in files_quality if img.endswith(rec_images_ext)]
            rec_image_paths.sort()
            metrics_dict = {}
            for rec_image_path in rec_image_paths:
                try:
                    rec_image_name = rec_image_path.split('/')[-1]
                    rec_image_name = rec_image_name.split('.')[0]
                    bin_rec_name = copy.deepcopy(rec_image_name)
                    if rec_image_name.find('_') != -1:
                        rec_image_name = rec_image_name.split('_')[1]
                    original_image_path = os.path.join(original_images_dir, dataset_name, rec_image_name + '.png')
                    original_image = read_image(original_image_path)
                    rec_image = read_image(rec_image_path)
                    bitstream_path = os.path.join(quality_dir, bin_rec_name + '.' + bin_rec_ext)
                    bitstream_size = os.path.getsize(bitstream_path) # in bytes
                    metrics = image_metrics.metric_image(rec_image, original_image, bitstream_size)
                    metrics_dict = image_metrics.append_to_metrics_dict(metrics, metrics_dict)
                    with open(csv_path, 'a') as f:
                        # f.write(f'{model_name},{dataset_name},{quality},{metrics["psnr"]},{metrics["ssim"]},{metrics["msssim"]},{metrics["lpips"]},{metrics["fid"]},{metrics["inception_score_mean"]},{metrics["inception_score_std"]},{metrics["bpp"]}\n')
                        f.write(f'{model_name},{dataset_name},{quality},{metrics["psnr"]},{metrics["ssim"]},{metrics["msssim"]},{metrics["lpips"]},,,,{metrics["bpp"]}\n')
                except Exception as e:
                    print(f'Could not compute metrics for {rec_image_path}: {e}')
            avg_metrics = image_metrics.get_avg_metrics(metrics_dict)
            with open(csv_avg_path, 'a') as f:
                # f.write(f'{model_name},{dataset_name},{quality},{avg_metrics["psnr"]},{avg_metrics["ssim"]},{avg_metrics["msssim"]},{avg_metrics["lpips"]},{avg_metrics["fid"]},{avg_metrics["inception_score_mean"]},{avg_metrics["inception_score_std"]},{avg_metrics["bpp"]}\n')
                f.write(f'{model_name},{dataset_name},{quality},{avg_metrics["psnr"]},{avg_metrics["ssim"]},{avg_metrics["msssim"]},{avg_metrics["lpips"]},,,,{avg_metrics["bpp"]}\n')

            


if __name__ == '__main__':
    rec_images_dir = '/home/ayman/repos/summary/images/jpeg-ai-2025-jan-high'
    rec_images_dir = '/home/ayman/repos/summary/images/jpeg-ai-2025-jan-simple'
    rec_images_dir = '/home/ayman/repos/summary/images/jpeg-ai-2025-jan-main'
    benchmark_rec_images(rec_images_dir)

                

            
    
    

