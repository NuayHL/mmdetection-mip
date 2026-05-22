import os
import shutil
import subprocess
from mmdet.apis import DetInferencer

def batch_vis(image_list, model_name, checkpoint='epoch_12.pth', exp_dir='work_dirs'):
    model_inference = DetInferencer(model=os.path.join(exp_dir, model_name, f'{model_name}.py'),
                                    weights=os.path.join(exp_dir, model_name, checkpoint))

    model_inference(image_list, batch_size=32, show=False, out_dir=f'visualization_interpIoU/{model_name}',
                    pred_score_thr=0.25)


def merge_images(work_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for model_folder in os.listdir(work_dir):
        model_path = os.path.join(work_dir, model_folder)
        vis_path = os.path.join(model_path, "vis")

        if os.path.isdir(model_path) and os.path.isdir(vis_path):
            for img_name in os.listdir(vis_path):
                img_path = os.path.join(vis_path, img_name)
                if os.path.isfile(img_path):
                    new_img_name = f"{os.path.splitext(img_name)[0]}_{model_folder}{os.path.splitext(img_name)[1]}"
                    new_img_path = os.path.join(output_dir, new_img_name)
                    shutil.copy(img_path, new_img_path)

if __name__ == "__main__":
    num_img = 1500
    coco_val_img_list = os.listdir('COCO/val2017')

    img_list = [os.path.join('COCO/val2017', img) for img in coco_val_img_list[:num_img]]
    # img_list = ['Img/JK.jpg', 'Img/coco_1.jpg']

    batch_vis(img_list, model_name='dino-4scale_r50_1xb8-12e_interpiou_coco')
    batch_vis(img_list, model_name='dino-4scale_r50_1xb8-12e_piou_coco')
    batch_vis(img_list, model_name='dino-4scale_r50_1xb8-12e_ciou_coco')
    batch_vis(img_list, model_name='dino-4scale_r50_1xb8-12e_coco')

    merge_images('visualization_interpIoU', 'visualization_interpIoU')

    # subprocess.run([
    #     "python", "tools/analysis_tools/browse_dataset.py",
    #     "configs/dino/inspection_val_coco.py",
    #     "--output-dir", "visualization_interpIoU/gt/",
    #     "--not-show"
    # ], check=True)
    
    gt_path = os.path.join('visualization_interpIoU', 'gt')
    for img_name in os.listdir(gt_path):
        img_path = os.path.join(gt_path, img_name)
        if os.path.isfile(img_path):
            new_img_name = f"{os.path.splitext(img_name)[0]}_{'gt'}{os.path.splitext(img_name)[1]}"
            new_img_path = os.path.join('visualization_interpIoU', new_img_name)
            shutil.copy(img_path, new_img_path)

    target_folder = 'visualization_interpIoU/comparison'
    os.makedirs(target_folder, exist_ok=True)
    files_in_folder = os.listdir('visualization_interpIoU')
    used_img = set()
    undet = dict()
    for file in files_in_folder:
        if os.path.isdir(os.path.join('visualization_interpIoU',file)):
            continue
        prefix, suffix = file.split('_', 1)
        if suffix != 'gt.jpg':
            used_img.add(prefix)
            shutil.move(os.path.join('visualization_interpIoU',file), os.path.join(target_folder, file))
            if prefix in undet.keys():
                shutil.move(os.path.join('visualization_interpIoU', undet[prefix]), os.path.join(target_folder, undet[prefix]))
                del undet[prefix]
        else:
            if prefix in used_img:
                shutil.move(os.path.join('visualization_interpIoU', file),
                            os.path.join(target_folder, file))
            else:
                undet[prefix] = file

    files_in_folder = os.listdir('visualization_interpIoU')
    for file in files_in_folder:
        file_name = os.path.join('visualization_interpIoU',file)
        if not os.path.isdir(file_name):
            os.remove(file_name)

