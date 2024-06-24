import os
import cv2
import torch
import time
import cuda_malloc
import folder_paths
import numpy as np
from PIL import Image
from pathlib import Path

from datetime import datetime
from omegaconf import OmegaConf
from torchvision import transforms
from diffusers import AutoencoderKL, DDIMScheduler
from huggingface_hub import snapshot_download,hf_hub_download
from transformers import CLIPVisionModelWithProjection

from ViViD.src.models.pose_guider import PoseGuider
from ViViD.src.models.unet_3d import UNet3DConditionModel
from ViViD.src.models.unet_2d_condition import UNet2DConditionModel
from ViViD.src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
from ViViD.src.utils.util import get_fps, read_frames, save_videos_grid

from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.utils_ootd import get_mask_location

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from densepose import add_densepose_config
from densepose.vis.extractor import DensePoseResultExtractor
from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer as Visualizer

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

now_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = folder_paths.get_output_directory()
input_dir = folder_paths.get_input_directory()

ckpt_dir = os.path.join(now_dir,"ckpts")
pretrained_vae_path = os.path.join(ckpt_dir,"sd-vae-ft-mse")
snapshot_download(repo_id="stabilityai/sd-vae-ft-mse",local_dir=pretrained_vae_path,allow_patterns=['*.json',"*.safetensors"])

pretrained_base_model_path = os.path.join(ckpt_dir, "sd-image-variations-diffusers")
snapshot_download(repo_id="lambdalabs/sd-image-variations-diffusers",local_dir=pretrained_base_model_path)

motion_module_path = os.path.join(ckpt_dir, "mm_sd_v15_v2.ckpt")
hf_hub_download(repo_id="guoyww/animatediff",filename="mm_sd_v15_v2.ckpt",local_dir=ckpt_dir)

image_encoder_path = os.path.join(pretrained_base_model_path,"image_encoder")


denoising_unet_path = os.path.join(ckpt_dir,"ViViD","denoising_unet.pth")
reference_unet_path = os.path.join(ckpt_dir,"ViViD","reference_unet.pth")
pose_guider_path = os.path.join(ckpt_dir,"ViViD","pose_guider.pth")

snapshot_download(repo_id="alibaba-yuanjing-aigclab/ViViD",local_dir=os.path.join(ckpt_dir,"ViViD"))


sam_model_path = os.path.join(ckpt_dir,"sam_vit_b_01ec64.pth")
hf_hub_download(repo_id="ybelkada/segment-anything",filename="sam_vit_b_01ec64.pth",subfolder="checkpoints",local_dir=ckpt_dir)


vae = None

class ViViD_Node:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "cloth_image_path": ("IMAGEPATH",),
                "model_video_path": ("VIDEO",),
                "category":( ['upper_body', 'lower_body', 'dresses'],{
                    "default": "upper_body",
                }),
                "W":("INT",{
                    "default": 384
                }),
                "H":("INT",{
                    "default": 512
                }),
                "L":("INT",{
                    "default": 24
                }),
                "seed":("INT",{
                    "default": 42
                }),
                "cfg":("FLOAT",{
                    "default": 3.5
                }),
                "steps":("INT",{
                    "default": 20
                }),
                "if_fp16":("BOOLEAN",{
                    "default": True
                })
            }
        }

    RETURN_TYPES = ("VIDEO",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "tryon"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_ViViD"

    def tryon(self,cloth_image_path,model_video_path,category,W,H,L,seed,cfg,steps,if_fp16):
        if if_fp16:
            weight_dtype = torch.float16
        else:
            weight_dtype = torch.float32
        device = "cuda" if cuda_malloc.cuda_malloc_supported() else "cpu"
        global vae
        if vae is None:
            vae = AutoencoderKL.from_pretrained(
                pretrained_vae_path,use_safetensors=True
            ).to(device, dtype=weight_dtype)

            reference_unet = UNet2DConditionModel.from_pretrained_2d(
                pretrained_base_model_path,
                subfolder="unet",
                unet_additional_kwargs={
                    "in_channels": 5,
                }
            ).to(dtype=weight_dtype, device=device)

            infer_config = OmegaConf.load(os.path.join(now_dir,"ViViD","configs","inference","inference.yaml"))

            denoising_unet = UNet3DConditionModel.from_pretrained_2d(
                pretrained_base_model_path,
                motion_module_path,
                subfolder="unet",
                unet_additional_kwargs=infer_config.unet_additional_kwargs,
            ).to(dtype=weight_dtype, device=device)

            pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
                dtype=weight_dtype, device=device
            )

            image_enc = CLIPVisionModelWithProjection.from_pretrained(
                image_encoder_path
            ).to(dtype=weight_dtype, device=device)

            sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
            scheduler = DDIMScheduler(**sched_kwargs)

            # load pretrained weights
            denoising_unet.load_state_dict(
                torch.load(denoising_unet_path, map_location="cpu"),
                strict=False,
            )
            reference_unet.load_state_dict(
                torch.load(reference_unet_path, map_location="cpu"),
            )

            pose_guider.load_state_dict(
                torch.load(pose_guider_path, map_location="cpu"),
            )

            pipe = Pose2VideoPipeline(
                vae=vae,
                image_encoder=image_enc,
                reference_unet=reference_unet,
                denoising_unet=denoising_unet,
                pose_guider=pose_guider,
                scheduler=scheduler,
            )
            pipe = pipe.to(device, dtype=weight_dtype)

            openpose_model = OpenPose(0)
            parsing_model = Parsing(0)

            ## densepose
            # Initialize Detectron2 configuration for DensePose
            cfg = get_cfg()
            add_densepose_config(cfg)
            cfg.merge_from_file("detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml")
            cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
            cfg.MODEL.DEVICE = device
            predictor = DefaultPredictor(cfg)

            ## segment_anying
            sam = sam_model_registry["vit_b"](checkpoint=sam_model_path)
            mask_generator = SamAutomaticMaskGenerator(sam)
            # masks = mask_generator.generate(<your_image>)


        generator = torch.manual_seed(seed)

        width, height = W, H
        clip_length = L 
        guidance_scale = cfg

        date_str = datetime.now().strftime("%Y%m%d")
        time_str = datetime.now().strftime("%H%M")
        save_dir_name = f"{time_str}--seed_{seed}-{W}x{H}"

        save_dir = os.path.join(out_dir,"ViViD",date_str,save_dir_name)
        os.makedirs(save_dir,exist_ok=True)

        transform = transforms.Compose(
            [transforms.Resize((height, width)), transforms.ToTensor()]
        )

        model_image_path = model_video_path

        src_fps = get_fps(model_image_path)
        model_name = os.path.basename(model_image_path)[:-4]
        agnostic_path=os.path.join(out_dir,"ViViD","agnostic",model_name)
        agn_mask_path=os.path.join(out_dir,"ViViD","agnostic_mask",model_name)
        densepose_path=os.path.join(out_dir,"ViViD","densepose",model_name)

        os.makedirs(agnostic_path,exist_ok=True)
        os.makedirs(agn_mask_path,exist_ok=True)
        os.makedirs(densepose_path,exist_ok=True)

        video_tensor_list=[]
        video_images=read_frames(model_image_path)

        agnostic_list=[]
        agn_mask_list=[]
        pose_list=[]
        for i, vid_image_pil in enumerate(video_images[:clip_length]):
            video_tensor_list.append(transform(vid_image_pil))

            ## ootd
            keypoints = openpose_model(vid_image_pil)
            model_parse, _ = parsing_model(vid_image_pil)
            mask, mask_gray = get_mask_location("hd",category, model_parse, keypoints)
            # mask = mask.resize((768, 1024), Image.NEAREST)
            # mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
            
            masked_vton_img = Image.composite(mask_gray, vid_image_pil, mask)
            masked_vton_img.save(os.path.join(agnostic_path,"%04d.jpg"%(i+1)))
            agnostic_list.append(masked_vton_img)
            agn_mask_list.append(mask)
            mask.save(os.path.join(agn_mask_path,"%04d.jpg"%(i+1)))

            vid_image_cv2 = cv2.cvtColor(np.array(vid_image_pil),cv2.COLOR_RGB2BGR)
            with torch.no_grad():
                outputs = predictor(vid_image_cv2)['instances']
        
            results = DensePoseResultExtractor()(outputs)
            cmap = cv2.COLORMAP_VIRIDIS
            # Visualizer outputs black for background, but we want the 0 value of
            # the colormap, so we initialize the array with that value
            arr = cv2.applyColorMap(np.zeros((height, width), dtype=np.uint8), cmap)
            out_frame = Visualizer(alpha=1, cmap=cmap).visualize(arr, results)        
            out_frame_pil = Image.fromarray(cv2.cvtColor(out_frame,cv2.COLOR_BGR2RGB))
            pose_list.append(out_frame_pil)
            out_frame_pil.save(os.path.join(densepose_path,"%04d.jpg"%(i+1)))


        video_tensor = torch.stack(video_tensor_list, dim=0)  # (f, c, h, w)
        video_tensor = video_tensor.transpose(0, 1)

        video_tensor = video_tensor.unsqueeze(0)

        cloth_name =  Path(cloth_image_path).stem
        cloth_image_pil = Image.open(cloth_image_path).convert("RGB")

        cloth_mask_path = os.path.join(out_dir,"ViViD",f"{cloth_name}_mask.jpg") 
        cloth_image_cv2 = cv2.imread(cloth_image_path)

        mask = mask_generator(cv2.cvtColor(cloth_image_cv2, cv2.COLOR_BGR2RGB))[0]
        cv2.imwrite(cloth_mask_path, mask["segmentation"] * 255)
        cloth_mask_pil = Image.open(cloth_mask_path).convert("RGB")

        pipeline_output = pipe(
            agnostic_list,
            agn_mask_list,
            cloth_image_pil,
            cloth_mask_pil,
            pose_list,
            width,
            height,
            clip_length,
            steps,
            guidance_scale,
            generator=generator,
        )
        video = pipeline_output.videos

        video = torch.cat([video], dim=0)
        outfile = os.path.join(out_dir, f"{model_name}_{cloth_name}_{H}x{W}_{int(guidance_scale)}_{time_str}.mp4")
        save_videos_grid(
            video,
            outfile,
            n_rows=1,
            fps=src_fps,
        )

        return (outfile,)


class LoadImagePath:
    @classmethod
    def INPUT_TYPES(s):
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.split('.')[-1].lower() in ['bmp','jpg','png','webp','jpeg'] ]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "AIFSH_ViViD"

    RETURN_TYPES = ("IMAGEPATH",)
    FUNCTION = "load_image"
    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        return (image_path,)


class LoadVideo:
    @classmethod
    def INPUT_TYPES(s):
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.split('.')[-1] in ["mp4", "webm","mkv","avi"]]
        return {"required":{
            "video":(files,),
        }}
    
    CATEGORY = "AIFSH_ViViD"
    DESCRIPTION = "hello world!"

    RETURN_TYPES = ("VIDEO")

    OUTPUT_NODE = False

    FUNCTION = "load_video"

    def load_video(self, video):
        video_path = os.path.join(input_dir,video)
        return (video_path,)

class PreViewVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
            "video":("VIDEO",),
        }}
    
    CATEGORY = "AIFSH_ViViD"
    DESCRIPTION = "hello world!"

    RETURN_TYPES = ()

    OUTPUT_NODE = True

    FUNCTION = "load_video"

    def load_video(self, video):
        video_name = os.path.basename(video)
        video_path_name = os.path.basename(os.path.dirname(video))
        return {"ui":{"video":[video_name,video_path_name]}}

