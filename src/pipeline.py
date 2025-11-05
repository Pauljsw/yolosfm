"""
Main Pipeline
End-to-end pipeline for YOLO + SFM 3D fusion.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional
import numpy as np

# Import modules
from calib_io import load_camera_info, load_poses
from align_depth_to_rgb import align_depth_to_rgb
from project_mask_to_A import project_all_masks
from fusion_3d import LabelFusion
from instance_merge import merge_pipeline
from measurement import measure_all_instances
from export_results import export_all_results
from utils import setup_logging, load_config, Timer, ensure_dir, list_files

logger = logging.getLogger(__name__)


class Pipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config_path: str):
        """
        Initialize pipeline.
        
        Args:
            config_path: Path to configuration YAML
        """
        self.config = load_config(config_path)
        self.setup_paths()
        
        # Load calibrations
        self.rgb_calib = load_camera_info(self.config['paths']['calib_rgb'])
        self.depth_calib = load_camera_info(self.config['paths']['calib_depth'])
        
        # Load poses
        poses_path = Path(self.config['paths']['sfm_dir']) / 'poses.json'
        self.poses = load_poses(str(poses_path))
        
        # Class information
        self.class_names = list(self.config['classes'].keys())
        self.num_classes = len(self.class_names)
        self.colors = self.config.get('colors', {})
        
        logger.info(f"Pipeline initialized with {len(self.poses)} images")
        logger.info(f"Classes: {self.class_names}")
    
    def setup_paths(self):
        """Setup and validate paths"""
        paths = self.config['paths']
        
        # Ensure output directories exist
        ensure_dir(paths['out_dir'])
        ensure_dir(f"{paths['out_dir']}/aligned_depth")
        ensure_dir(f"{paths['out_dir']}/fused")
        ensure_dir(f"{paths['out_dir']}/report")
        ensure_dir(paths['masks_dir'])
    
    def run_sfm(self):
        """
        Stage 0: Structure from Motion with COLMAP
        """
        logger.info("=" * 80)
        logger.info("Stage 0: Structure from Motion (COLMAP)")
        logger.info("=" * 80)
        
        try:
            from colmap_sfm import run_colmap_sfm_auto
        except ImportError:
            logger.error("colmap_sfm module not found")
            return
        
        with Timer("SFM"):
            rgb_dir = self.config['paths']['rgb_dir']
            sfm_dir = self.config['paths']['sfm_dir']
            
            ensure_dir(sfm_dir)
            
            # Get SFM config
            sfm_config = self.config.get('sfm', {})
            camera_model = sfm_config.get('camera_model', 'OPENCV')
            quality = sfm_config.get('quality', 'high')
            dense = sfm_config.get('dense', False)
            
            poses_output = f"{sfm_dir}/poses.json"
            
            logger.info(f"Running COLMAP on images in: {rgb_dir}")
            logger.info(f"Camera model: {camera_model}, Quality: {quality}")
            
            # Run COLMAP
            poses = run_colmap_sfm_auto(
                image_dir=rgb_dir,
                output_dir=sfm_dir,
                poses_json_output=poses_output,
                camera_model=camera_model,
                quality=quality,
                dense=dense
            )
            
            logger.info(f"SFM complete: {len(poses)} images reconstructed")
            logger.info(f"Poses saved to: {poses_output}")
            
            # Reload poses
            self.poses = poses
        
        logger.info("SFM stage completed")

    
    def run_alignment(self):
        """
        Stage 1: Align depth to RGB for all images.
        """
        logger.info("=" * 80)
        logger.info("Stage 1: Depth-to-RGB Alignment")
        logger.info("=" * 80)
        
        with Timer("Alignment"):
            # Get list of depth images
            depth_dir = self.config['paths']['depth_dir']
            depth_files = list_files(depth_dir, '.png')
            
            if not depth_files:
                logger.error(f"No depth images found in {depth_dir}")
                return
            
            logger.info(f"Found {len(depth_files)} depth images")
            
            align_config = self.config['align']
            bilateral_params = {
                'd': align_config.get('bilateral_d', 9),
                'sigma_color': align_config.get('bilateral_sigma_color', 75),
                'sigma_space': align_config.get('bilateral_sigma_space', 75)
            }
            
            output_dir = f"{self.config['paths']['out_dir']}/aligned_depth"
            
            for i, depth_file in enumerate(depth_files):
                image_id = Path(depth_file).stem
                
                logger.info(f"Processing [{i+1}/{len(depth_files)}]: {image_id}")
                
                # Load depth
                import cv2
                depth_img = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED).astype(np.float32)
                
                # Align
                aligned = align_depth_to_rgb(
                    depth_img,
                    self.rgb_calib.K,
                    self.rgb_calib.D,
                    self.depth_calib.K,
                    self.depth_calib.D,
                    rgb_size=(self.rgb_calib.width, self.rgb_calib.height),
                    depth_unit=align_config['in_depth_unit'],
                    hole_fill=align_config['hole_fill'],
                    joint_bilateral=align_config['joint_bilateral'],
                    bilateral_params=bilateral_params
                )
                
                # Save
                output_path = f"{output_dir}/{image_id}.png"
                # Save as float32 EXR or scaled uint16
                aligned_scaled = (aligned * 1000).astype(np.uint16)  # mm
                cv2.imwrite(output_path, aligned_scaled)
                
                logger.info(f"  Saved aligned depth: {output_path}")
        
        logger.info("Alignment stage completed")

    def _resolve_rgb_path(self, image_id: str) -> Optional[Path]:
        """Resolve the RGB image path for a given image identifier."""
        rgb_dir = Path(self.config['paths']['rgb_dir'])

        candidates = [
            rgb_dir / image_id,
            *(rgb_dir / f"{image_id}{ext}" for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'])
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return None

    def run_detection(self, reinfer_mode: str = 'auto'):
        """Stage 1.5: Run YOLO segmentation to produce mask JSON files."""
        logger.info("=" * 80)
        logger.info("Stage 1.5: YOLO Segmentation Inference")
        logger.info("=" * 80)

        image_ids = [Path(p).stem for p in self.poses.keys()]
        effective_mode = reinfer_mode
        if effective_mode == 'off':
            effective_mode = self.config.get('reinfer', {}).get('mode', 'auto')

        self.run_yolo_inference(image_ids, effective_mode)

    def run_yolo_inference(self, image_ids: List[str], reinfer_mode: str):
        """Execute YOLO segmentation inference for the provided image ids."""

        if reinfer_mode == 'off':
            logger.info("Reinference mode is 'off'; skipping YOLO inference.")
            return

        yolo_config = self.config.get('yolo')
        if not yolo_config:
            logger.warning("YOLO configuration missing in config file; skipping inference.")
            return

        try:
            from yolo_inference import YOLOSegmenter
        except ImportError as exc:
            logger.error("Failed to import YOLO inference utilities: %s", exc)
            return

        weights_path = yolo_config.get('weights')
        if not weights_path:
            logger.error("YOLO weights path not specified. Set 'yolo.weights' in the config file.")
            return

        weights_path = Path(weights_path)
        if not weights_path.exists():
            logger.error("YOLO weights not found at %s", weights_path)
            return

        masks_dir = Path(self.config['paths']['masks_dir'])
        ensure_dir(str(masks_dir))

        force = reinfer_mode == 'on'

        images_to_process = []
        for image_id in image_ids:
            image_path = self._resolve_rgb_path(image_id)
            if image_path is None:
                logger.warning("RGB image not found for %s; skipping YOLO inference.", image_id)
                continue

            mask_path = masks_dir / f"{image_id}.json"
            if mask_path.exists() and not force:
                logger.debug("Mask already exists for %s; skipping in auto mode.", image_id)
                continue

            images_to_process.append((image_id, image_path, mask_path))

        if not images_to_process:
            logger.info("No images require YOLO inference.")
            return

        logger.info("Running YOLO segmentation on %d images", len(images_to_process))

        with Timer("YOLO Inference"):
            inferencer = YOLOSegmenter(
                weights_path=str(weights_path),
                class_names=self.class_names,
                conf=yolo_config.get('conf', 0.25),
                iou=yolo_config.get('iou', 0.45),
                img_size=yolo_config.get('img_size', 1024),
                device=yolo_config.get('device'),
                max_det=yolo_config.get('max_det', 300)
            )

            for idx, (image_id, image_path, mask_path) in enumerate(images_to_process, start=1):
                logger.info("[%d/%d] Running YOLO on %s", idx, len(images_to_process), image_path.name)
                inferencer.process_image(str(image_path), str(mask_path))

        logger.info("YOLO inference completed. Masks saved to %s", masks_dir)

    def run_fusion(self, reinfer_mode: str = 'off'):
        """
        Stage 2: 3D Label Fusion

        Args:
            reinfer_mode: 'off', 'on', or 'auto'
        """
        logger.info("=" * 80)
        logger.info("Stage 2: 3D Label Fusion")
        logger.info("=" * 80)
        
        with Timer("3D Fusion"):
            fusion_config = self.config['fusion']
            
            # Initialize fusion
            voxel_size = fusion_config['voxel_size_cm'] / 100.0  # to meters
            fusion = LabelFusion(
                voxel_size=voxel_size,
                num_classes=self.num_classes,
                class_names=self.class_names,
                config=fusion_config
            )
            
            # Process each image
            aligned_depth_dir = f"{self.config['paths']['out_dir']}/aligned_depth"
            masks_dir = self.config['paths']['masks_dir']

            image_ids = [Path(p).stem for p in self.poses.keys()]

            effective_mode = reinfer_mode
            if effective_mode == 'off':
                effective_mode = self.config.get('reinfer', {}).get('mode', 'off')

            if effective_mode in ('on', 'auto'):
                self.run_yolo_inference(image_ids, effective_mode)

            for i, image_id in enumerate(image_ids):
                logger.info(f"Processing image [{i+1}/{len(image_ids)}]: {image_id}")
                
                # Load aligned depth
                depth_path = f"{aligned_depth_dir}/{image_id}.png"
                if not Path(depth_path).exists():
                    logger.warning(f"Aligned depth not found: {depth_path}")
                    continue
                
                import cv2
                aligned_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0  # to meters
                
                # Load masks
                masks_path = f"{masks_dir}/{image_id}.json"
                if not Path(masks_path).exists():
                    logger.warning(f"Masks not found: {masks_path}")
                    continue
                
                # Get pose
                pose_key = f"{image_id}.png"
                if pose_key not in self.poses:
                    # Try without extension
                    pose_key = image_id
                    if pose_key not in self.poses:
                        logger.warning(f"Pose not found for {image_id}")
                        continue
                
                pose = self.poses[pose_key]
                R = pose['R']
                t = pose['t']
                K = pose.get('K', self.rgb_calib.K)
                
                # Project masks to 3D
                projections = project_all_masks(
                    masks_path,
                    aligned_depth,
                    K,
                    self.rgb_calib.D,
                    R,
                    t,
                    self.class_names,
                    image_shape=(self.rgb_calib.height, self.rgb_calib.width)
                )
                
                # Fuse into voxel grid
                fusion.fuse_image(projections)
                
                logger.info(f"  Fused {len(projections)} masks")
            
            # Finalize fusion
            prob_thresh = fusion_config['prob_thresh']
            fusion_result = fusion.finalize(prob_thresh=prob_thresh)
            
            logger.info(f"Fusion complete: {len(fusion_result['labels'])} labeled voxels")
        
        # Stage 3: Instance Merging
        logger.info("=" * 80)
        logger.info("Stage 3: Instance Merging")
        logger.info("=" * 80)
        
        with Timer("Instance Merging"):
            merge_config = self.config['merge']
            
            instances = merge_pipeline(
                fusion_result,
                voxel_size,
                self.class_names,
                merge_config
            )
            
            logger.info(f"Merged to {len(instances)} instances")
        
        # Stage 4: Measurement
        logger.info("=" * 80)
        logger.info("Stage 4: Measurement")
        logger.info("=" * 80)
        
        with Timer("Measurement"):
            measure_config = self.config['measure']
            
            measurements = measure_all_instances(
                instances,
                voxel_size,
                measure_config
            )
            
            logger.info(f"Measured {len(measurements)} instances")
        
        # Stage 5: Export
        logger.info("=" * 80)
        logger.info("Stage 5: Export Results")
        logger.info("=" * 80)
        
        with Timer("Export"):
            output_dir = f"{self.config['paths']['out_dir']}/fused"
            
            export_all_results(
                fusion_result,
                instances,
                measurements,
                output_dir,
                self.class_names,
                self.config,
                self.colors
            )
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
    
    def run_report(self):
        """
        Generate final report.
        """
        logger.info("=" * 80)
        logger.info("Generating Final Report")
        logger.info("=" * 80)
        
        # Report already generated in export stage
        report_path = f"{self.config['paths']['out_dir']}/fused/report.md"
        
        if Path(report_path).exists():
            logger.info(f"Report available at: {report_path}")
        else:
            logger.warning("Report not found. Run fusion stage first.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='YOLO + SFM 3D Fusion Pipeline')

    parser.add_argument('command', choices=['sfm', 'align', 'infer', 'fuse3d', 'report', 'full'],
                       help='Pipeline command to run')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--reinfer', type=str, choices=['off', 'on', 'auto'], default='off',
                       help='Reinference mode for YOLO segmentation masks')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Optional log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    logger.info("=" * 80)
    logger.info("YOLO + SFM 3D Fusion Pipeline")
    logger.info("=" * 80)
    logger.info(f"Command: {args.command}")
    logger.info(f"Config: {args.config}")
    
    # Check config exists
    if not Path(args.config).exists():
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Initialize pipeline
    try:
        pipeline = Pipeline(args.config)
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
        sys.exit(1)
    
    # Run command
    try:
        if args.command == 'sfm':
            pipeline.run_sfm()
        
        elif args.command == 'align':
            pipeline.run_alignment()
        
        elif args.command == 'infer':
            reinfer_mode = args.reinfer if args.reinfer != 'off' else 'auto'
            pipeline.run_detection(reinfer_mode=reinfer_mode)

        elif args.command == 'fuse3d':
            pipeline.run_fusion(reinfer_mode=args.reinfer)
        
        elif args.command == 'report':
            pipeline.run_report()
        
        elif args.command == 'full':
            # Run full pipeline (skip SFM if poses already exist)
            poses_path = Path(pipeline.config['paths']['sfm_dir']) / 'poses.json'
            if not poses_path.exists():
                logger.info("Poses not found, running SFM first...")
                pipeline.run_sfm()
            pipeline.run_alignment()
            pipeline.run_detection(reinfer_mode=args.reinfer)
            pipeline.run_fusion(reinfer_mode=args.reinfer)
            pipeline.run_report()
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("=" * 80)
    logger.info("Pipeline execution completed")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
