import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"

import argparse
from omegaconf import OmegaConf
import random
import numpy as np
import torch
import time
import json
import logging
import matplotlib.pyplot as plt

import open_clip
from ultralytics import SAM, YOLOWorld

from src.habitat import pose_habitat_to_tsdf
from src.geom import get_cam_intr, get_scene_bnds
from src.tsdf_planner import TSDFPlanner, Frontier, SnapShot
from src.scene_aeqa import Scene
from src.utils import resize_image, get_pts_angle_aeqa
from src.query_vlm_aeqa import query_vlm_for_response
from src.logger_aeqa import Logger
from src.const import *
from src.habitat import pos_normal_to_habitat, pos_habitat_to_normal

OFFLINE_MODE = True

import cv2
def visualize_map(tsdf_planner: TSDFPlanner, pts, height=1.8, cam_pose=None):
    ft_map = np.zeros(
        (tsdf_planner._tsdf_vol_cpu.shape[0], tsdf_planner._tsdf_vol_cpu.shape[1], 3),
        dtype=np.uint8,
    ) + np.asarray([[[255, 255, 255]]], dtype=np.uint8)

    pts = pos_habitat_to_normal(pts)
    _, unoccupied_high = tsdf_planner.get_island_around_pts(pts, height=1.8)
    ft_map[unoccupied_high > 0] = [200, 200, 200]
    # ft_map[(tsdf_planner.unexplored == 0) & (unoccupied_high > 0)] = [194, 246, 198]
    
    height_voxel = int(height / tsdf_planner._voxel_size) + tsdf_planner.min_height_voxel
    explored_map = tsdf_planner._explore_vol_cpu[:, :, height_voxel]
    ft_map[explored_map.astype(np.bool_)] = [194, 246, 198]

    obstacle_map = tsdf_planner.get_obstacle_map(height=height)
    ft_map[obstacle_map > 0] = [100, 100, 100]
    
    if cam_pose is not None:
        cam_pos = cam_pose[:3, 3]
        cam_pos_in_voxel = tsdf_planner.normal2voxel(cam_pos)
        cv2.circle(ft_map, (int(cam_pos_in_voxel[1]), int(cam_pos_in_voxel[0])), 3, (255, 0, 0), -1)

    ft_map = cv2.resize(ft_map, dsize=(512, 512))
    # cv2.imshow(f"Obstacle Map", ft_map)        
    # cv2.waitKey(1)
    cv2.imwrite('obstacle_map_mem.png', ft_map)

    return ft_map

import csv
def load_questions(scene_id=None):
    init_path = '/home/shivin/Desktop/3dmem/data/mp3d/init_pos_mp3d.csv'
    scene2init = {}
    with open(init_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row if there is one
        
        for row in reader:
            _id = row[0]
            position = np.array([float(row[1]), float(row[2]), float(row[3])])
            scene2init[_id] = position

    path = '/home/shivin/Desktop/3dmem/data/mp3d/mp3d_qa'
    question_list = []

    file_list = sorted(os.listdir(path)) if scene_id is None else [f'{scene_id}.csv']
    for filename in file_list:
        with open(os.path.join(path, filename), 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header row if there is one
            count = 0
            for row in reader:
                _id = row[0]
                question = row[1]
                answer = row[2]
                metadata = row[3]
                question_id = f"{_id}_{count}"
                question_list.append({
                    'question': question,
                    'answer': answer,
                    'metadata': metadata,
                    'question_id': question_id,
                    'episode_history': _id,
                    'position': scene2init[_id],
                    'rotation': 0
                })
                count += 1
    return question_list

def answer_offline(question, scene, tsdf_planner, cfg):
    _original = cfg.egocentric_views
    cfg.egocentric_views = False
    vlm_response = query_vlm_for_response(
            question=question,
            scene=scene,
            tsdf_planner=tsdf_planner,
            rgb_egocentric_views=[],
            cfg=cfg,
            verbose=True,
        )
    cfg.egocentric_views = _original

    if vlm_response is None:
        logging.info(f"Offline VLM query failed!")
        return None

    return vlm_response


def main(cfg, start_ratio=0.0, end_ratio=1.0, scene_id=None):

    # load the default concept graph config
    cfg_cg = OmegaConf.load(cfg.concept_graph_config_path)
    OmegaConf.resolve(cfg_cg)

    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load dataset
    # questions_list = json.load(open(cfg.questions_list_path, "r"))
    questions_list = load_questions(scene_id=scene_id)
    # print(questions_list)
    
    total_questions = len(questions_list)
    # sort the data according to the question id
    # questions_list = sorted(questions_list, key=lambda x: x["question_id"])
    logging.info(f"Total number of questions: {total_questions}")
    # only process a subset of the questions
    questions_list = questions_list[
        int(start_ratio * total_questions) : int(end_ratio * total_questions)
    ]
    logging.info(f"number of questions after splitting: {len(questions_list)}")
    logging.info(f"question path: {cfg.questions_list_path}")

    # load detection and segmentation models
    detection_model = YOLOWorld(cfg.yolo_model_name)
    logging.info(f"Load YOLO model {cfg.yolo_model_name} successful!")

    sam_predictor = SAM(cfg.sam_model_name)  # UltraLytics SAM
    logging.info(f"Load SAM model {cfg.sam_model_name} successful!")

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", "laion2b_s34b_b79k"  # "ViT-H-14", "laion2b_s32b_b79k"
    )
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    logging.info(f"Load CLIP model successful!")

    # Initialize the logger
    logger = Logger(
        cfg.output_dir,
        start_ratio,
        end_ratio,
        len(questions_list),
        voxel_size=cfg.tsdf_grid_size,
    )

    # Run all questions
    questions_answered = []
    for question_idx, question_data in enumerate(questions_list):
        question_id = question_data["question_id"]
        scene_id = question_data["episode_history"]
        if question_id in logger.success_list or question_id in logger.fail_list:
            logging.info(f"Question {question_id} already processed")
            continue
        if any([invalid_scene_id in scene_id for invalid_scene_id in INVALID_SCENE_ID]):
            logging.info(f"Skip invalid scene {scene_id}")
            continue
        logging.info(f"\n========\nIndex: {question_idx} Scene: {scene_id}")

        question = question_data["question"]
        answer = question_data["answer"]

        # load scene
        if question_idx == 0:
            try:
                del scene
            except:
                pass
            scene = Scene(
                scene_id,
                cfg,
                cfg_cg,
                detection_model,
                sam_predictor,
                clip_model,
                clip_preprocess,
                clip_tokenizer,
            )

        pts = question_data["position"]
        angle = question_data["rotation"]


        if question_idx == 0:
            # initialize the TSDF
            tsdf_planner = TSDFPlanner(
                vol_bnds=get_scene_bnds(scene.pathfinder, floor_height=pts[1])[0],
                voxel_size=cfg.tsdf_grid_size,
                floor_height=pts[1],
                floor_height_offset=0,
                pts_init=pts,
                init_clearance=cfg.init_clearance * 2,
                save_visualization=cfg.save_visualization,
            )

        episode_dir, eps_chosen_snapshot_dir, eps_frontier_dir, eps_snapshot_dir = (
            logger.init_episode(
                question_id=question_id,
                init_pts_voxel=tsdf_planner.habitat2voxel(pts)[:2],
            )
        )

        tsdf_planner.reset()
        logging.info(f"\n\nQuestion id {question_id} initialization successful!")

        # run steps
        task_success = False
        cnt_step = -1
        pts_list = [pts.copy()]
        gpt_answer = None
        n_filtered_snapshots = 0
        while cnt_step < cfg.num_step - 1:
            cnt_step += 1
            logging.info(f"\n== step: {cnt_step}")

            if cnt_step == 0 and question_idx > 0 and OFFLINE_MODE:
                vlm_response = answer_offline(
                    question=question,
                    scene=scene,
                    tsdf_planner=tsdf_planner,
                    cfg=cfg,
                )

                if vlm_response is None:
                    continue
                max_point_choice, gpt_answer, n_filtered_snapshots, is_answer = vlm_response
                
                if is_answer:
                    logging.info(f"Question id {question_id} finished by offline VLM!")
                    task_success = True
                    snapshot_filename = max_point_choice.image.split(".")[0]
                    
                    for question_id_done in questions_answered:
                        old_eps_snapshot_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name, scene_id, question_id_done, "snapshot")
                        if os.path.exists(os.path.join(old_eps_snapshot_dir, max_point_choice.image)):
                            os.system(
                                f"cp {os.path.join(old_eps_snapshot_dir, max_point_choice.image)} {os.path.join(eps_chosen_snapshot_dir, f'snapshot_{snapshot_filename}.png')}"
                            )
                            break
                    break
            
            if cnt_step == 0 and task_success and OFFLINE_MODE:
                break

            # (1) Observe the surroundings, update the scene graph and occupancy map
            print("Step 1: Observing surroundings...")
            # Determine the viewing angles for the current step
            if cnt_step == 0:
                angle_increment = cfg.extra_view_angle_deg_phase_2 * np.pi / 180
                total_views = 1 + cfg.extra_view_phase_2
            else:
                angle_increment = cfg.extra_view_angle_deg_phase_1 * np.pi / 180
                total_views = 1 + cfg.extra_view_phase_1
            all_angles = [
                angle + angle_increment * (i - total_views // 2)
                for i in range(total_views)
            ]
            # Let the main viewing angle be the last one to avoid potential overwriting problems
            main_angle = all_angles.pop(total_views // 2)
            all_angles.append(main_angle)

            rgb_egocentric_views = []
            all_added_obj_ids = (
                []
            )  # Record all the objects that are newly added in this step
            for view_idx, ang in enumerate(all_angles):
                # For each view
                obs, cam_pose = scene.get_observation(pts, ang)
                rgb = obs["color_sensor"]
                depth = obs["depth_sensor"]

                obs_file_name = f"{question_idx}_{cnt_step}-view_{view_idx}.png"
                with torch.no_grad():
                    # Concept graph pipeline update
                    annotated_rgb, added_obj_ids, _ = scene.update_scene_graph(
                        image_rgb=rgb[..., :3],
                        depth=depth,
                        intrinsics=cam_intr,
                        cam_pos=cam_pose,
                        pts=pts,
                        pts_voxel=tsdf_planner.habitat2voxel(pts),
                        img_path=obs_file_name,
                        frame_idx=question_idx*total_views*cfg.num_step + cnt_step * total_views + view_idx,
                        target_obj_mask=None,
                    )
                    resized_rgb = resize_image(rgb, cfg.prompt_h, cfg.prompt_w)
                    scene.all_observations[obs_file_name] = resized_rgb
                    rgb_egocentric_views.append(resized_rgb)
                    if cfg.save_visualization:
                        plt.imsave(
                            os.path.join(eps_snapshot_dir, obs_file_name), annotated_rgb
                        )
                    else:
                        plt.imsave(os.path.join(eps_snapshot_dir, obs_file_name), rgb)
                    all_added_obj_ids += added_obj_ids

                # Clean up or merge redundant objects periodically
                scene.periodic_cleanup_objects(
                    frame_idx=question_idx*total_views*cfg.num_step + cnt_step * total_views + view_idx, pts=pts
                )

                # Update depth map, occupancy map
                tsdf_planner.integrate(
                    color_im=rgb,
                    depth_im=depth,
                    cam_intr=cam_intr,
                    cam_pose=pose_habitat_to_tsdf(cam_pose),
                    obs_weight=1.0,
                    margin_h=int(cfg.margin_h_ratio * img_height),
                    margin_w=int(cfg.margin_w_ratio * img_width),
                    explored_depth=cfg.explored_depth,
                )

            # (2) Update Memory Snapshots with hierarchical clustering
            print("Step 2: Updating memory snapshots...")
            # Choose all the newly added objects as well as the objects nearby as the cluster targets
            all_added_obj_ids = [
                obj_id for obj_id in all_added_obj_ids if obj_id in scene.objects
            ]
            for obj_id, obj in scene.objects.items():
                if (
                    np.linalg.norm(obj["bbox"].center[[0, 2]] - pts[[0, 2]])
                    < cfg.scene_graph.obj_include_dist + 0.5
                ):
                    all_added_obj_ids.append(obj_id)
            scene.update_snapshots(
                obj_ids=set(all_added_obj_ids), min_detection=cfg.min_detection
            )
            logging.info(
                f"Step {cnt_step}, update snapshots, {len(scene.objects)} objects, {len(scene.snapshots)} snapshots"
            )
            
            visualize_map(tsdf_planner, pts, height=1.5)

            # (3) Update the Frontier Snapshots
            print("Step 3: Updating frontier map...")
            update_success = tsdf_planner.update_frontier_map(
                pts=pts,
                cfg=cfg.planner,
                scene=scene,
                cnt_step=question_idx*cfg.num_step + cnt_step,
                save_frontier_image=cfg.save_visualization,
                eps_frontier_dir=eps_frontier_dir,
                prompt_img_size=(cfg.prompt_h, cfg.prompt_w),
            )
            if not update_success:
                logging.info("Warning! Update frontier map failed!")
                if cnt_step == 0:  # if the first step fails, we should stop
                    logging.info(
                        f"Question id {question_id} invalid: update_frontier_map failed!"
                    )
                    break

            # (4) Choose the next navigation point by querying the VLM
            print("Step 4: Querying VLM for next navigation point...")
            if cfg.choose_every_step:
                # if we choose to query vlm every step, we clear the target point every step
                if (
                    tsdf_planner.max_point is not None
                    and type(tsdf_planner.max_point) == Frontier
                ):
                    # reset target point to allow the model to choose again
                    tsdf_planner.max_point = None
                    tsdf_planner.target_point = None

            if tsdf_planner.max_point is None and tsdf_planner.target_point is None:
                # query the VLM for the next navigation point, and the reason for the choice
                vlm_response = query_vlm_for_response(
                    question=question,
                    scene=scene,
                    tsdf_planner=tsdf_planner,
                    rgb_egocentric_views=rgb_egocentric_views,
                    cfg=cfg,
                    verbose=True,
                )
                if vlm_response is None:
                    logging.info(
                        f"Question id {question_id} invalid: query_vlm_for_response failed!"
                    )
                    break

                max_point_choice, gpt_answer, n_filtered_snapshots, _ = vlm_response

                # set the vlm choice as the navigation target
                update_success = tsdf_planner.set_next_navigation_point(
                    choice=max_point_choice,
                    pts=pts,
                    objects=scene.objects,
                    cfg=cfg.planner,
                    pathfinder=scene.pathfinder,
                    random_position=False,
                )
                if not update_success:
                    logging.info(
                        f"Question id {question_id} invalid: set_next_navigation_point failed!"
                    )
                    break

            # (5) Agent navigate to the target point for one step
            print("Step 5: Agent navigating to the target point...")
            return_values = tsdf_planner.agent_step(
                pts=pts,
                angle=angle,
                objects=scene.objects,
                snapshots=scene.snapshots,
                pathfinder=scene.pathfinder,
                cfg=cfg.planner,
                path_points=None,
                save_visualization=cfg.save_visualization,
            )
            if return_values[0] is None:
                logging.info(f"Question id {question_id} invalid: agent_step failed!")
                break

            # update agent's position and rotation
            pts, angle, pts_voxel, fig, _, target_arrived = return_values
            pts_list.append(pts.copy())
            logger.log_step(pts_voxel=pts_voxel)
            logging.info(f"Current position: {pts}, {logger.explore_dist:.3f}")

            # sanity check about objects, scene graph, snapshots, ...
            scene.sanity_check(cfg=cfg)

            if cfg.save_visualization:
                # save the top-down visualization
                logger.save_topdown_visualization(
                    cnt_step=cnt_step,
                    fig=fig,
                )
                # save the visualization of vlm's choice at each step
                # logger.save_frontier_visualization(
                #     cnt_step=cnt_step,
                #     tsdf_planner=tsdf_planner,
                #     max_point_choice=max_point_choice,
                #     global_caption=f"{question}\n{answer}",
                # )

            # (6) Check if the agent has arrived at the target to finish the question
            print("Step 6: Checking if target is reached...")
            if type(max_point_choice) == SnapShot and target_arrived:
                # when the target is a snapshot, and the agent arrives at the target
                # we consider the question is finished and save the chosen target snapshot
                snapshot_filename = max_point_choice.image.split(".")[0]
                os.system(
                    f"cp {os.path.join(eps_snapshot_dir, max_point_choice.image)} {os.path.join(eps_chosen_snapshot_dir, f'snapshot_{snapshot_filename}.png')}"
                )

                task_success = True
                logging.info(
                    f"Question id {question_id} finished after arriving at target!"
                )
                break

        # save pts list and ptr voxel list
        np.savez(
            os.path.join(logger.episode_dir, "pts_list.npz"),
            pts_list=np.array(pts_list),
        )
        np.savez(
            os.path.join(logger.episode_dir, "pts_voxel_list.npz"),
            pts_voxel_list=np.array(logger.pts_voxels),
        )

        logger.log_episode_result(
            success=task_success,
            question_id=question_id,
            explore_dist=logger.explore_dist,
            gpt_answer=gpt_answer,
            n_filtered_snapshots=n_filtered_snapshots,
            n_total_snapshots=len(scene.snapshots),
            n_total_frames=len(scene.frames),
        )

        logging.info(f"Scene graph of question {question_id}:")
        logging.info(f"Question: {question}")
        logging.info(f"Answer: {answer}")
        logging.info(f"Prediction: {gpt_answer}")
        scene.print_scene_graph()

        # update the saved results after each episode
        logger.save_results()

        if not cfg.save_visualization:
            # clear up the stored images to save memory
            os.system(f"rm -r {episode_dir}")

        questions_answered.append(question_id)

    logger.save_results()
    # aggregate the results from different splits into a single file
    logger.aggregate_results()

    logging.info(f"All scenes finish")


if __name__ == "__main__":
    # Get config path
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", default="", type=str)
    parser.add_argument("--start_ratio", help="start ratio", default=0.0, type=float)
    parser.add_argument("--end_ratio", help="end ratio", default=1.0, type=float)
    parser.add_argument("--scene_id", help="only run this scene id", default=None, type=str)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)

    # Set up logging
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    # Initialize the logger
    if args.scene_id is not None:
        cfg.output_dir = os.path.join(cfg.output_dir, args.scene_id)

    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)  # recursive
    logging_path = os.path.join(
        str(cfg.output_dir), f"log_{args.start_ratio:.2f}_{args.end_ratio:.2f}.log"
    )

    os.system(f"cp {args.cfg_file} {cfg.output_dir}")

    class ElapsedTimeFormatter(logging.Formatter):
        def __init__(self, fmt=None, datefmt=None):
            super().__init__(fmt, datefmt)
            self.start_time = time.time()

        def formatTime(self, record, datefmt=None):
            elapsed_seconds = record.created - self.start_time
            hours, remainder = divmod(elapsed_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

    # Set up the logging format
    formatter = ElapsedTimeFormatter(fmt="%(asctime)s - %(message)s")

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    # Set the custom formatter
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)

    # run
    logging.info(f"***** Running {cfg.exp_name} *****")
    main(cfg, args.start_ratio, args.end_ratio, scene_id=args.scene_id)
