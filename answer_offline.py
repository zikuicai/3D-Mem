from src.query_vlm_aeqa import query_vlm_for_response

def answer_offline(question, scene, tsdf_planner, rgb_egocentric_views, cfg):
    vlm_response = query_vlm_for_response(
            question=question,
            scene=scene,
            tsdf_planner=tsdf_planner,
            rgb_egocentric_views=rgb_egocentric_views,
            cfg=cfg,
            verbose=True,
        )
    
    