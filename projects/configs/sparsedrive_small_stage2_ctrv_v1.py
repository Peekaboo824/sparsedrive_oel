"""E1: identical activated E0 plus scene-local current-model historical EKF.

No fixed pseudo-observation cache. Values below are initial experimental settings.
Class indices follow this config's class_names, NOT dataset default class order.
"""
_base_ = ['./sparsedrive_small_stage2_ctrv_e0.py']
model = dict(evo_cfg=dict(
    enable_dynamics_loss=True,
    dynamics_model='CTRV_EKF',
    lambda_dyn=0.1,
    ekf_history_length=5,
    ekf_min_observations=3,
    ekf_process_noise=[.1, .1, 1., .1, .1],
    ekf_measurement_noise=[.5, .5],
    ekf_initial_covariance=[1., 1., 25., 10., 10.],
    # [speed, heading, yaw-rate] variance floors (SI squared units).
    ekf_bootstrap_variance_floor=[4., .25, 1.],
    # Unobserved heading/rate stay conservative, especially for low speed.
    ekf_bootstrap_unobserved_variance=[25., 10., 10.],
    ekf_max_track_age=1.5,  # seconds since last measurement
    ekf_max_prediction_dt=1.5,
    ekf_motion_step_seconds=0.5,  # first motion point of the nuScenes model
    ekf_time_tolerance=0.05,
    ekf_association_gate=9.21,
    ekf_max_association_distance=6.,
    ekf_max_position_variance=25.,
    ekf_min_heading_speed=.5,
    ekf_detection_threshold=.4,
    dynamics_vehicle_class_ids=[0, 1, 2, 3, 4],
))
# This pipeline receives sensor_frame_info() ONLY, with no annotation fields.
dynamics_history_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipImage'),
    dict(type='NormalizeMultiviewImage',
         mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='NuScenesSparse4DAdaptor'),
    dict(type='Collect', keys=['img', 'timestamp', 'projection_mat', 'image_wh'],
         meta_keys=['T_global', 'T_global_inv', 'timestamp']),
]
data = dict(train=dict(dynamics_history_length=5,
                       dynamics_history_pipeline=dynamics_history_pipeline))
