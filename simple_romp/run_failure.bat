::python -m romp.main --mode=video --calc_smpl --render_mesh -i=d:/110598066/20230209video/CAM1/C0005.mp4 -o=../output/20230209video_cam1_romp/CAM1_C0005.mp4 --save_video --GPU 0
::python -m romp.main --mode=video --calc_smpl --render_mesh -i=d:/110598066/20230209video/CAM2/C0005.mp4 -o=../output/20230209video_cam2_romp/CAM2_C0005.mp4 --save_video --GPU 0
::python -m bev.main -m video -i=d:/110598066/20230209video/CAM1/C0005.mp4 -o=../output/20230209video_cam1_bev/CAM1_C0005.mp4 --save_video --GPU 0
::python -m bev.main -m video -i=../output/20230209video_cam1_bev/C0005_frames -o=../output/20230209video_cam1_bev/CAM1_C0005.mp4 --save_video --GPU 0
::python -m bev.main -m video -i=d:/110598066/20230209video/CAM2/C0005.mp4 -o=../output/20230209video_cam2_bev/CAM2_C0005.mp4 --save_video --GPU 0

python -m romp.main --mode=video --calc_smpl --render_mesh -i=d:/110598066/20230209video/CAM2/CAM2_C0005_cut.mp4 -o=../output/20230209video_cam2_romp_2/CAM2_C0005_cut.mp4 --save_video --GPU 0
::python -m bev.main -m video -i=d:/110598066/20230209video/CAM2/CAM2_C0005_cut.mp4 -o=../output/20230209video_cam2_bev_2/CAM2_C0005_cut.mp4 --save_video --GPU 0
