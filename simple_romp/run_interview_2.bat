::python -m romp.main --mode=video --calc_smpl --show_items=mesh,j3d --render_mesh -i=D:\110598028\2_people_interview.mp4 -o=../output/2_people_interview_a/2_people_interview_a_result.mp4 --save_video --GPU 0
::python -m romp.main --mode=video --calc_smpl -sc 0.05 -t --show_largest --show_items=mesh,j3d --render_mesh -i=D:\110598028\2_people_interview.mp4 -o=../output/2_people_interview_b/2_people_interview_b_result.mp4 --save_video --GPU 0
::python -m romp.main --mode=video --calc_smpl --render_mesh -sc 0.05 --temporal_optimize --show_items=mesh -i=D:\110598028\2_people_interview.mp4 -o=../output/2_people_interview_c/2_people_interview_c_result.mp4 --save_video --GPU 0
::python -m romp.main --mode=video --calc_smpl --render_mesh -sc 0.05 --temporal_optimize --show_items=mesh -i=D:\110598028\2_people_interview.mp4 -o=../output/2_people_interview_c_debug/2_people_interview_c_debug_result.mp4 --save_video --GPU 0
::python -m romp.main --mode=video --calc_smpl --render_mesh -sc 0.05 --temporal_optimize --show_items=mesh -i=D:\110598028\two_person_refine.mp4 -o=../output/two_person_refine/two_person_refine_result.mp4 --save_video --GPU 0

::python -m romp.main --mode=video --calc_smpl --render_mesh -sc 0.05 --temporal_optimize --show_items=mesh -i=D:\110598028\four_person_refine.mp4 -o=../output/four_person_refine/four_person_refine_result.mp4 --save_video --GPU 0
::python -m romp.main --mode=video --calc_smpl --render_mesh -sc 0.05 --temporal_optimize --show_items=mesh -i=D:\110598028\input_video_dataset\str_network.mp4 -o=../output/str_network/str_network.mp4 --save_video --GPU 0
::python -m romp.main --mode=video --calc_smpl --render_mesh -sc 0.05 --temporal_optimize --show_items=mesh -i=D:\110598028\input_video_dataset\str_network_short.mp4 -o=../output/str_network_short/str_network_short.mp4 --save_video --GPU 0
::python -m romp.main --mode=video --calc_smpl --render_mesh -sc 0.05 --show_items=mesh -i=D:\110598028\input_video_dataset\str_network_short.mp4 -o=../output/str_network_short3/str_network_short3.mp4 --save_video --GPU 0

::python -m romp.main --mode=video --calc_smpl --render_mesh -sc 0.2 --temporal_optimize --show_items=mesh -i=D:\110598028\input_video_dataset\three_person_refine.mp4 -o=../output/three_person_refine/three_person_refine_result.mp4 --save_video --GPU 0
python -m romp.main --mode=webcam --calc_smpl --temporal_optimize --smooth_coeff 0.2 --show --show_items=mesh  --GPU 0
