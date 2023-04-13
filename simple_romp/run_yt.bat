
::python -m bev.main -m video -i d:/110598028/videos/high_price_eggs.mp4 -o ../output/high_price_eggs_bev/high_price_eggs_bev_result.mp4 --save_video --GPU 0

::python -m romp.main --mode=video --calc_smpl --render_mesh -i=d:/110598028/videos/high_price_eggs.mp4 -o=../output/high_price_eggs_romp/high_price_eggs_romp_result.mp4 --save_video --GPU 0



python -m bev.main -m video -i d:/110598028/videos/persident.mp4 -o ../output/persident_bev/persident_bev_result.mp4 --save_video --GPU 0

python -m romp.main --mode=video --calc_smpl --render_mesh -i=d:/110598028/videos/persident.mp4 -o=../output/persident_romp/persident_romp_result.mp4 --save_video --GPU 0
