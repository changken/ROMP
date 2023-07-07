@REM python -m romp.main --mode=video --calc_smpl --render_mesh --show_items=tracking,mesh -i=D:/110598028/input_video_dataset/run_away/nick_japanese_more.mp4 -o=D:/110598028/ROMP/output/nick/results.mp4 --save_video -t -sc=3. --onnx
@REM 少了 --show_items=tracking,mesh
@REM python -m romp.main --mode=video --calc_smpl --render_mesh -i=D:/110598028/input_video_dataset/run_away/nick_japanese_more.mp4 -o=D:/110598028/ROMP/output/nick/results.mp4 --save_video -t -sc=3.

@REM python -m romp.main --mode=video --calc_smpl --render_mesh --show_items=tracking,mesh -i=D:/110598028/input_video_dataset/run_away/boss_imb_more.mp4 -o=D:/110598028/ROMP/output/imb_boss_more/results.mp4 --save_video -t -sc=3. --onnx
@REM python -m romp.main --mode=video --calc_smpl --render_mesh --show_items=tracking,mesh -i=D:/110598028/input_video_dataset/run_away/president_weihan_more.mp4 -o=D:/110598028/ROMP/output/president_weihan_more/results.mp4 --save_video -t -sc=3. --onnx
@REM python -m romp.main --mode=video --calc_smpl --render_mesh --show_items=tracking,mesh -i=D:/110598028/input_video_dataset/run_away/str_network_short.mp4 -o=D:/110598028/ROMP/output/str_network_short/results.mp4 --save_video -t -sc=3. --onnx

@REM python -m bev.main --mode video -i D:/110598028/input_video_dataset/run_away/nick_japanese_more.mp4 -o D:/110598028/ROMP/output/nick_BEV/BEV_results.mp4 --save_video --show_items=tracking,pj2d -t -sc=3.
@REM  少了 --show_items=tracking,pj2d
python -m bev.main --mode video -i D:/110598028/input_video_dataset/run_away/nick_japanese_more.mp4 -o D:/110598028/ROMP/output/nick_BEV/BEV_results.mp4 --save_video -t -sc=3.

@REM 有視覺化 onnx
@REM python -m romp.main --mode webcam --onnx --calc_smpl --render_mesh -t -sc=3. --show --show_items=tracking,mesh
@REM 有視覺化 僅有cuda cudnn
@REM python -m romp.main --mode webcam --calc_smpl --render_mesh -t -sc=3. --show --show_items=tracking,mesh
@REM 全部拔掉 onnx
@REM python -m romp.main --mode webcam --onnx -t -sc=3.
@REM 全部拔掉 onnx 連tracking都不用
@REM python -m romp.main --mode webcam --onnx

@REM 僅有cuda cudnn
@REM python -m romp.main --mode webcam -t -sc=3.
@REM 全部拔掉 僅有cuda cudnn 連tracking都不用
@REM python -m romp.main --mode webcam


@REM 有視覺化 cuda CUDNN 
@REM python -m bev.main --mode webcam  -t -sc=3. --show --show_items=tracking,mesh
@REM 全部拔掉
@REM python -m bev.main --mode webcam  -t -sc=3.
