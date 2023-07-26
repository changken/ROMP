# 有可視化
python -m romp.main --mode=video --calc_smpl --render_mesh --show_items=tracking,mesh -i=../input_video_dataset/nick_japanese_more.mp4 -o=../output/nick/results.mp4 --save_video -t -sc=3. --onnx
# 拔掉可視化
python -m romp.main --mode=video  -i=../input_video_dataset/nick_japanese_more.mp4 -o=../output/nick/results.mp4 --save_video -t -sc=3. --onnx
