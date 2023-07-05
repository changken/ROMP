trace2 -i D:\110598028\input_video_dataset\run_away\nick_japanese_more.mp4 --subject_num=2 --save_path D:\110598028\ROMP\output\nick_japanese --results_save_dir D:\110598028\ROMP\output\nick_japanese --smpl_path D:\.romp\SMPL_NEUTRAL.pth  --save_video

::python -m trace2.show --smpl_model_folder D:/110598028/smpl_model_data --preds_path ../output/nick_japanese/nick_japanese_more.mp4.npz --frame_dir ../output/nick_japanese/nick_japanese_more_frames --img_ext jpg
