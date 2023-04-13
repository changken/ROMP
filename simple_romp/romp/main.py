from .model import ROMPv1
import cv2
import numpy as np
import os
import sys
import os.path as osp
import torch
from torch import nn
import argparse

from .post_parser import SMPL_parser, body_mesh_projection2image, parsing_outputs
from .utils import img_preprocess, create_OneEuroFilter, euclidean_distance, check_filter_state, \
    time_cost, download_model, determine_device, ResultSaver, WebcamVideoStream, convert_cam_to_3d_trans,\
    wait_func, collect_frame_path, progress_bar, get_tracked_ids, get_tracked_ids3D, smooth_results, convert_tensor2numpy, save_video_results
from vis_human import setup_renderer, rendering_romp_bev_results
from .post_parser import CenterMap


def romp_settings(input_args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description='ROMP: Monocular, One-stage, Regression of Multiple 3D People')
    parser.add_argument('-m', '--mode', type=str, default='image',
                        help='Inferece mode, including image, video, webcam')
    parser.add_argument('-i', '--input', type=str, default=None,
                        help='Path to the input image / video')
    parser.add_argument('-o', '--save_path', type=str, default=osp.join(
        osp.expanduser("~"), 'ROMP_results'), help='Path to save the results')
    parser.add_argument('--GPU', type=int, default=0,
                        help='The gpu device number to run the inference on. If GPU=-1, then running in cpu mode')
    parser.add_argument('--onnx', action='store_true',
                        help='Whether to use ONNX for acceleration.')

    parser.add_argument('-t', '--temporal_optimize', action='store_true',
                        help='Whether to use OneEuro filter to smooth the results')
    parser.add_argument('--center_thresh', type=float, default=0.25,
                        help='The confidence threshold of positive detection in 2D human body center heatmap.')
    parser.add_argument('--show_largest', action='store_true',
                        help='Whether to show the largest person only')
    parser.add_argument('-sc', '--smooth_coeff', type=float, default=3.,
                        help='The smoothness coeff of OneEuro filter, the smaller, the smoother.')
    parser.add_argument('--calc_smpl', action='store_false',
                        help='Whether to calculate the smpl mesh from estimated SMPL parameters')
    parser.add_argument('--render_mesh', action='store_true',
                        help='Whether to render the estimated 3D mesh mesh to image')
    parser.add_argument('--renderer', type=str, default='sim3dr',
                        help='Choose the renderer for visualizaiton: pyrender (great but slow), sim3dr (fine but fast)')
    parser.add_argument('--show', action='store_true',
                        help='Whether to show the rendered results')
    parser.add_argument('--show_items', type=str, default='mesh',
                        help='The items to visualized, including mesh,pj2d,j3d,mesh_bird_view,mesh_side_view,center_conf. splited with ,')
    parser.add_argument('--save_video', action='store_true',
                        help='Whether to save the video results')
    parser.add_argument('--frame_rate', type=int, default=24,
                        help='The frame_rate of saved video results')
    parser.add_argument('--smpl_path', type=str, default=osp.join(osp.expanduser(
        "~"), '.romp', 'smpl_packed_info.pth'), help='The path of smpl model file')
    parser.add_argument('--model_path', type=str, default=osp.join(
        osp.expanduser("~"), '.romp', 'ROMP.pkl'), help='The path of ROMP checkpoint')
    parser.add_argument('--model_onnx_path', type=str, default=osp.join(
        osp.expanduser("~"), '.romp', 'ROMP.onnx'), help='The path of ROMP onnx checkpoint')
    parser.add_argument('--root_align', type=bool, default=False,
                        help='Please set this config as True to use the ROMP checkpoints trained by yourself.')
    parser.add_argument('--webcam_id', type=int,
                        default=0, help='The Webcam ID.')
    args = parser.parse_args(input_args)

    if not torch.cuda.is_available():
        args.GPU = -1
        args.temporal_optimize = False
    if args.show:
        args.render_mesh = True
    if args.render_mesh or args.show_largest:
        args.calc_smpl = True
    if not os.path.exists(args.smpl_path):
        smpl_url = 'https://github.com/Arthur151/ROMP/releases/download/V2.0/smpl_packed_info.pth'
        download_model(smpl_url, args.smpl_path, 'SMPL')
    if not os.path.exists(args.model_path):
        romp_url = 'https://github.com/Arthur151/ROMP/releases/download/V2.0/ROMP.pkl'
        download_model(romp_url, args.model_path, 'ROMP')
    if not os.path.exists(args.model_onnx_path) and args.onnx:
        romp_onnx_url = 'https://github.com/Arthur151/ROMP/releases/download/V2.0/ROMP.onnx'
        download_model(romp_onnx_url, args.model_onnx_path, 'ROMP')
    return args


default_settings = romp_settings(input_args=[])


class ROMP(nn.Module):
    def __init__(self, romp_settings):
        super(ROMP, self).__init__()
        self.settings = romp_settings
        self.tdevice = determine_device(self.settings.GPU)
        self._build_model_()
        self._initilization_()

    def _build_model_(self):
        # 使用gpu
        if not self.settings.onnx:
            model = ROMPv1().eval()
            model.load_state_dict(torch.load(
                self.settings.model_path, map_location=self.tdevice))
            model = model.to(self.tdevice)
            self.model = nn.DataParallel(model)
        # 使用onnx
        else:
            try:
                import onnxruntime
            except:
                print(
                    'To use onnx model, we need to install the onnxruntime python package. Please install it by youself if failed!')
                if not torch.cuda.is_available():
                    os.system('pip install onnxruntime')
                else:
                    os.system('pip install onnxruntime-gpu')
                import onnxruntime
            print('creating onnx model')
            self.ort_session = onnxruntime.InferenceSession(self.settings.model_onnx_path,
                                                            providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
            print('created!')

    def _initilization_(self):
        # 中心點偵測
        self.centermap_parser = CenterMap(
            conf_thresh=self.settings.center_thresh)

        # 如果有calc smpl，要先初始化smpl parser
        if self.settings.calc_smpl:
            self.smpl_parser = SMPL_parser(
                self.settings.smpl_path).to(self.tdevice)

        # 如果有要開啟temporal optimization，要先初始化相關工具 1歐元
        if self.settings.temporal_optimize:
            self._initialize_optimization_tools_()

        # 如果有要開啟render mesh，要先初始化相關工具
        if self.settings.render_mesh:
            self.visualize_items = self.settings.show_items.split(',')
            self.renderer = setup_renderer(name=self.settings.renderer)

    # 單張圖片的前向傳播
    def single_image_forward(self, image):
        # 先預處理
        input_image, image_pad_info = img_preprocess(image)

        # 使用onnx
        if self.settings.onnx:
            center_maps, params_maps = self.ort_session.run(
                None, {'image': input_image.numpy().astype(np.float32)})
            center_maps, params_maps = torch.from_numpy(center_maps).to(
                self.tdevice), torch.from_numpy(params_maps).to(self.tdevice)
        else:
            # 使用gpu
            center_maps, params_maps = self.model(input_image.to(self.tdevice))
        params_maps[:, 0] = torch.pow(1.1, params_maps[:, 0])
        parsed_results = parsing_outputs(
            center_maps, params_maps, self.centermap_parser)
        return parsed_results, image_pad_info

    # 初始化temporal optimization相關工具
    def _initialize_optimization_tools_(self):
        self.OE_filters = {}  # 先初始化一個空的一歐元過濾器
        if not self.settings.show_largest:  # 如果不是只顯示最大的
            try:
                from norfair import Tracker  # 先試試看有沒有安裝norfair
            except:
                print(
                    'To perform temporal optimization, installing norfair for tracking.')
                os.system('pip install norfair')  # 安裝norfair
                from norfair import Tracker
            # 初始化tracker
            self.tracker = Tracker(
                distance_function=euclidean_distance, distance_threshold=200)  # 120
            self.tracker_initialized = False

    # 做temporal optimization
    def temporal_optimization(self, outputs, signal_ID):
        # 如果是第一次偵測到這個人，就要初始化一個過濾器
        check_filter_state(self.OE_filters, signal_ID,
                           self.settings.show_largest, self.settings.smooth_coeff)
        # 如果只要顯示最大的
        if self.settings.show_largest:
            # 找到最大的那個
            max_id = torch.argmax(outputs['cam'][:, 0])
            # 做一歐元過濾
            outputs['smpl_thetas'], outputs['smpl_betas'], outputs['cam'] = \
                smooth_results(self.OE_filters[signal_ID],
                               outputs['smpl_thetas'][max_id], outputs['smpl_betas'][max_id], outputs['cam'][max_id])
            # 轉成batch的形式
            outputs['smpl_thetas'], outputs['smpl_betas'], outputs['cam'] = outputs['smpl_thetas'].unsqueeze(
                0), outputs['smpl_betas'].unsqueeze(0), outputs['cam'].unsqueeze(0)
        else:
            # 如果不是只顯示最大的，就要用tracker來做tracking
            pred_cams = outputs['cam']

            print(f"cam: {pred_cams.cpu().numpy()}")

            # 先把所有的detection都做出來
            from norfair import Detection
            detections = [Detection(points=cam[[2, 1]]*512)
                          for cam in pred_cams.cpu().numpy()]
            # 如果還沒初始化tracker，就先初始化
            if not self.tracker_initialized:
                # 初始化tracker
                for _ in range(8):
                    tracked_objects = self.tracker.update(
                        detections=detections)
            # 初始化完畢
            tracked_objects = self.tracker.update(detections=detections)
            #  如果沒有偵測到任何人，就直接回傳
            if len(tracked_objects) == 0:
                return outputs

            print(f"tracked_objects: {tracked_objects}")

            # 如果有偵測到人，就要把偵測到的人的id找出來
            tracked_ids = get_tracked_ids(detections, tracked_objects)

            print(f"tracked_ids: {tracked_ids}")

            # 做一歐元過濾
            for ind, tid in enumerate(tracked_ids):
                # 如果這個人還沒有過濾器，就要先初始化一個
                if tid not in self.OE_filters[signal_ID]:
                    self.OE_filters[signal_ID][tid] = create_OneEuroFilter(
                        self.settings.smooth_coeff)

                # 做一歐元過濾
                outputs['smpl_thetas'][ind], outputs['smpl_betas'][ind], outputs['cam'][ind] = \
                    smooth_results(self.OE_filters[signal_ID][tid],
                                   outputs['smpl_thetas'][ind], outputs['smpl_betas'][ind], outputs['cam'][ind])

            # 把偵測到的人的id存起來
            outputs['track_ids'] = np.array(tracked_ids).astype(np.int32)
        return outputs

    # 對單張圖片進行前向傳播
    @time_cost('ROMP')
    def forward(self, image, signal_ID=0, **kwargs):
        # 單張圖片的前向傳播
        outputs, image_pad_info = self.single_image_forward(image)
        if outputs is None:  # 如果沒有偵測到人體，直接回傳None
            return None
        if self.settings.temporal_optimize:  # 如果有開啟濾波器
            # 將結果傳入temporal optimization來處理濾波
            outputs = self.temporal_optimization(outputs, signal_ID)
        # 算位移向量
        outputs['cam_trans'] = convert_cam_to_3d_trans(outputs['cam'])
        if self.settings.calc_smpl:  # 如果有calc smpl
            # 將結果轉換成smpl的格式
            outputs = self.smpl_parser(
                outputs, root_align=self.settings.root_align)
            # 將smpl的結果轉換成mesh，然後投影在圖片上
            outputs.update(body_mesh_projection2image(
                outputs['joints'], outputs['cam'], vertices=outputs['verts'], input2org_offsets=image_pad_info))
        if self.settings.render_mesh:  # 如果有render mesh
            # 設定rendering的參數
            rendering_cfgs = {'mesh_color': 'identity', 'items': self.visualize_items,
                              'renderer': self.settings.renderer}  # 'identity'
            # render 網路輸出的結果
            outputs = rendering_romp_bev_results(
                self.renderer, outputs, image, rendering_cfgs)
        if self.settings.show:  # 如果有show的話
            # 顯示該次的結果
            cv2.imshow('rendered', outputs['rendered_image'])
            wait_func(self.settings.mode)
        # 將結果轉換成numpy
        return convert_tensor2numpy(outputs)


def main():
    args = romp_settings()
    romp = ROMP(args)
    if args.mode == 'image':
        saver = ResultSaver(args.mode, args.save_path)
        image = cv2.imread(args.input)
        outputs = romp(image)
        saver(outputs, args.input)

    if args.mode == 'video':
        frame_paths, video_save_path = collect_frame_path(
            args.input, args.save_path)
        saver = ResultSaver(args.mode, args.save_path)
        for frame_path in progress_bar(frame_paths):
            image = cv2.imread(frame_path)
            outputs = romp(image)
            saver(outputs, frame_path)
        save_video_results(saver.frame_save_paths)
        if args.save_video:
            saver.save_video(video_save_path, frame_rate=args.frame_rate)

    if args.mode == 'webcam':
        cap = WebcamVideoStream(args.webcam_id)
        cap.start()
        while True:
            frame = cap.read()
            outputs = romp(frame)
        cap.stop()


if __name__ == '__main__':
    main()
