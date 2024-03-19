import importlib
import os
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
import time
import cv2 as cv

from tqdm import tqdm
from pathlib import Path
import numpy as np
from PIL import Image
import tempfile

from lib.test.evaluation.tracker import Tracker
from segment_anything.build_sam import sam_model_registry
from segment_anything.predictor import SamPredictor
from gradio import processing_utils


class TrackerSAM(Tracker):
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """
    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, display_name: str = None, result_only=False, tracker_params=None):
        super().__init__(name, parameter_name, dataset_name, run_id, display_name, result_only, tracker_params)

        # sam predictor
        sam = sam_model_registry["vit_b"](checkpoint="/data/songtianhui.sth/projects/sam-hq/train/pretrained_checkpoint/sam_vit_b_01ec64.pth").cuda()
        self.sam_predictor = SamPredictor(sam)

    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """

        # params = self.get_parameters()
        params = self.params

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)

        # elif multiobj_mode == 'parallel':
        #     tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        output_boxes = []
        cap = cv.VideoCapture(videofilepath)
        fps = cap.get(cv.CAP_PROP_FPS)
        nframes = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        success, frame = cap.read()
        frame_disp = frame.copy()
        # cv.imshow(display_name, frame)

        output_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        mask_dir = tempfile.mkdtemp()
        # output_file = "output.mp4"
        out_writer = cv.VideoWriter(output_file.name, cv.VideoWriter_fourcc(*"mp4v"), 30, frameSize=(frame.shape[1], frame.shape[0]))

        def _build_init_info(box):
            return {'init_bbox': box}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        else:
            raise NotImplementedError("We haven't support cv_show now.")
            # while True:
            #     # cv.waitKey()
            #     frame_disp = frame.copy()
            #
            #     cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
            #                1.5, (0, 0, 0), 1)
            #
            #     x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
            #     init_state = [x, y, w, h]
            #     tracker.initialize(frame, _build_init_info(init_state))
            #     output_boxes.append(init_state)
            #     break
        frame_id = 0

        def get_mask(image, box, out_writer):
            nonlocal frame_id
            x1, y1, w, h = box
            box = [x1, y1, x1 + w, y1 + h]
            box = np.array(box)

            point = [[x1 + w * 0.5, y1 + h * 0.5]]
            point = np.array(point)
            point_label = np.array([1])

            self.sam_predictor.set_image(image, image_format="BGR")
            masks_np, iou_predictions_np, low_res_masks_np = self.sam_predictor.predict(point_coords=None,
                                                                                        point_labels=None,
                                                                                        box=box[None, :],
                                                                                        multimask_output=False)
            # masks_np, iou_predictions_np, low_res_masks_np = self.sam_predictor.predict(point_coords=point,
            #                                                                             point_labels=point_label,
            #                                                                             box=None,
            #                                                                             multimask_output=False)
            Image.fromarray((masks_np[0].astype(np.uint8) * 255)).save(os.path.join(mask_dir, "mask_{:04d}.png".format(frame_id)))
            masks_np_red = np.zeros((image.shape), dtype=np.uint8)
            masks_np_red[:, :, 2] = masks_np * 255
            blend_img = 0.5 * masks_np_red + 0.7 * image
            # cv.imwrite("debug/blend_{:04d}.png".format(frame_id), blend_img)
            cv.rectangle(blend_img, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)

            out_writer.write(np.clip(blend_img, 0, 255).astype(np.uint8))
            # Image.fromarray((blend_img.astype(np.uint8))).save("debug/blend_{:04d}.png".format(frame_id))

        get_mask(frame_disp, optional_box, out_writer)

        frames = []
        for _ in tqdm(range(nframes - 1), desc="Reading video"):
            success, frame = cap.read()
            frames.append(frame)

        # while True:
        for frame in tqdm(frames, desc="Tracking"):
            frame_id += 1
            # ret, frame = cap.read()

            if frame is None:
                break

            frame_disp = frame.copy()

            # Draw box
            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox']]    # x1y1wh
            output_boxes.append(state)

            # sam
            get_mask(frame_disp, state, out_writer)

        # When everything done, release the capture
        cap.release()
        # out_writer.release()
        # cv.destroyAllWindows()
        print("done")

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')

        # convert to h264
        # result_file = processing_utils.convert_video_to_playable_mp4(output_file)
        # cmd = f"ffmpeg -i {output_file} -vcodec h264 _{output_file}"
        # print(cmd)
        # os.system(cmd)
        # output_file = "_" + output_file

        return output_file.name, mask_dir
