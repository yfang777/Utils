from pathlib import Path
import os
import time
import numpy as np
import cv2
import open3d as o3d
from threadpoolctl import threadpool_limits
import multiprocess as mp
import threading
from threading import Lock




from .camera.multi_realsense import MultiRealsense
from .camera.single_realsense import SingleRealsense


class Perception(mp.Process):
    name = "Perception"

    def __init__(
        self,
        root,
        realsense: MultiRealsense | SingleRealsense, 
        capture_fps, 
        record_fps, 
        record_time, 
        process_func,
        exp_name=None,
        verbose=False,
    ):
        super().__init__()
        self.verbose = verbose

        self.capture_fps = capture_fps
        self.record_fps = record_fps
        self.record_time = record_time
        self.exp_name = exp_name
        self.root = Path(root)

        if self.exp_name is None:
            assert self.record_fps == 0

        self.realsense = realsense
        # self.perception_q = mp.SimpleQueue()
        self.perception_q = mp.Queue(maxsize=1)
        # self.perception_q.cancel_join_thread()

        self.process_func = process_func
        self.num_cam = len(realsense.cameras.keys())
        self.alive = mp.Value('b', False)
        self.record_restart = mp.Value('b', False)
        self.record_stop = mp.Value('b', False)
        self.record_failed = mp.Value('b', False)

    def log(self, msg):
        if self.verbose:
            print(f"\033[92m{self.name}: {msg}\033[0m")

    @property
    def can_record(self):
        return self.record_fps != 0

    def run(self):
        # limit threads
        threadpool_limits(1)
        cv2.setNumThreads(1)

        realsense = self.realsense
        capture_fps = self.capture_fps
        record_fps = self.record_fps
        record_time = self.record_time

        cameras_output = None
        recording_frame = float("inf")  # local record step index (since current record start), record fps
        record_start_frame = 0  # global step index (since process start), capture fps
        is_recording = False  # recording state flag
        timestamps_f = None

        eps_idx = 0
        index = 0
        while self.alive.value:
            index += 1
            cameras_output = realsense.get(out=cameras_output)
            get_time = time.time()
            timestamps = [cameras_output[i]['timestamp'].item() for i in range(self.num_cam)]
            if is_recording and not all([abs(timestamps[i] - timestamps[i+1]) < 0.05 for i in range(self.num_cam - 1)]):
                print(f"Captured at different timestamps: {[f'{x:.2f}' for x in timestamps]}")

            # treat captured time and record time as the same
            process_start_time = get_time
            process_out = self.process_func(cameras_output) if self.process_func is not None else cameras_output
            self.log(f"process time: {time.time() - process_start_time}")
            
            if not self.perception_q.full():
                self.perception_q.put(process_out) 

            # self.perception_q.put(process_out) 
            
            if self.can_record:
                if not is_recording and self.record_restart.value == True:
                    self.record_restart.value = False
                    self.record_failed.value = False

                    recording_frame = 0
                    record_start_time = get_time
                    record_start_frame = cameras_output[0]['step_idx'].item()
                    print("Start Recording")
                    print(self.root)

                    record_dir = self.root / self.exp_name / f"{record_start_time:.0f}"
                    print(record_dir)
                    os.makedirs(record_dir, exist_ok=True)
                    timestamps_f = open(f'{record_dir}/timestamps.txt', 'a')
                    
                    for i in range(self.num_cam):
                        os.makedirs(f'{record_dir}/camera_{i}/rgb', exist_ok=True)
                        os.makedirs(f'{record_dir}/camera_{i}/depth', exist_ok=True)
                    is_recording = True
                
                elif is_recording and (
                    self.record_stop.value == True or 
                    (recording_frame >= record_time * record_fps)
                ):
                    finish_time = get_time
                    if self.record_failed.value == False:
                        print(f"-------------- eps_idx {eps_idx} --------------")
                        print(f"is_recording {is_recording}, self.record_stop.value {self.record_stop.value}, recording time {recording_frame}, max recording time {record_time} * {record_fps}")
                        print(f"total time: {finish_time - record_start_time}")
                        print(f"fps: {recording_frame / (finish_time - record_start_time)}")
                        eps_idx += 1
                    else:
                        print(f"Recording failed, eps_idx {eps_idx}, recording time {recording_frame}, max recording time {record_time} * {record_fps}")
                        print(f"total time: {finish_time - record_start_time}")
                        print(f"fps: {recording_frame / (finish_time - record_start_time)}")
                        np.savetxt(record_dir / "failed.txt", np.array([1]), fmt="%d")

                    is_recording = False
                    timestamps_f.close()
                    self.record_restart.value = False
                    self.record_stop.value = False
                    self.record_failed.value = False
                else:
                    self.record_restart.value = False
                    self.record_stop.value = False

                # record the frame according to the record_fps
                if is_recording and cameras_output[0]['step_idx'].item() >= (recording_frame * (capture_fps // record_fps) + record_start_frame):
                    timestamps_f.write(' '.join(
                        [str(cameras_output[i]['step_idx'].item()) for i in range(self.num_cam)] + 
                        [str(np.round(cameras_output[i]['timestamp'].item() - record_start_time, 3)) for i in range(self.num_cam)] + 
                        [str(cameras_output[i]['timestamp'].item()) for i in range(self.num_cam)]
                    ) + '\n')
                    timestamps_f.flush()
                    for i in range(self.num_cam):
                        cv2.imwrite(f'{record_dir}/camera_{i}/rgb/{recording_frame:06}.jpg', cameras_output[i]['color'])
                        cv2.imwrite(f'{record_dir}/camera_{i}/depth/{recording_frame:06}.png', cameras_output[i]['depth'])
                    recording_frame += 1

                    output = self.perception_q.get()
    
        if self.can_record:
            if timestamps_f is not None and not timestamps_f.closed:
                timestamps_f.close()
            finish_time = time.time()

        print("Perception Exit Run")

    def start(self):
        self.alive.value = True
        super().start()

    def stop(self):
        self.alive.value = False
        self.perception_q.close()
    
    def set_record_start(self):
        if self.record_fps == 0:
            print("record disabled because record_fps is 0")
            assert self.record_restart.value == False
        else:
            self.record_restart.value = True
            # print("record restart cmd received")

    def set_record_stop(self):
        if self.record_fps == 0:
            print("record disabled because record_fps is 0")
            assert self.record_stop.value == False
        else:
            self.record_stop.value = True
            # print("record stop cmd received")
    
    def set_record_failed(self):
        if self.record_fps == 0:
            print("record disabled because record_fps is 0")
            assert self.record_failed.value == False
        else:
            self.record_failed.value = True
            print("record failed cmd received")


def perception_func_test():
    import multiprocessing, logging
    mp_logger = multiprocessing.log_to_stderr()
    mp_logger.setLevel(multiprocessing.SUBDEBUG)
    # def process_perception_out(perception):
    #     while perception.alive.value:
    #         try:
    #             output = perception.perception_q.get(timeout=0.1)
    #             print(f"Figure: {time.time()}")
    #             time.sleep(1)
    #         except Exception:
    #             continue


    # Create an instance of MultiRealsense or SingleRealsense
    realsense = MultiRealsense(
        resolution=(848, 480),
        capture_fps=30,
        enable_color=True,
        enable_depth=True,
        verbose=False
    )
    realsense.start(wait=True)
    realsense.restart_put(start_time=time.time())
    

    # Create an instance of the Perception class
    perception = Perception(
        realsense=realsense, 
        capture_fps=30, # should be the same as the camera fps
        record_fps=30, # 0 for no record
        record_time=10, # in seconds
        process_func=None,
        exp_name="test",
        verbose=False
    )

    # Start the perception process
    perception.start()
    print("Perception Start")
    perception.set_record_start()
    print("Perception Record Start")
    time.sleep(5)
    perception.set_record_stop()
    time.sleep(2)
    perception.stop()
    realsense.stop(wait=True)

    perception.kill()

    # exit
    print("Test finished")

if __name__ == "__main__":
    perception_func_test()
