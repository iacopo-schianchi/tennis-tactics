from processor import VideoProcessor

FPS = 30

vp = VideoProcessor('p1', 'p2', FPS)
vp.process("../assets/clip.mp4")