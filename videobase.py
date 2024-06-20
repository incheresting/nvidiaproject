#!/usr/bin/python3
import jetson_inference
import jetson_utils

import argparse
import sys
from jetson_inference import imageNet
from jetson_utils import videoSource, videoOutput, cudaFont, Log

# parse the command line
parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=imageNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())
parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="googlenet", help="model to use, can be:  googlenet, resnet-18, etc. (see --help for others)")
opt = parser.parse_args()

# Load the network
net = imageNet(opt.network)

input = videoSource(opt.input, argv=sys.argv)
output = videoOutput(opt.output, argv=sys.argv)
font = cudaFont()

# process frames until EOS or the user exits
while True:
    # capture the next image
    img = input.Capture()

    if img is None: # timeout
        continue  

    class_id, confidence = net.Classify(img)

    # draw predicted class labels
    classLabel = net.GetClassLabel(class_id)
    confidence *= 100.0

    print(f"imagenet:  {confidence:05.2f}% class #{class_id} ({classLabel})")
                         
    # render the image
    output.Render(img)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(net.GetNetworkName(), net.GetNetworkFPS()))

    # print out performance info
    net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break
#type in the terminal "python3 videobase.py /dev/video0 [desired_file_name].mp4"