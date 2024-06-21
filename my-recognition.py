#!/usr/bin/python3
import sys
import argparse

from jetson_inference import imageNet
from jetson_utils import videoSource, videoOutput, cudaFont, Log
parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=imageNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())
parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="googlenet", help="model to use, can be:  googlenet, resnet-18, etc. (see --help for others)")
opt = parser.parse_args()

food_protein_content = {
    'hen': 31.0,  # Chicken, broilers or fryers, meat only, cooked, roasted
    'cock': 31.0,  # Similar to chicken, using same value
    'ostrich, struthio camelus': 29.0,  # Ostrich, ground, raw
    'brambling, fringilla montifringilla': 25.0,  # Estimate based on game meat
    'ostrich': 29.0,  # Same as above
    'house sparrow, passer domesticus': 25.0,  # Estimate based on game meat
    'swan': 25.0,  # Estimate based on game meat
    'goldfinch': 25.0,  # Estimate based on game meat
    'chaffinch': 25.0,  # Estimate based on game meat
    'indian peafowl': 25.0,  # Estimate based on game meat
    'grey partridge, perdix perdix': 30.0,  # Partridge, cooked
    'capercaillie, tetrao urogallus': 30.0,  # Estimate based on similar game birds
    'great cormorant': 25.0,  # Estimate based on game meat
    'common pheasant': 25.0,  # Pheasant, raw, meat only
    'bald eagle': 25.0,  # Estimate based on game meat
    'common buzzard': 25.0,  # Estimate based on game meat
    'barn owl': 25.0,  # Estimate based on game meat
    'little owl': 25.0,  # Estimate based on game meat
    'common blackbird': 25.0,  # Estimate based on game meat
    'european robin': 25.0,  # Estimate based on game meat
    'common cuckoo': 25.0,  # Estimate based on game meat
    'white stork': 25.0,  # Estimate based on game meat
    'mute swan': 25.0,  # Estimate based on game meat
    'tufted duck': 25.0,  # Estimate based on game meat
    'great crested grebe': 25.0,  # Estimate based on game meat
    'snowy owl': 25.0,  # Estimate based on game meat
    'eurasian blue tit': 25.0,  # Estimate based on game meat
    'eurasian wren': 25.0,  # Estimate based on game meat
    'eurasian hobby': 25.0,  # Estimate based on game meat
    'eurasian sparrowhawk': 25.0,  # Estimate based on game meat
    'little grebe': 25.0,  # Estimate based on game meat
    'mallard': 28.0,  # Duck, meat only, raw
    'great spotted woodpecker': 25.0,  # Estimate based on game meat
    'white-backed woodpecker': 25.0,  # Estimate based on game meat
    'black woodpecker': 25.0,  # Estimate based on game meat
    'common starling': 25.0,  # Estimate based on game meat
    'woodpigeon': 25.0,  # Pigeon, meat only, raw
    'grey heron': 25.0,  # Estimate based on game meat
    'mew gull': 25.0,  # Estimate based on game meat
    'rock dove': 25.0,  # Pigeon, meat only, raw
    'northern raven': 25.0,  # Estimate based on game meat
    'magpie': 25.0,  # Estimate based on game meat
    'alpine chough': 25.0,  # Estimate based on game meat
    'house crow': 25.0,  # Estimate based on game meat
    'hooded crow': 25.0,  # Estimate based on game meat
    'rook': 25.0,  # Estimate based on game meat
    'jackdaw': 25.0,  # Estimate based on game meat
    'great tit': 25.0,  # Estimate based on game meat
    'willow tit': 25.0,  # Estimate based on game meat
    'coal tit': 25.0,  # Estimate based on game meat
    'marsh tit': 25.0,  # Estimate based on game meat
    'lesser whitethroat': 25.0,  # Estimate based on game meat
    'common whitethroat': 25.0,  # Estimate based on game meat
    'barn swallow': 25.0,  # Estimate based on game meat
    'long-tailed tit': 25.0,  # Estimate based on game meat
    'common swift': 25.0,  # Estimate based on game meat
    'rose-ringed parakeet': 25.0,  # Estimate based on game meat
    'goldcrest': 25.0,  # Estimate based on game meat
    'firecrest': 25.0,  # Estimate based on game meat
    'song thrush': 25.0,  # Estimate based on game meat
    'redwing': 25.0,  # Estimate based on game meat
    'eurasian nuthatch': 25.0,  # Estimate based on game meat
    'great reed warbler': 25.0,  # Estimate based on game meat
    'eurasian treecreeper': 25.0,  # Estimate based on game meat
    'common chiffchaff': 25.0,  # Estimate based on game meat
    'wood warbler': 25.0,  # Estimate based on game meat
    'willow warbler': 25.0,  # Estimate based on game meat
    'great grey owl': 25.0,  # Estimate based on game meat
    'eurasian eagle-owl': 25.0,  # Estimate based on game meat
    'long-eared owl': 25.0,  # Estimate based on game meat
    'eurasian scops owl': 25.0,  # Estimate based on game meat
    'eurasian pygmy owl': 25.0,  # Estimate based on game meat
    'eurasian dotterel': 25.0,  # Estimate based on game meat
    'grey plover': 25.0,  # Estimate based on game meat
    'common ringed plover': 25.0,  # Estimate based on game meat
    'little ringed plover': 25.0,  # Estimate based on game meat
    'eurasian curlew': 25.0,  # Estimate based on game meat
    'black-headed gull': 25.0,  # Estimate based on game meat
    'common gull': 25.0,  # Estimate based on game meat
    'great black-backed gull': 25.0,  # Estimate based on game meat
    'lesser black-backed gull': 25.0,  # Estimate based on game meat
    'herring gull': 25.0,  # Estimate based on game meat
    'caspian gull': 25.0,  # Estimate based on game meat
    'common tern': 25.0,  # Estimate based on game meat
    'arctic tern': 25.0,  # Estimate based on game meat
    'little tern': 25.0,  # Estimate based on game meat
    'black tern': 25.0,  # Estimate based on game meat
    'white-winged tern': 25.0,  # Estimate based on game meat
    'spinach': 2.9,
    'broccoli': 2.8,
    'kale': 4.3,
    'peas': 5.4,
    'asparagus': 2.2,
    'brussels sprouts': 3.4,
    'cauliflower': 1.9,
    'mushrooms': 3.1,
    'potatoes': 2.0,
    'sweet potatoes': 1.6,
    'avocado': 2.0,
    'guava': 2.6,
    'blackberries': 1.4,
    'orange': 1.2,
    'banana': 1.1,
    'apples': 0.3,
    'strawberries': 0.8,
    'blueberries': 0.7,
    'grapes': 0.6,
    'watermelon': 0.6,
    'almonds': 21.2,
    'peanuts': 25.8,
    'walnuts': 15.2,
    'cashews': 18.2,
    'chia seeds': 16.5,
    'flaxseeds': 18.3,
    'sunflower seeds': 20.8,
    'pumpkin seeds': 19.0,
    'sesame seeds': 17.0,
    'pistachios': 20.2,
    'quinoa': 4.4,
    'oats': 16.9,
    'brown rice': 2.6,
    'barley': 12.5,
    'millet': 11.0,
    'buckwheat': 13.3,
    'bulgur': 12.3,
    'wheat': 13.7,
    'corn (maize)': 9.4,
    'sorghum': 10.6,
    'lentils': 25.8,
    'chickpeas': 19.0,
    'black beans': 21.6,
    'kidney beans': 24.0,
    'soybeans': 36.5,
    'pinto beans': 21.4,
    'navy beans': 22.3,
    'edamame': 11.9,
    'mung beans': 24.0,
    'lemon': 1.1,
    'cherries': 1.0
}

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

    print(f"imagenet:  {confidence:05.2f}%  {classLabel} {food_protein_content[classLabel]} grams of protein per 100g ")
    if classLabel in food_protein_content:
        print("food_protein_content[classLabel] per 100g")
    font.OverlayText(img, text=f"{confidence:05.2f}% {classLabel} {food_protein_content[classLabel]} grams of protein per 100g", 
                         x=5, y=5 + (font.GetSize() + 5),
                         color=font.White, background=font.Gray40)              
    # render the image
    output.Render(img)

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break

# img = jetson_utils.loadImage(opt.filename)

# net = jetson_inference.imageNet(opt.network)

# class_idx, confidence = net.Classify(img)

# class_desc = net.GetClassDesc(class_idx)

