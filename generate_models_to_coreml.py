import torch
import torchvision
import torch.nn as nn
import coremltools as ct
import timm
import sys
import os

from core.utils import print_model_info
from core.scp_model import MobileFacenet
from core.model import MobileFacenet as base
import coremltools
import coremltools.proto.FeatureTypes_pb2 as ft
from coremltools.models.neural_network import quantization_utils
from coremltools.models.utils import rename_feature
from torch.utils.mobile_optimizer import optimize_for_mobile

NUM_CLASS = 3


def get_nn(spec):
    if spec.WhichOneof("Type") == "neuralNetwork":
        return spec.neuralNetwork
    elif spec.WhichOneof("Type") == "neuralNetworkClassifier":
        return spec.neuralNetworkClassifier
    elif spec.WhichOneof("Type") == "neuralNetworkRegressor":
        return spec.neuralNetworkRegressor
    else:
        raise ValueError("MLModel does not have a neural network")

    
def convert_to_pytorch(torch_model, example_input, filename):
    torch_model.eval()
    traced_model = torch.jit.trace(torch_model, example_input)

    # Convert to Core ML using the Unified Conversion API
    model = ct.convert(
        traced_model,
        inputs=[ct.ImageType(name="image", shape=example_input.shape)],
    )

    if not os.path.exists('mlmodels'):
        os.makedirs('mlmodels')

    model.save("mlmodels/" + filename)

def convert_mobilenetv2(width=1.0):
    # Load a pre-trained version of MobileNetV2
    torch_model = torchvision.models.mobilenet_v2(num_classes=NUM_CLASS, pretrained=False, width_mult=width)
    example_input = torch.rand(1, 3, 224, 224) 
    convert_to_pytorch(torch_model, example_input, "mobilenet_v2_" + str(width) + ".mlmodel")


def convert_mnasnet():
    torch_model = mnasnet0_5(pretrained=False)
    torch_model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True),nn.Linear(1280, NUM_CLASS))
    example_input = torch.rand(1, 3, 256, 256) 
    convert_to_pytorch(torch_model, example_input, "mnasnet0_5.mlmodel")

def convert_timm(m):
    torch_model = timm.create_model(m, pretrained=False, num_classes=NUM_CLASS, exportable=True)
    example_input = torch.rand(1, 3, 256, 256) 
    convert_to_pytorch(torch_model, example_input, "timm_" + str(m) + ".mlmodel")

def convert_all_timm():
    models = timm.list_models()

    for m  in models:
        try:
            print("Converting model : ", m)
            convert_timm(m)
        except:
            print ("Error when converting ", m)
            print ( sys.exc_info()[0])
    print(m)

def convert_mobilenetv3(width=1.0):
    
    with set_layer_config(scriptable=True, exportable=True, no_jit=True):
        torch_model = _gen_mobilenet_v3('mobilenetv3_small_100', width, pretrained=False, num_classes=NUM_CLASS)
        example_input = torch.rand(1, 3, 224, 224)
        convert_to_pytorch(torch_model, example_input, "mobilenet_v3_" + str(width)  +".mlmodel")

if __name__ == "__main__":
    model_in_file = None

    input_size = (112,96)

    model_name = "mobilenet"
    # model_in_file = "KR20192_coco2017_80/33/model/model.pt"
    # model_in_file = 'tnwls.pt'
    model_out_file = "mobilenet"

    torch_model = base()
    print_model_info(torch_model, (3, 112, 96))


    example_input = torch.rand(1, 3, 112, 96) 

    if model_in_file is not None:
        print ("loading weights ...")
        d = torch.load(model_in_file, map_location="cpu")
        torch_model.load_state_dict(d['model'])

    example_input = torch.rand(1, 3, 112, 96) 
    torch_model.eval()

    # Convert to Pytorch
    print ("Export to Pytorch ...")
    traced_model = torch.jit.trace(torch_model, example_input)
    scripted_model = torch.jit.script(traced_model)
    torchscript_model_optimized = optimize_for_mobile(scripted_model)

    torch.jit.save(torchscript_model_optimized, model_out_file + '.pt')


    print("Export to ONNX ...")
    input_names = ["image"]
    output_names = ["output"]

    torch.onnx.export(torch_model,
                      example_input,
                      model_out_file + ".onnx",
                      verbose=False,
                      input_names=input_names,
                      output_names=output_names)

    traced_model = torch.jit.trace(torch_model, example_input)
    class_labels = ["no_person", "one_person", "two_or_more_person"]
    model = ct.convert(
        traced_model,
        inputs=[ct.ImageType(name="image", shape=example_input.shape)], #name "input_1" is used in 'quickstart'
        # classifier_config = ct.ClassifierConfig(class_labels) # provide only if step 4 was performed

    )
    # model.author = 'Soojin Jang & Nicolas Monet from Clova Avatar Team'
    # model.short_description = 'Human Detector Lightweight Deep Neural Network'


    model.save(model_out_file + ".mlmodel")


    spec = coremltools.utils.load_spec(model_out_file + ".mlmodel")

    red_std = 58.395 / 255 #= 0.229
    green_std = 57.12 / 255 #= 0.224
    blue_std = 57.375 / 255 #= 0.225

    image_scale = 1 / 255.0
    red_bias = -123.68 / 255.0 #= -0.485
    green_bias = -116.779 / 255.0 #= -0.456
    blue_bias = -103.939 / 255.0 #= -0.406

    nn = get_nn(spec)

    #print(nn.preprocessing)

    pp = nn.preprocessing[0]

    pp.scaler.channelScale = 1 / 255.0
    pp.scaler.redBias = -0.485
    pp.scaler.greenBias = -0.456
    pp.scaler.blueBias = -0.406

    #normalized_r = r * image_scale + red_bias #= r / 255.0 - 0.485
    #normalized_g = g * image_scale + green_bias #= g / 255.0 - 0.456
    #normalized_b = b * image_scale + blue_bias #= b / 255.0 - 0.406

    #really_normalized_r = normalized_r / red_std #= normalized_r / 0.229
    #really_normalized_g = normalized_g / green_std #= normalized_g / 0.224
    #really_normalized_b = normalized_b / blue_std #= normalized_b / 0.225

    import copy
    old_layers = copy.deepcopy(nn.layers)

    del nn.layers[:]

    input_name = old_layers[0].input[0]
    new_layer_output = input_name + "_scaled"

    new_layer = nn.layers.add()
    new_layer.name = "input_scale_layer"
    new_layer.input.append(input_name)
    new_layer.output.append(new_layer_output)

    new_layer.scale.shapeScale.extend([3, 1, 1])
    new_layer.scale.scale.floatValue.extend([1/0.229, 1/0.224, 1/0.225])
    new_layer.scale.hasBias = False

    nn.layers.extend(old_layers)

    nn.layers[1].input[0] = new_layer_output

    coremltools.utils.save_spec(spec, model_out_file + ".mlmodel")
    # if model_in_file:
    #     coremltools.utils.save_spec(spec, "iOS/MLModelCamera/models/" + model_out_file + ".mlmodel")
    #     print ("ML Model copied to MLModel Camera demo app !")
    print ("Done.")

    model_fp32 = coremltools.models.MLModel(model_out_file + ".mlmodel")
    model_fp16 = quantization_utils.quantize_weights(model_fp32, nbits=16)

    model_fp16.save(model_out_file + "_FP16.mlmodel")
    # model_fp16.save("iOS/MLModelCamera/models/" + model_out_file + "_FP16.mlmodel")
