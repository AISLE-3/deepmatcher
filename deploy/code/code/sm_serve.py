import os
import sys
import json
import torch
import torch.nn.functional as F
import logging
import pandas as pd

from deepmatcher.models.core import MatchingModel
from deepmatcher.data.custom_dataset import DeepMatcherDataset
from deepmatcher.data.encoder.text_encoders import HFTextEncoder
from deepmatcher.data.encoder.image_encoders import DINOImageEncoder

from sagemaker_inference import decoder, encoder

CONFIG = {
    "text_encoder_identifier" : "albert-base-v2",
    "text_encoder_trainable" : False,
    "text_encoder_seq_to_vec" : "cls",
    "text_attrs" : ['title'],
    "image_attr" : '',
    "prefixes" : ["left_", "right_"]
}

print("defined config:")
print(CONFIG)

tokenizer = None

def model_fn(model_dir):
    print('executing model_fn')
    print('model_dir', model_dir)
    print('model_dir_contents:', os.listdir(model_dir))

    text_encoder_identifier = CONFIG["text_encoder_identifier"]
    text_encoder_trainable = CONFIG['text_encoder_trainable']
    text_encoder_seq_to_vec = CONFIG['text_encoder_seq_to_vec']
    text_encoder = HFTextEncoder(text_encoder_identifier, trainable=text_encoder_trainable, seqtovec=text_encoder_seq_to_vec)
    global tokenizer
    tokenizer = text_encoder.tokenizer
    # image_encoder = DINOImageEncoder()

    model = MatchingModel.load_from_state(
        os.path.join(model_dir, "model.pth"),
        text_encoder=text_encoder,
        image_encoder=None,
    )
    print(model)
    print('loaded model')
    model = model.eval()
    return model

def input_fn(input_data, content_type):
    """
    Args:
        request_body (str): Expecting this to be a json document that contains elements of pairs that are to be matched together, eg: [{left_id: "offer_id_666", left_title: "some title text", right_id: "offer_id_555", right_title: "more title text"},{..more..} ..]
        request_content_type (str): Expecting this to be application/json
    """
    assert tokenizer is not None

    print('input_fn:', 'input_data', input_data, 'type', type(input_data), 'content_type', content_type)
    input_data = decoder.decode(input_data, content_type)
    print('decoded input data:', input_data, 'type', type(input_data))
    # input_data = input_data.item()
    input_data = input_data.tolist()
    print('unpickled dtype:', type(input_data))
    if isinstance(input_data, str):
        input_data = json.loads(input_data)
    if isinstance(input_data, dict):
        input_data = [input_data]
    assert isinstance(input_data, list), f"expected dtype for input data : list but got : {type(input_data)}"
    assert isinstance(input_data[0], dict), f"expected dtype for iterable elements : dict but got : {type(input_data)}"

    data_df = pd.DataFrame(input_data)
    dataset =  DeepMatcherDataset(
        data_df=data_df,
        label_col=None,
        text_cols=CONFIG["text_attrs"],
        image_col=CONFIG["image_attr"],
        tokenizer=tokenizer,
        prefixes=CONFIG["prefixes"],
    )
    batch = [dataset[i] for i in range(len(dataset))]
    batch = dataset.collate_fn(batch)
    return batch['attrs']

def predict_fn(input_data, model):
    print('predict_fn', input_data)
    logits = model(input_data)
    print('logits', logits)
    preds = F.softmax(logits, dim=1)
    matches = torch.argmax(preds, dim=1)
    matches = matches.flatten().cpu().numpy()
    print('matches', matches)
    return matches
    
def output_fn(prediction, content_type):
    print('output_fn:', 'prediction:', prediction, 'content_type', content_type)
    try:
        return encoder.encode({"results" : list(prediction)}, content_type)
    except Exception as e:
        logging.exception(e)