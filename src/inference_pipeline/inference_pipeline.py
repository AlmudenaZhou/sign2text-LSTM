from ..video.live_video import video_live_for_inference
from ..training_pipeline.train_model import model_architecture


def load_model(model_filename):
    model = model_architecture()
    model.load_weights(model_filename)
    return model


def inference_pipeline_workflow(model_filename):
    model = load_model(model_filename)
    video_live_for_inference(model, camera_index=0)
