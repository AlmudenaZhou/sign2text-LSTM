from src.generate_dataset import generate_dataset
from src.video.live_video import video_with_landmarks
from src.training_pipeline.main import training_pipeline_workflow
from src.inference_pipeline.inference_pipeline import inference_pipeline_workflow


model_filename = "sign2text.h5"

video_with_landmarks()

generate_dataset()

training_pipeline_workflow(model_filename)

inference_pipeline_workflow(model_filename)
