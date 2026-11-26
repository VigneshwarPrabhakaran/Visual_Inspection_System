# Visual Inspection System

An AI-assisted visual inspection prototype for identifying visible defects in forged automotive components as images are captured at a conveyor inspection station.

## Project Overview

The system processes camera frames of a fuel rail and checks for defect regions. It uses two YOLO object-detection models in parallel: one receives the original RGB image, while the other receives a Canny edge representation of the same frame. Their predicted bounding boxes are compared using intersection over union (IoU). When the overlap passes a configured threshold, the prototype flags the region and displays a detection confidence score.

The project combines a conveyor concept designed by the hardware team with the vision pipeline developed by the AI team. The README focuses on the AI inspection workflow.

## Inspection Pipeline

1. Capture a frame of the component at the inspection point.
2. Send the original RGB frame to the first YOLO model.
3. Create a Canny edge image and send it to the second YOLO model.
4. Run both model branches in parallel and compare their detections using IoU.
5. Flag a defect when the bounding-box overlap exceeds the configured threshold, then show its location and confidence in the prototype interface.

IoU is used to compare the overlap between detections. The confidence shown for an individual prediction is not a measure of overall model accuracy.

## Data Collection and Model Development

- Images were collected manually from the target component.
- Images were labelled and augmented using Roboflow.
- Two YOLO models were trained with PyTorch.
- Visual Studio Code was used as the development and training environment.
- The models use complementary input representations: RGB appearance and Canny edges.

The project scope identifies forging-profile issues and visible surface defects such as cracks, dents, pores or porosity, edge defects, scratches, and surface irregularities. The exact classes represented in the current training dataset should be documented alongside the dataset or model configuration.

## Prototype Status

The current work demonstrates a parallel vision pipeline that localizes a candidate defect region and displays a confidence score. The conveyor design and the AI inspection pipeline were developed by separate parts of the team. This README does not claim production deployment, complete 360-degree inspection, automated part sorting, or validated industrial performance.

Aggregate evaluation results such as precision, recall, mAP, and false-acceptance or false-rejection rates should be added when they have been measured on a held-out test set. A confidence value from a single displayed detection should not be reported as model accuracy.

## Technology

- Python and PyTorch
- YOLO object detection
- Canny edge preprocessing
- Roboflow for image annotation and augmentation
- Visual Studio Code for development and training

## Reproducibility

To reproduce the results, the repository should include the training and inference scripts, dependency versions, model configuration and weights (or instructions to obtain them), the IoU threshold, and the commands used to train and run the prototype. Dataset size, class definitions, and train/validation/test split should also be documented where sharing the data is permitted.

## Future Work

- Report evaluation metrics on a clearly defined held-out dataset.
- Document the model versions, class labels, thresholds, and software environment.
- Evaluate sensitivity to lighting, camera position, component orientation, and surface variation.
- Continue integration between the conveyor hardware and the vision processing unit.
- Assess additional production requirements such as full inspection coverage, traceability, defect grading, and automated sorting.
