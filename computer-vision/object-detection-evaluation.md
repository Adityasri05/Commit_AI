# Evaluating Object Detection: IoU, Precision, Recall, and mAP

Object detection predicts both **what** an object is and **where** it is in an image. A prediction is usually a class label, a confidence score, and a bounding box. Evaluation must consider localization and classification together.

## Intersection over Union (IoU)

IoU measures the overlap between a predicted box (B_p) and a ground-truth box (B_g):

[
IoU(B_p,B_g)=rac{|B_pcap B_g|}{|B_pcup B_g|}
]

An IoU of 1 means perfect overlap; 0 means no overlap. A detection is commonly considered a match only when its class is correct and IoU exceeds a chosen threshold.

## Matching predictions to ground truth

For each image and class, sort predicted boxes by confidence. Match each prediction to an eligible, not-yet-matched ground-truth box according to the evaluation protocol.

- A correct class with sufficient overlap can be a true positive.
- A prediction with insufficient overlap or the wrong class is a false positive.
- A ground-truth object that was not matched is a false negative.
- A duplicate prediction for an object already matched is normally another false positive.

The one-to-one matching rule prevents a detector from receiving credit multiple times for the same object.

## Precision-recall and Average Precision

As the confidence threshold changes, precision and recall change. The precision-recall curve summarizes this trade-off.

**Average Precision (AP)** summarizes the precision-recall curve for a class under a particular IoU criterion and evaluation protocol. **Mean Average Precision (mAP)** averages AP across classes, and some benchmarks also average across multiple IoU thresholds.

For example, COCO-style AP averages over IoU thresholds from 0.50 to 0.95 in increments of 0.05. A reported value must name the benchmark and protocol; “mAP” alone is ambiguous.

## Do not confuse evaluation with post-processing

- **Confidence thresholding** decides which predictions to retain.
- **Non-maximum suppression (NMS)** removes highly overlapping predictions, usually within a class, based on confidence and IoU.
- **Evaluation matching** compares the retained predictions with ground-truth annotations using a benchmark-defined procedure.

Changing NMS or the confidence threshold changes the predictions being evaluated. Tune them on validation data, not on the final test benchmark.

## Dataset and annotation checks

1. Verify box coordinate format, image dimensions, and class-ID mapping.
2. Check that boxes are clipped or handled consistently at image boundaries.
3. Review missing, duplicate, and inconsistent annotations.
4. Split by video, location, camera, or capture session when neighboring frames are correlated.
5. Report per-class AP as well as an overall summary; rare classes can be hidden by the mean.
6. Inspect false positives and false negatives visually.

## Practice

Evaluate a detector at IoU 0.50 and across the COCO IoU range. Compare per-class AP, inspect duplicate detections, and measure how NMS settings affect validation results. Keep the test set untouched until the final configuration is chosen.

## References

- [COCO evaluation API](https://github.com/cocodataset/cocoapi)
- [TorchMetrics: Mean Average Precision for object detection](https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html)

**Key takeaway:** Detection quality depends on both class correctness and box localization. Always report the IoU and benchmark protocol behind an AP or mAP number.