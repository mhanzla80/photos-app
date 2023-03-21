import 'package:image/image.dart' as image_lib;
import "package:logging/logging.dart";
import "package:photos/services/object_detection/models/predictions.dart";
import "package:photos/services/object_detection/models/stats.dart";
import "package:photos/services/object_detection/tflite/classifier.dart";
import "package:tflite_flutter/tflite_flutter.dart";
import "package:tflite_flutter_helper/tflite_flutter_helper.dart";

class ClipImageEncoder extends Classifier {
  static final _logger = Logger("ClipImageEncoder");
  static const double threshold = 0.5;

  @override
  String get modelPath => "models/clip/clip-image.tflite";

  @override
  String get labelPath => "";

  @override
  int get inputSize => 224;

  @override
  Logger get logger => _logger;

  ClipImageEncoder({Interpreter? interpreter}) {
    loadModel(interpreter);
  }

  @override
  Predictions? predict(image_lib.Image image) {
    final predictStartTime = DateTime.now().millisecondsSinceEpoch;
    final preProcessStart = DateTime.now().millisecondsSinceEpoch;

    TensorImage inputImage = TensorImage(TfLiteType.float32);
    inputImage.loadImage(image_lib.grayscale(image));
    inputImage = getProcessedImage(inputImage);

    final preProcessElapsedTime =
        DateTime.now().millisecondsSinceEpoch - preProcessStart;
    final outputLocations = TensorBufferFloat(outputShapes[0]);
    final outputs = {
      0: outputLocations.buffer,
    };

    final inferenceTimeStart = DateTime.now().millisecondsSinceEpoch;
    interpreter.run(inputImage.buffer, outputs);
    final inferenceTimeElapsed =
        DateTime.now().millisecondsSinceEpoch - inferenceTimeStart;
    final predictElapsedTime =
        DateTime.now().millisecondsSinceEpoch - predictStartTime;
    return Predictions(
      [],
      Stats(
        predictElapsedTime,
        predictElapsedTime,
        inferenceTimeElapsed,
        preProcessElapsedTime,
      ),
    );
  }
}
