import 'package:image/image.dart' as image_lib;
import "package:logging/logging.dart";
import "package:photos/services/object_detection/models/predictions.dart";
import "package:photos/services/object_detection/models/stats.dart";
import "package:photos/services/object_detection/tflite/classifier.dart";
import "package:photos/services/object_detection/tflite/constants.dart";
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

    // TensorImage inputImage = TensorImage.fromImage(image);
    // inputImage = getProcessedImage(inputImage);

    // bool foundNonZeroValue = false;

    // for (final value in inputImage.getTensorBuffer().getDoubleList()) {
    //   if (value != 0) {
    //     foundNonZeroValue = true;
    //   }
    // }
    // _logger.info("Input foundNonZeroValue?" + foundNonZeroValue.toString());
    // _logger.info(
    //     "Input: " + inputImage.getTensorBuffer().getDoubleList().toString());

    final preProcessElapsedTime =
        DateTime.now().millisecondsSinceEpoch - preProcessStart;
    final output = TensorBufferFloat(outputShapes[0]);
    final outputs = {
      0: output.buffer,
    };
    final inferenceTimeStart = DateTime.now().millisecondsSinceEpoch;

    try {
      final input = TensorBufferFloat([1, 3, 224, 224]);
      input.loadList(kKnownValue, shape: [1, 3, 224, 224]);
      interpreter.run(input.buffer, outputs);
    } catch (e, s) {
      _logger.severe(e, s);
    }

    // foundNonZeroValue = false;

    // for (final value in output.getDoubleList()) {
    //   if (value != 0) {
    //     foundNonZeroValue = true;
    //   }
    // }
    // _logger.info("foundNonZeroValue?" + foundNonZeroValue.toString());

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
