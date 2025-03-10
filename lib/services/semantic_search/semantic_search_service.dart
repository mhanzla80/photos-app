import "dart:async";
import "dart:collection";

import "package:computer/computer.dart";
import "package:logging/logging.dart";
import "package:photos/core/cache/lru_map.dart";
import "package:photos/core/configuration.dart";
import "package:photos/core/event_bus.dart";
import "package:photos/db/files_db.dart";
import "package:photos/db/object_box.dart";
import "package:photos/events/diff_sync_complete_event.dart";
import 'package:photos/events/embedding_updated_event.dart';
import "package:photos/events/file_uploaded_event.dart";
import "package:photos/models/embedding.dart";
import "package:photos/models/file/file.dart";
import "package:photos/objectbox.g.dart";
import "package:photos/services/semantic_search/embedding_store.dart";
import "package:photos/services/semantic_search/frameworks/ml_framework.dart";
import 'package:photos/services/semantic_search/frameworks/onnx/onnx.dart';
import "package:photos/utils/file_util.dart";
import "package:photos/utils/local_settings.dart";
import "package:photos/utils/thumbnail_util.dart";

class SemanticSearchService {
  SemanticSearchService._privateConstructor();

  static final SemanticSearchService instance =
      SemanticSearchService._privateConstructor();
  static final Computer _computer = Computer.shared();
  static final LRUMap<String, List<double>> _queryCache = LRUMap(20);

  static const kModelName = "clip";
  static const kEmbeddingLength = 512;
  static const kScoreThreshold = 0.23;
  static const kShouldPushEmbeddings = false;

  final _logger = Logger("SemanticSearchService");
  final _queue = Queue<EnteFile>();
  final _cachedEmbeddings = <Embedding>[];
  final _mlFramework = ONNX();
  final _frameworkInitialization = Completer<void>();

  bool _hasInitialized = false;
  bool _isComputingEmbeddings = false;
  bool _isSyncing = false;
  Future<List<EnteFile>>? _ongoingRequest;
  PendingQuery? _nextQuery;

  get hasInitialized => _hasInitialized;

  Future<void> init() async {
    if (!LocalSettings.instance.hasEnabledMagicSearch()) {
      return;
    }
    if (_hasInitialized) {
      _logger.info("Initialized already");
      return;
    }
    _hasInitialized = true;
    await ObjectBox.instance.init();
    await EmbeddingStore.instance.init();
    _setupCachedEmbeddings();
    Bus.instance.on<DiffSyncCompleteEvent>().listen((event) {
      // Diff sync is complete, we can now pull embeddings from remote
      unawaited(sync());
    });
    if (Configuration.instance.hasConfiguredAccount() &&
        kShouldPushEmbeddings) {
      unawaited(EmbeddingStore.instance.pushEmbeddings());
    }

    // ignore: unawaited_futures
    _loadModels().then((v) async {
      _logger.info("Getting text embedding");
      await _getTextEmbedding("warm up text encoder");
      _logger.info("Got text embedding");
    });
    Bus.instance.on<FileUploadedEvent>().listen((event) async {
      _addToQueue(event.file);
    });
  }

  Future<void> release() async {
    if (_frameworkInitialization.isCompleted) {
      await _mlFramework.release();
    }
  }

  Future<void> sync() async {
    if (_isSyncing) {
      return;
    }
    _isSyncing = true;
    await EmbeddingStore.instance.pullEmbeddings();
    await _backFill();
    _isSyncing = false;
  }

  Future<List<EnteFile>> search(String query) async {
    if (!LocalSettings.instance.hasEnabledMagicSearch() ||
        !_frameworkInitialization.isCompleted) {
      return [];
    }
    if (_ongoingRequest == null) {
      _ongoingRequest = _getMatchingFiles(query).then((result) {
        _ongoingRequest = null;
        if (_nextQuery != null) {
          final next = _nextQuery;
          _nextQuery = null;
          search(next!.query).then((nextResult) {
            next.completer.complete(nextResult);
          });
        }

        return result;
      });
      return _ongoingRequest!;
    } else {
      // If there's an ongoing request, create or replace the nextCompleter.
      await _nextQuery?.completer.future
          .timeout(const Duration(seconds: 0)); // Cancels the previous future.
      _nextQuery = PendingQuery(query, Completer<List<EnteFile>>());
      return _nextQuery!.completer.future;
    }
  }

  Future<IndexStatus> getIndexStatus() async {
    return IndexStatus(
      _cachedEmbeddings.length,
      (await _getFileIDsToBeIndexed()).length,
    );
  }

  Future<void> clearIndexes() async {
    await ObjectBox.instance
        .getEmbeddingBox()
        .query(
          Embedding_.model.equals(
            _mlFramework.getFrameworkName() + "-" + kModelName,
          ),
        )
        .build()
        .removeAsync();
    _logger.info("Indexes cleared for ${_mlFramework.getFrameworkName()}");
  }

  void _setupCachedEmbeddings() {
    ObjectBox.instance
        .getEmbeddingBox()
        .query(
          Embedding_.model.equals(
            "ggml" + "-" + kModelName,
          ),
        )
        .watch(triggerImmediately: true)
        .map((query) => query.find())
        .listen((embeddings) {
      _logger.info("Updated embeddings: " + embeddings.length.toString());
      _cachedEmbeddings.clear();
      _cachedEmbeddings.addAll(embeddings);
      Bus.instance.fire(EmbeddingUpdatedEvent());
    });
  }

  Future<void> _backFill() async {
    if (!LocalSettings.instance.hasEnabledMagicSearch() ||
        !MLFramework.kImageEncoderEnabled) {
      return;
    }
    await _frameworkInitialization.future;
    _logger.info("Attempting backfill");
    final fileIDs = await _getFileIDsToBeIndexed();
    final files = await FilesDB.instance.getUploadedFiles(fileIDs);
    _logger.info(files.length.toString() + " to be embedded");
    // await _cacheThumbnails(files);
    _queue.addAll(files);
    unawaited(_pollQueue());
  }

  Future<void> _cacheThumbnails(List<EnteFile> files) async {
    int counter = 0;
    const batchSize = 100;
    for (var i = 0; i < files.length;) {
      final futures = <Future>[];
      for (var j = 0; j < batchSize && i < files.length; j++, i++) {
        futures.add(getThumbnail(files[i]));
      }
      await Future.wait(futures);
      counter += futures.length;
      _logger.info("$counter/${files.length} thumbnails cached");
    }
  }

  Future<List<int>> _getFileIDsToBeIndexed() async {
    final uploadedFileIDs = await FilesDB.instance
        .getOwnedFileIDs(Configuration.instance.getUserID()!);
    final embeddedFileIDs = _cachedEmbeddings.map((e) => e.fileID).toSet();
    uploadedFileIDs.removeWhere(
      (id) => embeddedFileIDs.contains(id),
    );
    return uploadedFileIDs;
  }

  Future<void> clearQueue() async {
    _queue.clear();
  }

  Future<List<EnteFile>> _getMatchingFiles(String query) async {
    final textEmbedding = await _getTextEmbedding(query);

    final queryResults = await _getScores(textEmbedding);

    final filesMap = await FilesDB.instance
        .getFilesFromIDs(queryResults.map((e) => e.id).toList());
    final results = <EnteFile>[];
    for (final result in queryResults) {
      if (filesMap.containsKey(result.id)) {
        results.add(filesMap[result.id]!);
      }
    }

    _logger.info(results.length.toString() + " results");

    return results;
  }

  void _addToQueue(EnteFile file) {
    if (!LocalSettings.instance.hasEnabledMagicSearch()) {
      return;
    }
    _logger.info("Adding " + file.toString() + " to the queue");
    _queue.add(file);
    _pollQueue();
  }

  Future<void> _loadModels() async {
    _logger.info("Initializing ML framework");
    try {
      await _mlFramework.init();
      _frameworkInitialization.complete();
    } catch (e, s) {
      _logger.severe("ML framework initialization failed", e, s);
    }
    _logger.info("ML framework initialized");
  }

  Future<void> _pollQueue() async {
    if (_isComputingEmbeddings) {
      return;
    }
    _isComputingEmbeddings = true;

    while (_queue.isNotEmpty) {
      await computeImageEmbedding(_queue.removeLast());
    }

    _isComputingEmbeddings = false;
  }

  Future<void> computeImageEmbedding(EnteFile file) async {
    if (!MLFramework.kImageEncoderEnabled) {
      return;
    }
    if (!_frameworkInitialization.isCompleted) {
      return;
    }
    try {
      final filePath = (await getFile(file))!.path;
      _logger.info("Running clip over $file");
      final result = await _mlFramework.getImageEmbedding(filePath);
      if (result.length != kEmbeddingLength) {
        _logger.severe("Discovered incorrect embedding for $file - $result");
        return;
      }
      // dev.log(result.toString());
      // dev.log(computeScore(result, webEmbedding).toString());
      // dev.log(computeScore(result, pyEmbedding).toString());
      // dev.log(computeScore(pyEmbedding, webEmbedding).toString());

      final embedding = Embedding(
        fileID: file.uploadedFileID!,
        model: _mlFramework.getFrameworkName() + "-" + kModelName,
        embedding: result,
      );
      await EmbeddingStore.instance.storeEmbedding(
        file,
        embedding,
      );
    } catch (e, s) {
      _logger.severe(e, s);
    }
  }

  Future<List<double>> _getTextEmbedding(String query) async {
    _logger.info("Searching for " + query);
    final cachedResult = _queryCache.get(query);
    if (cachedResult != null) {
      return cachedResult;
    }
    try {
      final result = await _mlFramework.getTextEmbedding(query);
      _queryCache.put(query, result);
      return result;
    } catch (e) {
      _logger.severe("Could not get text embedding", e);
      return [];
    }
  }

  Future<List<QueryResult>> _getScores(List<double> textEmbedding) async {
    final startTime = DateTime.now();
    final List<QueryResult> queryResults = await _computer.compute(
      computeBulkScore,
      param: {
        "imageEmbeddings": _cachedEmbeddings,
        "textEmbedding": textEmbedding,
      },
      taskName: "computeBulkScore",
    );
    final endTime = DateTime.now();
    _logger.info(
      "computingScores took: " +
          (endTime.millisecondsSinceEpoch - startTime.millisecondsSinceEpoch)
              .toString() +
          "ms",
    );
    return queryResults;
  }
}

List<QueryResult> computeBulkScore(Map args) {
  final queryResults = <QueryResult>[];
  final imageEmbeddings = args["imageEmbeddings"] as List<Embedding>;
  final textEmbedding = args["textEmbedding"] as List<double>;
  for (final imageEmbedding in imageEmbeddings) {
    final score = computeScore(
      imageEmbedding.embedding,
      textEmbedding,
    );
    if (score >= SemanticSearchService.kScoreThreshold) {
      queryResults.add(QueryResult(imageEmbedding.fileID, score));
    }
  }

  queryResults.sort((first, second) => second.score.compareTo(first.score));
  return queryResults;
}

double computeScore(List<double> imageEmbedding, List<double> textEmbedding) {
  assert(
    imageEmbedding.length == textEmbedding.length,
    "The two embeddings should have the same length",
  );
  double score = 0;
  for (int index = 0; index < imageEmbedding.length; index++) {
    score += imageEmbedding[index] * textEmbedding[index];
  }
  return score;
}

class QueryResult {
  final int id;
  final double score;

  QueryResult(this.id, this.score);
}

class PendingQuery {
  final String query;
  final Completer<List<EnteFile>> completer;

  PendingQuery(this.query, this.completer);
}

class IndexStatus {
  final int indexedItems, pendingItems;

  IndexStatus(this.indexedItems, this.pendingItems);
}
