import "dart:async";

import "package:flutter/material.dart";
import "package:flutter/scheduler.dart";
import "package:logging/logging.dart";
import "package:photos/core/event_bus.dart";
import "package:photos/events/clear_and_unfocus_search_bar_event.dart";
import "package:photos/events/tab_changed_event.dart";
import "package:photos/models/search/index_of_indexed_stack.dart";
import "package:photos/models/search/search_result.dart";
import "package:photos/services/search_service.dart";
import "package:photos/theme/ente_theme.dart";
import "package:photos/ui/viewer/search/search_suffix_icon_widget.dart";
import "package:photos/utils/date_time_util.dart";
import "package:photos/utils/debouncer.dart";

class SearchWidget extends StatefulWidget {
  const SearchWidget({Key? key}) : super(key: key);

  @override
  State<SearchWidget> createState() => SearchWidgetState();
}

class SearchWidgetState extends State<SearchWidget> {
  static final ValueNotifier<Stream<List<SearchResult>>?>
      searchResultsStreamNotifier = ValueNotifier(null);

  ///This stores the query that is being searched for. When going to other tabs
  ///when searching, this state gets disposed and when coming back to the
  ///search tab, this query is used to populate the search bar.
  static String query = "";
  //Debouncing + querying
  static final isLoading = ValueNotifier(false);
  final _searchService = SearchService.instance;
  final _debouncer = Debouncer(const Duration(milliseconds: 200));
  final Logger _logger = Logger((SearchWidgetState).toString());
  late FocusNode focusNode;
  StreamSubscription<TabDoubleTapEvent>? _tabDoubleTapEvent;
  double _bottomPadding = 0.0;
  double _distanceOfWidgetFromBottom = 0;
  GlobalKey widgetKey = GlobalKey();
  TextEditingController textController = TextEditingController();
  late final StreamSubscription<ClearAndUnfocusSearchBar>
      _clearAndUnfocusSearchBar;

  @override
  void initState() {
    super.initState();
    focusNode = FocusNode();
    _tabDoubleTapEvent =
        Bus.instance.on<TabDoubleTapEvent>().listen((event) async {
      debugPrint("Firing now ${event.selectedIndex}");
      if (mounted && event.selectedIndex == 3) {
        focusNode.requestFocus();
      }
    });

    SchedulerBinding.instance.addPostFrameCallback((_) {
      //This buffer is for doing this operation only after SearchWidget's
      //animation is complete.
      Future.delayed(const Duration(milliseconds: 300), () {
        final RenderBox box =
            widgetKey.currentContext!.findRenderObject() as RenderBox;
        final heightOfWidget = box.size.height;
        final offsetPosition = box.localToGlobal(Offset.zero);
        final y = offsetPosition.dy;
        final heightOfScreen = MediaQuery.sizeOf(context).height;
        _distanceOfWidgetFromBottom = heightOfScreen - (y + heightOfWidget);
      });

      textController.addListener(textControllerListener);
    });

    //Populate the serach tab with the latest query when coming back
    //to the serach tab.
    textController.text = query;

    _clearAndUnfocusSearchBar =
        Bus.instance.on<ClearAndUnfocusSearchBar>().listen((event) {
      textController.clear();
      focusNode.unfocus();
    });
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _bottomPadding =
        (MediaQuery.viewInsetsOf(context).bottom - _distanceOfWidgetFromBottom);
    if (_bottomPadding < 0) {
      _bottomPadding = 0;
    }
  }

  @override
  void dispose() {
    _debouncer.cancelDebounce();
    focusNode.dispose();
    _tabDoubleTapEvent?.cancel();
    textController.removeListener(textControllerListener);
    textController.dispose();
    _clearAndUnfocusSearchBar.cancel();
    super.dispose();
  }

  Future<void> textControllerListener() async {
    isLoading.value = true;
    _debouncer.run(() async {
      if (mounted) {
        query = textController.text;
        IndexOfStackNotifier().isSearchQueryEmpty = query.isEmpty;
        searchResultsStreamNotifier.value =
            _getSearchResultsStream(context, query);
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final colorScheme = getEnteColorScheme(context);
    return RepaintBoundary(
      key: widgetKey,
      child: Padding(
        padding: EdgeInsets.only(bottom: _bottomPadding),
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 12),
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 2, horizontal: 8),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: Container(
                color: colorScheme.backgroundBase,
                child: Container(
                  height: 44,
                  color: colorScheme.fillFaint,
                  child: TextFormField(
                    controller: textController,
                    focusNode: focusNode,
                    style: Theme.of(context).textTheme.titleMedium,
                    // Below parameters are to disable auto-suggestion
                    enableSuggestions: false,
                    autocorrect: false,
                    // Above parameters are to disable auto-suggestion
                    decoration: InputDecoration(
                      // hintText: S.of(context).searchHintText,
                      hintText: "Search",
                      filled: true,
                      contentPadding: const EdgeInsets.symmetric(
                        vertical: 10,
                      ),
                      border: const UnderlineInputBorder(
                        borderSide: BorderSide.none,
                      ),
                      focusedBorder: const UnderlineInputBorder(
                        borderSide: BorderSide.none,
                      ),
                      prefixIconConstraints: const BoxConstraints(
                        maxHeight: 44,
                        maxWidth: 44,
                        minHeight: 44,
                        minWidth: 44,
                      ),
                      suffixIconConstraints: const BoxConstraints(
                        maxHeight: 44,
                        maxWidth: 44,
                        minHeight: 44,
                        minWidth: 44,
                      ),
                      prefixIcon: Hero(
                        tag: "search_icon",
                        child: Icon(
                          Icons.search,
                          color: colorScheme.strokeFaint,
                        ),
                      ),
                      /*Using valueListenableBuilder inside a stateful widget because this widget is only rebuild when
                      setState is called when deboucncing is over and the spinner needs to be shown while debouncing */
                      suffixIcon: ValueListenableBuilder(
                        valueListenable: isLoading,
                        builder: (
                          BuildContext context,
                          bool isSearching,
                          Widget? child,
                        ) {
                          return SearchSuffixIcon(
                            isSearching,
                          );
                        },
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Future<List<SearchResult>> getSearchResultsForQuery(
    BuildContext context,
    String query,
  ) async {
    final Completer<List<SearchResult>> completer = Completer();

    _debouncer.run(
      () {
        return _getSearchResultsFromService(context, query, completer);
      },
    );

    return completer.future;
  }

  Future<void> _getSearchResultsFromService(
    BuildContext context,
    String query,
    Completer completer,
  ) async {
    final List<SearchResult> allResults = [];
    if (query.isEmpty) {
      completer.complete(allResults);
      return;
    }
    try {
      if (_isYearValid(query)) {
        final yearResults = await _searchService.getYearSearchResults(query);
        allResults.addAll(yearResults);
      }

      final holidayResults =
          await _searchService.getHolidaySearchResults(context, query);
      allResults.addAll(holidayResults);

      final fileTypeSearchResults =
          await _searchService.getFileTypeResults(context, query);
      allResults.addAll(fileTypeSearchResults);

      final captionAndDisplayNameResult =
          await _searchService.getCaptionAndNameResults(query);
      allResults.addAll(captionAndDisplayNameResult);

      final fileExtnResult =
          await _searchService.getFileExtensionResults(query);
      allResults.addAll(fileExtnResult);

      final locationResult = await _searchService.getLocationResults(query);
      allResults.addAll(locationResult);

      final collectionResults =
          await _searchService.getCollectionSearchResults(query);
      allResults.addAll(collectionResults);

      final monthResults =
          await _searchService.getMonthSearchResults(context, query);
      allResults.addAll(monthResults);

      final possibleEvents =
          await _searchService.getDateResults(context, query);
      allResults.addAll(possibleEvents);

      final magicResults =
          await _searchService.getMagicSearchResults(context, query);
      allResults.addAll(magicResults);

      final contactResults =
          await _searchService.getContactSearchResults(query);
      allResults.addAll(contactResults);
    } catch (e, s) {
      _logger.severe("error during search", e, s);
    }
    completer.complete(allResults);
  }

  Stream<List<SearchResult>> _getSearchResultsStream(
    BuildContext context,
    String query,
  ) {
    int resultCount = 0;
    final maxResultCount = _isYearValid(query) ? 12 : 11;
    final streamController = StreamController<List<SearchResult>>();

    if (query.isEmpty) {
      streamController.sink.add([]);
      streamController.close();
      return streamController.stream;
    }

    void onResultsReceived(List<SearchResult> results) {
      streamController.sink.add(results);
      resultCount++;
      if (resultCount == maxResultCount) {
        streamController.close();
      }
    }

    if (_isYearValid(query)) {
      _searchService.getYearSearchResults(query).then((yearSearchResults) {
        onResultsReceived(yearSearchResults);
      });
    }

    _searchService.getHolidaySearchResults(context, query).then(
      (holidayResults) {
        onResultsReceived(holidayResults);
      },
    );

    _searchService.getFileTypeResults(context, query).then(
      (fileTypeSearchResults) {
        onResultsReceived(fileTypeSearchResults);
      },
    );

    _searchService.getCaptionAndNameResults(query).then(
      (captionAndDisplayNameResult) {
        onResultsReceived(captionAndDisplayNameResult);
      },
    );

    _searchService.getFileExtensionResults(query).then(
      (fileExtnResult) {
        onResultsReceived(fileExtnResult);
      },
    );

    _searchService.getLocationResults(query).then(
      (locationResult) {
        onResultsReceived(locationResult);
      },
    );

    _searchService.getCityResults(query).then(
      (results) {
        onResultsReceived(results);
      },
    );

    _searchService.getCollectionSearchResults(query).then(
      (collectionResults) {
        onResultsReceived(collectionResults);
      },
    );

    _searchService.getMonthSearchResults(context, query).then(
      (monthResults) {
        onResultsReceived(monthResults);
      },
    );

    _searchService.getDateResults(context, query).then(
      (possibleEvents) {
        onResultsReceived(possibleEvents);
      },
    );

    _searchService.getMagicSearchResults(context, query).then(
      (magicResults) {
        onResultsReceived(magicResults);
      },
    );

    _searchService.getContactSearchResults(query).then(
      (contactResults) {
        onResultsReceived(contactResults);
      },
    );

    return streamController.stream;
  }

  bool _isYearValid(String year) {
    final yearAsInt = int.tryParse(year); //returns null if cannot be parsed
    return yearAsInt != null && yearAsInt <= currentYear;
  }
}
