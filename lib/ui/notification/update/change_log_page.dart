import 'package:flutter/material.dart';
import "package:photos/generated/l10n.dart";
import 'package:photos/services/update_service.dart';
import 'package:photos/theme/ente_theme.dart';
import 'package:photos/ui/components/buttons/button_widget.dart';
import 'package:photos/ui/components/divider_widget.dart';
import 'package:photos/ui/components/models/button_type.dart';
import 'package:photos/ui/components/title_bar_title_widget.dart';
import 'package:photos/ui/notification/update/change_log_entry.dart';

class ChangeLogPage extends StatefulWidget {
  const ChangeLogPage({
    Key? key,
  }) : super(key: key);

  @override
  State<ChangeLogPage> createState() => _ChangeLogPageState();
}

class _ChangeLogPageState extends State<ChangeLogPage> {
  @override
  void initState() {
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    final enteColorScheme = getEnteColorScheme(context);
    return Scaffold(
      appBar: null,
      body: Container(
        color: enteColorScheme.backgroundElevated,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const SizedBox(
              height: 36,
            ),
            Container(
              alignment: Alignment.centerLeft,
              child: const Padding(
                padding: EdgeInsets.symmetric(horizontal: 16.0),
                child: TitleBarTitleWidget(
                  title: "What's new",
                ),
              ),
            ),
            const SizedBox(
              height: 24,
            ),
            Expanded(child: _getChangeLog()),
            const DividerWidget(
              dividerType: DividerType.solid,
            ),
            SafeArea(
              child: Padding(
                padding: const EdgeInsets.only(
                  left: 16.0,
                  right: 16,
                  top: 16,
                  bottom: 8,
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    ButtonWidget(
                      buttonType: ButtonType.trailingIconPrimary,
                      buttonSize: ButtonSize.large,
                      labelText: S.of(context).continueLabel,
                      icon: Icons.arrow_forward_outlined,
                      onTap: () async {
                        await UpdateService.instance.hideChangeLog();
                        if (mounted && Navigator.of(context).canPop()) {
                          Navigator.of(context).pop();
                        }
                      },
                    ),
                    const SizedBox(
                      height: 8,
                    ),
                    ButtonWidget(
                      buttonType: ButtonType.trailingIconSecondary,
                      buttonSize: ButtonSize.large,
                      labelText: S.of(context).rateTheApp,
                      icon: Icons.favorite_rounded,
                      iconColor: enteColorScheme.primary500,
                      onTap: () async {
                        await UpdateService.instance.launchReviewUrl();
                      },
                    ),
                    const SizedBox(height: 8),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _getChangeLog() {
    final scrollController = ScrollController();
    final List<ChangeLogEntry> items = [];
    items.add(
      ChangeLogEntry(
        "Explore with the new Search Tab ✨",
        'Introducing a dedicated search tab with distinct sections for effortless discovery.\n'
            '\nYou can now discover items that come under different Locations, Moments, Contacts, Photo descriptions, Albums and File types with ease.\n',
      ),
    );

    return Container(
      padding: const EdgeInsets.only(left: 16),
      child: Scrollbar(
        controller: scrollController,
        thumbVisibility: true,
        thickness: 2.0,
        child: ListView.builder(
          physics: const BouncingScrollPhysics(),
          itemBuilder: (context, index) {
            return Padding(
              padding: const EdgeInsets.only(right: 16.0),
              child: ChangeLogEntryWidget(entry: items[index]),
            );
          },
          itemCount: items.length,
        ),
      ),
    );
  }
}
