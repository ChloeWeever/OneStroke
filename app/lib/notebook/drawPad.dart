import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:perfect_freehand/perfect_freehand.dart';
import 'package:tdesign_flutter/tdesign_flutter.dart';
import 'toolbar.dart';

class DrawPadPage extends StatefulWidget {
  const DrawPadPage({super.key});

  @override
  State<DrawPadPage> createState() => _DrawPadPageState();
}

class _DrawPadPageState extends State<DrawPadPage> {
  StrokeOptions options = StrokeOptions(
    size: 16,
    thinning: 0.7,
    smoothing: 0.5,
    streamline: 0.5,
    start: StrokeEndOptions.start(
      taperEnabled: true,
      customTaper: 0.0,
      cap: true,
    ),
    end: StrokeEndOptions.end(taperEnabled: true, customTaper: 0.0, cap: true),
    simulatePressure: true,
    isComplete: false,
  );

  int currentPage = 0;

  // All lines drawn.
  List<ValueNotifier<List<Stroke>>> allLines = [];

  void nextPage() {
    if (currentPage >= allLines.length) {
      allLines.add(ValueNotifier(lines.value));
    } else {
      allLines[currentPage] = ValueNotifier(lines.value);
    }
    currentPage++;
    if (currentPage >= allLines.length) {
      lines.value = [];
    } else {
      lines.value = allLines[currentPage].value;
    }
    setState(() {}); // 添加这行代码以触发页面重建
  }

  void previousPage() {
    if (currentPage > 0) {
      if (currentPage == allLines.length) {
        allLines.add(ValueNotifier(lines.value));
      } else {
        allLines[currentPage] = ValueNotifier(lines.value);
      }
      currentPage--;
      lines.value = allLines[currentPage].value;
    }
    setState(() {}); // 添加这行代码以触发页面重建
  }

  int totalPage() {
    return allLines.length < currentPage+1 ? currentPage+1 : allLines.length;
  }

  /// Previous lines drawn.
  ValueNotifier<List<Stroke>> lines = ValueNotifier(<Stroke>[]);

  /// The current line being drawn.
  final line = ValueNotifier<Stroke?>(null);

  /// 控制 Toolbar 的显示
  bool _isToolbarVisible = false;
  bool _canDraw = true;

  void clear() => setState(() {
    lines.value = [];
    line.value = null;
  });

  void undo() => setState(() {
    if (lines.value.isNotEmpty) {
      lines.value = List.from(lines.value)..removeLast();
    }
  });

  void onPointerDown(PointerDownEvent details) {
    if (_canDraw) {
      final supportsPressure = details.kind == PointerDeviceKind.stylus;
      options = options.copyWith(simulatePressure: !supportsPressure);

      final localPosition = details.localPosition;
      final point = PointVector(
        localPosition.dx,
        localPosition.dy,
        supportsPressure ? details.pressure : null,
      );

      line.value = Stroke([point]);
    }
  }

  void onPointerMove(PointerMoveEvent details) {
    if (_canDraw) {
      final supportsPressure = details.pressureMin < 1;
      final localPosition = details.localPosition;
      final point = PointVector(
        localPosition.dx,
        localPosition.dy,
        supportsPressure ? details.pressure : null,
      );

      line.value = Stroke([...line.value!.points, point]);
    }
  }

  void onPointerUp(PointerUpEvent details) {
    if (_canDraw) {
      lines.value = [...lines.value, line.value!];
      line.value = null;
    }
  }

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    return Scaffold(
      appBar: AppBar(
        title: const Text('DrawPad'),
        backgroundColor: Color.fromRGBO(42, 130, 228, 0.7),
        actions: [
          IconButton(icon: Icon(Icons.undo), onPressed: undo),
          IconButton(onPressed: previousPage, icon: Icon(Icons.arrow_back_ios)),
          IconButton(onPressed: nextPage, icon: Icon(Icons.arrow_forward_ios)),
        ],
      ),
      body: Row(
        children: [
          Expanded(
            child: Column(
              children: [
                SizedBox(height: 10),
                TDBottomTabBar(
                  TDBottomTabBarBasicType.icon,
                  componentType: TDBottomTabBarComponentType.label,
                  outlineType: TDBottomTabBarOutlineType.capsule,
                  useVerticalDivider: true,
                  navigationTabs: [
                    TDBottomTabBarTabConfig(
                      selectedIcon: Icon(Icons.edit),
                      unselectedIcon: Icon(Icons.edit_outlined),
                      //tabText: '',
                      onTap: () {
                        setState(() {
                          _isToolbarVisible = false;
                          _canDraw = true;
                        });
                      },
                    ),
                    TDBottomTabBarTabConfig(
                      selectedIcon: Icon(Icons.cleaning_services),
                      unselectedIcon: Icon(Icons.cleaning_services_outlined),
                      //tabText: '',
                      onTap: () {
                        setState(() {
                          _isToolbarVisible = false;
                          _canDraw = false;
                        });
                      },
                    ),
                    TDBottomTabBarTabConfig(
                      selectedIcon: Icon(Icons.add_a_photo),
                      unselectedIcon: Icon(Icons.add_a_photo_outlined),
                      //tabText: '',
                      onTap: () {
                        setState(() {
                          _isToolbarVisible = false;
                          _canDraw = false;
                        });
                      },
                    ),
                    TDBottomTabBarTabConfig(
                      selectedIcon: Icon(Icons.add_comment),
                      unselectedIcon: Icon(Icons.add_comment_outlined),
                      //tabText: '',
                      onTap: () {
                        setState(() {
                          _isToolbarVisible = false;
                          _canDraw = false;
                        });
                      },
                    ),
                    TDBottomTabBarTabConfig(
                      selectedIcon: Icon(Icons.more),
                      unselectedIcon: Icon(Icons.more_outlined),
                      //tabText: '',
                      onTap: () {
                        setState(() {
                          _isToolbarVisible = true;
                          _canDraw = true;
                        });
                      },
                    ),
                  ],
                ),
                if (_isToolbarVisible)
                  Toolbar(
                    options: options,
                    updateOptions: setState,
                    clear: clear,
                  ),
              ],
            ),
          ),
          // 添加分割线
          Container(width: 1, color: Colors.grey),
          Expanded(
            child: Listener(
              onPointerDown: onPointerDown,
              onPointerMove: onPointerMove,
              onPointerUp: onPointerUp,
              child: Container(
                decoration: BoxDecoration(
                  image: DecorationImage(
                    image: AssetImage(
                      'assets/images/paper.webp',
                    ), // 假设背景图片路径为 'assets/background.png'
                    // fit: BoxFit.cover,
                  ),
                ),
                child: Stack(
                  children: [
                    Positioned.fill(
                      child: ValueListenableBuilder(
                        valueListenable: lines,
                        builder: (context, lines, _) {
                          return CustomPaint(
                            painter: StrokePainter(
                              color: colorScheme.onSurface,
                              lines: lines,
                              options: options,
                            ),
                          );
                        },
                      ),
                    ),
                    Positioned.fill(
                      child: ValueListenableBuilder(
                        valueListenable: line,
                        builder: (context, line, _) {
                          return CustomPaint(
                            painter: StrokePainter(
                              color: colorScheme.onSurface,
                              lines: line == null ? [] : [line],
                              options: options,
                            ),
                          );
                        },
                      ),
                    ),
                    // 添加页码显示
                    Positioned(
                      bottom: 10,
                      right: 10,
                      child: Text(
                        '${currentPage + 1} / ${totalPage()}',
                        style: TextStyle(color: Colors.black, fontSize: 16),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    lines.dispose();
    line.dispose();
    super.dispose();
  }
}

class StrokePainter extends CustomPainter {
  const StrokePainter({
    required this.color,
    required this.lines,
    required this.options,
  });

  final Color color;
  final List<Stroke> lines;
  final StrokeOptions options;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()..color = color;

    for (final line in lines) {
      final outlinePoints = getStroke(line.points, options: options);

      if (outlinePoints.isEmpty) {
        continue;
      } else if (outlinePoints.length < 2) {
        // If the path only has one point, draw a dot.
        canvas.drawCircle(outlinePoints.first, options.size / 2, paint);
      } else {
        final path = Path();
        path.moveTo(outlinePoints.first.dx, outlinePoints.first.dy);
        for (int i = 0; i < outlinePoints.length - 1; ++i) {
          final p0 = outlinePoints[i];
          final p1 = outlinePoints[i + 1];
          path.quadraticBezierTo(
            p0.dx,
            p0.dy,
            (p0.dx + p1.dx) / 2,
            (p0.dy + p1.dy) / 2,
          );
        }
        // You'll see performance improvements if you cache this Path
        // instead of creating a new one every paint.
        canvas.drawPath(path, paint);
      }
    }
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => true;
}

class Stroke {
  final List<PointVector> points;

  const Stroke(this.points);
}

void main() {
  runApp(
    MaterialApp(
      title: 'Drawing App',
      debugShowCheckedModeBanner: false,
      themeMode: ThemeMode.system,
      theme: ThemeData(
        brightness: Brightness.light,
        colorSchemeSeed: Colors.blue,
      ),
      darkTheme: ThemeData(
        brightness: Brightness.dark,
        colorSchemeSeed: Colors.blue,
      ),
      home: const DrawPadPage(),
    ),
  );
}
