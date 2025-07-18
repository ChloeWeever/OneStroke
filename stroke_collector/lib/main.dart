import 'dart:typed_data';
import 'dart:ui' as ui;
import 'dart:ui';
import 'package:flutter/rendering.dart';
import 'package:image_gallery_saver/image_gallery_saver.dart';
import 'package:permission_handler/permission_handler.dart';
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
  List<String> Message = [
    "请在右侧写出‘卢’并翻页",
    "请在右侧写出‘文’并翻页",
    "请在右侧写出‘仃’并翻页",
    "请在右侧写出‘杆’并翻页",
    "请在右侧写出‘才’并翻页",
    "请在右侧写出‘尽’并翻页",
    "请在右侧写出‘半’并翻页",
    "请在右侧写出‘何’并翻页",
    "请在右侧写出‘亲’并翻页",
    "请在右侧写出‘走’并翻页",
    "请在右侧写出‘级’并翻页",
    "请在右侧写出‘卖’并翻页",
    "请在右侧写出‘废’并翻页",
    "请在右侧写出‘鲦’并翻页",
    "请在右侧写出‘氢’并翻页",
    "请在右侧写出‘陈’并翻页",
    "请在右侧写出‘畅’并翻页",
    "请在右侧写出‘欢’并翻页",
    "请在右侧写出‘尴’并翻页",
    "请在右侧写出‘她’并翻页",
    "请写出你名字的第一个字并翻页",
    "请写出你名字的第二个字并翻页",
    "请写出你名字的第三个字并翻页(如果没有请随意写一个字)",
    "请在右侧写出‘一’并翻页",
    "请按下保存键",
  ];

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
    return allLines.length < currentPage + 1
        ? currentPage + 1
        : allLines.length;
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

  void restart() async {
    // 弹出确认对话框
    bool? confirmed = await showDialog<bool>(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text('确认重启'),
          content: const Text('您确定要清空所有数据吗？'),
          actions: <Widget>[
            TextButton(
              child: const Text('取消'),
              onPressed: () {
                Navigator.of(context).pop(false);
              },
            ),
            TextButton(
              child: const Text('确认'),
              onPressed: () {
                Navigator.of(context).pop(true);
              },
            ),
          ],
        );
      },
    );
    if (confirmed == true) {
      setState(() {
        allLines.clear();
        lines.value = [];
        line.value = null;
        currentPage = 0;
      });
    }
  }

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

  // 添加一个 GlobalKey 用于获取绘制区域的 RenderObject
  final GlobalKey _globalKey = GlobalKey();

  Future<void> _exportAllImage() async {
    // 弹出确认对话框
    bool? confirmed = await showDialog<bool>(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text('确认保存'),
          content: const Text('您已完成书写了吗？'),
          actions: <Widget>[
            TextButton(
              child: const Text('取消'),
              onPressed: () {
                Navigator.of(context).pop(false);
              },
            ),
            TextButton(
              child: const Text('确认'),
              onPressed: () {
                Navigator.of(context).pop(true);
              },
            ),
          ],
        );
      },
    );
    if (confirmed == true) {
      for (int i = 0; i < allLines.length; i++) {
        lines.value = allLines[i].value;
        setState(() {});
        await _exportImage(i);
        for (int k = 0; k < allLines[i].value.length; k++) {
          lines.value = [allLines[i].value[k]];
          setState(() {});
          await _exportImage(i);
        }
      }
      lines.value = allLines[currentPage].value;
      setState(() {});
    }
  }

  // 添加导出图片的方法
  Future<void> _exportImage([int? pageIndex]) async {
    // 获取 RenderObject
    final RenderRepaintBoundary boundary =
        _globalKey.currentContext!.findRenderObject() as RenderRepaintBoundary;

    // 将 RenderObject 转换为图像
    final ui.Image image = await boundary.toImage();
    final ByteData? byteData = await image.toByteData(
      format: ui.ImageByteFormat.png,
    );
    if (byteData == null) {
      print('Failed to convert image to byte data');
      return;
    }

    // 请求存储权限
    final status = await Permission.storage.request();
    if (status.isGranted) {
      // 保存图像到相册
      final result = await ImageGallerySaver.saveImage(
        byteData.buffer.asUint8List(),
      );
      if (result['isSuccess']) {
        print('Image saved successfully');
      } else {
        print('Failed to save image');
      }
    } else {
      print('Storage permission denied');
    }
  }

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    return Scaffold(
      appBar: AppBar(
        title: const Text('DrawPad'),
        backgroundColor: const Color.fromRGBO(42, 130, 228, 0.7),
        actions: [
          IconButton(icon: const Icon(Icons.undo), onPressed: undo),
          IconButton(
            onPressed: previousPage,
            icon: const Icon(Icons.arrow_back_ios),
          ),
          IconButton(
            onPressed: nextPage,
            icon: const Icon(Icons.arrow_forward_ios),
          ),
          // 添加导出按钮
          IconButton(icon: const Icon(Icons.save), onPressed: _exportAllImage),
          // 添加重启按钮
          IconButton(icon: const Icon(Icons.restart_alt), onPressed: restart),
        ],
      ),
      body: Row(
        children: [
          Expanded(
            child: Column(
              children: [
                const SizedBox(height: 10),
                TDBottomTabBar(
                  TDBottomTabBarBasicType.icon,
                  componentType: TDBottomTabBarComponentType.label,
                  outlineType: TDBottomTabBarOutlineType.capsule,
                  useVerticalDivider: true,
                  navigationTabs: [
                    TDBottomTabBarTabConfig(
                      selectedIcon: const Icon(Icons.edit),
                      unselectedIcon: const Icon(Icons.edit_outlined),
                      //tabText: '',
                      onTap: () {
                        setState(() {
                          _isToolbarVisible = false;
                          _canDraw = true;
                        });
                      },
                    ),
                    TDBottomTabBarTabConfig(
                      selectedIcon: const Icon(Icons.more),
                      unselectedIcon: const Icon(Icons.more_outlined),
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
                Text(
                  Message[currentPage < Message.length
                      ? currentPage
                      : Message.length - 1],
                ),
                Expanded(
                  child: ListView.builder(
                    itemCount: lines.value.length,
                    itemBuilder: (context, index) {
                      final stroke = lines.value[index];
                      return ListTile(
                        title: Text('Line ${index + 1}'),
                        subtitle: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text('Points: ${stroke.points.length}'),
                            Column(
                              children:
                                  stroke.points.map((point) {
                                    return Row(
                                      children: [
                                        Text('Point: (${point.x}, ${point.y})'),
                                        if (point.pressure != null)
                                          Text(', Pressure: ${point.pressure}'),
                                      ],
                                    );
                                  }).toList(),
                            ),
                          ],
                        ),
                      );
                    },
                  ),
                ),
              ],
            ),
          ),
          Container(width: 1, color: Colors.grey),
          Expanded(
            child: Listener(
              onPointerDown: onPointerDown,
              onPointerMove: onPointerMove,
              onPointerUp: onPointerUp,
              child: RepaintBoundary(
                key: _globalKey,
                child: Container(
                  decoration: const BoxDecoration(
                    image: DecorationImage(
                      image: AssetImage('assets/images/paper.webp'),
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
                      Positioned(
                        bottom: 10,
                        right: 10,
                        child: Text(
                          '${currentPage + 1} / ${totalPage()}',
                          style: const TextStyle(
                            color: Colors.black,
                            fontSize: 16,
                          ),
                        ),
                      ),
                    ],
                  ),
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
    final redPaint = Paint()..color = Colors.red; // 添加红色画笔

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
        // 绘制落笔点
        canvas.drawCircle(outlinePoints[outlinePoints.length - 7], 3, redPaint);
        if (outlinePoints.length >= 2) {
          // 确保至少有两个点
          final midIndex = (outlinePoints.length / 2).toInt();
          canvas.drawCircle(outlinePoints[midIndex], 3, redPaint);
        }
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
