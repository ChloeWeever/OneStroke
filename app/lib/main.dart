import 'package:english_words/english_words.dart';
import 'package:flutter/material.dart';
import 'forum.dart';
import 'package:provider/provider.dart';
import 'shop.dart';
import 'main/recomend.dart';
import 'person.dart';
import 'notebook.dart';
import 'notebook/drawPad.dart';
import 'main/ttf.dart';
import 'main/copybook.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (context) => MyAppState(),
      child: MaterialApp(
        title: 'Namer App',
        theme: ThemeData(
          useMaterial3: true,
          colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepOrange),
        ),
        home: MyHomePage(),
        routes: {
          //'/recomend': (context) => RecomendPage(),
        },
      ),
    );
  }
}

class MyAppState extends ChangeNotifier {
  var current = WordPair.random();
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key});

  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  int _currentIndex = 0;

  final List<Widget> _children = [
    MainContent(),
    ShopPage(),
    NoteBookPage(),
    ForumPage(),
    PersonPage(),
  ];

  void onTabTapped(int index) {
    setState(() {
      _currentIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Row(
        children: [
          NavigationRail(
            selectedIndex: _currentIndex,
            onDestinationSelected: (int index) {
              setState(() {
                _currentIndex = index;
              });
            },
            groupAlignment: 0.0,
            backgroundColor: Color.fromRGBO(67, 108, 156, 0.4),
            indicatorColor: Colors.white,
            labelType: NavigationRailLabelType.all,
            destinations: [
              NavigationRailDestination(
                icon: Icon(Icons.home_outlined, size: 35.0),
                selectedIcon: Icon(Icons.home, size: 35.0),
                label: Text('首页', style: TextStyle(fontSize: 15.0)),
              ),
              NavigationRailDestination(
                icon: Icon(Icons.shop_outlined, size: 35.0),
                selectedIcon: Icon(Icons.shop, size: 35.0),
                label: Text('商城', style: TextStyle(fontSize: 15.0)),
              ),
              NavigationRailDestination(
                icon: Icon(Icons.book_outlined, size: 35.0),
                selectedIcon: Icon(Icons.book, size: 35.0),
                label: Text('笔记本', style: TextStyle(fontSize: 15.0)),
              ),
              NavigationRailDestination(
                icon: Icon(Icons.forum_outlined, size: 35.0),
                selectedIcon: Icon(Icons.forum, size: 35.0),
                label: Text('论坛', style: TextStyle(fontSize: 15.0)),
              ),
              NavigationRailDestination(
                icon: Icon(Icons.person_outlined, size: 35.0),
                selectedIcon: Icon(Icons.person, size: 35.0),
                label: Text('我的', style: TextStyle(fontSize: 15.0)),
              ),
            ],
          ),
          Expanded(
            child: _children[_currentIndex],
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        backgroundColor: Color.fromRGBO(67, 108, 156, 0.4),
        onPressed: () {
          // 添加按钮点击事件
          Navigator.push(
            context,
            MaterialPageRoute(builder: (context) => DrawPadPage()),
          );
        },
        child: Icon(Icons.add),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.endFloat,
    );
  }
}

class MainContent extends StatefulWidget {
  const MainContent({super.key});

  @override
  _MainContentState createState() => _MainContentState();
}

class _MainContentState extends State<MainContent> {
  List<RecomendData> recomendData = [];
  List<CopyBookData> copyBookData = [
    CopyBookData(),
    CopyBookData(),
    CopyBookData(),
    CopyBookData(),
    CopyBookData(),
    CopyBookData(),
    CopyBookData(),
    CopyBookData(),
    CopyBookData(),
    CopyBookData(),
  ];

  @override
  void initState() {
    super.initState();
    _fetchRecomendData();
  }

  Future<void> _fetchRecomendData() async {
    try {
      final response = await http
          .get(Uri.parse('https://dm.ecnucpp.cn:8081/GetrecomendData'));
      if (response.statusCode == 200) {
        setState(() {
          recomendData = List<RecomendData>.from(
            json.decode(response.body).map((x) => RecomendData.fromJson(x)),
          );
        });
      } else {
        setState(() {
          recomendData = [
            RecomendData(0,
                "https://www.teamrockstars.nl/wp-content/uploads/2023/06/flutter-Featured-Blog-Image2.jpg"),
            RecomendData(1,
                "https://www.ideamotive.co/hs-fs/hubfs/2088x1252_infografika_pros_and_cons_4%20(1).png?width=2088&name=2088x1252_infografika_pros_and_cons_4%20(1).png"),
            RecomendData(2,
                "https://uploads-ssl.webflow.com/6377bf360873283fad488724/638ca82a95fb434e6f42a283_Flutter.png"),
          ];
        });
      }
    } catch (e) {
      print('Error fetching data: $e');
      setState(() {
        recomendData = [
          RecomendData(0,
              "https://www.teamrockstars.nl/wp-content/uploads/2023/06/flutter-Featured-Blog-Image2.jpg"),
          RecomendData(1,
              "https://www.ideamotive.co/hs-fs/hubfs/2088x1252_infografika_pros_and_cons_4%20(1).png?width=2088&name=2088x1252_infografika_pros_and_cons_4%20(1).png"),
          RecomendData(2,
              "https://uploads-ssl.webflow.com/6377bf360873283fad488724/638ca82a95fb434e6f42a283_Flutter.png"),
        ];
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    var appState = context.watch<MyAppState>();

    return Scaffold(
      body: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SizedBox(height: 40),
            Text("  今日推荐",
                style: TextStyle(fontSize: 50.0, fontWeight: FontWeight.bold)),
            RecomendArea(recomendData: recomendData),
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text("  精选字体",
                        style: TextStyle(
                            fontSize: 50.0, fontWeight: FontWeight.bold)),
                    SizedBox(height: 20),
                    TtfArea(),
                    SizedBox(height: 40),
                  ],
                ),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text("  精品字帖",
                          style: TextStyle(
                              fontSize: 50.0, fontWeight: FontWeight.bold)),
                      SizedBox(height: 20),
                      Padding(
                          padding: EdgeInsets.only(left: 30.0),
                          child: Container(
                            height: 300,
                            decoration: BoxDecoration(
                              color: Colors.white,
                              borderRadius: BorderRadius.circular(12.0),
                              boxShadow: [
                                BoxShadow(
                                  color: Colors.black.withOpacity(0.3),
                                  blurRadius: 10.0,
                                  offset: Offset(0, 5),
                                ),
                              ],
                            ),
                            child:
                                CopyBookArea(copybookData: copyBookData),
                          ))
                    ],
                  ),
                ),
                SizedBox(width: 10),
              ],
            )
          ],
        ),
      ),
    );
  }
}

class RecomendArea extends StatelessWidget {
  List<RecomendData> recomendData = [];
  RecomendArea({super.key, required this.recomendData});

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<List<RecomendData>>(
      future: Future.value(recomendData),
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return Center(child: CircularProgressIndicator());
        } else if (snapshot.hasError) {
          return Center(child: Text('Error: ${snapshot.error}'));
        } else if (!snapshot.hasData || snapshot.data!.isEmpty) {
          return Center(child: Text('No data'));
        } else {
          return SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: Row(
              children: snapshot.data!.map((data) {
                return Column(
                  children: [
                    SizedBox(height: 25),
                    Row(
                      children: [
                        SizedBox(width: 20),
                        GestureDetector(
                          onTap: () {
                            Navigator.push(
                              context,
                              MaterialPageRoute(
                                builder: (contex) => RecomendPage(
                                  id: data.id,
                                ),
                              ),
                            );
                          },
                          child: Image.network(
                            data.imgUrl,
                            width: 400,
                            height: 250,
                            fit: BoxFit.cover,
                          ),
                        ),
                        SizedBox(width: 20)
                      ],
                    ),
                    SizedBox(height: 25),
                  ],
                );
              }).toList(),
            ),
          );
        }
      },
    );
  }
}

class TtfArea extends StatelessWidget {
  const TtfArea({super.key});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: EdgeInsets.only(left: 30.0),
      child: Container(
        width: 300,
        height: 300,
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(12.0),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.3),
              blurRadius: 10.0,
              offset: Offset(0, 5),
            ),
          ],
        ),
        child: Column(
          children: [
            Row(
              children: [
                SizedBox(
                  width: 150,
                  child: Column(
                    children: [
                      SizedBox(height: 25),
                      GestureDetector(
                        onTap: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (contex) => TtfPage(
                                ttfList: [Ttf(1, "楷书")],
                              ),
                            ),
                          );
                        },
                        child: Container(
                          width: 80,
                          height: 80,
                          decoration: BoxDecoration(
                            color: Color.fromRGBO(128, 197, 224, 0.24),
                            borderRadius: BorderRadius.circular(100.0),
                          ),
                          child: Image.asset('assets/images/1.webp'),
                        ),
                      ),
                      SizedBox(height: 5),
                      Text("楷书",
                          style: TextStyle(
                              fontSize: 25,
                              fontFamily: 'SimSun',
                              fontWeight: FontWeight.bold))
                    ],
                  ),
                ),
                SizedBox(
                  width: 150,
                  child: Column(
                    children: [
                      SizedBox(height: 25),
                      GestureDetector(
                        onTap: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (contex) => TtfPage(
                                ttfList: [Ttf(2, "行书")],
                              ),
                            ),
                          );
                        },
                        child: Container(
                          width: 80,
                          height: 80,
                          decoration: BoxDecoration(
                            color: Color.fromRGBO(128, 197, 224, 0.24),
                            borderRadius: BorderRadius.circular(100.0),
                          ),
                          child: Image.asset('assets/images/pen.webp'),
                        ),
                      ),
                      SizedBox(height: 5),
                      Text("行书",
                          style: TextStyle(
                              fontSize: 25,
                              fontFamily: 'SimSun',
                              fontWeight: FontWeight.bold))
                    ],
                  ),
                ),
              ],
            ),
            Row(
              children: [
                SizedBox(
                  width: 150,
                  child: Column(
                    children: [
                      SizedBox(height: 20),
                      GestureDetector(
                        onTap: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (contex) => TtfPage(
                                ttfList: [Ttf(3, "奶酪体")],
                              ),
                            ),
                          );
                        },
                        child: Container(
                          width: 80,
                          height: 80,
                          decoration: BoxDecoration(
                            color: Color.fromRGBO(128, 197, 224, 0.24),
                            borderRadius: BorderRadius.circular(100.0),
                          ),
                          child: Image.asset('assets/images/1.webp'),
                        ),
                      ),
                      SizedBox(height: 5),
                      Text("奶酪体",
                          style: TextStyle(
                              fontSize: 25,
                              fontFamily: 'SimSun',
                              fontWeight: FontWeight.bold))
                    ],
                  ),
                ),
                SizedBox(
                  width: 150,
                  child: Column(
                    children: [
                      SizedBox(height: 20),
                      GestureDetector(
                        onTap: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (contex) => TtfPage(
                                ttfList: [
                                  Ttf(0, "all"),
                                  Ttf(1, "楷书"),
                                  Ttf(2, "行书"),
                                  Ttf(3, "奶酪体")
                                ],
                              ),
                            ),
                          );
                        },
                        child: Container(
                          width: 80,
                          height: 80,
                          decoration: BoxDecoration(
                            color: Color.fromRGBO(128, 197, 224, 0.24),
                            borderRadius: BorderRadius.circular(100.0),
                          ),
                          child: Image.asset('assets/images/hua.webp'),
                        ),
                      ),
                      SizedBox(height: 5),
                      Text("更多",
                          style: TextStyle(
                              fontSize: 25,
                              fontFamily: 'SimSun',
                              fontWeight: FontWeight.bold))
                    ],
                  ),
                ),
              ],
            )
          ],
        ),
      ),
    );
  }
}

class CopyBookArea extends StatelessWidget {
  final List<CopyBookData> copybookData;

  const CopyBookArea({super.key, required this.copybookData});

  @override
  Widget build(BuildContext context) {
    // 获取父组件的宽度
    double parentWidth = MediaQuery.of(context).size.width;

    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SizedBox(
          height: 300,
          width: 360,
          child: Column(
            children: [
              SizedBox(
                height: 150, // 设置明确的高度
                child: ListView.builder(
                  scrollDirection: Axis.horizontal, // 水平滚动
                  itemCount: 3,
                  itemBuilder: (context, index) {
                    return SizedBox(
                      width: 120,
                      child: Column(
                        children: [
                          SizedBox(height: 15),
                          GestureDetector(
                            onTap: () => {
                              Navigator.push(
                                context,
                                MaterialPageRoute(
                                  builder: (contex) => CopybookPage(
                                      id: copybookData[index].id),
                                ),
                              ),
                            },
                            child: Image.network(
                              copybookData[index].imgUrl,
                              height: 120,
                              width: 90,
                              fit: BoxFit.cover,
                            ),
                          ),
                          SizedBox(height: 15),
                        ],
                      ),
                    );
                  },
                ),
              ),
              SizedBox(
                height: 150, // 设置明确的高度
                child: ListView.builder(
                  scrollDirection: Axis.horizontal, // 水平滚动
                  itemCount: 3,
                  itemBuilder: (context, index) {
                    return SizedBox(
                      width: 120,
                      child: Column(
                        children: [
                          SizedBox(height: 15),
                          GestureDetector(
                            onTap: () => {
                              Navigator.push(
                                context,
                                MaterialPageRoute(
                                  builder: (contex) => CopybookPage(
                                      id: copybookData[index + 3].id),
                                ),
                              ),
                            },
                            child: Image.network(
                              copybookData[index + 3].imgUrl,
                              height: 120,
                              width: 90,
                              fit: BoxFit.cover,
                            ),
                          ),
                          SizedBox(height: 15),
                        ],
                      ),
                    );
                  },
                ),
              ),
            ],
          ),
        ),
        Expanded(child: OrientationBuilder(
          builder: (context, orientation) {
            if (orientation == Orientation.portrait) {
              return Container(); // 当屏幕为竖直方向时不显示任何内容
            } else {
              return Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  SizedBox(
                    height: 300,
                    child: ListView.builder(
                        itemCount: copybookData.length - 6,
                        itemBuilder: (context, index) {
                          return Column(
                            children: [
                              Padding(
                                padding: EdgeInsets.only(left: 3.0),
                                child: GestureDetector(
                                  onTap: () => {
                                    Navigator.push(
                                      context,
                                      MaterialPageRoute(
                                        builder: (contex) => CopybookPage(
                                            id: copybookData[index + 6]
                                                .id),
                                      ),
                                    ),
                                  },
                                  child: Container(
                                    width: 340,
                                    height: 140,
                                    decoration: BoxDecoration(
                                      color: Colors.white,
                                      borderRadius: BorderRadius.circular(12.0),
                                      boxShadow: [
                                        BoxShadow(
                                          color: Colors.black.withOpacity(0.3),
                                          blurRadius: 10.0,
                                          offset: Offset(0, 5),
                                        ),
                                      ],
                                    ),
                                    child: Row(
                                      crossAxisAlignment:
                                          CrossAxisAlignment.start,
                                      children: [
                                        SizedBox(
                                          height: 140,
                                          child: Column(
                                            crossAxisAlignment:
                                                CrossAxisAlignment.start,
                                            children: [
                                              SizedBox(
                                                height: 10,
                                              ),
                                              Row(
                                                crossAxisAlignment:
                                                    CrossAxisAlignment.start,
                                                children: [
                                                  SizedBox(
                                                    width: 10,
                                                  ),
                                                  Image.network(
                                                    copybookData[index + 6]
                                                        .authorImgUrl,
                                                    width: 35,
                                                    height: 35,
                                                  ),
                                                  SizedBox(
                                                    width: 10,
                                                  ),
                                                  Column(
                                                    crossAxisAlignment:
                                                        CrossAxisAlignment
                                                            .start,
                                                    children: [
                                                      Text(
                                                        copybookData[index + 6]
                                                            .author,
                                                        style: TextStyle(
                                                            fontSize: 15,
                                                            fontWeight:
                                                                FontWeight
                                                                    .bold),
                                                      ),
                                                      Text(
                                                        copybookData[index + 6]
                                                            .time,
                                                        style: TextStyle(
                                                            fontSize: 10),
                                                      ),
                                                    ],
                                                  )
                                                ],
                                              ),
                                              SizedBox(
                                                height: 5,
                                              ),
                                              Row(
                                                crossAxisAlignment:
                                                    CrossAxisAlignment.start,
                                                children: [
                                                  SizedBox(
                                                    width: 55,
                                                  ),
                                                  Column(
                                                    crossAxisAlignment:
                                                        CrossAxisAlignment
                                                            .start,
                                                    children: [
                                                      SizedBox(
                                                        width: 180,
                                                        child: Text(
                                                          copybookData[
                                                                  index + 6]
                                                              .content,
                                                          maxLines: 3,
                                                          style: TextStyle(
                                                              fontSize: 15,
                                                              fontWeight:
                                                                  FontWeight
                                                                      .bold),
                                                        ),
                                                      ),
                                                      SizedBox(
                                                        width: 180,
                                                        child: Text(
                                                          "...",
                                                          maxLines: 1,
                                                          style: TextStyle(
                                                              fontSize: 15,
                                                              fontWeight:
                                                                  FontWeight
                                                                      .bold),
                                                        ),
                                                      )
                                                    ],
                                                  ),
                                                ],
                                              )
                                            ],
                                          ),
                                        ),
                                        SizedBox(
                                          width: 10,
                                        ),
                                        Column(
                                          children: [
                                            SizedBox(
                                              height: 10,
                                            ),
                                            Image.network(
                                              copybookData[index + 6].imgUrl,
                                              height: 120,
                                              width: 90,
                                              fit: BoxFit.cover,
                                            )
                                          ],
                                        )
                                      ],
                                    ),
                                  ),
                                ),
                              ),
                              SizedBox(height: 20)
                            ],
                          );
                        }),
                  )
                ],
              );
            }
          },
        )),
      ],
    );
  }
}

class RecomendData {
  int id = 0;
  String imgUrl = "";

  RecomendData(this.id, this.imgUrl);

  factory RecomendData.fromJson(Map<String, dynamic> json) {
    return RecomendData(
      json['id'],
      json['imgUrl'],
    );
  }
}

class CopyBookData {
  int id = 0;
  int authorId = 0;
  String author = "书画";
  String authorImgUrl = "http://dm.ecnucpp.cn:8080/1pen/user/0/portrait.webp";
  String imgUrl = "http://dm.ecnucpp.cn:8080/1pen/copybook/copybook.webp";
  String time = "2025-2-9";
  String content = "奶酪体新字帖";

  CopyBookData() {
    id = 0;
    authorId = 0;
    author = "书画";
    authorImgUrl = "http://dm.ecnucpp.cn:8080/1pen/user/0/portrait.webp";
    imgUrl = "http://dm.ecnucpp.cn:8080/1pen/copybook/copybook.webp";
    time = "2025-2-9";
    content =
        "奶酪体新字帖jhkhkhkjhjkghjfvjgvchchchgcthycthvjvsdfafargasadjakjgkjhahjkfbabfkagbkfhgajkhfjkahfkjahskjsfhaksjfha";
  }

  CopyBookData.withParams({
    required this.id,
    required this.authorId,
    required this.author,
    required this.authorImgUrl,
    required this.imgUrl,
    required this.time,
    required this.content,
  });
}
