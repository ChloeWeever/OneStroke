import 'package:flutter/material.dart';

class ForumPage extends StatelessWidget {
  ForumPage({super.key});

  final List<ForumData> hotForum = [
    ForumData(
      id: 1,
      title: '论坛1',
      imageUrl: 'http://dm.ecnucpp.cn:8080/1pen/recomend/0/0.webp',
    ),
    ForumData(
      id: 2,
      title: '论坛2',
      imageUrl: 'http://dm.ecnucpp.cn:8080/1pen/recomend/0/0.webp',
    ),
    ForumData(
      id: 3,
      title: '论坛3',
      imageUrl: 'http://dm.ecnucpp.cn:8080/1pen/recomend/0/0.webp',
    ),
    ForumData(
      id: 4,
      title: '论坛4',
      imageUrl: 'http://dm.ecnucpp.cn:8080/1pen/recomend/0/0.webp',
    ),
    ForumData(
      id: 5,
      title: '论坛5',
      imageUrl: 'http://dm.ecnucpp.cn:8080/1pen/recomend/0/0.webp',
    ),
    ForumData(
      id: 6,
      title: '论坛6',
      imageUrl: 'http://dm.ecnucpp.cn:8080/1pen/recomend/0/0.webp',
    ),
    ForumData(
      id: 7,
      title: '论坛7',
      imageUrl: 'http://dm.ecnucpp.cn:8080/1pen/recomend/0/0.webp',
    ),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Row(
        children: [
          SizedBox(
            width: MediaQuery.of(context).orientation == Orientation.portrait ? MediaQuery.of(context).size.width * 0.9 : MediaQuery.of(context).size.width * 0.46,
            child: Column(
              children: [
                Expanded(
                  child: Padding(
                    padding: EdgeInsets.all(8),
                    child: Container(
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
                      child: const Center(child: Text('广告活动区')),
                    ),
                  ),
                ),
                Expanded(
                  child: Padding(
                    padding: EdgeInsets.all(8),
                    child: Container(
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
                              Padding(
                                padding: EdgeInsets.only(left: 30, top: 15),
                                child: Text(
                                  "热门论坛",
                                  style: TextStyle(
                                    fontSize: 30,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                              ),
                            ],
                          ),
                          SizedBox(height: 10),
                          Expanded(
                            child: GridView.builder(
                              gridDelegate:
                                  SliverGridDelegateWithFixedCrossAxisCount(
                                    crossAxisCount: 3,
                                    crossAxisSpacing: 8.0,
                                    mainAxisSpacing: 8.0,
                                  ),
                              itemCount: hotForum.length,
                              itemBuilder: (context, index) {
                                return Column(
                                  children: [
                                    ClipRRect(
                                      borderRadius: BorderRadius.circular(8.0),
                                      child: Image.network(
                                        hotForum[index].imageUrl,
                                        height: 80,
                                        width: 80,
                                        fit: BoxFit.cover,
                                      ),
                                    ),
                                    SizedBox(height: 10),
                                    Text(
                                      hotForum[index].title,
                                      style: TextStyle(
                                        fontSize: 15,
                                        fontWeight: FontWeight.bold,
                                      ),
                                    ),
                                  ],
                                );
                              },
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
          OrientationBuilder(
            builder: (context, orientation) {
              if (MediaQuery.of(context).orientation == Orientation.portrait) {
                return Container();
              }
              else {
                return SizedBox(
                  width: MediaQuery.of(context).size.width * 0.47,
                  child: Padding(
                    padding: EdgeInsets.all(8),
                    child: Container(
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
                      child: const Center(child: Text('Right')),
                    ),
                  ),
                );
              }
            },
          ),
        ],
      ),
    );
  }
}

class ForumData {
  final int id;
  final String title;
  final String imageUrl;

  ForumData({required this.id, required this.title, required this.imageUrl});
}
