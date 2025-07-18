import 'package:flutter/material.dart';
import 'notebook/drawPad.dart';

class NoteBookPage extends StatelessWidget {
  List<Notebook> notebookList = [
    Notebook(),
    Notebook(),
    Notebook(),
  ];
  NoteBookPage({super.key}) {
    // TODO: get notebooklist from server
  }
  @override
  Widget build(BuildContext context) {
    return Scaffold(
        body: SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
              padding: EdgeInsets.only(left: 40, top: 30),
              child: Text(
                " 我的练字本",
                style: TextStyle(fontSize: 35, fontWeight: FontWeight.bold),
              )),
          OrientationBuilder(
            builder: (context, orientation) {
              return GridView.builder(
                shrinkWrap: true,
                physics: NeverScrollableScrollPhysics(),
                gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                  crossAxisCount: MediaQuery.of(context).orientation ==
                          Orientation.landscape
                      ? 4
                      : 3,
                  crossAxisSpacing: 10,
                  mainAxisSpacing: 10,
                ),
                itemCount: notebookList.length,
                itemBuilder: (context, copyBookIndex) {
                  return Padding(
                    padding: EdgeInsets.only(
                        right: 10, left: 10, top: 10, bottom: 10),
                    child: GestureDetector(
                        /*onTap: () {
                      Navigator.push(
                          context,
                          MaterialPageRoute(
                              builder: (context) => CopybookPage(
                                    id: this
                                        .ttfList[index]
                                        .copyBookList[copyBookIndex]
                                        .id,
                                  )));
                    },*/
                        child: Column(
                      children: [
                        GestureDetector(
                          onTap: () {
                            Navigator.push(
                                context,
                                MaterialPageRoute(
                                    builder: (context) => DrawPadPage()));
                          },
                          child: Image.network(
                            notebookList[copyBookIndex].imgUrl,
                            fit: BoxFit.cover,
                            width: MediaQuery.of(context).orientation ==
                                    Orientation.landscape
                                ? 170
                                : 150,
                            height: MediaQuery.of(context).orientation ==
                                    Orientation.landscape
                                ? 200
                                : 170,
                          ),
                        ),
                        Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            SizedBox(width: 40),
                            Text(
                              notebookList[copyBookIndex].title,
                              style: TextStyle(
                                  fontSize: 20, fontWeight: FontWeight.bold),
                            ),
                          ],
                        ),
                        Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            SizedBox(width: 40),
                            Text(
                              notebookList[copyBookIndex].createTime,
                              style: TextStyle(fontSize: 10),
                            ),
                          ],
                        )
                      ],
                    )),
                  );
                },
              );
            },
          ),
        ],
      ),
    ));
  }
}

class Notebook {
  int id = 0;
  String imgUrl = "http://dm.ecnucpp.cn:8080/1pen/notebook/notebook.webp";
  String createTime = "2025-1-27";
  String title = "我的练字本";
  Notebook() {
    title = title + id.toString();
  }
  Notebook.withParams({
    required this.id,
    required this.imgUrl,
    required this.createTime,
    required this.title,
  }) {
    id = id;
    imgUrl = imgUrl;
    createTime = createTime;
    title = title;
    if (title == "") {
      title = "我的练字本${this.id}";
    }
  }
}
