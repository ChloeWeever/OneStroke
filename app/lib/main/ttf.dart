import 'package:flutter/material.dart';
import 'copybook.dart';

class TtfPage extends StatelessWidget {
  List<Ttf> ttfList = [
    Ttf(0, "奶酪体"),
  ];
  TtfPage({super.key, required this.ttfList}) {
    // TODO: get ttf info
  }
  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(
          title: Text('精选字体'),
          backgroundColor: Color.fromRGBO(42, 130, 228, 0.7),
        ),
        body: ListView.builder(
            itemCount: ttfList.length,
            itemBuilder: (context, index) {
              return Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Padding(
                      padding: EdgeInsets.only(left: 40, top: 30),
                      child: Text(
                        ttfList[index].ttfName,
                        style: TextStyle(
                            fontSize: 35, fontWeight: FontWeight.bold),
                      )),
                  SizedBox(
                    height: 20,
                  ),
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
                        itemCount: ttfList[index].copyBookList.length,
                        itemBuilder: (context, copyBookIndex) {
                          return Padding(
                            padding: EdgeInsets.only(
                                right: 10, left: 10, top: 10, bottom: 10),
                            child: GestureDetector(
                              onTap: () {
                                Navigator.push(
                                    context,
                                    MaterialPageRoute(
                                        builder: (context) => CopybookPage(
                                              id: ttfList[index]
                                                  .copyBookList[copyBookIndex]
                                                  .id,
                                            )));
                              },
                              child: Image.network(
                                ttfList[index]
                                    .copyBookList[copyBookIndex]
                                    .imgUrl,
                              ),
                            ),
                          );
                        },
                      );
                    },
                  ),
                ],
              );
            }));
  }
}

class Ttf {
  int id = 0;
  String ttfName = "";
  List<CopyBook> copyBookList = [
    CopyBook(),
    CopyBook(),
    CopyBook(),
    CopyBook(),
    CopyBook(),
    CopyBook(),
    CopyBook(),
    CopyBook(),
    CopyBook(),
    CopyBook(),
  ]; // example
  Ttf(this.id, this.ttfName);
  Ttf.withParams({
    required this.id,
    required this.ttfName,
    required this.copyBookList,
  });
}

class CopyBook {
  int id = 0;
  String imgUrl = "http://dm.ecnucpp.cn:8080/1pen/copybook/copybook.webp";

  CopyBook() {
    id = 0;
    imgUrl = "http://dm.ecnucpp.cn:8080/1pen/copybook/copybook.webp";
  }

  CopyBook.withParams({
    required this.id,
    required this.imgUrl,
  });
}
