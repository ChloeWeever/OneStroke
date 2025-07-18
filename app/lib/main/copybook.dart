import 'package:flutter/material.dart';

class CopybookPage extends StatelessWidget {
  final int id;
  final String title = "奶酪体新字帖";
  final int authorId = 0;
  final String authorName = " painter";
  final String imgUrl = "http://dm.ecnucpp.cn:8080/1pen/copybook/copybook.webp";
  final List<Comment> CommentList = [
    Comment(),
    Comment(),
    Comment(),
    Comment(),
    Comment(),
    Comment(),
    Comment(),
    Comment(),
    Comment(),
    Comment(),
  ];
  CopybookPage({super.key, required this.id}) {
    // TODO: get data from server
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(
          backgroundColor: Color.fromRGBO(42, 130, 228, 0.7),
        ),
        body: Center(
          child: OrientationBuilder(builder: (context, orientation) {
            if (MediaQuery.of(context).orientation == Orientation.portrait) {
              return Column(
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
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Padding(
                            padding: EdgeInsets.only(left: 30.0, top: 20),
                            child: Row(
                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                              children: [
                                Text(title,
                                    style: TextStyle(
                                        fontSize: 35,
                                        fontWeight: FontWeight.bold)),
                                Padding(
                                  padding: EdgeInsets.only(right: 30.0),
                                  child: Row(
                                    children: [
                                      ElevatedButton(
                                        style: ElevatedButton.styleFrom(
                                          backgroundColor: Colors.blue,
                                          foregroundColor: Colors.white,
                                        ),
                                        onPressed: () {
                                          // TODO: Add to notebook logic
                                        },
                                        child: Text(
                                          "加入本子",
                                          style: TextStyle(
                                              fontSize: 17,
                                              color: Colors.yellow,
                                              fontWeight: FontWeight.bold),
                                        ),
                                      ),
                                      ElevatedButton(
                                        style: ElevatedButton.styleFrom(
                                          backgroundColor: Colors.white,
                                          foregroundColor: Colors.blue,
                                          shape: CircleBorder(),
                                        ),
                                        onPressed: () {
                                          showModalBottomSheet(
                                            context: context,
                                            backgroundColor:
                                                const Color.fromARGB(
                                                    255, 203, 235, 250),
                                            builder: (BuildContext context) {
                                              return SizedBox(
                                                height: MediaQuery.of(context)
                                                        .size
                                                        .height *
                                                    0.6,
                                                child: Center(
                                                  child: Padding(
                                                    padding: EdgeInsets.all(8),
                                                    child: ListView.builder(
                                                        itemCount: CommentList
                                                            .length,
                                                        itemBuilder:
                                                            (context, index) {
                                                          return Column(
                                                            crossAxisAlignment:
                                                                CrossAxisAlignment
                                                                    .start,
                                                            children: [
                                                              if (index == 0)
                                                                Padding(
                                                                  padding:
                                                                      EdgeInsets
                                                                          .all(
                                                                              10),
                                                                  child:
                                                                      TextField(
                                                                    decoration:
                                                                        InputDecoration(
                                                                      hintText:
                                                                          "  请输入评论",
                                                                      border:
                                                                          OutlineInputBorder(
                                                                             borderRadius: BorderRadius.circular(50.0),
                                                                          ),
                                                                    ),
                                                                    onSubmitted:
                                                                        (value) {
                                                                      // TODO: 处理评论提交逻辑
                                                                    },
                                                                  ),
                                                                ),
                                                              Padding(
                                                                  padding: EdgeInsets
                                                                      .only(
                                                                          left:
                                                                              20.0,
                                                                          top:
                                                                              20),
                                                                  child: Row(
                                                                      children: [
                                                                        Image
                                                                            .network(
                                                                          CommentList[index]
                                                                              .authorImgUrl,
                                                                          height:
                                                                              50,
                                                                          width:
                                                                              50,
                                                                        ),
                                                                        SizedBox(
                                                                          width:
                                                                              13,
                                                                        ),
                                                                        Column(
                                                                          crossAxisAlignment:
                                                                              CrossAxisAlignment.start,
                                                                          children: [
                                                                            Row(
                                                                              children: [
                                                                                Text(CommentList[index].author,
                                                                                    style: TextStyle(
                                                                                      fontSize: 25,
                                                                                      fontWeight: FontWeight.w900,
                                                                                    )),
                                                                                SizedBox(
                                                                                  width: 6,
                                                                                ),
                                                                                if (index == 0)
                                                                                  Container(
                                                                                    decoration: BoxDecoration(
                                                                                      color: Colors.white,
                                                                                      border: Border.all(color: Colors.blue),
                                                                                      borderRadius: BorderRadius.circular(6.4),
                                                                                    ),
                                                                                    child: Padding(
                                                                                      padding: EdgeInsets.only(left: 8, right: 8, top: 1, bottom: 1),
                                                                                      child: Text(
                                                                                        "作者",
                                                                                        style: TextStyle(
                                                                                          fontSize: 16,
                                                                                          fontWeight: FontWeight.bold,
                                                                                          color: Colors.blue,
                                                                                        ),
                                                                                      ),
                                                                                    ),
                                                                                  ),
                                                                              ],
                                                                            ),
                                                                            Text(
                                                                              CommentList[index].time,
                                                                              style: TextStyle(color: Colors.grey),
                                                                            )
                                                                          ],
                                                                        ),
                                                                      ])),
                                                              Padding(
                                                                padding: EdgeInsets
                                                                    .only(
                                                                        left:
                                                                            83.0,
                                                                        top: 10,
                                                                        right:
                                                                            20),
                                                                child: Text(
                                                                    CommentList[
                                                                            index]
                                                                        .content,
                                                                    style: TextStyle(
                                                                        fontSize:
                                                                            20)),
                                                              ),
                                                              Padding(
                                                                padding:
                                                                    EdgeInsets
                                                                        .only(
                                                                  right: 10.0,
                                                                ),
                                                                child: Row(
                                                                  crossAxisAlignment:
                                                                      CrossAxisAlignment
                                                                          .end,
                                                                  children: [
                                                                    Expanded(
                                                                        child:
                                                                            Container()),
                                                                    ElevatedButton(
                                                                      style: ElevatedButton
                                                                          .styleFrom(
                                                                        backgroundColor:
                                                                            Colors.white,
                                                                        foregroundColor:
                                                                            Colors.blue,
                                                                        shape:
                                                                            CircleBorder(),
                                                                      ),
                                                                      onPressed:
                                                                          () {
                                                                        // TODO: Like logic
                                                                      },
                                                                      child:
                                                                          Icon(
                                                                        Icons
                                                                            .thumb_up,
                                                                        color: Colors
                                                                            .blue,
                                                                        size:
                                                                            20,
                                                                      ),
                                                                    ),
                                                                    ElevatedButton(
                                                                      style: ElevatedButton
                                                                          .styleFrom(
                                                                        backgroundColor:
                                                                            Colors.blue,
                                                                        foregroundColor:
                                                                            Colors.white,
                                                                        shape:
                                                                            CircleBorder(),
                                                                      ),
                                                                      onPressed:
                                                                          () {
                                                                        // TODO: Reply logic
                                                                      },
                                                                      child:
                                                                          Icon(
                                                                        Icons
                                                                            .comment_bank,
                                                                        color: Colors
                                                                            .white,
                                                                        size:
                                                                            20,
                                                                      ),
                                                                    ),
                                                                  ],
                                                                ),
                                                              ),
                                                              Divider(
                                                                height: 50,
                                                                color:
                                                                    Colors.grey,
                                                              ),
                                                            ],
                                                          );
                                                        }),
                                                  ),
                                                ),
                                              );
                                            },
                                          );
                                        },
                                        child: Icon(
                                          Icons.comment,
                                          color: Colors.blue,
                                          size: 20,
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ],
                            ),
                          ),
                          Expanded(
                              child: Center(
                            child: Image.network(
                              imgUrl,
                              fit: BoxFit.cover,
                              height: MediaQuery.of(context).size.height * 0.8,
                            ),
                          ))
                        ],
                      ),
                    ),
                  ))
                ],
              );
            } else {
              return Row(children: [
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
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Padding(
                          padding: EdgeInsets.only(left: 30.0, top: 20),
                          child: Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              Text(title,
                                  style: TextStyle(
                                      fontSize: 35,
                                      fontWeight: FontWeight.bold)),
                              Padding(
                                padding: EdgeInsets.only(right: 30.0),
                                child: Row(
                                  children: [
                                    ElevatedButton(
                                      style: ElevatedButton.styleFrom(
                                        backgroundColor: Colors.blue,
                                        foregroundColor: Colors.white,
                                      ),
                                      onPressed: () {
                                        // TODO: Add to notebook logic
                                      },
                                      child: Text(
                                        "加入本子",
                                        style: TextStyle(
                                            fontSize: 17,
                                            color: Colors.yellow,
                                            fontWeight: FontWeight.bold),
                                      ),
                                    ),
                                    ElevatedButton(
                                      style: ElevatedButton.styleFrom(
                                        backgroundColor: Colors.white,
                                        foregroundColor: Colors.blue,
                                        shape: CircleBorder(),
                                      ),
                                      onPressed: () {
                                        // TODO: Comment logic
                                      },
                                      child: Icon(
                                        Icons.comment,
                                        color: Colors.blue,
                                        size: 20,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ],
                          ),
                        ),
                        Expanded(
                            child: Center(
                          child: Image.network(
                            imgUrl,
                            fit: BoxFit.cover,
                            height: MediaQuery.of(context).size.height * 0.7,
                          ),
                        ))
                      ],
                    ),
                  ),
                )),
                Expanded(
                    child: Padding(
                  padding: EdgeInsets.all(8),
                  child: Container(
                    decoration: BoxDecoration(
                      color: Color.fromARGB(255, 203, 235, 250),
                      borderRadius: BorderRadius.circular(12.0),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withOpacity(0.3),
                          blurRadius: 10.0,
                          offset: Offset(0, 5),
                        ),
                      ],
                    ),
                    child: ListView.builder(
                        itemCount: CommentList.length,
                        itemBuilder: (context, index) {
                          return Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Padding(
                                  padding: EdgeInsets.only(left: 20.0, top: 20),
                                  child: Row(children: [
                                    Image.network(
                                      CommentList[index].authorImgUrl,
                                      height: 50,
                                      width: 50,
                                    ),
                                    SizedBox(
                                      width: 13,
                                    ),
                                    Column(
                                      crossAxisAlignment:
                                          CrossAxisAlignment.start,
                                      children: [
                                        Row(
                                          children: [
                                            Text(CommentList[index].author,
                                                style: TextStyle(
                                                  fontSize: 25,
                                                  fontWeight: FontWeight.w900,
                                                )),
                                            SizedBox(
                                              width: 6,
                                            ),
                                            if (index == 0)
                                              Container(
                                                decoration: BoxDecoration(
                                                  color: Colors.white,
                                                  border: Border.all(
                                                      color: Colors.blue),
                                                  borderRadius:
                                                      BorderRadius.circular(
                                                          6.4),
                                                ),
                                                child: Padding(
                                                  padding: EdgeInsets.only(
                                                      left: 8,
                                                      right: 8,
                                                      top: 1,
                                                      bottom: 1),
                                                  child: Text(
                                                    "作者",
                                                    style: TextStyle(
                                                      fontSize: 16,
                                                      fontWeight:
                                                          FontWeight.bold,
                                                      color: Colors.blue,
                                                    ),
                                                  ),
                                                ),
                                              ),
                                          ],
                                        ),
                                        Text(
                                          CommentList[index].time,
                                          style: TextStyle(color: Colors.grey),
                                        )
                                      ],
                                    ),
                                  ])),
                              Padding(
                                padding: EdgeInsets.only(
                                    left: 83.0, top: 10, right: 20),
                                child: Text(CommentList[index].content,
                                    style: TextStyle(fontSize: 20)),
                              ),
                              Padding(
                                padding: EdgeInsets.only(
                                  right: 10.0,
                                ),
                                child: Row(
                                  crossAxisAlignment: CrossAxisAlignment.end,
                                  children: [
                                    Expanded(child: Container()),
                                    ElevatedButton(
                                      style: ElevatedButton.styleFrom(
                                        backgroundColor: Colors.white,
                                        foregroundColor: Colors.blue,
                                        shape: CircleBorder(),
                                      ),
                                      onPressed: () {
                                        // TODO: Like logic
                                      },
                                      child: Icon(
                                        Icons.thumb_up,
                                        color: Colors.blue,
                                        size: 20,
                                      ),
                                    ),
                                    ElevatedButton(
                                      style: ElevatedButton.styleFrom(
                                        backgroundColor: Colors.blue,
                                        foregroundColor: Colors.white,
                                        shape: CircleBorder(),
                                      ),
                                      onPressed: () {
                                        // TODO: Reply logic
                                      },
                                      child: Icon(
                                        Icons.comment_bank,
                                        color: Colors.white,
                                        size: 20,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                              Divider(
                                height: 50,
                                color: Colors.grey,
                              ),
                            ],
                          );
                        }),
                  ),
                )),
              ]);
            }
          }),
        ));
  }
}

class Comment {
  int id = 0;
  int authorId = 0;
  String author = "书画";
  String authorImgUrl = "http://dm.ecnucpp.cn:8080/1pen/user/0/portrait.webp";
  String time = "2025-2-9";
  String content = "奶酪体新字帖";
  Comment() {
    id = 0;
    authorId = 0;
    author = "书画";
    authorImgUrl = "http://dm.ecnucpp.cn:8080/1pen/user/0/portrait.webp";
    time = "2025-2-9";
    content =
        "奶酪体新字帖jhkhkhkjhjkghjfvjgvchchchgcthycthvjvsdfafargasadjakjgkjhahjkfbabfkagbkfhgajkhfjkahfkjahskjsfhaksjfhaadfadfadfadfafafadfadfadfadfafadfadfadfadfadfadfafdafadfadfa";
  }

  Comment.withParams({
    required this.id,
    required this.authorId,
    required this.author,
    required this.authorImgUrl,
    required this.time,
    required this.content,
  });
}
